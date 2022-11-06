import sys, os, cv2, utils, pickle, argparse
import numpy as np
import scipy.stats as stat
from tracking_extra import Cell
import shutil, datetime
from utils import order_wells_correctly
import pprint, shutil, collections
from utils import numericalSort

'''
Loop through dictionary and add each cell's parameters to tab-delim txt.
'''

header_line = True
line_counter = 0

# With a dictionary structure keyed on timepoint, with a list of (id, cnt) tuples.
# inputs:   time_dictionary = {'T0': [(cnt_ind, cell_obj), ...], 'T1': [], 'T2': [], ...'Tn': []}
#           file_list = utils.make_filelist(path, well_id)
#           where to write output

def get_cell_params(cell_records, time, well, txt_f, headers, read_path, var_dict):
    '''
    Initialize parameters and write to file.
    cell_record[Time] = [cnt_index, cnt_instance]
    '''

    global header_line
    global line_counter

    shape_features = headers[headers.index('BlobArea'):headers.index('Convexity')+1]
    intensity_features = headers[headers.index('PixelIntensityTotal'):headers.index('Sci_PlateID')]
    # descriptor_features = headers[headers.index('Descriptors')]
    pixels_features = headers[headers.index('CellPatchPixels')]

    cnt_indices = [cell_record[0] for cell_record in cell_records if len(cell_record[1].cnt) > 5]
    cnt_params = [cell_record[1].calculate_cnt_parameters() for cell_record in cell_records if len(cell_record[1].cnt) > 5]

    # print 'Index, param examples:', cnt_indices[0], cnt_params[0]

    iterator, iter_list = utils.set_iterator(var_dict)

    all_ch_int_stats = []
    ch_images = get_all_ch_images(read_path, time, well, var_dict)
    # pprint.pprint(ch_images)

    for cell_record in cell_records:
        # print "This is the cell_record:", cell_record
        if len(cell_record[1].cnt) <= 5:
            continue
        else:
            try: cell_record[1].collect_all_ch_intensities(ch_images)
            except IndexError:
                print '-Will be excluded (index, time, well):', cell_record, time, well
                continue
            all_ch_int_stats.append(
                cell_record[1].all_ch_int_stats)
            # pprint.pprint(cell_record[1].all_ch_int_stats)

    for color in ch_images.keys():

        # Switched loop order to keep bursts together for objects.
        for cnt_ind, cnt_param, ch_int_stats in zip(
                cnt_indices, cnt_params, all_ch_int_stats):
            for frame in ch_images[color].keys():

                params = [str(cnt_param[header])
                    for header in shape_features]

                params.insert(0, str(len(cnt_indices)))
                params.insert(1, str(cnt_ind))
                params.insert(2, str(color))
                intensities = [str(ch_int_stats[color][frame][header])
                    for header in intensity_features]
                params.extend(intensities)
                row = well[0]; col = well[1:]
                params.extend([var_dict['PlateID'], well, row, col, time[1:]])

                # For returning to the descriptor work
                # descriptor = [str(item) for item in ch_int_stats[color][frame]['Descriptors']]
                # params.extend(descriptor)
                cell_patch = [str(item) for item in ch_int_stats[color][frame]['CellPatchPixels']]
                params.extend(cell_patch)

                if iterator != None:
                    params.append(frame)

                if header_line:
                    # Header: column names
                    num_pixels = map(str, range(1,len(cell_patch)))
                    pixel_names = ['Pixel'+nu for nu in num_pixels]
                    txt_f.write(','.join(headers+pixel_names))
                    txt_f.write('\n')
                    header_line = False

                txt_f.write(','.join(params))
                txt_f.write('\n')
                line_counter = line_counter + 1

    return txt_f

def extract_to_delim(write_path, time_dict, headers, txt_f, well, read_path, var_dict):
    '''Write all time points to a tab-delim text file.'''

    for time, cell_records in time_dict.items():
        print 'For well', well, ':', 'Treating time point', time
        print "Number of records:", len(cell_records)
        if len(cell_records)<1:
            continue
        get_cell_params(cell_records, time, well, txt_f, headers, read_path, var_dict)
    # get_cell_params(time_dict[var_dict['TimePoints'][0]], "T0", well, txt_f, headers, read_path, var_dict)


    # pprint.pprint(time_dict)

def get_all_ch_images(read_path, time, well, var_dict):
    '''
    Using the var_dict iterators (channels and bursts or depths)
    Gets all chanels and frames associated with a particular timepoint and well.
    Returns a dictionary: ch_images[color][frame] = image.
    This dictionary will be passed to Cell object to collect all intensity statistics.
    '''

    iterator, iter_list = utils.set_iterator(var_dict)
    # print iterator, iter_list

    # ch_images = {}
    ch_images = collections.OrderedDict()
    for channel in var_dict['Channels']:

        selector = utils.make_selector(well=well, timepoint=time, channel=channel)
        # image_pointers = utils.make_filelist(read_path, selector)
        image_pointers = utils.make_filelist_wells(read_path, selector)
        # print len(image_pointers), selector

        if len(image_pointers) == 0:
            # print 'No images were found for:', selector
            continue
        elif len(image_pointers) == 1:
            frame = ''
            ch_images[channel] = collections.OrderedDict()
            assert len(image_pointers)<=1, 'More than one image per timepoint, well, channel, frame.'
            ch_images[channel][frame] = cv2.imread(image_pointers[0], -1)

        # Keep architecture constant for frames or no frames
        # For RAM caution:
        # assert len(image_pointers)<500, 'Caution: May run into RAM constraints.'

        else:
            # ch_images[channel] = {}
            ch_images[channel] = collections.OrderedDict()
            for frame in iter_list:
                if frame == '0':
                    continue
                selector = utils.make_selector(
                    iterator=iterator, well=well, timepoint=time, channel=channel, frame=frame)
                # burst_images = utils.make_filelist(read_path, selector)
                burst_images = utils.make_filelist_wells(read_path, selector)
                assert len(burst_images)<=1, 'More than one image per timepoint, well, channel, frame.'

                if len(burst_images) == 0:
                    # print 'No frames/images were found for:', selector
                    continue

                ch_images[channel][frame] = cv2.imread(burst_images[0], -1)

        # Quality control
        # pprint.pprint(ch_images)

    return ch_images


def extract_cell_info(var_dict, read_path, write_path):
    '''
    Main point of entry.
    '''

    headers = ['ObjectCount', 'ObjectLabelsFound', 'MeasurementTag',
    'BlobArea','Extent','AspectRatio','BlobPerimeter',
    'Radius','BlobCentroidX','BlobCentroidY','BlobCircularity',
    'Angle','MajorAxis','MinorAxis','Spread','Convexity',
    'PixelIntensityTotal','PixelIntensityMinimum','PixelIntensityMaximum',
    'PixelIntensityMean','PixelIntensityStdDev','PixelIntensityVariance',
    'PixelIntensityMeanSD','PixelIntensityRangeSD','PixelIntensity1Percentile',
    'PixelIntensity5Percentile','PixelIntensity10Percentile','PixelIntensity25Percentile',
    'PixelIntensity50Percentile','PixelIntensity75Percentile','PixelIntensity90Percentile',
    'PixelIntensity95Percentile','PixelIntensity99Percentile','PixelIntensityInterquartileRange',
    'PixelIntensitySkewness','PixelIntensityKurtosis','EdgeIntensityTotal',
    'EdgeIntensityMaximum','EdgeIntensityMinimum','EdgeIntensityMean','EdgeIntensityVariance',
    'EdgeIntensityStdDev','EdgeIntensityMeanSD','EdgeIntensityRangeSD', 'EdgeIntensitySkewness',
    'EdgeIntensityKurtosis','CellCenterIntensity','LineIntensityTotal','LineIntensityMaximum',
    'LineIntensityMinimum','LineIntensityMean','LineIntensityVariance','LineIntensityStdDev',
    'LineIntensityMeanSD','LineIntensityRangeSD','LineIntensitySkewness', 'LineIntensityKurtosis',
    'LineIntensitySlope','LineIntensityRsquared','LineIntensity25PercentRaw',
    'LineIntensity50PercentRaw','LineIntensity75PercentRaw','LineIntensity99PercentRaw',
    'LineIntensity25PercentCum','LineIntensity50PercentCum','LineIntensity75PercentCum',
    'LineIntensity99PercentCum', 'HuOne', 'HuTwo', 'HuThree', 'HuFour', 'HuFive', 'HuSix', 'HuSeven',
    'GaborMean','GaborVariance','ScharrMean','ScharrVariance',
    'HesianXXMean','HesianXXVariance','HesianXYMean','HesianXYVariance','HesianYYMean',
    'HesianYYVariance','DohMean','DohVariance','LaplacianMean','LaplacianVariance',
    'EntropyMean','EntropyVariance',
    # 'CorrMaxX', 'CorrMaxY',
    'Sci_PlateID','Sci_WellID', 'RowID', 'ColumnID', 'Timepoint',
    # 'Descriptors' # For returning to the descriptor work
    'CellPatchPixels'
    ]

    # Add frame header if relevant
    iterator, iter_list  = utils.set_iterator(var_dict)
    if iterator != None:
        headers.append(iterator)

    txt_f = open(os.path.join(write_path, 'cell_data.csv'), 'w')
    # # Header: column names
    # txt_f.write(','.join(headers))
    # txt_f.write('\n')

    for well in var_dict['Wells']:

        time_dictionary = var_dict['TrackedCells'][well]
        # print 'Time dictionary for', well
        # pprint.pprint(time_dictionary.items())
        extract_to_delim(
            write_path, time_dictionary, headers, txt_f, well, read_path, var_dict)

    txt_f.close()

if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Extract intensity info from original images.")
    parser.add_argument("input_dict",
        help="Load input variable dictionary")
    parser.add_argument("input_path",
        help="Folder path to input data.")
    parser.add_argument("output_path",
        help="Folder path to ouput results.")
    parser.add_argument("output_dict",
        help="Write variable dictionary.")
    args = parser.parse_args()

    # ----Load path dict-------------------------
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))

    # ----Initialize parameters------------------
    read_path = args.input_path
    write_path = args.output_path
    outfile = args.output_dict
    var_dict['Wells'].sort(key=order_wells_correctly)

    # Initiate STAR detector
    star = cv2.FeatureDetector_create("STAR")
    # Initiate BRIEF extractor
    brief = cv2.DescriptorExtractor_create("BRIEF")

    # ----Handle correct output is passed in-----
    try: var_dict['TrackedCells']
    except KeyError: print 'Confirm that result from cell tracking step is passed in to local variabes dictionary.'
    # raise

    # ----Confirm given folders exist--
    assert os.path.exists(read_path), 'Confirm the given path for data exists.'
    assert os.path.exists(write_path), 'Confirm the given path for results exists.'

    # ----Run extract cell info------------------
    start_time = datetime.datetime.utcnow()

    extract_cell_info(var_dict, read_path, write_path)

    end_time = datetime.datetime.utcnow()
    print 'Extract cell info run time:', end_time-start_time
    # ----Output for user and save dict----------
    print 'Overlays were created for each time point.'
    print 'Output was written to:'
    print write_path
    print 'Number of lines in CSV:', line_counter

    # Handles sending output to Galaxy viewer, which is expensive and not used
    # outfile = 'outfile'
    # orig_outfile = os.path.join(write_path, 'cell_data.csv')
    # new_outfile = os.path.join(write_path, 'cell_data2.csv')
    # shutil.copyfile(orig_outfile, new_outfile)
    # outfile = shutil.move(new_outfile, outfile)
    # Returns the dictionary, as with other modules
    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    # outfile = os.rename('var_dict.p', outfile)
    outfile = shutil.move('var_dict.p', outfile)
    utils.save_user_args_to_csv(args, write_path, 'extract_extra')


