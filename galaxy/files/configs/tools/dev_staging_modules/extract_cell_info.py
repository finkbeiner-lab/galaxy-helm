import sys, os, cv2, pickle, argparse, utils
import numpy as np
import scipy.stats as stat
from tracking import Cell
import shutil, datetime
# import galaxy.tools.dev_staging_modules.utils as utils
# from galaxy.tools.dev_staging_modules.tracking import Cell
from utils import order_wells_correctly
import pprint, shutil, collections
# from utils import numericalSort
import warnings
import pandas as pd
# stats.skew throws nonsense runtimewarning on object 93 of image PID20150217_BioP7asynA_T0_0_F7_1_FITC-DFTrCy5_BG_MONTAGE_ALIGNED_CROPPED.tif
# while executing pixel_intensity_skewness = stats.skew(obj_img_cnt_intensities)
# Error message as folows:
# /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/scipy/stats/stats.py:993: RuntimeWarning:
# invalid value encountered in double_scalars
# vals = np.where(zero, 0, m3 / m2**1.5)
# In order to depress the error on history pane of Galaxy, suppress the RuntimeWarning message
warnings.filterwarnings("ignore", category=RuntimeWarning)

'''
Loop through dictionary and add each cell's parameters to tab-delim txt.
'''

# With a dictionary structure keyed on timepoint, with a list of (id, cnt) tuples.
# inputs:   time_dictionary = {'T0': [(cnt_ind, cell_obj), ...], 'T1': [], 'T2': [], ...'Tn': []}
#           file_list = utils.make_filelist(path, well_id)
#           where to write output

def get_cell_params(cell_records, time, well, txt_f, headers, read_path, var_dict):
    '''
    Initialize parameters and write to file.
    '''

    cnt_indices = [cell_record[0] for cell_record in cell_records if len(cell_record[1].cnt) > 5]
    cnt_params = [cell_record[1].calculate_cnt_parameters() for cell_record in cell_records if len(cell_record[1].cnt) > 5]

    iterator, iter_list = utils.set_iterator(var_dict)

    all_ch_int_stats = []
    ch_images = get_all_ch_images(read_path, time, well, var_dict)
    # pprint.pprint(ch_images)

    for cell_record in cell_records:
        if len(cell_record[1].cnt) <= 5:
            continue
        else:
            cell_record[1].collect_all_ch_intensities(ch_images)
            all_ch_int_stats.append(
                cell_record[1].all_ch_int_stats)
            # pprint.pprint(cell_record[1].all_ch_int_stats)

    for color in ch_images.keys():

        # Switched loop order to keep bursts together for objects.
        for cnt_ind, cnt_param, ch_int_stats in zip(
                cnt_indices, cnt_params, all_ch_int_stats):
            for frame in ch_images[color].keys():

                params = [str(cnt_param[header])
                    for header in headers[3:13]]

                params.insert(0, str(len(cnt_indices)))
                params.insert(1, str(cnt_ind))
                params.insert(2, str(color))
                intensities = [str(ch_int_stats[color][frame][header])
                    # for header in headers[11:28]]
                    for header in headers[headers.index('PixelIntensityMaximum'):headers.index('Sci_PlateID')]]
                params.extend(intensities)
                row = well[0]; col = well[1:]
                params.extend([var_dict['PlateID'], well, row, col, time[1:]])

                if iterator != None:
                    params.append(frame)

                txt_f.write(','.join(params))
                txt_f.write('\n')

    return txt_f

def extract_to_delim(write_path, time_dict, headers, txt_f, well, read_path, var_dict):
    '''Write all time points to a tab-delim text file.'''

    for time, cell_records in time_dict.items():
        print('For well', well, ':', 'Treating time point', time)
        get_cell_params(cell_records, time, well, txt_f, headers, read_path, var_dict)

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
    'BlobArea', 'BlobPerimeter', 'Radius', 'BlobCentroidX',
    'BlobCentroidY', 'BlobCentroidX_RefIntWeighted', 'BlobCentroidY_RefIntWeighted',
    'BlobCircularity', 'Spread', 'Convexity',
    'PixelIntensityMaximum', 'PixelIntensityMinimum',
    'PixelIntensityMean', 'PixelIntensityVariance',
    'PixelIntensityStdDev', 'PixelIntensity1Percentile',
    'PixelIntensity5Percentile','PixelIntensity10Percentile',
    'PixelIntensity25Percentile', 'PixelIntensity50Percentile',
    'PixelIntensity75Percentile', 'PixelIntensity90Percentile',
    'PixelIntensity95Percentile', 'PixelIntensity99Percentile',
    'PixelIntensitySkewness', 'PixelIntensityKurtosis',
    'PixelIntensityInterquartileRange', 'PixelIntensityTotal', 'Sci_PlateID',
    'Sci_WellID', 'RowID', 'ColumnID', 'Timepoint']

    # Add frame header if relevant
    iterator, iter_list  = utils.set_iterator(var_dict)
    if iterator != None:
        headers.append(iterator)

    txt_f = open(os.path.join(write_path, 'cell_data.csv'), 'w')
    # Header: column names
    txt_f.write(','.join(headers))
    txt_f.write('\n')

    for well in var_dict['Wells']:

        try:
            time_dictionary = var_dict['TrackedCells'][well]
            # print 'Time dictionary for', well
            # pprint.pprint(time_dictionary.items())
        except KeyError:
            print("Skipping well", well, "(no files were found).")
        else:
            extract_to_delim(
                write_path, time_dictionary, headers, txt_f, well, read_path, var_dict)

    txt_f.close()


def merge_elapsed_hours(var_dict, write_path):
    cell_data_path = os.path.join(write_path, 'cell_data.csv')
    cell_data = pd.read_csv(cell_data_path)
    timepoint_hours = pd.DataFrame(
            {'Timepoint': [int(x.replace('T', '')) for x in var_dict['TimePoints']],
                           'ElapsedHours': var_dict['ElapsedHours']})
    cell_data = pd.merge(cell_data, timepoint_hours, on='Timepoint', how='left')
    cell_data.to_csv(cell_data_path, index=False)


if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Extract intensity info from original images.")
    parser.add_argument("input_dict",
        help="Load input variable dictionary", default = '/mnt/finkbeinerlab/robodata/zach/input_dict')
    parser.add_argument("--input_image_path",
        help="Folder path to input data.", default = '')
    parser.add_argument("--output_results_path",
        help="Folder path to ouput results.", default = '')
    parser.add_argument("output_dict",
        help="Write variable dictionary.")
    args = parser.parse_args()

    # ----Load path dict-------------------------
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))

    # ----Initialize parameters------------------
    read_path = utils.get_path(args.input_image_path, var_dict['GalaxyOutputPath'], 'AlignedImages')
    print('Input path: %s' % read_path)

    write_path = utils.get_path(args.output_results_path, var_dict['GalaxyOutputPath'], '')
    print('Output path: %s' % write_path)
    outfile = args.output_dict
    var_dict['Wells'].sort(key=utils.order_wells_correctly)

    # ----Handle correct output is passed in-----
    try: var_dict['TrackedCells']
    except KeyError: print('Confirm that result from cell tracking step is passed in to local variabes dictionary.')
    # raise

    # ----Confirm given folders exist--
    assert os.path.exists(read_path), 'Confirm the given path for data exists.'
    assert os.path.exists(write_path), 'Confirm the given path for results exists.'

    # ----Run extract cell info------------------
    start_time = datetime.datetime.utcnow()

    extract_cell_info(var_dict, read_path, write_path)

    try:
        merge_elapsed_hours(var_dict, write_path)
    except:
        pass

    end_time = datetime.datetime.utcnow()
    print('Extract cell info run time:', end_time-start_time)
    # ----Output for user and save dict----------

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
    timestamp = utils.update_timestring()
    utils.save_user_args_to_csv(args, write_path, 'extract'+'_'+timestamp)



