import cv2, sys, utils, os, shutil, pprint
import numpy as np


def calc_neurite_area(image_pointer, percent_threshold):
    '''
    Reads in image converts it to mask based on percent_threshold.
    Counts number of white pixels in the generated mask.
    '''

    thresh = (1-float(percent_threshold))
    # img = cv2.imread(image_pointer, -1)
    img = cv2.imread(image_pointer, 0)
    ret, mask = cv2.threshold(
        img, int(img.max()*thresh), img.max(), cv2.THRESH_BINARY)
    neurite_area = sum(sum(mask==255))

    print 'Neurite area:', neurite_area
    return neurite_area

def add_area_to_csv(neurite_area, well, time, plateID, txt_f):
    '''
    Write well, time, Sci_PlateID, to csv.
    '''

    image_params = [plateID, well, well[0], well[1:], time[1:], str(neurite_area)] 

    txt_f.write(','.join(image_params))
    txt_f.write('\n')


def get_wells(all_files):
    '''
    Use appropriate well token to collect the ordered set of wells.
    '''

    wells = []
    for one_file in all_files:
        well = os.path.basename(one_file).split('_')[4]
        wells.append(well)
        wells = list(set(wells))
        wells.sort(key=utils.order_wells_correctly)

    print 'Wells:', wells
    return wells

def get_timepoints(all_files):
    '''
    Use appropriate timepoint token to collect the ordered set of timepoints.
    '''

    timepoints = []
    for one_file in all_files:
        well = os.path.basename(one_file).split('_')[2]
        timepoints.append(well)
        timepoints = sorted(list(set(timepoints)))

    print 'Timepoints:', timepoints
    return timepoints

def make_neurite_img_pointer(image_pointer):
    '''
    Takes given stack reference, splits it into single images.
    Returns pointer to neurite image.
    '''
    output_filenames_root = image_pointer.split('.')[0]
    output_filenames = output_filenames_root+'-%d.tif'
    utils.split_stack_magically(image_pointer, output_filenames)
    neurite_image_pointer = output_filenames_root+'-1.tif'
    return neurite_image_pointer

def clean_up_singles(neurite_image_pointer):
    '''
    Removes single files after treatment.
    '''
    single_image_root = os.path.basename(
        neurite_image_pointer).split('.')[0][:-1]
    single_image_path = os.path.dirname(neurite_image_pointer)
    single_files = utils.make_filelist(single_image_path, single_image_root)
    for img_file in single_files:
        os.remove(img_file)

def neurite_area_per_image(input_path, percent_threshold, write_path):
    '''
    Generates a csv with neurite areas for each probability image.
    '''

    all_files = utils.make_filelist(input_path, 'PID')
    wells = get_wells(all_files)
    timepoints = get_timepoints(all_files)
    plateID = '_'.join(os.path.basename(all_files[0]).split('_')[0:2])

    headers = [
        'PlateID','WellID','RowID', 'ColumnID', 
        'Timepoint', 'Neurite_Area']

    txt_f = open(os.path.join(write_path, 'neurite_areas.csv'), 'w')
    txt_f.write(','.join(headers))
    txt_f.write('\n')

    for well in wells:
        for time in timepoints:
            identifier = utils.make_selector(
                well=well, timepoint=time)
            image_files = utils.make_filelist(
                input_path, identifier)
            # print 'Number of files:', len(image_files)
            # pprint.pprint([os.path.basename(im) for im in image_files])
            assert len(image_files) <= 1, 'Multiple image files, selector is weak.'
            if len(image_files) == 0:
                continue
            image_pointer = image_files[0]
            neurite_image_pointer = make_neurite_img_pointer(image_pointer)
            # print 'Image pointer:', os.path.basename(image_pointer)
            # print 'Neurite image pointer:', os.path.basename(neurite_image_pointer)
            neurite_area = calc_neurite_area(
                neurite_image_pointer, percent_threshold)
            add_area_to_csv(neurite_area, well, time, plateID, txt_f)
            clean_up_singles(neurite_image_pointer)
    
    txt_f.close()


if __name__ == '__main__':

    input_path = sys.argv[1]
    percent_threshold = sys.argv[2]
    write_path = input_path
    outfile = sys.argv[3]

    neurite_area_per_image(input_path, percent_threshold, write_path)

    orig_outfile = os.path.join(write_path, 'neurite_areas.csv')
    new_outfile = os.path.join(write_path, 'neurite_areas2.csv')
    shutil.copyfile(orig_outfile, new_outfile)
    outfile = shutil.move(new_outfile, outfile)

