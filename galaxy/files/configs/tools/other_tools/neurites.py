'''
Takes two images: 
1. Raw image 
2. Foreground image (mike's probability output)
Returns a difference image: foreground image - the segmented cells (and/or debris) image. 
This segmentation step is different than the one producing CELLMASK to capture debris.

@Usage
Takes path to probability images and path to segmentation images.
Takes path to write mask and csv output.
Reads in probability image, thresholds at midpoint to keep foreground.
Subtracts cell-segmented mask from foreground image.
Returns difference image to output path.
Sums number of white pixels in the neurite mask.
'''

import cv2, utils, sys, os, argparse
import numpy as np
import pickle, datetime, shutil, pprint
import datetime

def find_cells(img, percent_int_thresh=0.1, img_is_mask=False):
    '''
    Resets depth of image to 'uint8'.
    Finds contours in given image.
    '''

    ret, mask = cv2.threshold(
        img, int(img.max()*percent_int_thresh), img.max(), cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_contours(contours, small=10, large=3500, ecn=.1, verbose=True):
    '''
    Filters contours based on size and eccentricity.
    Will learn other important parameters.
    '''

    contours_kept = []
    for cnt in contours:
        if len(cnt) > 5 and cv2.contourArea(cnt) > small \
            and cv2.contourArea(cnt) < large:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            ecc = np.sqrt(1-((MA)**2/(ma)**2))
            if ecc > ecn:
                contours_kept.append(cnt)

    if verbose:
        print 'Kept', len(contours_kept), \
            '/', len(contours), 'contours.'

    return contours_kept

def make_cell_mask(img_pointer, contours_kept):
    '''
    Draw kept cells onto image.
    '''
    
    img = cv2.imread(img_pointer, 0)
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    new_name = utils.extract_file_name(img_pointer)+'_CELLSMASK'

    cv2.drawContours(mask, contours_kept, -1, 255, -1)
    cv2.drawContours(mask, contours_kept, -1, 255, 5)

    return new_name, mask

def make_neurite_mask_get_sum_pixels(img_pointer, mask_cells, mask_foreground):
    '''
    Subtracts two masks. Sums total pixels in difference mask.
    '''

    diff_mask = np.subtract(mask_foreground, mask_cells)
    new_name = utils.extract_file_name(img_pointer)+'_NEURITEMASK'
    neurite_area = sum(sum(diff_mask==255))

    return new_name, diff_mask, neurite_area

def add_area_to_csv(neurite_area, well, time, plateID, txt_f):
    '''
    Write well, time, Sci_PlateID, to csv.
    '''

    image_params = [plateID, well, well[0], well[1:], time[1:], str(neurite_area)] 

    txt_f.write(','.join(image_params))
    txt_f.write('\n')

def write_neurite_area_to_csv(raw_img_path, mask_path, fg_img_path, verbose=False, mask=False):
    '''
    Generates a csv with neurite areas for each probability image.
    '''

    all_files = utils.make_filelist(raw_img_path, 'PID')
    wells = utils.get_wells(all_files)
    timepoints = utils.get_timepoints(all_files)
    plateID = '_'.join(os.path.basename(all_files[0]).split('_')[0:2])

    headers = [
        'PlateID','WellID','RowID', 'ColumnID', 
        'Timepoint', 'Neurite_Area']

    txt_f = open(os.path.join(mask_path, 'neurite_areas.csv'), 'w')
    txt_f.write(','.join(headers))
    txt_f.write('\n')

    for well in wells:
        for time in timepoints:
            identifier = utils.make_selector(
                well=well, timepoint=time)
            raw_img_files = utils.make_filelist(
                raw_img_path, identifier)
            fg_img_files = utils.make_filelist(
                fg_img_path, identifier)
            if verbose:
                print 'Number of raw files:', len(raw_img_files)
                pprint.pprint([os.path.basename(im) for im in raw_img_files])
                print 'Number of foreground files:', len(fg_img_files)
                pprint.pprint([os.path.basename(im) for im in fg_img_files])
            assert len(raw_img_files) <= 1, 'Multiple image files, selector is weak.'
            assert len(fg_img_files) <= 1, 'Multiple image files, selector is weak.'
            if len(raw_img_files) == 0:
                continue
            raw_img_pointer = raw_img_files[0]
            if len(raw_img_files) == 0:
                continue
            fg_img_pointer = fg_img_files[0]
            
            if mask:
                mask_cells = cv2.imread(raw_img_pointer, 0)
            else:
                raw_img = cv2.imread(raw_img_pointer, 0)
                contours = find_cells(raw_img)
                contours_kept = filter_contours(contours)
                mask_name, mask_cells = make_cell_mask(raw_img_pointer, contours_kept)
                mask_pointer_name = utils.make_file_name(mask_path, mask_name)
                cv2.imwrite(mask_pointer_name, mask_cells)

            foreground_img = cv2.imread(fg_img_pointer, 0)
            ret, mask_foreground = cv2.threshold(
                foreground_img, foreground_img.max()-10, 
                foreground_img.max(), cv2.THRESH_BINARY)
            neur_mask_name, neurite_mask, neurite_area = make_neurite_mask_get_sum_pixels(
                raw_img_pointer, mask_cells, mask_foreground)
            neur_mask_pointer_name = utils.make_file_name(mask_path, neur_mask_name)
            cv2.imwrite(neur_mask_pointer_name, neurite_mask)
            add_area_to_csv(neurite_area, well, time, plateID, txt_f)
    
    txt_f.close()


if __name__ == '__main__':

    # ----Parse arguments--------------
    parser = argparse.ArgumentParser(
        description="Find number of intersections per junction.")
    parser.add_argument("input_path", 
        help="Folder path to cell masks or raw data.")
    parser.add_argument("output_path", 
        help="Folder path to write output difference masks and csvs.")
    parser.add_argument("foreground_path", 
        help="Folder path to foreground images.")
    parser.add_argument("output_file", 
        help="Place holder.")
    args = parser.parse_args()

    # ----I/O--------------------------
    raw_path = args.input_path
    mask_path = args.output_path
    fg_bg_path = args.foreground_path
    if not os.path.exists(mask_path):
        os.makedirs(mask_path)

    # ----Confirm given folders exist--
    assert os.path.exists(raw_path), 'Confirm the given path for raw image data exists.'
    assert os.path.exists(mask_path), 'Confirm the given path for cell masks and csv output exists.'
    assert os.path.exists(fg_bg_path), 'Confirm the given path for foreground/background images exists.'

    # ----Run neurite counter--------------------
    start_time = datetime.datetime.utcnow()

    write_neurite_area_to_csv(raw_path, mask_path, fg_bg_path)

    end_time = datetime.datetime.utcnow()
    print 'Neurite-counting run time:', end_time-start_time

    print 'Cell masks, neurite masks, and csv output can be found in:'
    print mask_path

