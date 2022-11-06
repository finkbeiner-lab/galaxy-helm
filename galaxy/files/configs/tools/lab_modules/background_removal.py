'''
Functions dealing with background subtraction.
The main function is background_removal().
Takes paths and information about raw images.
Return background-corrected images and background images for QC.
'''

import cv2, sys, os, utils
import numpy as np
import argparse, pickle
import datetime, pprint, shutil


def calc_median_img(well_time_channel_list, resolution):
    '''Creates image of smae dimensions as source and contains
    well_time_channel_list median at each pixel.'''
    img_list = [cv2.imread(img_pointer, resolution)
        for img_pointer in well_time_channel_list]
    median_img = np.median(np.array(img_list), axis=0)

    return median_img

def divide_median(img, well_median_img):
    '''
    Divides the well_median_value from each pixel value in image.
    '''
    div_img = 500*(img / well_median_img)
    if div_img.max() > 2**16:
        factor = ((2**16)-1) / div_img.max()
        div_img = factor * (img / well_median_img)

    # max_med = well_median_img.max()
    # div_img = 1000*(((img - well_median_img)+max_med)/well_median_img)
    # print 'Max div img', div_img.max()

    return div_img

def med_correct_bg(well_images, dest_path, resolution, well_median_img, bg_rm_type):
    '''Correct image by subtracting median image. Write to disk.'''

    assert len(well_images) != 0, 'No images available.'
    for well_image in well_images:
        # Set up the image
        img = cv2.imread(well_image, resolution)

        img_name = utils.extract_file_name(well_image)
        bg_img_name = utils.make_file_name(
            dest_path, img_name+'_BG')

        if bg_rm_type == 'division':
            corrected_img = divide_median(img, well_median_img)
            bg_img_name = utils.make_file_name(
                dest_path, img_name+'_BGd')
        elif bg_rm_type == 'subtraction':
            well_median_img = well_median_img.astype(np.uint16)
            corrected_img = cv2.subtract(img, well_median_img)
            bg_img_name = utils.make_file_name(
                dest_path, img_name+'_BGs')
        else:
            print "Correction type is not recognized."

        assert corrected_img.min() >= 0, corrected_img.min()
        assert corrected_img.max() <= 2**16, corrected_img.max()
        corrected_img_16 = corrected_img.astype(np.uint16)
        cv2.imwrite(bg_img_name, corrected_img_16)
    return img_name


def subtract_bg(well_images, resolution, dest_path, qc_dest_path, bg_rm_type):
    '''
    Subtracts background from each set of images.
    '''
    # Subtract background from each channel
    well_median_img = calc_median_img(well_images, resolution)
    # Saving the background image
    img_name = med_correct_bg(
        well_images, dest_path, resolution, well_median_img, bg_rm_type)
    split_name = img_name.split('_')
    # 0 = PID, 1 = experiment, 2 = timepoint, 4 = well, 6 = channel
    img_id = [split_name[n] for n in [0, 1, 2, 4, 6]]
    bg_img_name = utils.make_file_name(
        qc_dest_path, '_'.join(img_id)+'_BGMEDIAN')
    bg_img_img = well_median_img.astype(np.uint16)
    cv2.imwrite(bg_img_name, bg_img_img)

def background_removal(var_dict, source_path, dest_path, qc_dest_path, bg_rm_type, verbose=False):
    '''Main point of entry'''
    resolution = var_dict['Resolution']

    # Select an iterator, if needed
    iterator, iter_list = utils.set_iterator(var_dict)
    print 'What will be iterated:', iterator
    if iterator=='TimeBursts':
        token_num = 3
    elif iterator=='ZDepths':
        token_num = 8

    num_panels = var_dict["NumberHorizontalImages"] * var_dict["NumberVerticalImages"]

    for well in var_dict['Wells']:
        for timepoint in var_dict['TimePoints']:
            for channel in var_dict['Channels']:
                print 'Working on (well, timepoint, channel):', well, timepoint, channel

                # Get relevant images
                if var_dict['RoboNumber']==0:
                    all_well_images = utils.get_selected_files(
                        source_path, well, timepoint, channel, robonum=0)

                    if verbose:
                        print 'Well-Channel-Time-Filtered Files'
                        pprint.pprint([os.path.basename(img_pntr) for img_pntr in all_well_images])

                else:
                    selector = utils.make_selector(
                        iterator=iterator, well=well, timepoint=timepoint, channel=channel)
                    if channel == 'Cy5':
                        selector = timepoint+'_*'+well+'_*'+'_'+channel
                    all_well_images = utils.make_filelist(source_path, selector)

                    if verbose:
                        print 'Images used:'
                        pprint.pprint([os.path.basename(wellimg) for wellimg in all_well_images])

                # Handles no images
                if len(all_well_images) == 0:
                    continue

                elif len(all_well_images)%num_panels != 0:
                    # print 'Selector generated:', selector,
                    print 'Number of images not factor of number of panels.'
                    print 'Number of images found:', len(all_well_images), 'not', num_panels
                    print 'Images used:'
                    pprint.pprint([os.path.basename(wellimg) for wellimg in all_well_images])

                # Handles null iterators (no bursts or depths)
                elif len(all_well_images) == num_panels:
                    subtract_bg(all_well_images, resolution, dest_path, qc_dest_path, bg_rm_type)

                # Handles iterators (bursts or depths)
                elif len(all_well_images) > num_panels and len(all_well_images)%num_panels == 0:

                    for frame in iter_list:
                        print 'Current frame:', frame
                        # Get relevant images
                        if var_dict['RoboNumber']==0:
                            well_images = utils.get_frame_files(
                                all_well_images, frame, token_num)
                            print "Files that will be used to generate median image with", iterator, frame
                            pprint.pprint([os.path.basename(fname) for fname in well_images])

                        else:
                            selector = utils.make_selector(
                                iterator, well, timepoint, channel, frame)
                            well_images = utils.make_filelist(source_path, selector)

                        if len(well_images) == 0:
                            continue

                        subtract_bg(well_images, resolution, dest_path, qc_dest_path, bg_rm_type)

                # Catch other
                else:
                    print 'Warning: All images will be used to calculate median image (might be many)'
                    subtract_bg(all_well_images, resolution, dest_path, qc_dest_path, bg_rm_type)


    # For select_analysis_module input, set var_dict['OutputPath']
    var_dict["OutputPath"] = dest_path


if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Correct background uneveness.")
    parser.add_argument("input_dict",
        help="Load input variable dictionary")
    parser.add_argument("input_path",
        help="Folder path to input data.")
    parser.add_argument("output_path",
        help="Folder path to ouput results.")
    parser.add_argument("qc_path",
        help="Folder path to qc of results.")
    parser.add_argument("bg_removal_type",
        help="division or subtraction")
    parser.add_argument("output_dict",
        help="Write variable dictionary.")
    args = parser.parse_args()

    # ----Load path dict-------------------------
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))

    # ----Initialize parameters------------------
    raw_image_data = args.input_path
    write_path = args.output_path
    bg_img_path = args.qc_path
    bg_rm_type = args.bg_removal_type
    outfile = args.output_dict

    # ----Confirm given folders exist--
    assert os.path.exists(raw_image_data), 'Confirm the given path for data exists.'
    assert os.path.exists(write_path), 'Confirm the given path for results exists.'
    assert os.path.exists(bg_img_path), 'Confirm the given path for qc output exists.'

    # ----Run background subtraction-------------
    start_time = datetime.datetime.utcnow()

    background_removal(
        var_dict, raw_image_data, write_path, bg_img_path, bg_rm_type)

    end_time = datetime.datetime.utcnow()
    print 'Background correction run time:', end_time-start_time

    # ----Output for user and save dict----------
    print 'Raw images were background-corrected.'
    print 'Output was written to:'
    print write_path
    print 'Median background images written to:'
    print bg_img_path

    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    # outfile = os.rename('var_dict.p', outfile)
    outfile = shutil.move('var_dict.p', outfile)
    utils.save_user_args_to_csv(args, write_path, 'background_removal')
