'''
Stiches images together given robo number specification and matrix dimentions.
Validated: Montaging order is correct. Sizes match.
'''

import os, sys, utils, cv2
import numpy as np
import argparse, pickle, shutil
import datetime, pprint

def get_montage_order(var_dict):
    '''Takes montage dimensions and returns a montage seqeunce.'''
    horizontal_num_images = var_dict['NumberHorizontalImages']
    vertical_num_images = var_dict['NumberVerticalImages']
    robo_number = var_dict['RoboNumber']
    imaging_mode = var_dict['ImagingMode']

    total_matrix = horizontal_num_images*vertical_num_images
    all_image_indices = range(1, total_matrix)

    rows = total_matrix/horizontal_num_images
    cols = total_matrix/vertical_num_images

    # 'Montage_order': [13, 14, 15, 16, 12, 11, 10, 9, 5, 6, 7, 8, 4, 3, 2, 1]
    if robo_number == 3 or robo_number==4 or robo_number==0:
        # Column order is reversed
        # Starting with last column...
        # Row order is alternated between inverted and original
        montage_order = range(1, (rows*cols)+1)
        counter = 1
        for r_ind in reversed(range(1, rows+1)):
            for c_ind in range(1, cols+1):
                if rows%2 != 0:
                    if r_ind%2 != 0: #inverted col order
                        # print r_ind, (cols+1)-c_ind, counter, (cols*(r_ind-1))+((cols+1)-c_ind), 'inverted order'
                        montage_order[((cols*(r_ind-1))+((cols+1)-c_ind))-1] = counter
                    else: #kept col order
                        # print r_ind, c_ind, counter, (cols*(r_ind-1))+c_ind, 'kept order'
                        montage_order[((cols*(r_ind-1))+c_ind)-1] = counter
                else:
                    if r_ind%2 == 0: #inverted col order
                        # print r_ind, (cols+1)-c_ind, counter, (cols*(r_ind-1))+((cols+1)-c_ind), 'inverted order'
                        montage_order[((cols*(r_ind-1))+((cols+1)-c_ind))-1] = counter
                    else: #kept col order
                        # print r_ind, c_ind, counter, (cols*(r_ind-1))+c_ind, 'kept order'
                        montage_order[((cols*(r_ind-1))+c_ind)-1] = counter
                counter = counter+1
        return montage_order


    # # 'Montage_order': [4, 3, 2, 1, 5, 6, 7, 8, 12, 11, 10, 9, 13, 14, 15, 16]
    # if robo_number == 4 and imaging_mode=="epi":
    #     # Column order is reversed
    #     # Starting with last column...
    #     # Row order is alternated between preserved and inverted
    #     montage_order = range(1, (rows*cols)+1)
    #     counter = 1
    #     for r_ind in reversed(range(1, rows+1)):
    #         for c_ind in range(1, cols+1):
    #             if rows%2 != 0:
    #                 if r_ind%2 == 0: #inverted col order
    #                     # print r_ind, (cols+1)-c_ind, counter, (cols*(r_ind-1))+((cols+1)-c_ind), 'inverted order'
    #                     montage_order[((cols*(r_ind-1))+((cols+1)-c_ind))-1] = counter
    #                 else: #kept col order
    #                     # print r_ind, c_ind, counter, (cols*(r_ind-1))+c_ind, 'kept order'
    #                     montage_order[((cols*(r_ind-1))+c_ind)-1] = counter
    #             else:
    #                 if r_ind%2 != 0: #inverted col order
    #                     # print r_ind, (cols+1)-c_ind, counter, (cols*(r_ind-1))+((cols+1)-c_ind), 'inverted order'
    #                     montage_order[((cols*(r_ind-1))+((cols+1)-c_ind))-1] = counter
    #                 else: #kept col order
    #                     # print r_ind, c_ind, counter, (cols*(r_ind-1))+c_ind, 'kept order'
    #                     montage_order[((cols*(r_ind-1))+c_ind)-1] = counter
    #             counter = counter+1
    #     return montage_order

    # else:
    #     print "No stitching pattern available for this robo."


def stich_images_together(var_dict, montage_files, stiched_path, verbose=False):

    '''
    Takes info about microscope, stiching dimensions, path and images.
    Returns montaged image for each specified well.
    '''
    horizontal_num_images = var_dict['NumberHorizontalImages']
    vertical_num_images = var_dict['NumberVerticalImages']
    robo_number = var_dict['RoboNumber']
    resolution = var_dict['Resolution']
    pixel_overlap = var_dict['ImagePixelOverlap']
    imaging_mode = var_dict['ImagingMode']

    assert robo_number == 0 or robo_number == 3 or robo_number == 4, 'Choose existing robo number. '

    # Robo 4 will need different magnification selection
    robo_dict = {}
    robo_dict[2] = {'Pixel_overlap': 40, 'Flip': False, 'Image_dimensions': (1392, 1040)}
    robo_dict[3] = {'Pixel_overlap': 30, 'Flip': True, 'Image_dimensions': (1024, 1024)}
    robo_dict[4] = {'Pixel_overlap': 30, 'Flip': True, 'Image_dimensions': (2048, 2048)}
    robo_dict[5] = {'Pixel_overlap': 30, 'Flip': True, 'Image_dimensions': (2048, 2048)}
    robo_dict[0] = {'Pixel_overlap': 30, 'Flip': True, 'Image_dimensions': (2048, 2048)}
    # if robo_number == 4 and imaging_mode=="epi":
    #     robo_dict[4] = {'Pixel_overlap': 30, 'Flip': False, 'Image_dimensions': (2048, 2048)}
    # if robo_number == 4 and imaging_mode=="confocal":
    #     robo_dict[4] = {'Pixel_overlap': 30, 'Flip': False, 'Image_dimensions': (2048, 2048)}

    # Set up microscope parameters
    montage_order = get_montage_order(var_dict)
    horizontal_image_overlap = vertical_image_overlap = pixel_overlap
    flip = robo_dict[robo_number]['Flip']
    # im_rows, im_cols = robo_dict[robo_number]['Image_dimensions']
    im_rows, im_cols = cv2.imread(montage_files[0], 0).shape

    # Set up montaging parameters
    total_matrix = horizontal_num_images*vertical_num_images
    total_rows = horizontal_num_images*im_rows - horizontal_image_overlap*(horizontal_num_images-1)
    total_cols = vertical_num_images*im_cols - vertical_image_overlap*(vertical_num_images-1)

    if total_matrix != len(montage_files):
        print 'Given', len(montage_files), 'files instead of', total_matrix, '.'
        assert total_matrix == len(montage_files), 'Confirm that montage matrix dimensions are set correctly.'

    # Handling 8bit vs 16bit
    d_type = np.uint16 if resolution == -1 else np.uint8 if resolution == 0 else None
    grey_val = 30000 if resolution == -1 else 127 if resolution == 0 else None
    assert d_type == np.uint16 or d_type == np.uint8, 'Unsupported intensity depth.'

    montaged_image = grey_val*np.ones((total_rows, total_cols), dtype=d_type)
    # print 'Montaged image dimensions:', montaged_image.shape

    row_starter = 0
    col_starter = 0

    for im_num in montage_order[0:total_matrix]:

        # Handling 8bit vs 16bit
        img = cv2.imread(montage_files[im_num-1], resolution)

        # Handling flipping
        if flip == True:
            img = cv2.flip(img, 0)

        assert img.shape == (im_rows, im_cols), 'Is the correct Robo# selected?'

        montaged_image[row_starter:(row_starter+im_rows), col_starter:(col_starter+im_cols)] = img
        if col_starter+im_cols <= total_cols:
            row_starter = row_starter
            col_starter = col_starter+im_cols-vertical_image_overlap
        if col_starter+im_cols > total_cols:
            row_starter = row_starter+im_rows-horizontal_image_overlap
            col_starter = 0

        # print 'Image number:', im_num
        # cv2.imshow('Overlap', utils.width_resizer(
        #   montaged_image, 500))
        # cv2.waitKey(0)

    img_name = utils.extract_file_name(montage_files[0])
    img_name = utils.make_file_name(
        stiched_path, img_name+'_MN')

    # Handling 8bit vs 16bit
    cv2.imwrite(img_name, montaged_image)

    # cv2.imshow('Overlap', utils.width_resizer(montaged_image, 500))
    # cv2.waitKey(0)
    cv2.destroyAllWindows()

    if verbose == True:
        print 'Dimensions with', str(horizontal_image_overlap),\
            'pixels overlap:', montaged_image.shape

def montage(var_dict, path_bgcorr_images, path_montaged_images, verbose=False):
    '''Main point of entry.'''

    resolution = var_dict['Resolution']
    robo_number = var_dict["RoboNumber"]
    num_horiz = var_dict["NumberHorizontalImages"]
    num_vert = var_dict["NumberVerticalImages"]
    pixel_overlap = var_dict["ImagePixelOverlap"]
    num_panels = num_horiz*num_vert

    # Select an iterator, if needed
    iterator, iter_list = utils.set_iterator(var_dict)
    if iterator=='TimeBursts':
        token_num = 3
    elif iterator=='ZDepths':
        token_num = 8

    print 'What will be iterated:', iterator

    for well in var_dict['Wells']:
        for timepoint in var_dict['TimePoints']:
            for channel in var_dict['Channels']:
                
                # Get relevant images
                if var_dict['RoboNumber']==0:
                    all_montage_files = utils.get_selected_files(
                        path_bgcorr_images, well, timepoint, channel, robonum=0)

                    if verbose:
                        print 'Well-Channel-Time-Filtered Files'
                        pprint.pprint(all_montage_files)

                else:
                    identifier = utils.make_selector(
                        iterator, well, timepoint, channel)
                    all_montage_files = utils.make_filelist(
                        path_bgcorr_images, identifier)

                # Handles no images
                if len(all_montage_files) == 0:
                    continue

                # Handles null iterators (no bursts or depths)
                elif len(all_montage_files) == num_panels:
                    stich_images_together(
                        var_dict, all_montage_files, path_montaged_images)

                # Handles iterators (bursts or depths)
                elif len(all_montage_files) > num_panels:

                    for frame in iter_list:
                        if frame == '0':
                            continue
                        
                        if var_dict['RoboNumber']==0:
                            montage_files = utils.get_frame_files(
                                all_montage_files, frame, token_num)
                            print "Files that will be montaged with", iterator, frame
                            pprint.pprint([os.path.basename(fname) for fname in montage_files])
                        else:
                            identifier = utils.make_selector(
                                iterator=iterator, well=well, timepoint=timepoint,
                                channel=channel, frame=frame)
                            montage_files = utils.make_filelist(
                                path_bgcorr_images, identifier)

                        if len(montage_files) == 0:
                            continue

                        if len(montage_files)%num_panels != 0 or len(montage_files) > num_panels:
                            print 'Selector generated:', identifier,
                            print 'Number of images:', len(montage_files)
                            print 'Images used:'
                            pprint.pprint([os.path.basename(wellimg) for wellimg in montage_files])
                            assert len(montage_files)%num_panels == 0, 'Did not find all files.'
                            assert len(montage_files) == num_panels, 'Wrong number of files.'

                        stich_images_together(
                            var_dict, montage_files, path_montaged_images)

                # Catch other
                else:
                    print 'Unusual number of images found. Not stitching.'
                    pprint.pprint([os.path.basename(wellimg) for wellimg in montage_files])

    # For select_analysis_module input, set OutPath
    var_dict["OutputPath"] = path_montaged_images


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
    parser.add_argument("output_dict",
        help="Write variable dictionary.")
    args = parser.parse_args()

    # ----Load path dict-------------------------
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))

    # ----Initialize parameters--------
    path_bgcorr_images = args.input_path
    path_montaged_images = args.output_path
    outfile = args.output_dict

    # ----Confirm given folders exist--
    assert os.path.exists(path_bgcorr_images), 'Confirm the given path for data exists.'
    assert os.path.exists(path_montaged_images), 'Confirm the given path for results exists.'

    # ----Run montage----------------------------
    start_time = datetime.datetime.utcnow()

    montage(var_dict, path_bgcorr_images, path_montaged_images)

    end_time = datetime.datetime.utcnow()
    print 'Montage run time:', end_time-start_time

    # ----Output for user and save dict----
    print 'Background-corrected images were montaged.'
    print 'Output was written to:'
    print path_montaged_images

    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    # outfile = os.rename('var_dict.p', outfile)
    outfile = shutil.move('var_dict.p', outfile)
