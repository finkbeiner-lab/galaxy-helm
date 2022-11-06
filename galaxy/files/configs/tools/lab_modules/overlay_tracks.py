'''
Loops through dictionary and add contour and cell_number label to each image.
'''

import cv2, utils, sys, datetime, argparse
import numpy as np
import pickle, os, shutil, pprint
from tracking import Cell

# With a dictionary structure keyed on timepoint, with a list of (id, cnt) tuples.
# inputs:   time_dictionary = {'T0': [(cnt_ind, cell_obj), ...], 'T1': [], 'T2': [], ...'Tn': []}
#           file_list = utils.make_filelist(path, well_id)
#           where to write output

def add_label(cell_records, img):
    '''
    Add cell number next to cell.
    '''

    cnt_indices = [record[0] for record in cell_records]
    cnt_centroids = [record[1].get_circle()[0] for record in cell_records]
    cnt_radii = [record[1].get_circle()[1] for record in cell_records]
    cnt_contours = [record[1].cnt for record in cell_records]

    for cnt_ind, cnt_centroid, cnt in zip(
        cnt_indices, cnt_centroids, cnt_contours):
        if cnt_ind=='e':
            continue
        if cnt_ind=='n':
            continue
    # for cnt_ind, cnt_centroid, cnt_radius in zip(
    #     cnt_indices, cnt_centroids, cnt_radii):

        cnt_cntr_x = int(cnt_centroid[0])+10
        cnt_cntr_y = int(cnt_centroid[1])-10
        # print str(cnt_ind), ':', cnt_cntr_x, cnt_cntr_y

        font = cv2.FONT_HERSHEY_SIMPLEX
        cnt_center = (int(cnt_centroid[0]), int(cnt_centroid[1]))

        # cv2.circle(img, cnt_center, int(cnt_radius), (0, 0, 255), 2)

        # updated from (0, 0, 255) to 255 for mask
        cv2.drawContours(img, [cnt], 0, 255, 5)
        cv2.putText(img, str(cnt_ind),
            (cnt_cntr_x, cnt_cntr_y),
            font, 4, 255, 5, cv2.CV_AA)

    return img

def add_tracks_blackout_noncells(time_dictionary, img_list, mask_list, write_path, var_dict):
    '''
    Opens image that takes the cell labels.
    Writes out two images:
        - Image with labels and drawn out contours
        - Image with black outside of contour areas (masked image)
    '''

    # Take here a list of images with the same time_ref and keep the first one.

    for time_ref, cell_records in time_dictionary.items():
        # print 'The current time:', time_ref, time_ref+'_'
        for img_pointer, mask_pointer in zip(img_list, mask_list):
            # print 'The current image:', os.path.basename(img_pointer), os.path.basename(mask_pointer)
            if time_ref+'_' in mask_pointer:
                # print time_ref, mask_pointer
                img = cv2.imread(mask_pointer, -1)
                img_dims = img.shape[0:2]
                # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                over_mask = np.zeros(img_dims, dtype=np.uint16)
                # cell_mask = cv2.imread(mask_pointer, -1)

                orig_name = utils.extract_file_name(mask_pointer)
                over_img_name = utils.make_file_name(write_path, orig_name+'_OL')
                # under_img_name = utils.make_file_name(write_path, orig_name+'_MASKED')
                well_name = os.path.basename(mask_pointer).split('_')[4]
                over_img_name = utils.reroute_imgpntr_to_wells(over_img_name, well_name)

                over_mask = add_label(cell_records, over_mask)
                # under_mask = cv2.bitwise_and(img, img, mask=cell_mask)

                # cv2.imwrite(under_img_name, under_mask)
                if var_dict["MorphologyChannel"] in mask_pointer:
                    cv2.imwrite(over_img_name, over_mask)

                # cv2.imshow('Labeled cell_records', utils.width_resizer(img, 500))
                # cv2.waitKey(0)
                # cv2.destroyAllWindows()

def overlay_tracks(var_dict, path_to_images, write_path, path_to_masks, verbose=False):
    '''
    Main point of entry.
    '''
    resolution = var_dict['Resolution']

    for well in var_dict['Wells']:
        print 'Well:', well

        try:
            time_dictionary = var_dict['TrackedCells'][well]
            # print time_dictionary.keys()
        except KeyError:
            print "Skipping well", well, "(no files were found)."
        else:
            for channel in var_dict["Channels"]:

                selector = utils.make_selector(well=well, channel=channel)

                # files_to_label = utils.make_filelist(path_to_images, selector)
                # masks_to_use = utils.make_filelist(path_to_masks, well+'_*MASK.tif')
                # # masks_to_use = utils.make_filelist(path_to_masks, well+'_*MASK*')
                files_to_label = utils.make_filelist_wells(path_to_images, selector)
                masks_to_use = utils.make_filelist_wells(path_to_masks, well+'_*MASK.tif')

                if verbose:
                    print "files_to_label"
                    pprint.pprint([os.path.basename(ftl) for ftl in files_to_label])
                    print "masks_to_use"
                    pprint.pprint([os.path.basename(ftl) for ftl in masks_to_use])

                if len(files_to_label) == 0 or len(masks_to_use) == 0:
                    continue
                if len(files_to_label) != len(masks_to_use):
                    selected_files_to_label = []
                    for time_point in time_dictionary.keys():
                        # print 'The time_point:', time_point
                        time_files = [el for el in files_to_label if time_point in el]
                        # pprint.pprint(time_files)
                        if len(time_files) == 0:
                            continue
                        ind = int(len(time_files)/2)
                        # print 'Index for fime_files...', ind
                        selected_files_to_label.append(time_files[ind])

                    if verbose:
                        print 'Files to label:', len(selected_files_to_label)
                        pprint.pprint(selected_files_to_label)
                        print 'Masks to use:', len(masks_to_use)
                        pprint.pprint(masks_to_use)
                    files_to_label = selected_files_to_label
                    # assert len(files_to_label) == len(masks_to_use), 'Mismatch between number of images and masks.'

                add_tracks_blackout_noncells(
                    time_dictionary, files_to_label, masks_to_use, write_path, var_dict)

    # For select_analysis_module input, set var_dict['OutputPath']
    var_dict["OutputPath"] = write_path


if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Overlay cell annotation on original images.")
    parser.add_argument("input_dict",
        help="Load input variable dictionary")
    parser.add_argument("input_path",
        help="Folder path to input data.")
    parser.add_argument("output_path",
        help="Folder path to ouput results.")
    parser.add_argument("mask_path",
        help="Folder path to cell masks.")
    parser.add_argument("output_dict",
        help="Write variable dictionary.")
    args = parser.parse_args()

    # ----Load path dict-------------------------
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))

    # ----Initialize parameters------------------
    path_to_images = args.input_path
    path_to_masks = args.mask_path
    write_path = args.output_path
    outfile = args.output_dict

    # ----Handle correct output is passed in-----
    try: var_dict['TrackedCells']
    except KeyError: print 'Confirm that result from cell tracking step is passed in to local variabes dictionary.'

    # ----Confirm given folders exist--
    assert os.path.exists(path_to_images), 'Confirm the given path for image data exists.'
    assert os.path.exists(path_to_masks), 'Confirm the given path for cell masks exists.'
    assert os.path.exists(write_path), 'Confirm the given path for output exists.'

    # ----Run overlay tracks---------------------
    start_time = datetime.datetime.utcnow()

    overlay_tracks(var_dict, path_to_images, write_path, path_to_masks)

    end_time = datetime.datetime.utcnow()
    print 'Overlay tracks run time:', end_time-start_time
    # ----Output for user and save dict----------
    print 'Overlays were created for each time point.'
    print 'Output was written to:'
    print write_path

    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    # outfile = os.rename('var_dict.p', outfile)
    outfile = shutil.move('var_dict.p', outfile)
    utils.save_user_args_to_csv(args, write_path, 'overlay_tracks')



