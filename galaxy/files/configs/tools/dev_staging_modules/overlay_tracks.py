'''
Loops through dictionary and add contour and cell_number label to each image.
'''

import cv2, sys, datetime, argparse, utils
import numpy as np
# import galaxy.tools.dev_staging_modules.utils as utils
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
            font, 4, 255, 5, cv2.LINE_AA)

    return img

def add_tracks_blackout_noncells(cell_records, img_pointer, mask_pointer, write_path, var_dict):
    '''
    Opens image that takes the cell labels.
    Writes out two images:
        - Image with labels and drawn out contours
        - Image with black outside of contour areas (masked image)
    '''

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
    cv2.imwrite(over_img_name, over_mask)


def overlay_tracks(var_dict, image_tokens, write_path, masks_tokens, verbose=False):
    '''
    Main point of entry.
    '''
    resolution = var_dict['Resolution']

    for well in var_dict['Wells']:
        print('Well:', well)

        for timepoint in var_dict['TimePoints']:
            print('Timepoint:', timepoint)

            try:
                time_dictionary = var_dict['TrackedCells'][well][timepoint]

            except KeyError:
                print("Skipping well", well, "timepoint", timepoint, "(no files were found).")
            else:
                morphology_channel = var_dict['MorphologyChannel']

                img_pointer = utils.get_filename(image_tokens, well, timepoint,
                                                 morphology_channel, var_dict['RoboNumber'])
                mask_pointer = utils.get_filename(masks_tokens, well, timepoint,
                                                  morphology_channel, var_dict['RoboNumber'])

                if verbose:
                    print(img_pointer)
                    print(mask_pointer)

                add_tracks_blackout_noncells(
                    time_dictionary, img_pointer[0], mask_pointer[0], write_path, var_dict)

    # For select_analysis_module input, set var_dict['OutputPath']
    var_dict["OutputPath"] = write_path


if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Overlay cell annotation on original images.")
    parser.add_argument("input_dict",
        help="Load input variable dictionary")
    parser.add_argument("--input_image_path",
        help="Folder path to input data.", default = '')
    parser.add_argument("--output_results_path",
        help="Folder path to ouput results.", default = '')
    parser.add_argument("--mask_path",
        help="Folder path to cell masks.", default = '')
    parser.add_argument("output_dict",
        help="Write variable dictionary.")
    args = parser.parse_args()

    # ----Load path dict-------------------------
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))

    # ----Initialize parameters------------------
    path_to_images = utils.get_path(args.input_image_path, var_dict['GalaxyOutputPath'], 'AlignedImages')
    print('Images input path: %s' % path_to_images)

    path_to_masks = utils.get_path(args.mask_path, var_dict['GalaxyOutputPath'], 'CellMasks')
    print('Cell Masks input path: %s' % path_to_masks)

    write_path = utils.get_path(args.output_results_path, var_dict['GalaxyOutputPath'], 'OverlaysTablesResults')
    print('Overlays output path: %s' % write_path)

    outfile = args.output_dict

    # ----Handle correct output is passed in-----
    try: var_dict['TrackedCells']
    except KeyError: print('Confirm that result from cell tracking step is passed in to local variabes dictionary.')

    # ----Confirm given folders exist--
    assert os.path.exists(path_to_images), 'Confirm the path for image data exists (%s)' % path_to_images
    assert os.path.exists(path_to_masks), 'Confirm the path for cell masks exists (%s)' % path_to_masks
    assert os.path.exists(write_path), 'Confirm the path for output exists (%s)' % write_path

    if var_dict['DirStructure'] == 'root_dir':
        image_paths = [os.path.join(path_to_images, name) for name in os.listdir(path_to_images) if
                       name.endswith('.tif') and '_FIDUCIARY_' not in name]
        mask_paths = [os.path.join(path_to_masks, name) for name in os.listdir(path_to_masks) if
                     name.endswith('_CELLMASK.tif') and '_FIDUCIARY_' not in name]

    elif var_dict['DirStructure'] == 'sub_dir':
        # Only traverse root and immediate subdirectories for images
        relevant_img_dirs = [path_to_images] + [os.path.join(path_to_images, name) for name in
                                                os.listdir(path_to_images) if
                                                os.path.isdir(os.path.join(path_to_images, name))]
        image_paths = [os.path.join(relevant_dir, name) for relevant_dir in relevant_img_dirs for name in
                       os.listdir(relevant_dir) if
                       name.endswith('.tif') and '_FIDUCIARY_' not in name]

        relevant_mask_dirs = [path_to_masks] + [os.path.join(path_to_masks, name) for name in
                                                os.listdir(path_to_masks) if
                                                os.path.isdir(os.path.join(path_to_masks, name))]
        mask_paths = [os.path.join(relevant_dir, name) for relevant_dir in relevant_mask_dirs for name in
                     os.listdir(relevant_dir) if
                     name.endswith('_CELLMASK.tif') and '_FIDUCIARY_' not in name]

    else:
        raise Exception('Unknown Directory Structure!')

    image_tokens = utils.tokenize_files(image_paths)
    masks_tokens = utils.tokenize_files(mask_paths)

    # ----Run overlay tracks---------------------
    start_time = datetime.datetime.utcnow()

    overlay_tracks(var_dict, image_tokens, write_path, masks_tokens, verbose=True)

    end_time = datetime.datetime.utcnow()
    print('Overlay tracks run time:', end_time-start_time)
    # ----Output for user and save dict----------
    print('Overlays were created for each time point.')

    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    # outfile = os.rename('var_dict.p', outfile)
    outfile = shutil.move('var_dict.p', outfile)
    timestamp = utils.update_timestring()
    utils.save_user_args_to_csv(args, write_path, 'overlay_tracks'+'_'+timestamp)