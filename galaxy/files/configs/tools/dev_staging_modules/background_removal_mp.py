import os
import datetime
import math
import argparse
import pickle
import numpy as np
import cv2
from libtiff import TIFF
import multiprocessing
import re
import utils


NUMBER_OF_PROCESSORS = math.floor(multiprocessing.cpu_count()/2)


INPUT_PATH = ''
OUTPUT_PATH = ''
QC_PATH = ''
BG_REMOVAL_TYPE = ''
ROBO_NUMBER = None
IMAGING_MODE = ''
VALID_WELLS = []
VALID_TIMEPOINTS = []
DTYPE = 'uint16'


print("TEST")

def get_image_tokens_list(input_dir, robo_num, imaging_mode, valid_wells, valid_timepoints, bg_well):
    ''' Get image file token list
    Args:
      input_montaged_dir: Input dir. each image file is Montaged time point separated.
      robo_num: Which Robo microscope
      imaging_mode: Confocal or epi

    Time separated image name examples(4 naming types):
    Robo3:
      PID20150217_BioP7asynA_T0_0_A1_1_RFP-DFTrCy5_BG_MONTAGE_ALIGNED_CROPPED.tif
      PID20150904_PGPSTest_T1_8_A7_MONTAGE_RFP-DFTrCy5.tif
      PID20150217_BioP7asynA_T1_12-3_A10_1_RFP-DFTrCy5_MN.tif
    Robo4 epi:
      PID20160706_MerckMitoPlate23_T0_12_A11_1_Cyan_DAPI-FITC_525_1_0.0_ANDORZYLA120XELWD.tif
    Robo4 confocal:
      PID20160706_MerckMitoPlate23_T0_12_A11_1_488_561_Empty_525_1_0.0_AndorZyla120XELWD.tif
    Robo4 latest:
      PID20160706_MerckMitoPlate23_T0_0-0_A11_1_Epi-DAPI_0.0_0_1.0.tif

    '''
    stack_dict = {}

    # use os.walk() to recursively iterate through a directory and all its subdirectories
    # image_paths = [os.path.join(root, name) for root, dirs, files in os.walk(input_dir) for name in files if name.endswith('.tif') and '_FIDUCIARY_' not in name]
    image_paths = ''
    if var_dict['DirStructure'] == 'root_dir':
        image_paths = [os.path.join(input_dir, name) for name in os.listdir(input_dir) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
    elif var_dict['DirStructure'] == 'sub_dir':
        # Only traverse root and immediate subdirectories for images
        relevant_dirs = [input_dir] + [os.path.join(input_dir, name) for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]
        image_paths = [os.path.join(relevant_dir, name) for relevant_dir in relevant_dirs for name in os.listdir(relevant_dir) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
    else:
        raise Exception('Unknown Directory Structure!')

    # Robo3 naming
    # Example: PID20150217_BioP7asynA_T1_12-3_A10_1_RFP-DFTrCy5_MN.tif
    if robo_num == 3:
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            name_tokens = image_name.split('_')
            pid_token = name_tokens[0]
            experiment_name_token = name_tokens[1]
            timepoint_token = name_tokens[2]
            if timepoint_token not in valid_timepoints:
                continue
            # Check burst
            burst_idx_token = None
            if '-' in name_tokens[3]:
                numofhours_token, burst_idx_token = name_tokens[3].split('-')
                numofhours_token = float(numofhours_token)
                burst_idx_token = int(burst_idx_token)
            else:
                numofhours_token = int(name_tokens[3])
                burst_idx_token = None
            well_id_token = name_tokens[4]
            montage_idx = int(name_tokens[5])
            if well_id_token not in valid_wells:
                continue
            channel_token = name_tokens[6].replace('.tif', '')
            z_idx_token = None

            # Well ID example: H12
            experiment_well_channel_timepoint_burst_z_key = (experiment_name_token, well_id_token, channel_token, timepoint_token, burst_idx_token, z_idx_token)

            if experiment_well_channel_timepoint_burst_z_key in stack_dict:
                stack_dict[experiment_well_channel_timepoint_burst_z_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token, montage_idx])
            else:
                stack_dict[experiment_well_channel_timepoint_burst_z_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token, montage_idx]]
    # Robo4 epi naming
    # Example: PID20160706_MerckMitoPlate23_T0_12_A11_1_Cyan_DAPI-FITC_525_1_0.0_ANDORZYLA120XELWD.tif
    elif robo_num == 4 and imaging_mode == 'epi':
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            name_tokens = image_name.split('_')
            pid_token = name_tokens[0]
            experiment_name_token = name_tokens[1]
            timepoint_token = name_tokens[2]
            if timepoint_token not in valid_timepoints:
                continue
            # Check burst
            burst_idx_token = None
            if '-' in name_tokens[3]:
                numofhours_token, burst_idx_token = name_tokens[3].split('-')
                numofhours_token = float(numofhours_token)
                burst_idx_token = int(burst_idx_token)
            else:
                numofhours_token = int(name_tokens[3])
                burst_idx_token = None
            well_id_token = name_tokens[4]
            montage_idx = int(name_tokens[5])
            if well_id_token not in valid_wells:
                continue
            channel_token = name_tokens[6]
            z_idx_token = int(name_tokens[9])

            # Well ID example: H12
            experiment_well_channel_timepoint_burst_z_key = (experiment_name_token, well_id_token, channel_token, timepoint_token, burst_idx_token, z_idx_token)


            if experiment_well_channel_timepoint_burst_z_key in stack_dict:
                stack_dict[experiment_well_channel_timepoint_burst_z_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token, montage_idx])
            else:
                stack_dict[experiment_well_channel_timepoint_burst_z_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token, montage_idx]]
    # Robo4 confocal naming
    # Example: PID20160706_MerckMitoPlate23_T0_12_A11_1_488_561_Empty_525_1_0.0_AndorZyla120XELWD.tif
    elif robo_num == 4 and imaging_mode == 'confocal':
        for i in range(len(image_paths)):
            image_path = image_paths[i]
            image_name = os.path.basename(image_path)
            name_tokens = image_name.split('_')
            pid_token = name_tokens[0]
            experiment_name_token = name_tokens[1]
            timepoint_token = name_tokens[2]
            if timepoint_token not in valid_timepoints:
                continue
            # Check burst
            burst_idx_token = None
            if '-' in name_tokens[3]:
                numofhours_token, burst_idx_token = name_tokens[3].split('-')
                numofhours_token = float(numofhours_token)
                burst_idx_token = int(burst_idx_token)
            else:
                numofhours_token = int(name_tokens[3])
                burst_idx_token = None
            well_id_token = name_tokens[4]
            montage_idx = int(name_tokens[5])
            if well_id_token not in valid_wells:
                continue
            # Find the Z-step marker position
            z_step_pos = None
            if i == 0:
                for idx, e in reversed(list(enumerate(name_tokens))):
                    if name_tokens[idx].isdigit():
                        continue
                    else:
                        try:
                            float(name_tokens[idx])
                            z_step_pos = idx
                        except ValueError:
                            continue

            channel_token = name_tokens[z_step_pos-2]
            z_idx_token = int(name_tokens[z_step_pos-1])

            # Well ID example: H12
            experiment_well_channel_timepoint_burst_z_key = (experiment_name_token, well_id_token, channel_token, timepoint_token, burst_idx_token, z_idx_token)


            if experiment_well_channel_timepoint_burst_z_key in stack_dict:
                stack_dict[experiment_well_channel_timepoint_burst_z_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token, montage_idx])
            else:
                stack_dict[experiment_well_channel_timepoint_burst_z_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token, montage_idx]]
    # Robo4 latest naming Robo0
    # Example: PID20160706_MerckMitoPlate23_T0_0-0_A11_1_Epi-DAPI_0.0_0_1.0.tif
    elif robo_num == 0:
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            name_tokens = image_name.split('_')
            pid_token = name_tokens[0]
            experiment_name_token = name_tokens[1]
            timepoint_token = name_tokens[2]
            if timepoint_token not in valid_timepoints:
                continue
            # Check burst
            burst_idx_token = None
            if '-' in name_tokens[3]:
                numofhours_token, burst_idx_token = name_tokens[3].split('-')
                numofhours_token = float(numofhours_token)
                burst_idx_token = int(burst_idx_token)
            else:
                numofhours_token = int(name_tokens[3])
                burst_idx_token = None
            well_id_token = name_tokens[4]
            montage_idx = int(name_tokens[5])
            if well_id_token not in valid_wells:
                continue
            channel_token = name_tokens[6]
            z_idx_token = int(name_tokens[8])

            # Well ID example: H12
            experiment_well_channel_timepoint_burst_z_key = (experiment_name_token, well_id_token, channel_token, timepoint_token, burst_idx_token, z_idx_token)


            if experiment_well_channel_timepoint_burst_z_key in stack_dict:
                stack_dict[experiment_well_channel_timepoint_burst_z_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token, montage_idx])
            else:
                stack_dict[experiment_well_channel_timepoint_burst_z_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token, montage_idx]]
    else:
        raise Exception('Unknown RoboNumber!')

    # Get dict for background well
    bg_well_dict = {}
    if bg_well:
        for idx in stack_dict:
            if idx[1] == bg_well:
                for el in stack_dict[idx]:
                    # experiment_well_channel_timepoint_burst_z_montageidx_key
                    k = idx + (el[9],)
                    bg_well_dict[k] = el[0]

    return [stack_dict[experiment_well_channel_timepoint_burst_z_key] for experiment_well_channel_timepoint_burst_z_key in sorted(stack_dict)], bg_well_dict

def background_correction(image_stack_list):

    # Get median image
    stack_matrix = []
    for im_tokens in image_stack_list:
        slice_img = cv2.imread(im_tokens[0], -1)
        if slice_img is None:
            raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (os.path.basename(im_tokens[0])))
        else:
            stack_matrix.append(slice_img)
    median_image = np.median(stack_matrix, axis=0).astype(DTYPE)

    # Get QC image
    bg_image_name = '_'.join([image_stack_list[0][1], image_stack_list[0][2], image_stack_list[0][5], image_stack_list[0][3], image_stack_list[0][4]]) + '_BGMEDIAN.tif'
    tif_output = TIFF.open(os.path.join(QC_PATH, bg_image_name), mode='w')
    tif_output.write_image(median_image, compression='lzw')
    del tif_output

    for idx, raw_image in enumerate(stack_matrix):
        if BG_WELL:
            background_corrected_image_name = os.path.basename(image_stack_list[idx][0]).replace('.tif', '_BGsw.tif')
            rel_key = (image_stack_list[idx][2], BG_WELL, image_stack_list[idx][4], image_stack_list[idx][5], image_stack_list[idx][8], image_stack_list[idx][7], image_stack_list[idx][9])
            if rel_key in BG_WELL_DICT:
                rel_bg_well_img = cv2.imread(BG_WELL_DICT[rel_key], -1)
                background_corrected_image = cv2.subtract(raw_image, rel_bg_well_img)
            else:
                background_corrected_image = raw_image
                print('Warning: Background image for %s does NOT exist! This will keep the image as it is.' % os.path.basename(image_stack_list[idx][0]))
        else:
            if BG_REMOVAL_TYPE == 'division':
                background_corrected_image_name = os.path.basename(image_stack_list[idx][0]).replace('.tif', '_BGd.tif')
                background_corrected_image = 500*(raw_image / median_image)
                if background_corrected_image.max() > 2**16:
                    factor = ((2**16)-1) / background_corrected_image.max()
                    background_corrected_image = factor * (raw_image / median_image)

            elif BG_REMOVAL_TYPE == 'subtraction':
                background_corrected_image_name = os.path.basename(image_stack_list[idx][0]).replace('.tif', '_BGs.tif')
                background_corrected_image = cv2.subtract(raw_image, median_image)

            else:
                print("Correction type is not recognized.")

        assert background_corrected_image.min() >= 0, background_corrected_image.min()
        assert background_corrected_image.max() <= 2**16, background_corrected_image.max()
        # Make sure dtype back to DTYPE
        background_corrected_image = background_corrected_image.astype(DTYPE)
        output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(OUTPUT_PATH, background_corrected_image_name), image_stack_list[idx][3])
        tif_output = TIFF.open(output_img_location, mode='w')
        tif_output.write_image(background_corrected_image, compression='lzw')
        del tif_output


def multiprocess_background_correction():
    global DTYPE
    global BG_WELL_DICT
    input_image_stack_list, BG_WELL_DICT = get_image_tokens_list(INPUT_PATH, ROBO_NUMBER, IMAGING_MODE, VALID_WELLS, VALID_TIMEPOINTS, BG_WELL)

    # Get global DTYPE from first image
    first_img = cv2.imread(input_image_stack_list[0][0][0], -1)
    if first_img is None:
        raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (os.path.basename(input_image_stack_list[0][0][0])))
    DTYPE = first_img.dtype

    # The following problem only happens on Mac OSX.
    # Disable multithreading in OpenCV for main thread to avoid problems after fork
    # Otherwise any cv2 function call in worker process will hang!!
    # cv2.setNumThreads(0)

    # Initialize workers pool
    workers_pool = multiprocessing.Pool(processes=NUMBER_OF_PROCESSORS)

    # Use chunksize to speed up dispatching, and make sure chunksize is roundup integer
    chunk_size = int(math.ceil(len(input_image_stack_list)/float(NUMBER_OF_PROCESSORS)))

    # Feed data to workers in parallel
    # 99999 is timeout, can be 1 or 99999 etc, used for KeyboardInterrupt multiprocessing
    # map_results = workers_pool.map_async(background_correction, input_image_stack_list, chunksize=chunk_size).get(99999)

    # Use pool.imap to avoid MemoryError, but KeyboardInterrupt not work
    map_results = workers_pool.imap(background_correction, input_image_stack_list, chunksize=chunk_size)

    # Must have these to get return from subprocesses, otherwise all the Exceptions in subprocesses will not throw
    for r in map_results:
        pass

    # # Single instance test
    # print montage(input_image_stack_list[0])

    workers_pool.close()
    workers_pool.join()

def panel_batch_subtraction():
    batch_size = int(str.strip(args.batch_size))
    panels = list(range(1, var_dict['NumberHorizontalImages'] * var_dict['NumberVerticalImages'] + 1))

    image_stack_list, BG_WELL_DICT = get_image_tokens_list(INPUT_PATH, ROBO_NUMBER, IMAGING_MODE, VALID_WELLS, VALID_TIMEPOINTS, BG_WELL)

    # get channels
    channels = var_dict['Channels']

    # get rows and columns
    rows = natural_sort(set([re.search(r'[A-P]', x).group(0) for x in VALID_WELLS]))
    columns = natural_sort(set([re.search(r'\d{1,2}', x).group(0) for x in VALID_WELLS]))

    # build list of wells in the imaging order (i.e. snaking across the plate)
    wells_ordered = []
    for r in rows:
        for c in columns:
            well = r + c
            if any(well in x for x in VALID_WELLS):
                wells_ordered.extend([well])
        columns = list(reversed(columns))

    # reorder image list by imaging order so that batches contain neighboring wells
    image_stack_list.sort(key=lambda x:sum(wells_ordered.index(i[3]) for i in x))

    for tp in VALID_TIMEPOINTS:
        for ch in channels:
            # subset list by images from current channel and timepoint
            image_stack_list_ch = [[tokens for tokens in montage if tokens[4] == ch if tokens[5] == tp] for montage in image_stack_list]

            # remove empty lists for non-matching channels (couldn't figure out how to do it in the above list comprehension)
            image_stack_list_ch = [x for x in image_stack_list_ch if x]

            # check that number of wells is >= batch size
            assert len(image_stack_list_ch) >= batch_size, 'Number of wells (%i) must be greater or equal to batch size (%i)' % (len(image_stack_list_ch), batch_size)

            # divide image stack list into sublists by batch
            imgs_batched = [image_stack_list_ch[i:i + batch_size] for i in range(0, len(image_stack_list_ch), batch_size)]

            # if last batch is smaller than batch_size, then take last batch starting from end of the image list
            if len(imgs_batched[-1]) != batch_size:
                imgs_batched.pop()
                imgs_batched.append(image_stack_list_ch[-batch_size:])

            for batch in imgs_batched:
                for p in panels:
                    # Get images for creating median image
                    panel_stack = [[tokens for tokens in montage if tokens[9] == p][0] for montage in batch]

                    panel_imgs = []
                    for img_tokens in panel_stack:
                        img = cv2.imread(img_tokens[0], -1)
                        if img is None:
                            raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (os.path.basename(img_tokens[0])))
                        else:
                            panel_imgs.append(img)

                    if fill_edges:
                        edge_mask_imgs = []
                        for img_tokens in panel_stack:
                            # open corresponding edge mask
                            edge_mask_path = '_'.join([img_tokens[1], img_tokens[2], img_tokens[5], str(img_tokens[6]), img_tokens[3], str(img_tokens[9]), edge_masks_ch, 'EDGEMASK.tif'])
                            edge_mask_path = utils.reroute_imgpntr_to_wells(os.path.join(masks_path, edge_mask_path), img_tokens[3])
                            assert os.path.exists(edge_mask_path), 'Unable to find edge mask: %s' % edge_mask_path
                            edge_mask = cv2.imread(edge_mask_path, -1)
                            edge_mask = np.expand_dims(edge_mask, axis=0)

                            edge_mask_imgs.append(edge_mask)

                        edge_mask_imgs = np.concatenate(edge_mask_imgs, axis=0)

                    # Create median image from batch of panels
                    median_img = np.median(panel_imgs, axis=0).astype(DTYPE)

                    if fill_edges:
                        # find smallest common well area per batch of image masks
                        common_area_mask = np.prod(edge_mask_imgs, axis=0).astype(DTYPE)

                        # erode common area mask to clean up edges
                        kernel = np.ones((40,40), np.uint8)
                        common_area_mask = cv2.erode(common_area_mask, kernel, iterations=erode_its)

                        # set median image pixel outside of common area to zero
                        median_img[common_area_mask == 0] = 0

                    # Save median image
                    bg_wells = '-'.join([panel_stack[0][3], panel_stack[-1][3]])
                    bg_image_name = '_'.join([panel_stack[0][1], panel_stack[0][2], bg_wells, panel_stack[0][4], panel_stack[0][5], str(panel_stack[0][9]), 'BGMEDIAN.tif'])
                    utils.create_dir(os.path.join(QC_PATH, '_PanelBatchSubtractionMedianImages'))
                    tif_output = TIFF.open(os.path.join(QC_PATH, '_PanelBatchSubtractionMedianImages', bg_image_name), mode='w')
                    tif_output.write_image(median_img, compression='lzw')
                    del tif_output

                    # Subtract median image from raw image
                    for i, raw_img in enumerate(panel_imgs):
                        if fill_edges:
                            # create mask of interwell area and set this region to pixel value of zero
                            raw_img[common_area_mask == 0] = 0

                        # subtract thresholded median image from thresholded raw image
                        if fill_edges:
                            background_corrected_image_name = os.path.basename(panel_stack[i][0]).replace('.tif', '_BGpbsf.tif')
                        else:
                            background_corrected_image_name = os.path.basename(panel_stack[i][0]).replace('.tif', '_BGpbs.tif')
                        background_corrected_image = cv2.subtract(raw_img, median_img)

                        if fill_edges:
                            # shuffle pixels of background corrected image to use a fill for interwell space
                            orig_shape = background_corrected_image.shape
                            shuffled_img = np.copy(background_corrected_image[common_area_mask != 0])
                            # this gets skipped if the image doesn't have well edges
                            if shuffled_img.size > 0:
                                np.random.shuffle(shuffled_img)
                                area_deficit_factor = int(math.ceil(float(background_corrected_image.size)/float(shuffled_img.size)))
                                shuffled_img = np.tile(shuffled_img, area_deficit_factor)[:background_corrected_image.size]
                                shuffled_img = np.reshape(shuffled_img, orig_shape)
                                background_corrected_image[common_area_mask == 0] = shuffled_img[common_area_mask == 0]

                        assert background_corrected_image.min() >= 0, background_corrected_image.min()
                        assert background_corrected_image.max() <= 2**16, background_corrected_image.max()

                        # save background subtracted image
                        background_corrected_image = background_corrected_image.astype(DTYPE)
                        output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(OUTPUT_PATH, background_corrected_image_name), panel_stack[i][3])
                        tif_output = TIFF.open(output_img_location, mode='w')
                        tif_output.write_image(background_corrected_image, compression='lzw')
                        del tif_output

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Correct background uneveness.")
    parser.add_argument("input_dict",
        help="Load input variable dictionary")
    parser.add_argument("bg_removal_type",
        help="division or subtraction")
    parser.add_argument("batch_size",
        help="size of batch for subtraction by panel batch")
    parser.add_argument("fill_edges",
        help="Bool option to detect and fill interwell space")
    parser.add_argument("erode_its",
        help="Integer for degeree of erosion after thresholding")
    parser.add_argument("--input_path", default = '',
        help="Folder path to input data.")
    parser.add_argument("--output_path", default = '',
        help="Folder path to ouput results.")
    parser.add_argument("--qc_path", default = '',
        help="Folder path to qc of results.")
    parser.add_argument("--chosen_bg_well",
        help="Background well.")
    parser.add_argument("--masks_path", default = '',
        help="Folder path to masks used for fill well edges.")
    parser.add_argument("output_dict",
        help="Write variable dictionary.")
    args = parser.parse_args()

    # ----Load path dict-------------------------
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))

    # ----Initialize parameters------------------
    bg_removal_type = args.bg_removal_type
    print('Background correction type: %s' % bg_removal_type)

    bg_well = str.strip(args.chosen_bg_well) if args.chosen_bg_well else None

    outfile = args.output_dict

    raw_image_data = utils.get_path(args.input_path, var_dict['InputPath'], '')
    print('Input path: %s' % raw_image_data)

    write_path = utils.get_path(args.output_path, var_dict['GalaxyOutputPath'], 'BackgroundCorrected')
    print('Background corrected output path: %s' % write_path)

    bg_img_path = utils.get_path(args.qc_path, var_dict['GalaxyOutputPath'], 'QualityControl')
    print('Quality control output path: %s' % bg_img_path)

    if args.fill_edges == 'True' and bg_removal_type == 'subtraction_panel_batch':
        erode_its = int(str.strip(args.erode_its))
        assert erode_its > 0, 'Amount of erosion must be greater than zero.'
        masks_path = utils.get_path(args.masks_path, var_dict['GalaxyOutputPath'], 'EdgeMasks')
        assert os.path.exists(masks_path), 'Confirm the path for edge masks exists (%s)' % masks_path
        ch_pos = utils.get_channel_token(var_dict['RoboNumber'])
        edge_masks_ch = utils.tokenize_files(utils.get_all_files_all_subdir(masks_path))[0][ch_pos+1]
        fill_edges = True
        var_dict['FillEdges'] = True

        print(('Fill well edges: %s' % str(fill_edges)))
        print(('Masks path: %s' % str(masks_path)))
        print(('Edge masks channel: %s' % str(edge_masks_ch)))
        print(('Amount of erosion: %i' % erode_its))
    else:
        fill_edges = False


    # ----Confirm given folders exist--
    assert os.path.exists(raw_image_data), 'Confirm the path for data exists (%s)' % raw_image_data
    assert os.path.exists(write_path), 'Confirm the path for results exists (%s)' % write_path
    assert os.path.exists(bg_img_path), 'Confirm the path for qc output exists (%s)' % bg_img_path

    # Save bg_removal_type and bg_well to dict
    var_dict['bg_removal_type'] = bg_removal_type
    var_dict['bg_well'] = bg_well


    INPUT_PATH = raw_image_data
    OUTPUT_PATH = write_path
    QC_PATH = bg_img_path
    BG_REMOVAL_TYPE = bg_removal_type
    BG_WELL = bg_well
    ROBO_NUMBER = int(var_dict['RoboNumber'])
    IMAGING_MODE = var_dict['ImagingMode']
    VALID_WELLS = var_dict['Wells']
    VALID_TIMEPOINTS = var_dict['TimePoints']

    # ----Run background subtraction-------------s
    start_time = datetime.datetime.utcnow()

    if bg_removal_type == 'subtraction_panel_batch':
        panel_batch_subtraction()
    else:
        multiprocess_background_correction()

    end_time = datetime.datetime.utcnow()
    print('Background correction run time:', end_time-start_time)
    var_dict['DirStructure'] = 'sub_dir'

    # ----Output for user and save dict----------

    # Save dict to file
    pickle.dump(var_dict, open(outfile, 'wb'))
    timestamp = utils.update_timestring()
    utils.save_user_args_to_csv(args, write_path, 'background_removal_mp'+'_'+timestamp)
