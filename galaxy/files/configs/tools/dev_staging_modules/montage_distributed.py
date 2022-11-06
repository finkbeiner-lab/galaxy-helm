import os
import datetime
import math
import argparse
import pickle
import cv2
import multiprocessing
import dispy
import utils

NUMBER_OF_PROCESSORS = math.floor(multiprocessing.cpu_count() / 2)

INPUT_BGCORRECTION_PATH = ''
OUTPT_MONTAGED_PATH = ''
ROBO_NUMBER = None
IMAGING_MODE = ''
VALID_WELLS = []
VALID_TIMEPOINTS = []
HORIZONTAL_NUM_IMAGES = 0
VERTICAL_NUM_IMAGES = 0
PIXEL_OVERLAP = 0
DTYPE = 'uint16'


def get_image_tokens_list(input_montaged_dir, robo_num, imaging_mode, valid_wells, valid_timepoints):
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

    image_paths = [os.path.join(input_montaged_dir, name) for name in os.listdir(input_montaged_dir) if
                   name.endswith('.tif') and '_FIDUCIARY_' not in name]

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
            experiment_well_channel_timepoint_burst_z_key = (
            experiment_name_token, well_id_token, channel_token, timepoint_token, burst_idx_token, z_idx_token)

            if experiment_well_channel_timepoint_burst_z_key in stack_dict:
                stack_dict[experiment_well_channel_timepoint_burst_z_key].append(
                    [image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token,
                     numofhours_token, z_idx_token, burst_idx_token, montage_idx])
            else:
                stack_dict[experiment_well_channel_timepoint_burst_z_key] = [
                    [image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token,
                     numofhours_token, z_idx_token, burst_idx_token, montage_idx]]
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
            experiment_well_channel_timepoint_burst_z_key = (
            experiment_name_token, well_id_token, channel_token, timepoint_token, burst_idx_token, z_idx_token)

            if experiment_well_channel_timepoint_burst_z_key in stack_dict:
                stack_dict[experiment_well_channel_timepoint_burst_z_key].append(
                    [image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token,
                     numofhours_token, z_idx_token, burst_idx_token, montage_idx])
            else:
                stack_dict[experiment_well_channel_timepoint_burst_z_key] = [
                    [image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token,
                     numofhours_token, z_idx_token, burst_idx_token, montage_idx]]
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

            channel_token = name_tokens[z_step_pos - 2]
            z_idx_token = int(name_tokens[z_step_pos - 1])

            # Well ID example: H12
            experiment_well_channel_timepoint_burst_z_key = (
            experiment_name_token, well_id_token, channel_token, timepoint_token, burst_idx_token, z_idx_token)

            if experiment_well_channel_timepoint_burst_z_key in stack_dict:
                stack_dict[experiment_well_channel_timepoint_burst_z_key].append(
                    [image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token,
                     numofhours_token, z_idx_token, burst_idx_token, montage_idx])
            else:
                stack_dict[experiment_well_channel_timepoint_burst_z_key] = [
                    [image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token,
                     numofhours_token, z_idx_token, burst_idx_token, montage_idx]]
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
            experiment_well_channel_timepoint_burst_z_key = (
            experiment_name_token, well_id_token, channel_token, timepoint_token, burst_idx_token, z_idx_token)

            if experiment_well_channel_timepoint_burst_z_key in stack_dict:
                stack_dict[experiment_well_channel_timepoint_burst_z_key].append(
                    [image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token,
                     numofhours_token, z_idx_token, burst_idx_token, montage_idx])
            else:
                stack_dict[experiment_well_channel_timepoint_burst_z_key] = [
                    [image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token,
                     numofhours_token, z_idx_token, burst_idx_token, montage_idx]]
    else:
        raise Exception('Unknowed RoboNumber!')

    return [stack_dict[experiment_well_channel_timepoint_burst_z_key] for experiment_well_channel_timepoint_burst_z_key
            in sorted(stack_dict)]


def montage(image_stack_list, HORIZONTAL_NUM_IMAGES, VERTICAL_NUM_IMAGES, PIXEL_OVERLAP, DTYPE, OUTPT_MONTAGED_PATH):
    import os
    import numpy as np
    import cv2
    from libtiff import TIFF

    # Check if all the panels exist
    if len(image_stack_list) != HORIZONTAL_NUM_IMAGES * VERTICAL_NUM_IMAGES:
        print('[Error!!] Number of images(%d) DO NOT match Number of panels(%d)' % (
        len(image_stack_list), HORIZONTAL_NUM_IMAGES * VERTICAL_NUM_IMAGES))
        for im in image_stack_list:
            print(im)
        raise Exception('[Error!!] Number of images(%d) DO NOT match Number of panels(%d)' % (
        len(image_stack_list), HORIZONTAL_NUM_IMAGES * VERTICAL_NUM_IMAGES))

    # z_idx_token may be the same with montage_idx, to avoid accidental misreplacement of both,
    # we combine previous well_id_token to make sure the replacement is correct.
    montaged_filename = os.path.basename(image_stack_list[0][0]).replace('.tif', '_MN.tif').replace(
        '_' + image_stack_list[0][3] + '_' + str(image_stack_list[0][9]) + '_', '_' + image_stack_list[0][3] + '_0_')

    # Sort by montage index
    image_stack_list = sorted(image_stack_list, key=lambda x: int(x[9]))

    # Get standard pannel pixel width and height from first image
    pannel_height, pannel_width = None, None
    fs_img = cv2.imread(image_stack_list[0][0], -1)
    if fs_img is None:
        raise Exception(
            '%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (
                os.path.basename(image_stack_list[0][0])))
    else:
        pannel_height, pannel_width = fs_img.shape

    # Width and height after montage
    montaged_width = pannel_width * HORIZONTAL_NUM_IMAGES - PIXEL_OVERLAP * (HORIZONTAL_NUM_IMAGES - 1)
    montaged_height = pannel_height * VERTICAL_NUM_IMAGES - PIXEL_OVERLAP * (VERTICAL_NUM_IMAGES - 1)

    # Build a frame of zeros first
    montaged_image = np.zeros((montaged_height, montaged_width), dtype=DTYPE)

    # Get montage order indexes relative to regular left to right and top to bottom order.
    # like
    # 3 2 1                 1 2 3
    # 4 5 6     relative to 4 5 6
    # 9 8 7                 7 8 9
    # But here we start from 0
    reversed_flag = True
    for idx_j, j in enumerate(range(VERTICAL_NUM_IMAGES)):
        if reversed_flag:
            enumerate_list = reversed(range(HORIZONTAL_NUM_IMAGES * j, HORIZONTAL_NUM_IMAGES * (j + 1)))
        else:
            enumerate_list = range(HORIZONTAL_NUM_IMAGES * j, HORIZONTAL_NUM_IMAGES * (j + 1))

        for idx_i, i in enumerate(enumerate_list):
            coordinate_x = 0
            coordinate_y = 0
            if idx_i != 0:
                coordinate_x = idx_i * (pannel_width - PIXEL_OVERLAP)
            if idx_j != 0:
                coordinate_y = idx_j * (pannel_height - PIXEL_OVERLAP)

                # Read current pannel
            current_pannel = cv2.imread(image_stack_list[i][0], -1)
            if current_pannel is None:
                raise Exception(
                    '%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (
                        os.path.basename(image_stack_list[i][0])))

            # Put pannel to the right position in the frame
            montaged_image[coordinate_y:coordinate_y + pannel_height,
            coordinate_x:coordinate_x + pannel_width] = current_pannel

        reversed_flag = not reversed_flag

    tif_output = TIFF.open(os.path.join(OUTPT_MONTAGED_PATH, montaged_filename), mode='w')
    # Note the compression parameter
    tif_output.write_image(montaged_image, compression='lzw')
    # flushes data to disk
    del tif_output

    # Coupled with distribted job() to catch exception
    return 'success'


def multiprocess_montage():
    global DTYPE
    input_image_stack_list = get_image_tokens_list(INPUT_BGCORRECTION_PATH, ROBO_NUMBER, IMAGING_MODE, VALID_WELLS,
                                                   VALID_TIMEPOINTS)

    # Get global DTYPE from first image
    first_img = cv2.imread(input_image_stack_list[0][0][0], -1)
    if first_img is None:
        raise Exception(
            '%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (
                os.path.basename(input_image_stack_list[0][0][0])))
    DTYPE = first_img.dtype

    # The following problem only happens on Mac OSX.
    # Disable multithreading in OpenCV for main thread to avoid problems after fork
    # Otherwise any cv2 function call in worker process will hang!!
    # cv2.setNumThreads(0)

    # Run dispy shared cluster and submit jobs
    cluster = dispy.SharedJobCluster(montage, scheduler_node='fb-image-compute01')
    jobs = []
    for i in range(len(input_image_stack_list)):
        job = cluster.submit(input_image_stack_list[i], HORIZONTAL_NUM_IMAGES, VERTICAL_NUM_IMAGES, PIXEL_OVERLAP,
                             DTYPE, OUTPT_MONTAGED_PATH)
        job.id = i
        jobs.append(job)
    cluster.wait()  # waits until all jobs finish
    # Must get job return to catch exception, dispy will not throw error on failed job by default
    for map_job in jobs:
        r = map_job()
        if not r:
            raise Exception(map_job.exception)

    cluster.print_status()  # shows which nodes executed how many jobs etc.


if __name__ == '__main__':
    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Montage MP.")
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

    # For select_analysis_module input, set OutPath
    var_dict["OutputPath"] = path_montaged_images

    # ----Confirm given folders exist--
    assert os.path.exists(path_bgcorr_images), 'Confirm the given path for data exists.'
    assert os.path.exists(path_montaged_images), 'Confirm the given path for results exists.'

    INPUT_BGCORRECTION_PATH = path_bgcorr_images
    OUTPT_MONTAGED_PATH = path_montaged_images
    ROBO_NUMBER = int(var_dict['RoboNumber'])
    IMAGING_MODE = var_dict['ImagingMode']
    VALID_WELLS = var_dict['Wells']
    VALID_TIMEPOINTS = var_dict['TimePoints']

    HORIZONTAL_NUM_IMAGES = var_dict['NumberHorizontalImages']
    VERTICAL_NUM_IMAGES = var_dict['NumberVerticalImages']
    resolution = var_dict['Resolution']
    PIXEL_OVERLAP = var_dict['ImagePixelOverlap']

    # ----Run montage----------------------------
    start_time = datetime.datetime.utcnow()

    multiprocess_montage()

    end_time = datetime.datetime.utcnow()
    print('Montage run time:', end_time - start_time)

    # ----Output for user and save dict----
    print('Background-corrected images were montaged.')
    print('Output was written to:')
    print(path_montaged_images)

    # Save dict to file
    with open(outfile, 'wb') as ofile:
        pickle.dump(var_dict, ofile)

    timestamp = utils.update_timestring()
    utils.save_user_args_to_csv(args, path_montaged_images, 'montage_distributed' + '_' + timestamp)