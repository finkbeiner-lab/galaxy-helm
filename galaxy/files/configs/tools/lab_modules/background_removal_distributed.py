import os
import datetime
import math
import argparse
import pickle
import numpy as np
import cv2
from libtiff import TIFF
import multiprocessing
import dispy
import utils



NUMBER_OF_PROCESSORS = multiprocessing.cpu_count()/2


INPUT_PATH = ''
OUTPT_PATH = ''
QC_PATH = ''
BG_REMOVAL_TYPE = ''
ROBO_NUMBER = None
IMAGING_MODE = ''
VALID_WELLS = []
VALID_TIMEPOINTS = []
DTYPE = 'uint16'
DIR_STRUCTURE = ''



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
    if DIR_STRUCTURE == 'root_dir':
        image_paths = [os.path.join(input_dir, name) for name in os.listdir(input_dir) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
    elif DIR_STRUCTURE == 'sub_dir':
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


def background_correction(image_stack_list, DTYPE, BG_REMOVAL_TYPE, QC_PATH, OUTPUT_PATH, BG_WELL_DICT, BG_WELL):
    import os
    import pickle
    import numpy as np
    import cv2
    from libtiff import TIFF

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
    med_image_name = '_'.join([image_stack_list[0][1], image_stack_list[0][2], image_stack_list[0][5], image_stack_list[0][3], image_stack_list[0][4]]) + '_BGMEDIAN.tif'
    med_img_location = utils.reroute_imgpntr_to_wells(os.path.join(QC_PATH, med_image_name), image_stack_list[0][3])
    tif_output = TIFF.open(med_img_location, mode='w')
    tif_output.write_image(median_image, compression='lzw')
    del tif_output

    for im_tokens in image_stack_list:
        raw_image = cv2.imread(im_tokens[0], -1)
        if BG_WELL:
            background_corrected_image_name = os.path.basename(im_tokens[0]).replace('.tif', '_BGsw.tif')
            rel_key = (im_tokens[2], BG_WELL, im_tokens[4], im_tokens[5], im_tokens[8], im_tokens[7], im_tokens[9])
            if rel_key in BG_WELL_DICT:
                rel_bg_well_img = cv2.imread(BG_WELL_DICT[rel_key], -1)
                background_corrected_image = raw_image - rel_bg_well_img
                # Make all negative subtraction result to be zero
                background_corrected_image[background_corrected_image<0] = 0
            else:
                background_corrected_image = raw_image
                print 'Warning: Background image for %s does NOT exist! This will keep the image as it is.' % os.path.basename(im_tokens[0])
        else:
            if BG_REMOVAL_TYPE == 'division':
                background_corrected_image_name = os.path.basename(im_tokens[0]).replace('.tif', '_BGd.tif')
                background_corrected_image = 500*(raw_image / median_image)
                if background_corrected_image.max() > 2**16:
                    factor = ((2**16)-1) / background_corrected_image.max()
                    background_corrected_image = factor * (raw_image / median_image)

            elif BG_REMOVAL_TYPE == 'subtraction':
                background_corrected_image_name = os.path.basename(im_tokens[0]).replace('.tif', '_BGs.tif')
                background_corrected_image = cv2.subtract(raw_image, median_image)

            else:
                print "Correction type is not recognized."

        assert background_corrected_image.min() >= 0, background_corrected_image.min()
        assert background_corrected_image.max() <= 2**16, background_corrected_image.max()
        # Make sure dtype back to DTYPE
        background_corrected_image = background_corrected_image.astype(DTYPE)
        output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(OUTPUT_PATH, background_corrected_image_name), image_stack_list[0][3])
        # output_img_location = ''
        # if DIR_STRUCTURE == 'root_dir':
        #     output_img_location = os.path.join(OUTPUT_PATH, background_corrected_image_name)
        # elif DIR_STRUCTURE == 'sub_dir':
        #     output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(OUTPUT_PATH, background_corrected_image_name), image_stack_list[0][3])
        # else:
        #     raise Exception('Unknown Directory Structure!')
        tif_output = TIFF.open(output_img_location, mode='w')
        tif_output.write_image(background_corrected_image, compression='lzw')
        del tif_output
    # Coupled with distribted job() to catch exception
    return 'success'


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

    # Run dispy shared cluster and submit jobs
    cluster = dispy.SharedJobCluster(background_correction, scheduler_node='fb-image-compute01')
    jobs = []
    for i in xrange(len(input_image_stack_list)):
        job = cluster.submit(input_image_stack_list[i], DTYPE, BG_REMOVAL_TYPE, QC_PATH, OUTPUT_PATH, BG_WELL_DICT, BG_WELL)
        job.id = i
        jobs.append(job)
    cluster.wait() # waits until all jobs finish
    # Must get job return to catch exception, dispy will not throw error on failed job by default
    for map_job in jobs:
        r = map_job()
        if not r:
            raise Exception(map_job.exception)
    cluster.print_status()  # shows which nodes executed how many jobs etc.


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
    parser.add_argument("--chosen_bg_well",
        help="Background well.")
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
    bg_removal_type = args.bg_removal_type
    bg_well = str.strip(args.chosen_bg_well) if args.chosen_bg_well else None
    outfile = args.output_dict


    # ----Confirm given folders exist--
    assert os.path.exists(raw_image_data), 'Confirm the given path for data exists.'
    assert os.path.exists(write_path), 'Confirm the given path for results exists.'
    assert os.path.exists(bg_img_path), 'Confirm the given path for qc output exists.'

    # Save bg_removal_type to dict
    var_dict['bg_removal_type'] = bg_removal_type
    var_dict['bg_well'] = bg_well


    INPUT_PATH = raw_image_data
    OUTPUT_PATH = write_path
    QC_PATH = bg_img_path
    BG_REMOVAL_TYPE = bg_removal_type
    BG_WELL = bg_well
    DIR_STRUCTURE = var_dict['DirStructure']
    ROBO_NUMBER = int(var_dict['RoboNumber'])
    IMAGING_MODE = var_dict['ImagingMode']
    VALID_WELLS = var_dict['Wells']
    VALID_TIMEPOINTS = var_dict['TimePoints']

    # ----Run background subtraction-------------
    start_time = datetime.datetime.utcnow()

    multiprocess_background_correction()

    end_time = datetime.datetime.utcnow()
    print 'Background correction run time:', end_time-start_time

    # ----Output for user and save dict----------
    print 'Raw images were background-corrected.'
    print 'Output was written to:'
    print write_path
    print 'Median background images written to:'
    print bg_img_path
    var_dict['DirStructure'] = 'sub_dir'

    # Save dict to file
    pickle.dump(var_dict, open(outfile, 'wb'))
    utils.save_user_args_to_csv(args, write_path, 'background_removal_distributed')
