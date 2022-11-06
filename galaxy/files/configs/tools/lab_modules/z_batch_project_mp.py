import os
import datetime
import math
import argparse
import pickle
import numpy as np
import cv2
from libtiff import TIFF
import multiprocessing
import shutil
import utils



NUMBER_OF_PROCESSORS = multiprocessing.cpu_count()/2

INPUT_PATH = ''
OUTPUT_PATH = ''
PROJECTION_TYPE = ''
ROBO_NUMBER = ''
IMAGING_MODE = ''
VALID_WELLS = []
VALID_TIMEPOINTS = []
SELECTED_CHANNELS = []
DTYPE = 'uint16'
DIR_STRUCTURE = ''



def get_image_tokens_list(input_dir, robo_num, imaging_mode, valid_wells, valid_timepoints):
    ''' Get image file token list
    Args:
      input_dir: Input dir. each image file is Montaged time point separated or Raw.
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
    Robo4 latest(Robo 0):
      PID20160706_MerckMitoPlate23_T0_0-0_A11_1_Epi-DAPI_0.0_0_1.0.tif

    '''
    stack_dict = {}

    # use os.walk() to recursively iterate through a directory and all its subdirectories
    # image_paths = [os.path.join(root, name) for root, dirs, files in os.walk(input_dir) for name in files if name.endswith('.tif') and '_FIDUCIARY_' not in name]

    # image_paths = [os.path.join(input_dir, name) for name in os.listdir(input_dir) if name.endswith('.tif') and '_FIDUCIARY_' not in name]

    image_paths = ''
    if DIR_STRUCTURE == 'root_dir':
        image_paths = [os.path.join(input_dir, name) for name in os.listdir(input_dir) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
    elif DIR_STRUCTURE == 'sub_dir':
        # Only traverse root and immediate subdirectories for images
        relevant_dirs = [input_dir] + [os.path.join(input_dir, name) for name in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, name))]
        image_paths = [os.path.join(relevant_dir, name) for relevant_dir in relevant_dirs for name in os.listdir(relevant_dir) if name.endswith('.tif') and '_FIDUCIARY_' not in name]

    else:
        raise Exception('Unknown Directory Structure!')


    assert len(image_paths)>0, 'Input path has no files.'

    # Robo4 epi naming
    # Example: PID20160706_MerckMitoPlate23_T0_12_A11_1_Cyan_DAPI-FITC_525_1_0.0_ANDORZYLA120XELWD.tif
    if robo_num == 4 and imaging_mode == 'epi':
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
            experiment_well_channel_timepoint_burst_panel_key = (experiment_name_token, well_id_token, channel_token, timepoint_token, burst_idx_token, montage_idx)


            if experiment_well_channel_timepoint_burst_panel_key in stack_dict:
                stack_dict[experiment_well_channel_timepoint_burst_panel_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token, montage_idx])
            else:
                stack_dict[experiment_well_channel_timepoint_burst_panel_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token, montage_idx]]
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
            experiment_well_channel_timepoint_burst_panel_key = (experiment_name_token, well_id_token, channel_token, timepoint_token, burst_idx_token, montage_idx)


            if experiment_well_channel_timepoint_burst_panel_key in stack_dict:
                stack_dict[experiment_well_channel_timepoint_burst_panel_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token, montage_idx])
            else:
                stack_dict[experiment_well_channel_timepoint_burst_panel_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token, montage_idx]]
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
            experiment_well_channel_timepoint_burst_panel_key = (experiment_name_token, well_id_token, channel_token, timepoint_token, burst_idx_token, montage_idx)


            if experiment_well_channel_timepoint_burst_panel_key in stack_dict:
                stack_dict[experiment_well_channel_timepoint_burst_panel_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token, montage_idx])
            else:
                stack_dict[experiment_well_channel_timepoint_burst_panel_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token, montage_idx]]
    else:
        raise Exception('%s is not supported!' % robo_num)

    return [stack_dict[experiment_well_channel_timepoint_burst_panel_key] for experiment_well_channel_timepoint_burst_panel_key in sorted(stack_dict)]

def project_in_z(image_stack_list):
    image_stack_list.sort(key=lambda x:int(x[7]))
    current_channel = image_stack_list[0][4]

    # Run projection in selected channels
    if current_channel in SELECTED_CHANNELS and len(image_stack_list)>1:
        init_image_name = os.path.basename(image_stack_list[0][0])
        init_image = cv2.imread(image_stack_list[0][0], -1)
        if init_image is None:
            raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % init_image_name)
        else:
            img_height, img_width = init_image.shape
            img_type = init_image.dtype

        rst_arr = np.zeros((img_height, img_width), dtype=img_type)
        rst_img_name = ''
        if PROJECTION_TYPE == 'max_proj':
            rst_img_name = init_image_name.replace('.tif', '_ZMAX.tif')
            for im_tokens in image_stack_list:
                cur_image_name = os.path.basename(im_tokens[0])
                cur_image = cv2.imread(im_tokens[0], -1)
                if cur_image is None:
                    raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % cur_image_name)
                if cur_image.dtype != img_type:
                    raise Exception('Image %s has %s bit depth, not matching with depth of other images %s' % (cur_image_name, cur_image.dtype, img.type))
                rst_arr = np.maximum(rst_arr, cur_image)
                # Round values in array and cast as 16-bit integer
                # rst_arr = np.array(np.round(rst_arr), dtype=img_type)
        elif PROJECTION_TYPE == 'avg_proj':
            rst_img_name = init_image_name.replace('.tif', '_ZAVG.tif')
            for im_tokens in image_stack_list:
                cur_image_name = os.path.basename(im_tokens[0])
                cur_image = cv2.imread(im_tokens[0], -1)
                if cur_image is None:
                    raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (cur_image_name))
                if cur_image.dtype != img_type:
                    raise Exception('Image %s has %s bit depth, not matching with depth of other images %s' % (cur_image_name, cur_image.dtype, img.type))
                rst_arr = rst_arr + cur_image
            rst_arr = rst_arr/len(image_stack_list)
            # Round values in array and cast as 16-bit integer
            # rst_arr = np.array(np.round(rst_arr), dtype=img_type)
        elif PROJECTION_TYPE == 'sum_proj':
            rst_img_name = init_image_name.replace('.tif', '_ZSUM.tif')
            for im_tokens in image_stack_list:
                cur_image_name = os.path.basename(im_tokens[0])
                cur_image = cv2.imread(im_tokens[0], -1)
                if cur_image is None:
                    raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (cur_image_name))
                if cur_image.dtype != img_type:
                    raise Exception('Image %s has %s bit depth, not matching with depth of other images %s' % (cur_image_name, cur_image.dtype, img.type))
                rst_arr = rst_arr + cur_image
        else:
            raise Exception("Projection Type '%s' not recognised!" % PROJECTION_TYPE)

        output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(OUTPUT_PATH, rst_img_name), image_stack_list[0][3])
        # output_img_location = ''
        # if DIR_STRUCTURE == 'root_dir':
        #     output_img_location = os.path.join(OUTPUT_PATH, rst_img_name)
        # elif DIR_STRUCTURE == 'sub_dir':
        #     output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(OUTPUT_PATH, rst_img_name), image_stack_list[0][3])
        # else:
        #     raise Exception('Unknown Directory Structure!')
        tif_output = TIFF.open(output_img_location, mode='w')
        tif_output.write_image(rst_arr, compression='lzw')
        del tif_output

    # Just copy the images to the destination if not selected
    else:
        for im_tokens in image_stack_list:
            shutil.copy2(im_tokens[0], OUTPUT_PATH)

    return 'success'


def multiprocess_project_in_z():
    global DTYPE
    input_image_stack_list = get_image_tokens_list(INPUT_PATH, ROBO_NUMBER, IMAGING_MODE, VALID_WELLS, VALID_TIMEPOINTS)

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
    map_results = workers_pool.imap(project_in_z, input_image_stack_list, chunksize=chunk_size)

    # Must have these to get return from subprocesses, otherwise all the Exceptions in subprocesses will not throw
    for r in map_results:
        pass

    # # Single instance test
    # print project_in_z(input_image_stack_list[0])

    workers_pool.close()
    workers_pool.join()


if __name__ == '__main__':

    # Argument parsing
    parser = argparse.ArgumentParser(description="Project z images.")
    parser.add_argument("input_dict",
        help="Load input variable dictionary")
    parser.add_argument("input_path",
        help="Folder path to input data.")
    parser.add_argument("output_path",
        help="Folder path to ouput results.")
    parser.add_argument("projection_type",
        help="Specify type of projection: max or mean.")
    parser.add_argument("channel_list",
        help="List channels to project, separated by commas")
    parser.add_argument("outfile",
        help="Name of output dictionary.")
    args = parser.parse_args()



    # ----Load path dict-------------------------
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))

    # Set up I/O parameters
    INPUT_PATH = args.input_path
    OUTPUT_PATH = args.output_path
    PROJECTION_TYPE = args.projection_type
    var_dict['ProjectionType'] = PROJECTION_TYPE

    channel_list = args.channel_list
    channel_list = channel_list.replace(" ","")
    user_channels = list(set(channel_list.split(',')))

    outfile = args.outfile
    DIR_STRUCTURE = var_dict['DirStructure']
    ROBO_NUMBER = int(var_dict['RoboNumber'])
    IMAGING_MODE = var_dict['ImagingMode']
    VALID_WELLS = var_dict['Wells']
    VALID_TIMEPOINTS = var_dict['TimePoints']

    var_dict['UserChannels'] = []
    for user_channel in user_channels:
        if user_channel in var_dict['Channels']:
            var_dict['UserChannels'].append(user_channel)
        else:
            raise Exception('Selected Channel %s does not exist' % user_channel)
    SELECTED_CHANNELS = var_dict['UserChannels']


    print 'Wells:', var_dict['Wells']
    print 'Time points:', var_dict['TimePoints']
    print 'Depths:', var_dict['Depths']
    print 'Available channels:', var_dict['Channels']
    print 'Selected channels', var_dict['UserChannels']

    # Confirm given folders exist
    assert os.path.exists(INPUT_PATH), 'Confirm the given path to input data exists.'
    assert os.path.exists(OUTPUT_PATH), 'Confirm the given path for results output exists.'

    # ----Run projections------------------------
    start_time = datetime.datetime.utcnow()

    multiprocess_project_in_z()

    end_time = datetime.datetime.utcnow()
    print 'Tracking run time:', end_time-start_time

    # ----Output for user and save dict----------
    print 'Stacks were unstacked.'
    print 'Output from this step is an encoded mask written to:'
    print OUTPUT_PATH
    var_dict['DirStructure'] = 'sub_dir'

    # Save dict to file
    pickle.dump(var_dict, open(outfile, 'wb'))
    utils.save_user_args_to_csv(args, OUTPUT_PATH, 'z_batch_project_mp')

