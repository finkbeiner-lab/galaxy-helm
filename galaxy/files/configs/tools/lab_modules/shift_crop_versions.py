import os, sys, cv2, argparse

import math
from libtiff import TIFF
import numpy as np
import pickle, datetime, pprint, shutil
import multiprocessing
import dispy
import galaxy.tools.dev_staging_modules.utils as utils




def get_image_tokens_list(input_montaged_dir, robo_num, imaging_mode):
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
    image_paths = [os.path.join(root, name) for root, dirs, files in os.walk(input_montaged_dir) for name in files if name.endswith('.tif')]
    
    # Robo3 naming
    # Example: PID20150217_BioP7asynA_T1_12-3_A10_1_RFP-DFTrCy5_MN.tif
    if robo_num == 3:
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            name_tokens = image_name.split('_')
            pid_token = name_tokens[0]
            experiment_name_token = name_tokens[1]
            timepoint_token = name_tokens[2]
            if timepoint_token not in VALID_TIMEPOINTS:
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
            if well_id_token not in VALID_WELLS:
                continue
            channel_token = name_tokens[6].replace('.tif', '')   
            CHANNEL_SET.add(channel_token) 
            z_idx_token = None

            # Split well id token to make sorting easier
            # Well ID example: H12
            experiment_well_key = (experiment_name_token, well_id_token[0], int(well_id_token[1:]))

            if experiment_well_key in stack_dict:
                stack_dict[experiment_well_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token])
            else:
                stack_dict[experiment_well_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token]]  
    # Robo4 epi naming
    # Example: PID20160706_MerckMitoPlate23_T0_12_A11_1_Cyan_DAPI-FITC_525_1_0.0_ANDORZYLA120XELWD.tif  
    elif robo_num == 4 and imaging_mode == 'epi':
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            name_tokens = image_name.split('_')
            pid_token = name_tokens[0]
            experiment_name_token = name_tokens[1]
            timepoint_token = name_tokens[2]
            if timepoint_token not in VALID_TIMEPOINTS:
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
            if well_id_token not in VALID_WELLS:
                continue
            channel_token = name_tokens[6]
            CHANNEL_SET.add(channel_token) 
            z_idx_token = int(name_tokens[9])

            # Split well id token to make sorting easier
            # Well ID example: H12
            experiment_well_key = (experiment_name_token, well_id_token[0], int(well_id_token[1:]))

            if experiment_well_key in stack_dict:
                stack_dict[experiment_well_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token])
            else:
                stack_dict[experiment_well_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token]]  
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
            if timepoint_token not in VALID_TIMEPOINTS:
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
            if well_id_token not in VALID_WELLS:
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
            CHANNEL_SET.add(channel_token) 
            z_idx_token = int(name_tokens[z_step_pos-1])

            # Split well id token to make sorting easier
            # Well ID example: H12
            experiment_well_key = (experiment_name_token, well_id_token[0], int(well_id_token[1:]))

            if experiment_well_key in stack_dict:
                stack_dict[experiment_well_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token])
            else:
                stack_dict[experiment_well_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token]]  
    # Robo4 latest naming Robo0
    # Example: PID20160706_MerckMitoPlate23_T0_0-0_A11_1_Epi-DAPI_0.0_0_1.0.tif
    elif robo_num == 0:      
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            name_tokens = image_name.split('_')
            pid_token = name_tokens[0]
            experiment_name_token = name_tokens[1]
            timepoint_token = name_tokens[2]
            if timepoint_token not in VALID_TIMEPOINTS:
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
            if well_id_token not in VALID_WELLS:
                continue
            channel_token = name_tokens[6]  
            CHANNEL_SET.add(channel_token) 
            z_idx_token = int(name_tokens[8])

            # Split well id token to make sorting easier
            # Well ID example: H12
            experiment_well_key = (experiment_name_token, well_id_token[0], int(well_id_token[1:]))

            if experiment_well_key in stack_dict:
                stack_dict[experiment_well_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token])
            else:
                stack_dict[experiment_well_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token]]  
    else:
        raise Exception('Unknowed RoboNumber!')                    
        
    return [stack_dict[ewkey] for ewkey in sorted(stack_dict)]


def multiprocess_shift_crop(image_stack_experiment_well):     
    ''' Worker process for single well
    args:
      image_stack_experiment_well:  a list of time series images tokens for the one experiment-well, including possible multiple channels
    
    image_stack format example: [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token], ...]    

    '''  
    # Get image resolution

    first_image = cv2.imread(image_stack_experiment_well[0][0], -1)
    first_image_filename = os.path.basename(image_stack_experiment_well[0][0])
    if first_image is None:
        raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (first_image_filename))
    img_height, img_width = first_image.shape   
    bit_depth = first_image.dtype 

    # Get maximum x, y shift in current well
    shift_dict_of_well = SHIFTS[(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3])] 
    shifts_of_well = [shift_dict_of_well[tp] for tp in shift_dict_of_well]
    max_shift_x = int(max(shifts_of_well, key=lambda item:item[1])[1])
    min_shift_x = int(min(shifts_of_well, key=lambda item:item[1])[1])
    max_shift_y = int(max(shifts_of_well, key=lambda item:item[0])[0])
    min_shift_y = int(min(shifts_of_well, key=lambda item:item[0])[0])
    shift_out_of_bound_tag = False
    if abs(max_shift_x) >= img_width or abs(min_shift_x) >= img_width or abs(max_shift_y) >= img_height or abs(min_shift_y) >= img_height:
        shift_out_of_bound_tag = True
        print('Warning! Maximum shift of Well: %s is larger than width/height of image, cropping will result complete black image!' % (image_stack_experiment_well[0][3]))
    # print 'max_shift_x:%s, min_shift_x:%s, max_shift_y:%s, min_shift_y:%s' %(max_shift_x, min_shift_x, max_shift_y, min_shift_y)

    # Caculate the vilid area
    if max_shift_x > 0:
        start_x = max_shift_x
    else:
        start_x = 0 # no cropping at the start

    if min_shift_x < 0:
        end_x = img_width + min_shift_x
    else:
        end_x = img_width # no cropping at the end     

    if max_shift_y > 0:
        start_y = max_shift_y
    else:
        start_y = 0 # no cropping at the start             

    if min_shift_y < 0:
        end_y = img_height + min_shift_y
    else:
        end_y = img_height # no cropping at the end
    if start_x >= end_x or start_y >= end_y:
        shift_out_of_bound_tag = True
        print('Warning! Maximum shift through timepoints of Well: %s is larger than width/height of image, cropping will result complete black image!' % (image_stack_experiment_well[0][3]))

    # print 'start_x:%s, end_x:%s, start_y:%s, end_y:%s' %(start_x, end_x, start_y, end_y)

    # Group images to make the output in order
    # Dictionary key group by channel
    channel_dict = {} 
    for tks in image_stack_experiment_well:
        if tks[4] in channel_dict:
            channel_dict[tks[4]].append(tks)
        else:
            channel_dict[tks[4]] = [tks]

    # Dictionary key group by timepoint
    for ch in channel_dict:
        timepoint_dict = {}
        tks_list_in_channel = channel_dict[ch]
        for tks in tks_list_in_channel:
            if tks[5] in timepoint_dict:
                timepoint_dict[tks[5]].append(tks)
            else:
                timepoint_dict[tks[5]] = [tks]     
        # Sort timepoint_dict by z_idx_token and burst_idx_token
        for t in timepoint_dict:
            # int(value or 0) will use 0 in the case when you provide any value that Python considers False, such as None, 0, [], "",
            timepoint_dict[t] = sorted(timepoint_dict[t], key=lambda x: (int(x[7] or 0), int(x[8] or 0)))
        channel_dict[ch] = timepoint_dict       

    for ch in channel_dict:
        # Sort Tx (e.g. T8) in order and loop
        sorted_timepoint_keys = sorted(channel_dict[ch], key=lambda x: int(x[1:]))
        # Not loop last item to avoid idx+1 index overflow
        for idx, t in enumerate(sorted_timepoint_keys):       
            # For all the z, bursts
            for ix, item in enumerate(channel_dict[ch][t]): 
                aligned_image_filename = os.path.basename(item[0])
                if not shift_out_of_bound_tag: 
                    # Python: cv2.imread(filename[, flags])  
                    # <0 Return the loaded image as is (with alpha channel).
                    aligned_image = cv2.imread(item[0], -1)

                    if aligned_image is None:
                        raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (aligned_image_filename))                 
                     
                    cropped_image = aligned_image[start_y:end_y, start_x:end_x]
                    tif_output = TIFF.open(os.path.join(PATH_SHIFT_CROPPED_IMAGES, aligned_image_filename.replace('.tif', '_CROPPED.tif')), mode='w')
                    # Note the compression parameter
                    tif_output.write_image(cropped_image, compression='lzw')
                    # Flushes data to disk
                    del tif_output 
                else:
                    image_zeros = np.zeros((img_height, img_width), dtype=bit_depth)  
                    tif_output = TIFF.open(os.path.join(PATH_SHIFT_CROPPED_IMAGES, aligned_image_filename.replace('.tif', '_CROPPEDBLACK.tif')), mode='w')
                    # Note the compression parameter
                    tif_output.write_image(image_zeros, compression='lzw')
                    # Flushes data to disk
                    del tif_output   

    


def distributed_shift_crop(image_stack_experiment_well, path_shift_cropped_images, shifts):     
    ''' Worker process for single well
    args:
      image_stack_experiment_well:  a list of time series images tokens for the one experiment-well, including possible multiple channels
    
    image_stack format example: [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token], ...]    

    '''     
    import cv2
    from libtiff import TIFF 
    import os
    import sys
    import numpy as np

    # Get image resolution
    first_image = cv2.imread(image_stack_experiment_well[0][0], -1)
    first_image_filename = os.path.basename(image_stack_experiment_well[0][0])
    if first_image is None:
        raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (first_image_filename))
    img_height, img_width = first_image.shape   
    bit_depth = first_image.dtype 

    # Get maximum x, y shift in current well
    shift_dict_of_well = shifts[(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3])] 
    shifts_of_well = [shift_dict_of_well[tp] for tp in shift_dict_of_well]
    max_shift_x = int(max(shifts_of_well, key=lambda item:item[1])[1])
    min_shift_x = int(min(shifts_of_well, key=lambda item:item[1])[1])
    max_shift_y = int(max(shifts_of_well, key=lambda item:item[0])[0])
    min_shift_y = int(min(shifts_of_well, key=lambda item:item[0])[0])
    shift_out_of_bound_tag = False
    if abs(max_shift_x) >= img_width or abs(min_shift_x) >= img_width or abs(max_shift_y) >= img_height or abs(min_shift_y) >= img_height:
        shift_out_of_bound_tag = True
        print('Warning! Maximum shift of Well: %s is larger than width/height of image, cropping will result complete black image!' % (image_stack_experiment_well[0][3]))
    # print 'max_shift_x:%s, min_shift_x:%s, max_shift_y:%s, min_shift_y:%s' %(max_shift_x, min_shift_x, max_shift_y, min_shift_y)

    # Caculate the vilid area
    if max_shift_x > 0:
        start_x = max_shift_x
    else:
        start_x = 0 # no cropping at the start

    if min_shift_x < 0:
        end_x = img_width + min_shift_x
    else:
        end_x = img_width # no cropping at the end     

    if max_shift_y > 0:
        start_y = max_shift_y
    else:
        start_y = 0 # no cropping at the start             

    if min_shift_y < 0:
        end_y = img_height + min_shift_y
    else:
        end_y = img_height # no cropping at the end
    if start_x >= end_x or start_y >= end_y:
        shift_out_of_bound_tag = True
        print('Warning! Maximum shift through timepoints of Well: %s is larger than width/height of image, cropping will result complete black image!' % (image_stack_experiment_well[0][3]))

    # print 'start_x:%s, end_x:%s, start_y:%s, end_y:%s' %(start_x, end_x, start_y, end_y)

    # Group images to make the output in order
    # Dictionary key group by channel
    channel_dict = {} 
    for tks in image_stack_experiment_well:
        if tks[4] in channel_dict:
            channel_dict[tks[4]].append(tks)
        else:
            channel_dict[tks[4]] = [tks]

    # Dictionary key group by timepoint
    for ch in channel_dict:
        timepoint_dict = {}
        tks_list_in_channel = channel_dict[ch]
        for tks in tks_list_in_channel:
            if tks[5] in timepoint_dict:
                timepoint_dict[tks[5]].append(tks)
            else:
                timepoint_dict[tks[5]] = [tks]     
        # Sort timepoint_dict by z_idx_token and burst_idx_token
        for t in timepoint_dict:
            # int(value or 0) will use 0 in the case when you provide any value that Python considers False, such as None, 0, [], "",
            timepoint_dict[t] = sorted(timepoint_dict[t], key=lambda x: (int(x[7] or 0), int(x[8] or 0)))
        channel_dict[ch] = timepoint_dict       

    for ch in channel_dict:
        # Sort Tx (e.g. T8) in order and loop
        sorted_timepoint_keys = sorted(channel_dict[ch], key=lambda x: int(x[1:]))
        # Not loop last item to avoid idx+1 index overflow
        for idx, t in enumerate(sorted_timepoint_keys):       
            # For all the z, bursts
            for ix, item in enumerate(channel_dict[ch][t]): 
                aligned_image_filename = os.path.basename(item[0])
                if not shift_out_of_bound_tag: 
                    # Python: cv2.imread(filename[, flags])  
                    # <0 Return the loaded image as is (with alpha channel).
                    aligned_image = cv2.imread(item[0], -1)

                    if aligned_image is None:
                        raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (aligned_image_filename))                 
                     
                    cropped_image = aligned_image[start_y:end_y, start_x:end_x]
                    tif_output = TIFF.open(os.path.join(path_shift_cropped_images, aligned_image_filename.replace('.tif', '_CROPPED.tif')), mode='w')
                    # Note the compression parameter
                    tif_output.write_image(cropped_image, compression='lzw')
                    # Flushes data to disk
                    del tif_output 
                else:
                    image_zeros = np.zeros((img_height, img_width), dtype=bit_depth)  
                    tif_output = TIFF.open(os.path.join(path_shift_cropped_images, aligned_image_filename.replace('.tif', '_CROPPEDBLACK.tif')), mode='w')
                    # Note the compression parameter
                    tif_output.write_image(image_zeros, compression='lzw')
                    # Flushes data to disk
                    del tif_output   
    # Coupled with distribted job() to catch exception    
    return 'success'    



def main_shift_crop():
    # Multiprocessing 
    if PROCESS_MODE == 'multiprocessing':
        input_image_stack_list = get_image_tokens_list(PATH_ALIGNED_IMAGES, ROBO_NUMBER, IMAGING_MODE)
        # print input_image_stack_list
        # Initialize workers pool
        # There is a High Memory Usage issue Using Python Multiprocessing. 
        # The solution essentially was to restart individual worker processes after a fixed number of tasks. 
        # The Pool class in python takes maxtasksperchild as an argument.
        workers_pool = multiprocessing.Pool(processes=NUMBER_OF_PROCESSORS, maxtasksperchild=2)

        # Use chunksize to speed up dispatching, and make sure chunksize is roundup integer
        chunk_size = int(math.ceil(len(input_image_stack_list)/float(NUMBER_OF_PROCESSORS)))

        # Feed data to workers in parallel
        # 99999 is timeout, can be 1 or 99999 etc, used for KeyboardInterrupt multiprocessing
        # map_results = workers_pool.map_async(multiprocess_shift_crop, input_image_stack_list, chunksize=chunk_size).get(99999) 

        # Use pool.imap to avoid MemoryError, but KeyboardInterrupt not work
        map_results = workers_pool.imap(multiprocess_shift_crop, input_image_stack_list, chunksize=chunk_size)

        # Must have these to get return from subprocesses, otherwise all the Exceptions in subprocesses will not throw
        for r in map_results:
            pass
        workers_pool.close()
        workers_pool.join()
    # Distributed    
    elif PROCESS_MODE == 'distributed':
        input_image_stack_list = get_image_tokens_list(PATH_ALIGNED_IMAGES, ROBO_NUMBER, IMAGING_MODE)
        
        # Run dispy shared cluster and submit jobs
        cluster = dispy.SharedJobCluster(distributed_shift_crop, scheduler_node='fb-image-compute01')
        jobs = []

        for i in range(len(input_image_stack_list)):

            job = cluster.submit(input_image_stack_list[i], PATH_SHIFT_CROPPED_IMAGES, SHIFTS)
            job.id = i
            jobs.append(job)
        cluster.wait() # waits until all jobs finish
        # Must get job return to catch exception, dispy will not throw error on failed job by default
        for map_job in jobs:
            r = map_job()
            if not r:
                raise Exception(map_job.exception)
        cluster.print_status()  # shows which nodes executed how many jobs etc.
    else:
        raise Exception('No such process mode: %s' % PROCESS_MODE)    
    

if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Crop images to common matrix.")
    parser.add_argument("input_dict", 
        help="Load input variable dictionary")
    parser.add_argument("process_mode", 
        help="Process mode")
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

    # ----Initialize parameters------------------
    PROCESS_MODE = args.process_mode
    PATH_ALIGNED_IMAGES = args.input_path
    PATH_SHIFT_CROPPED_IMAGES = args.output_path
    var_dict["OutputPath"] = PATH_SHIFT_CROPPED_IMAGES
    outfile = args.output_dict
    resolution = var_dict['Resolution']
    MORPHOLOGY_CHANNEL = var_dict["MorphologyChannel"]
    VALID_WELLS = var_dict['Wells']
    VALID_TIMEPOINTS = var_dict['TimePoints']
    ROBO_NUMBER = int(var_dict['RoboNumber'])
    IMAGING_MODE = var_dict['ImagingMode']
    CHANNEL_SET = set()
    SHIFTS = var_dict['CalculatedShift']

    NUMBER_OF_PROCESSORS = multiprocessing.cpu_count()/2

    # ----Confirm given folders exist--
    assert os.path.exists(PATH_ALIGNED_IMAGES), 'Confirm the given path for data exists.'
    assert os.path.exists(PATH_SHIFT_CROPPED_IMAGES), 'Confirm the given path for results exists.'

    # ----Crop shift-------------------------
    start_time = datetime.datetime.utcnow()

    main_shift_crop()

    end_time = datetime.datetime.utcnow()

    # ----Screen print for user----------
    print('Crop run time:', end_time-start_time)
    print('Aligned images were cropped.')
    print('Output was written to:')
    print(PATH_SHIFT_CROPPED_IMAGES)

    # Save dict to file
    with open(outfile, 'wb') as ofile: 
        pickle.dump(var_dict, ofile)     

