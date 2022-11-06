import numpy as np
# For some reason, either libtiff or PIL distorted/autoscaled some kind of images while reading in,
# opencv is the most robust one to read the image intensities as it is. And we use libtiff to write lzw compression
# image since only libtiff can write lzw compression image.
import cv2
import skimage
from skimage.feature import register_translation
import imreg_dft as ird
from libtiff import TIFF
import os
from operator import itemgetter
import sys
import multiprocessing, logging
import time
import math
from skimage import transform
import pickle, datetime, argparse
import galaxy.tools.dev_staging_modules.utils as utils


# mpl = multiprocessing.log_to_stderr()
# mpl.setLevel(logging.INFO)
NUMBER_OF_PROCESSORS = multiprocessing.cpu_count()/2

INPUT_MONTAGED_PATH = ''
OUTPUT_ALIGNED_PATH = ''
MORPHOLOGY_CHANNEL = ''
VALID_WELLS = []
VALID_TIMEPOINTS = []
CHANNEL_SET = set()
LOG_INFO = {}
SHIFTS = {}
ALIGNMENT_ALGORITHM = ''
ROBO_NUMBER = None
IMAGING_MODE = ''
DIR_STRUCTURE = ''


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
    # image_paths = [os.path.join(root, name) for root, dirs, files in os.walk(input_montaged_dir) for name in files if name.endswith('.tif')]
    image_paths = ''
    if DIR_STRUCTURE == 'root_dir':
        image_paths = [os.path.join(input_montaged_dir, name) for name in os.listdir(input_montaged_dir) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
    elif DIR_STRUCTURE == 'sub_dir':
        # Only traverse root and immediate subdirectories for images
        relevant_dirs = [input_montaged_dir] + [os.path.join(input_montaged_dir, name) for name in os.listdir(input_montaged_dir) if os.path.isdir(os.path.join(input_montaged_dir, name))]
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




def register_stack(image_stack_experiment_well):
    ''' Worker process for single well
    args:
      image_stack_experiment_well:  a list of time series images tokens for the one experiment-well, including possible multiple channels

    image_stack format example: [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token], ...]

    '''
    shift_logs = []
    suspicious_misalignment_logs = []
    asymmetric_missing_image_logs = []
    # Dictionary key by channel
    channel_dict = {}
    for tks in image_stack_experiment_well:
        if tks[4] in channel_dict:
            channel_dict[tks[4]].append(tks)
        else:
            channel_dict[tks[4]] = [tks]

    # Dictionary key by timepoint
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



    # Process morphology channel first, then use the calculated shift to apply to other channels
    morphology_timepoint_dict = channel_dict[MORPHOLOGY_CHANNEL]
    num_of_timepoints = len(morphology_timepoint_dict)

    processing_log = "Processing [experiment: %s, well: %s, channel: %s]" % (image_stack_experiment_well[0][2], image_stack_experiment_well[0][3], MORPHOLOGY_CHANNEL)
    print(processing_log)
    # current_experiment_well_log.append(processing_log)

    fixed_image = None
    moving_image = None

    # Load previous calculated shifts if exist
    pre_calculated_shift_dict = {}
    if SHIFTS and (image_stack_experiment_well[0][2], image_stack_experiment_well[0][3]) in SHIFTS:
        pre_calculated_shift_dict = SHIFTS[(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3])]
        # Sort Tx (e.g. T8) in order and loop
        sorted_morphology_timepoint_keys = sorted(morphology_timepoint_dict, key=lambda x: int(x[1:]))
        # Not loop last item to avoid idx+1 index overflow
        for idx, t in enumerate(sorted_morphology_timepoint_keys[:-1]):
            # For all the z, bursts
            for ix, item in enumerate(morphology_timepoint_dict[t]):
                # Only calc shift at first image in current timepoint, then propogate to other images
                if ix == 0:
                    # Write first time point image as fixed image
                    if idx == 0:
                        shift_for_cur_timepoint = [0, 0]
                        # shift_dict[t] = shift_for_cur_timepoint
                        # fixed_image = TIFF.open(item[0], mode='r')
                        # fixed_image = fixed_image.read_image()


                        # Python: cv2.imread(filename[, flags])
                        # <0 Return the loaded image as is (with alpha channel).
                        fixed_image = cv2.imread(item[0], -1)
                        fixed_image_filename = os.path.basename(item[0])
                        if fixed_image is None:
                            raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (fixed_image_filename))

                        output_img_location = ''
                        if DIR_STRUCTURE == 'root_dir':
                            output_img_location = os.path.join(OUTPUT_ALIGNED_PATH, fixed_image_filename.replace('.tif', '_ALIGNED.tif'))
                        elif DIR_STRUCTURE == 'sub_dir':
                            output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(OUTPUT_ALIGNED_PATH, fixed_image_filename.replace('.tif', '_ALIGNED.tif')), image_stack_experiment_well[0][3])
                        else:
                            raise Exception('Unknown Directory Structure!')
                        tif_output = TIFF.open(output_img_location, mode='w')
                        tif_output.write_image(fixed_image, compression='lzw')
                        del tif_output


                    # moving_image = TIFF.open(morphology_timepoint_dict[sorted_morphology_timepoint_keys[idx+1]][ix][0], mode='r')
                    # moving_image = moving_image.read_image()
                    moving_image = cv2.imread(morphology_timepoint_dict[sorted_morphology_timepoint_keys[idx+1]][ix][0], -1)
                    moving_image_filename = os.path.basename(morphology_timepoint_dict[sorted_morphology_timepoint_keys[idx+1]][ix][0])
                    if moving_image is None:
                        raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (moving_image_filename))
                    bit_depth = moving_image.dtype
                    # print fixed_image_filename
                    # print moving_image_filename



                    if sorted_morphology_timepoint_keys[idx+1] not in pre_calculated_shift_dict:
                        raise Exception('Previous calculated shifts for %s, %s, %s does not exists' % (image_stack_experiment_well[0][2], image_stack_experiment_well[0][3], sorted_morphology_timepoint_keys[idx+1]))

                    shift_for_cur_timepoint = pre_calculated_shift_dict[sorted_morphology_timepoint_keys[idx+1]]
                    # Shift back to fixed. Note the transform usage is opposite to normal. For example this shift from T1 to T0(fixed target) is [x, y],
                    # parameter in transform.warp should be reversed as [-x, -y]
                    tform = transform.SimilarityTransform(translation=(-shift_for_cur_timepoint[1], -shift_for_cur_timepoint[0]))
                    # With preserve_range=True, the original range of the data will be preserved, even though the output is a float image
                    # with the original pixel value preserved. Otherwise default pixel value is [0, 1] for float
                    corrected_image = transform.warp(moving_image, tform, preserve_range=True)

                    # print "before", type(corrected_image), corrected_image.dtype, corrected_image.shape
                    # print "before", corrected_image.max(), corrected_image.min()

                    # transform.warp default returns double float64 ndarray, have to convert back to original bit depth
                    corrected_image = corrected_image.astype(bit_depth, copy=False)

                    # print "after:", type(corrected_image), corrected_image.dtype, corrected_image.shape
                    # print "after", corrected_image.max(), corrected_image.min()


                    # Output the corrected images to file
                    output_img_location = ''
                    if DIR_STRUCTURE == 'root_dir':
                        output_img_location = os.path.join(OUTPUT_ALIGNED_PATH, moving_image_filename.replace('.tif', '_ALIGNED.tif'))
                    elif DIR_STRUCTURE == 'sub_dir':
                        output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(OUTPUT_ALIGNED_PATH, moving_image_filename.replace('.tif', '_ALIGNED.tif')), image_stack_experiment_well[0][3])
                    else:
                        raise Exception('Unknown Directory Structure!')
                    tif_output = TIFF.open(output_img_location, mode='w')
                    tif_output.write_image(corrected_image, compression='lzw')
                    del tif_output

                    # Move to next slice
                    fixed_image = moving_image
                    fixed_image_filename = moving_image_filename
                # Apply the shift to the other iamges(zs, bursts) in current timepoint
                else:
                    # other_zb_image = TIFF.open(item[0], mode='r')
                    # other_zb_image = other_zb_image.read_image()
                    other_zb_image = cv2.imread(item[0], -1)
                    other_zb_image_filename = os.path.basename(item[0])
                    if other_zb_image is None:
                        raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (other_zb_image_filename))
                    if idx != 0:
                        bit_depth =other_zb_image.dtype
                        tform = transform.SimilarityTransform(translation=(-pre_calculated_shift_dict[t][1], -pre_calculated_shift_dict[t][0]))
                        other_zb_image = transform.warp(other_zb_image, tform, preserve_range=True)
                        other_zb_image = other_zb_image.astype(bit_depth, copy=False)

                    output_img_location = ''
                    if DIR_STRUCTURE == 'root_dir':
                        output_img_location = os.path.join(OUTPUT_ALIGNED_PATH, other_zb_image_filename.replace('.tif', '_ALIGNED.tif'))
                    elif DIR_STRUCTURE == 'sub_dir':
                        output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(OUTPUT_ALIGNED_PATH, other_zb_image_filename.replace('.tif', '_ALIGNED.tif')), image_stack_experiment_well[0][3])
                    else:
                        raise Exception('Unknown Directory Structure!')
                    tif_output = TIFF.open(output_img_location, mode='w')
                    tif_output.write_image(other_zb_image, compression='lzw')
                    del tif_output

        # Reduce memory consumption. Maybe help garbage collection
        fixed_image = None
        moving_image = None

        # Apply the same shift to the other channels(Assuming the Microscope is done with position first imaging method)
        for chl in channel_dict:
            if chl != MORPHOLOGY_CHANNEL:
                other_channel_timepoint_dict = channel_dict[chl]
                other_channel_log = "Applying shift to other channels [experiment: %s, well: %s, channel: %s]" % (image_stack_experiment_well[0][2], image_stack_experiment_well[0][3], chl)
                print(other_channel_log)
                # current_experiment_well_log.append(other_channel_log)

                # Sort Tx (e.g. T8) in order and loop
                sorted_other_channel_timepoint_keys = sorted(other_channel_timepoint_dict, key=lambda x: int(x[1:]))
                # No idx+1, so enumerate all timepoints
                for idx, t in enumerate(sorted_other_channel_timepoint_keys):
                    # Check if current image has related morphology shift calculated, in case of asymmetric images
                    if t in pre_calculated_shift_dict:
                        # For all the z, bursts
                        for ix, item in enumerate(other_channel_timepoint_dict[t]):
                            # other_channel_image = TIFF.open(item[0], mode='r')
                            # other_channel_image = other_channel_image.read_image()
                            other_channel_image = cv2.imread(item[0], -1)
                            other_channel_image_filename = os.path.basename(item[0])
                            if other_channel_image is None:
                                raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (other_channel_image_filename))
                            if idx != 0:
                                bit_depth =other_channel_image.dtype
                                tform = transform.SimilarityTransform(translation=(-pre_calculated_shift_dict[t][1], -pre_calculated_shift_dict[t][0]))
                                other_channel_image = transform.warp(other_channel_image, tform, preserve_range=True)
                                other_channel_image = other_channel_image.astype(bit_depth, copy=False)

                            output_img_location = ''
                            if DIR_STRUCTURE == 'root_dir':
                                output_img_location = os.path.join(OUTPUT_ALIGNED_PATH, other_channel_image_filename.replace('.tif', '_ALIGNED.tif'))
                            elif DIR_STRUCTURE == 'sub_dir':
                                output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(OUTPUT_ALIGNED_PATH, other_channel_image_filename.replace('.tif', '_ALIGNED.tif')), image_stack_experiment_well[0][3])
                            else:
                                raise Exception('Unknown Directory Structure!')
                            tif_output = TIFF.open(output_img_location, mode='w')
                            tif_output.write_image(other_channel_image, compression='lzw')
                            del tif_output

                    else:
                        print('!!----------- Warning ----------!!')
                        asymmetric_missing_image_log = 'Related morphology channel image does not exist for [experiment: %s, well: %s, channel: %s, timepoint: %s], can not aligned!!\n' %(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3], chl, t)
                        print(asymmetric_missing_image_log)
                        # asymmetric_missing_image_logs.append(asymmetric_missing_image_log)

        # Return dict of current well log
        # current_experiment_well_log.extend(suspicious_misalignments_log)
        return [{(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3][0], int(image_stack_experiment_well[0][3][1:])): (shift_logs, suspicious_misalignment_logs, asymmetric_missing_image_logs)}, {(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3]): pre_calculated_shift_dict}]
    else:
        if SHIFTS:
            raise Exception('Previous calculated shifts for %s, %s does not exists' % (image_stack_experiment_well[0][2], image_stack_experiment_well[0][3]))
        else:
            shift_dict = {}
            # Sort Tx (e.g. T8) in order and loop
            sorted_morphology_timepoint_keys = sorted(morphology_timepoint_dict, key=lambda x: int(x[1:]))
            # Not loop last item to avoid idx+1 index overflow
            for idx, t in enumerate(sorted_morphology_timepoint_keys[:-1]):
                # For all the z, bursts
                for ix, item in enumerate(morphology_timepoint_dict[t]):
                    # Only calc shift at first image in current timepoint, then propogate to other images
                    if ix == 0:
                        # Write first time point image as fixed image
                        if idx == 0:
                            shift_for_cur_timepoint = [0, 0]
                            shift_dict[t] = shift_for_cur_timepoint
                            # fixed_image = TIFF.open(item[0], mode='r')
                            # fixed_image = fixed_image.read_image()


                            # Python: cv2.imread(filename[, flags])
                            # <0 Return the loaded image as is (with alpha channel).
                            fixed_image = cv2.imread(item[0], -1)
                            fixed_image_filename = os.path.basename(item[0])
                            if fixed_image is None:
                                raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (fixed_image_filename))

                            output_img_location = ''
                            if DIR_STRUCTURE == 'root_dir':
                                output_img_location = os.path.join(OUTPUT_ALIGNED_PATH, fixed_image_filename.replace('.tif', '_ALIGNED.tif'))
                            elif DIR_STRUCTURE == 'sub_dir':
                                output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(OUTPUT_ALIGNED_PATH, fixed_image_filename.replace('.tif', '_ALIGNED.tif')), image_stack_experiment_well[0][3])
                            else:
                                raise Exception('Unknown Directory Structure!')
                            tif_output = TIFF.open(output_img_location, mode='w')
                            tif_output.write_image(fixed_image, compression='lzw')
                            del tif_output


                        # moving_image = TIFF.open(morphology_timepoint_dict[sorted_morphology_timepoint_keys[idx+1]][ix][0], mode='r')
                        # moving_image = moving_image.read_image()
                        moving_image = cv2.imread(morphology_timepoint_dict[sorted_morphology_timepoint_keys[idx+1]][ix][0], -1)
                        moving_image_filename = os.path.basename(morphology_timepoint_dict[sorted_morphology_timepoint_keys[idx+1]][ix][0])
                        if moving_image is None:
                            raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (moving_image_filename))
                        bit_depth = moving_image.dtype
                        # print fixed_image_filename
                        # print moving_image_filename

                        fix_by_dft = False
                        while True:
                            # Calulate shift
                            shift = None

                            if ALIGNMENT_ALGORITHM == 'cross_correlation' or (ALIGNMENT_ALGORITHM == 'cross_correlation_dft_combo' and not fix_by_dft):
                                # pixel precision. Subpixel precision does not help while pixel prcision misalign, and it increases computation time
                                shift, error, phasediff = register_translation(fixed_image, moving_image)
                            elif ALIGNMENT_ALGORITHM == 'dft' or (ALIGNMENT_ALGORITHM == 'cross_correlation_dft_combo' and fix_by_dft):
                                # by DFT algorithm
                                dft_result_dict = ird.translation(fixed_image, moving_image)
                                shift = dft_result_dict['tvec']
                                success_number = dft_result_dict['success']
                            else:
                                raise Exception('error type of ALIGNMENT_ALGORITHM.')


                            # Shift sum for current timepoint to first timepoint (Tx --> T0)
                            shift_for_cur_timepoint = [ y+x for y, x in zip(shift_dict[sorted_morphology_timepoint_keys[idx]], shift)]
                            shift_dict[sorted_morphology_timepoint_keys[idx+1]] = shift_for_cur_timepoint

                            shift_log = ''
                            if ALIGNMENT_ALGORITHM == 'cross_correlation' or (ALIGNMENT_ALGORITHM == 'cross_correlation_dft_combo' and not fix_by_dft):
                                shift_log = "Detected subpixel offset[%s, %s, %s][%s --> %s: (x: %s, y: %s)] [%s --> %s: (x: %s, y:%s)] [error:%s; phasediff:%s]" %(morphology_timepoint_dict[t][ix][2], morphology_timepoint_dict[t][ix][3], morphology_timepoint_dict[t][ix][4], sorted_morphology_timepoint_keys[idx+1], t, shift[1], shift[0],  sorted_morphology_timepoint_keys[idx+1], sorted_morphology_timepoint_keys[0], shift_for_cur_timepoint[1], shift_for_cur_timepoint[0], error, phasediff)
                                print(shift_log)
                                shift_logs.append(shift_log)
                            elif ALIGNMENT_ALGORITHM == 'dft' or (ALIGNMENT_ALGORITHM == 'cross_correlation_dft_combo' and fix_by_dft):
                                shift_log = "Detected subpixel offset[%s, %s, %s][%s --> %s: (x: %s, y: %s)] [%s --> %s: (x: %s, y:%s)] [success number:%s]" %(morphology_timepoint_dict[t][ix][2], morphology_timepoint_dict[t][ix][3], morphology_timepoint_dict[t][ix][4], sorted_morphology_timepoint_keys[idx+1], t, shift[1], shift[0],  sorted_morphology_timepoint_keys[idx+1], sorted_morphology_timepoint_keys[0], shift_for_cur_timepoint[1], shift_for_cur_timepoint[0], success_number)
                                if ALIGNMENT_ALGORITHM == 'cross_correlation_dft_combo':
                                    shift_log = '[Switched to DFT]' + shift_log
                                    print(shift_log)
                                    shift_logs[-1] = shift_logs[-1] + '\n' + shift_log
                                    shift_logs.append(shift_log)
                                else:
                                    print(shift_log)
                                    shift_logs.append(shift_log)

                            else:
                                print('error type of ALIGNMENT_ALGORITHM.')


                            # If the shift is dramatic, add to suspicious misalignmenet list
                            y_threshold_shift = fixed_image.shape[0]/9
                            x_threshold_shift = fixed_image.shape[1]/9
                            if (abs(shift[0]) >= y_threshold_shift and abs(shift_for_cur_timepoint[0]) >= y_threshold_shift) or (abs(shift[1]) >= x_threshold_shift and abs(shift_for_cur_timepoint[1]) >= x_threshold_shift):
                                suspicious_misalignment_log = ''
                                if ALIGNMENT_ALGORITHM == 'cross_correlation' or (ALIGNMENT_ALGORITHM == 'cross_correlation_dft_combo' and not fix_by_dft):
                                    suspicious_misalignment_log = "Suspicious Misalignment: [%s, %s, %s][%s --> %s: (x: %s, y: %s)] [%s --> %s: (x: %s, y:%s)] [error:%s; phasediff:%s]" %(morphology_timepoint_dict[t][ix][2], morphology_timepoint_dict[t][ix][3], morphology_timepoint_dict[t][ix][4], sorted_morphology_timepoint_keys[idx+1], t, shift[1], shift[0],  sorted_morphology_timepoint_keys[idx+1], sorted_morphology_timepoint_keys[0], shift_for_cur_timepoint[1], shift_for_cur_timepoint[0], error, phasediff)
                                    if ALIGNMENT_ALGORITHM == 'cross_correlation_dft_combo':
                                        fix_by_dft = True
                                    print(suspicious_misalignment_log)
                                    suspicious_misalignment_logs.append(suspicious_misalignment_log)

                                elif ALIGNMENT_ALGORITHM == 'dft' or (ALIGNMENT_ALGORITHM == 'cross_correlation_dft_combo' and fix_by_dft):
                                    suspicious_misalignment_log = "Suspicious Misalignment: [%s, %s, %s][%s --> %s: (x: %s, y: %s)] [%s --> %s: (x: %s, y:%s)] [success number:%s]" %(morphology_timepoint_dict[t][ix][2], morphology_timepoint_dict[t][ix][3], morphology_timepoint_dict[t][ix][4], sorted_morphology_timepoint_keys[idx+1], t, shift[1], shift[0],  sorted_morphology_timepoint_keys[idx+1], sorted_morphology_timepoint_keys[0], shift_for_cur_timepoint[1], shift_for_cur_timepoint[0], success_number)
                                    if ALIGNMENT_ALGORITHM == 'cross_correlation_dft_combo':
                                        fix_by_dft = False
                                        suspicious_misalignment_log = '[Switched to DFT]' + suspicious_misalignment_log
                                        print(suspicious_misalignment_log)
                                        suspicious_misalignment_logs[-1] = suspicious_misalignment_logs[-1] + '\n' + suspicious_misalignment_log
                                    else:
                                        print(suspicious_misalignment_log)
                                        suspicious_misalignment_logs.append(suspicious_misalignment_log)
                                else:
                                    print('error type of ALIGNMENT_ALGORITHM.')



                            # If Not suspicious misaligned, break the loop
                            else:
                                break
                            # If already fixed by DFT once, break
                            if not fix_by_dft:
                                break

                        # Shift back to fixed. Note the transform usage is opposite to normal. For example this shift from T1 to T0(fixed target) is [x, y],
                        # parameter in transform.warp should be reversed as [-x, -y]
                        tform = transform.SimilarityTransform(translation=(-shift_for_cur_timepoint[1], -shift_for_cur_timepoint[0]))
                        # With preserve_range=True, the original range of the data will be preserved, even though the output is a float image
                        # with the original pixel value preserved. Otherwise default pixel value is [0, 1] for float
                        corrected_image = transform.warp(moving_image, tform, preserve_range=True)

                        # print "before", type(corrected_image), corrected_image.dtype, corrected_image.shape
                        # print "before", corrected_image.max(), corrected_image.min()

                        # transform.warp default returns double float64 ndarray, have to convert back to original bit depth
                        corrected_image = corrected_image.astype(bit_depth, copy=False)

                        # print "after:", type(corrected_image), corrected_image.dtype, corrected_image.shape
                        # print "after", corrected_image.max(), corrected_image.min()


                        # Output the corrected images to file
                        output_img_location = ''
                        if DIR_STRUCTURE == 'root_dir':
                            output_img_location = os.path.join(OUTPUT_ALIGNED_PATH, moving_image_filename.replace('.tif', '_ALIGNED.tif'))
                        elif DIR_STRUCTURE == 'sub_dir':
                            output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(OUTPUT_ALIGNED_PATH, moving_image_filename.replace('.tif', '_ALIGNED.tif')), image_stack_experiment_well[0][3])
                        else:
                            raise Exception('Unknown Directory Structure!')
                        tif_output = TIFF.open(output_img_location, mode='w')
                        tif_output.write_image(corrected_image, compression='lzw')
                        del tif_output

                        # Move to next slice
                        fixed_image = moving_image
                        fixed_image_filename = moving_image_filename
                    # Apply the shift to the other iamges(zs, bursts) in current timepoint
                    else:
                        # other_zb_image = TIFF.open(item[0], mode='r')
                        # other_zb_image = other_zb_image.read_image()
                        other_zb_image = cv2.imread(item[0], -1)
                        other_zb_image_filename = os.path.basename(item[0])
                        if other_zb_image is None:
                            raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (other_zb_image_filename))
                        if idx != 0:
                            bit_depth =other_zb_image.dtype
                            tform = transform.SimilarityTransform(translation=(-shift_dict[t][1], -shift_dict[t][0]))
                            other_zb_image = transform.warp(other_zb_image, tform, preserve_range=True)
                            other_zb_image = other_zb_image.astype(bit_depth, copy=False)

                        output_img_location = ''
                        if DIR_STRUCTURE == 'root_dir':
                            output_img_location = os.path.join(OUTPUT_ALIGNED_PATH, other_zb_image_filename.replace('.tif', '_ALIGNED.tif'))
                        elif DIR_STRUCTURE == 'sub_dir':
                            output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(OUTPUT_ALIGNED_PATH, other_zb_image_filename.replace('.tif', '_ALIGNED.tif')), image_stack_experiment_well[0][3])
                        else:
                            raise Exception('Unknown Directory Structure!')
                        tif_output = TIFF.open(output_img_location, mode='w')
                        tif_output.write_image(other_zb_image, compression='lzw')
                        del tif_output

            # Reduce memory consumption. Maybe help garbage collection
            fixed_image = None
            moving_image = None

            # Apply the same shift to the other channels(Assuming the Microscope is done with position first imaging method)
            for chl in channel_dict:
                if chl != MORPHOLOGY_CHANNEL:
                    other_channel_timepoint_dict = channel_dict[chl]
                    other_channel_log = "Applying shift to other channels [experiment: %s, well: %s, channel: %s]" % (image_stack_experiment_well[0][2], image_stack_experiment_well[0][3], chl)
                    print(other_channel_log)
                    # current_experiment_well_log.append(other_channel_log)

                    # Sort Tx (e.g. T8) in order and loop
                    sorted_other_channel_timepoint_keys = sorted(other_channel_timepoint_dict, key=lambda x: int(x[1:]))
                    # No idx+1, so enumerate all timepoints
                    for idx, t in enumerate(sorted_other_channel_timepoint_keys):
                        # Check if current image has related morphology shift calculated, in case of asymmetric images
                        if t in shift_dict:
                            # For all the z, bursts
                            for ix, item in enumerate(other_channel_timepoint_dict[t]):
                                # other_channel_image = TIFF.open(item[0], mode='r')
                                # other_channel_image = other_channel_image.read_image()
                                other_channel_image = cv2.imread(item[0], -1)
                                other_channel_image_filename = os.path.basename(item[0])
                                if other_channel_image is None:
                                    raise Exception('%s is corrupted, please run Corrupted Images Detector to find out all the corrupted images.' % (other_channel_image_filename))
                                if idx != 0:
                                    bit_depth =other_channel_image.dtype
                                    tform = transform.SimilarityTransform(translation=(-shift_dict[t][1], -shift_dict[t][0]))
                                    other_channel_image = transform.warp(other_channel_image, tform, preserve_range=True)
                                    other_channel_image = other_channel_image.astype(bit_depth, copy=False)

                                output_img_location = ''
                                if DIR_STRUCTURE == 'root_dir':
                                    output_img_location = os.path.join(OUTPUT_ALIGNED_PATH, other_channel_image_filename.replace('.tif', '_ALIGNED.tif'))
                                elif DIR_STRUCTURE == 'sub_dir':
                                    output_img_location = utils.reroute_imgpntr_to_wells(os.path.join(OUTPUT_ALIGNED_PATH, other_channel_image_filename.replace('.tif', '_ALIGNED.tif')), image_stack_experiment_well[0][3])
                                else:
                                    raise Exception('Unknown Directory Structure!')
                                tif_output = TIFF.open(output_img_location, mode='w')
                                tif_output.write_image(other_channel_image, compression='lzw')
                                del tif_output
                        else:
                            print('!!----------- Warning ----------!!')
                            asymmetric_missing_image_log = 'Related morphology channel image does not exist for [experiment: %s, well: %s, channel: %s, timepoint: %s], can not aligned!!\n' %(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3], chl, t)
                            print(asymmetric_missing_image_log)
                            asymmetric_missing_image_logs.append(asymmetric_missing_image_log)

            # Return dict of current well log
            # current_experiment_well_log.extend(suspicious_misalignments_log)
            return [{(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3][0], int(image_stack_experiment_well[0][3][1:])): (shift_logs, suspicious_misalignment_logs, asymmetric_missing_image_logs)}, {(image_stack_experiment_well[0][2], image_stack_experiment_well[0][3]): shift_dict}]


def variety_alignments():
    input_image_stack_list = get_image_tokens_list(INPUT_MONTAGED_PATH, ROBO_NUMBER, IMAGING_MODE)
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
    # map_results = workers_pool.map_async(register_stack, input_image_stack_list, chunksize=chunk_size).get(99999)

    # Use pool.imap to avoid MemoryError, but KeyboardInterrupt not work
    map_results = workers_pool.imap(register_stack, input_image_stack_list, chunksize=chunk_size)

    # Single instance test
    # register_stack(input_image_stack_list[0])
    for r in map_results:
        LOG_INFO.update(r[0])
        SHIFTS.update(r[1])
    workers_pool.close()
    workers_pool.join()


if __name__ == '__main__':
    # # --- Command line test ---
    # start_time = datetime.datetime.utcnow()
    # INPUT_MONTAGED_PATH = '/Users/guangzhili/GladStone/AutoAlignment/data/BioP7asynA/MontagedImages'
    # OUTPUT_ALIGNED_PATH = '/Users/guangzhili/GladStone/AutoAlignment/data/BioP7asynA/AlignedCCG'
    # MORPHOLOGY_CHANNEL = 'RFP-DFTrCy5'
    # VALID_WELLS = ['A10', 'B10', 'C10', 'D10', 'E10', 'F10', 'G10', 'H10']
    # # VALID_WELLS = ['A10',  'F10']

    # # VALID_TIMEPOINTS = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
    # # VALID_TIMEPOINTS = ['T0', 'T1', 'T2', 'T3', 'T4']
    # VALID_TIMEPOINTS = ['T0', 'T1', 'T2']
    # ALIGNMENT_ALGORITHM = 'cross_correlation_dft_combo'
    # ROBO_NUMBER = 3
    # # epi or confocal
    # IMAGING_MODE = 'epi'

    # try:
    #     os.makedirs(OUTPUT_ALIGNED_PATH)
    # except OSError:
    #     if not os.path.isdir(OUTPUT_ALIGNED_PATH):
    #         raise

    # # Run alignment
    # variety_alignments()

    # end_time = datetime.datetime.utcnow()
    # print 'Alignment correction run time:', end_time-start_time





    # --- For Galaxy run ---
    start_time = datetime.datetime.utcnow()
    # Parser
    parser = argparse.ArgumentParser(
        description="Variety Alignments")
    parser.add_argument("input_dict",
        help="Load input variable dictionary.")
    parser.add_argument("alignment_algorithm",
        help="Algorithm for alignment.")
    parser.add_argument("input_path",
        help="Folder path to input data.")
    parser.add_argument("output_path",
        help="Folder path to ouput results.")
    parser.add_argument("output_dict",
        help="Write variable dictionary.")
    parser.add_argument("--shift",
        help="Previous calculated shifts.")
    args = parser.parse_args()

    # Load path dict
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))

    # Initialize parameters
    ALIGNMENT_ALGORITHM = args.alignment_algorithm
    INPUT_MONTAGED_PATH = args.input_path
    OUTPUT_ALIGNED_PATH = args.output_path
    MORPHOLOGY_CHANNEL = var_dict["MorphologyChannel"]
    VALID_WELLS = var_dict['Wells']
    VALID_TIMEPOINTS = var_dict['TimePoints']
    DIR_STRUCTURE = var_dict['DirStructure']
    ROBO_NUMBER = int(var_dict['RoboNumber'])
    IMAGING_MODE = var_dict['ImagingMode']
    CHANNEL_SET = set()
    outfile = args.output_dict

    if args.shift and args.shift.strip():
        SHIFTS = pickle.load(open(args.shift.strip(), 'rb'))

    # Make sure more than two timepoints to align
    assert len(VALID_TIMEPOINTS) > 1, 'Less than two time points, no need to use alignment module.'

    # Confirm given input/output folders exist
    assert os.path.isdir(INPUT_MONTAGED_PATH), 'Confirm the given path for input data exists.'
    assert os.path.isdir(OUTPUT_ALIGNED_PATH), 'Confirm the given path for output results exists.'

    # # Create output folder
    # try:
    #     os.makedirs(OUTPUT_ALIGNED_PATH)
    # except OSError:
    #     if not os.path.isdir(OUTPUT_ALIGNED_PATH):
    #         raise

    # Run alignment
    variety_alignments()

    # Print Total process time
    end_time = datetime.datetime.utcnow()
    print('Alignment correction run time:', end_time-start_time)


    # Output for user
    print('Montaged images were aligned.')
    print('Output was written to:')
    print(OUTPUT_ALIGNED_PATH)
    print('Check out %s_ResultLog.txt for detail log.' % ALIGNMENT_ALGORITHM)

    # For select_analysis_module input, set var_dict['OutputPath']
    var_dict["OutputPath"] = OUTPUT_ALIGNED_PATH

    # Save calculated shift to Galaxy dict
    var_dict['CalculatedShift'] = SHIFTS

    # Save dict to file
    with open(outfile, 'wb') as ofile:
        pickle.dump(var_dict, ofile)

    # If no previous calculated shifts
    if not (args.shift and args.shift.strip()):
        # Output console log info to file
        with open(os.path.join(OUTPUT_ALIGNED_PATH, '%s_ResultLog.txt' % ALIGNMENT_ALGORITHM), 'wb') as logfile:
            logfile.write('Alignmet Algorithm %s Result:\n\n' % ALIGNMENT_ALGORITHM)
            log_values = [LOG_INFO[ewkey] for ewkey in sorted(LOG_INFO)]

            # Write the shift log order by timepoint
            for idx in range(len(VALID_TIMEPOINTS)-1):
                for log_tuple in log_values:
                    if log_tuple[0] != [] and idx < len(log_tuple[0]):
                        logfile.write(log_tuple[0][idx]+'\n')
            # Write suspicious misalignment order by well
            for log_tuple in log_values:
                for suspicious_misalignment in log_tuple[1]:
                    logfile.write(suspicious_misalignment+'\n')
            # Write asymmetic missing images order by well
            for log_tuple in log_values:
                for asymmetric_missing_image in log_tuple[2]:
                    logfile.write(asymmetric_missing_image+'\n')
        # Save the shift to dict file
        with open(os.path.join(OUTPUT_ALIGNED_PATH, 'calculated_shift_%s.dict' % time.strftime('%Y%m%d-%H%M%S')), 'wb') as shiftfile:
            pickle.dump(SHIFTS, shiftfile)

    utils.save_user_args_to_csv(args, OUTPUT_ALIGNED_PATH, 'variety_alignments_mp')