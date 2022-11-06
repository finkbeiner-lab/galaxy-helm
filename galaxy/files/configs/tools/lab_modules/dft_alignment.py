import numpy as np
import skimage
import imreg_dft as ird
from libtiff import TIFF 
import os
from operator import itemgetter
import sys
import multiprocessing
import time
from skimage import transform
import pickle, datetime, argparse



NUMBER_OF_PROCESSORS = multiprocessing.cpu_count()

INPUT_MONTAGED_PATH = ''
OUTPUT_ALIGNED_PATH = '' 
MORPHOLOGY_CHANNEL = ''
VALID_WELLS = []
VALID_TIMEPOINTS = []
CHANNEL_SET = set()
LOG_INFO = {}




def get_image_stack_list():
    '''
    Input is Montaged time point separated images
    '''    
    # Time separated image name example(for references):
    #  PID20150217_BioP7asynA_T0_0_A1_1_RFP-DFTrCy5_BG_MONTAGE_ALIGNED_CROPPED.tif
    #  PID20150904_PGPSTest_T1_8_A7_MONTAGE_RFP-DFTrCy5.tif
    #  PID20150217_BioP7asynA_T1_12_A10_1_RFP-DFTrCy5_MN.tif
    stack_dict = {}

    # use os.walk() to recursively iterate through a directory and all its subdirectories
    image_paths = [os.path.join(root, name) for root, dirs, files in os.walk(INPUT_MONTAGED_PATH) for name in files if name.endswith('.tif')]
    for image_path in image_paths:
        image_name = os.path.basename(image_path)
        name_tokens = image_name.split('_')
        pid_token = name_tokens[0]
        experiment_name_token = name_tokens[1]
        timepoint_token = name_tokens[2]
        if timepoint_token not in VALID_TIMEPOINTS:
            continue
        numofhours_token = int(name_tokens[3])
        well_id_token = name_tokens[4]
        if well_id_token not in VALID_WELLS:
            continue
        # Handle possible(Pipeline_Pilot_Source) last token RFP-DFTrCy5.tif, in that case, get rid of .tif suffix
        channel_token = name_tokens[6].replace('.tif', '')   
        CHANNEL_SET.add(channel_token) 
        # Split well id token to make sorting easier
        # Well ID example: H12
        experiment_well_key = (experiment_name_token, well_id_token[0], int(well_id_token[1:]))

        if experiment_well_key in stack_dict:
            stack_dict[experiment_well_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token])
        else:
            stack_dict[experiment_well_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token]]    
        
    return [stack_dict[ewkey] for ewkey in sorted(stack_dict)]
 

def register_stack(image_stack_experiment_well):     
    '''
    Worker process
    image_stack:  a list of time series images tokens for the one experiment-well, including possible multiple channels
    image_stack format example: [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token], ...]    

    '''     
    current_experiment_well_log = []
    suspicious_misalignments_log = []
    channel_dict = {} 
    for tks in image_stack_experiment_well:
        if tks[4] in channel_dict:
            channel_dict[tks[4]].append(tks)
        else:
            channel_dict[tks[4]] = [tks]

    # Order by timepoints
    for ch in channel_dict:
        channel_dict[ch] = sorted(channel_dict[ch], key=itemgetter(6))           

    
    # Process morphology channel first, then use the calculated shift to apply to other channels
    morphology_stack = channel_dict[MORPHOLOGY_CHANNEL]
    num_of_timepoints = len(morphology_stack)

    processing_log = "Processing [experiment: %s, well: %s, channel: %s]" % (morphology_stack[0][2], morphology_stack[0][3], MORPHOLOGY_CHANNEL)
    print processing_log
    current_experiment_well_log.append(processing_log)

    fixed_image = None
    moving_image = None
    shift_for_cur_timepoint = [0, 0]
    shift_list = []

 
    for idx in range(num_of_timepoints-1):
        # Write first time point image as fixed image
        if idx == 0:
            fixed_image = TIFF.open(morphology_stack[idx][0], mode='r')
            fixed_image = fixed_image.read_image()
            fixed_image_filename = os.path.basename(morphology_stack[idx][0])
            tif_output = TIFF.open(os.path.join(OUTPUT_ALIGNED_PATH, fixed_image_filename.replace('.tif', '_ALIGNED.tif')), mode='w')
            # Note the compression parameter
            tif_output.write_image(fixed_image, compression='lzw')
            # flushes data to disk
            del tif_output 

        
        moving_image = TIFF.open(morphology_stack[idx+1][0], mode='r')
        moving_image = moving_image.read_image()
        
        moving_image_filename = os.path.basename(morphology_stack[idx+1][0])
        bit_depth = moving_image.dtype
        # print fixed_image_filename
        # print moving_image_filename

        # Calulate shift 
        # by DFT algorithm
        shift, success_number = ird.translation(fixed_image, moving_image)
        

        # Shift sum for current timepoint to first timepoint
        shift_for_cur_timepoint = [ y+x for y, x in zip(shift_for_cur_timepoint, shift)]
        shift_list.append(shift_for_cur_timepoint)

        shift_log = "Detected subpixel offset[%s, %s, %s][%s --> %s: (x: %s, y: %s)] [%s --> %s: (x: %s, y:%s)] [success number:%s]" %(morphology_stack[0][2], morphology_stack[0][3], morphology_stack[0][4], morphology_stack[idx+1][5], morphology_stack[idx][5], shift[1], shift[0],  morphology_stack[idx+1][5], morphology_stack[0][5], shift_for_cur_timepoint[1], shift_for_cur_timepoint[0], success_number)
        print shift_log
        current_experiment_well_log.append(shift_log)

        # If the shift is dramatic, add to suspicious misalignmenet list
        y_threshold_shift = fixed_image.shape[0]/9
        x_threshold_shift = fixed_image.shape[1]/9
        if (abs(shift[0]) >= y_threshold_shift and abs(shift_for_cur_timepoint[0]) >= y_threshold_shift) or (abs(shift[1]) >= x_threshold_shift and abs(shift_for_cur_timepoint[1]) >= x_threshold_shift):
            suspicious_misalignment = "Suspicious Misalignment: [%s, %s, %s][%s --> %s: (x: %s, y: %s)] [%s --> %s: (x: %s, y:%s)] [success number:%s]" %(morphology_stack[0][2], morphology_stack[0][3], morphology_stack[0][4], morphology_stack[idx+1][5], morphology_stack[idx][5], shift[1], shift[0],  morphology_stack[idx+1][5], morphology_stack[0][5], shift_for_cur_timepoint[1], shift_for_cur_timepoint[0], success_number)
            print suspicious_misalignment
            suspicious_misalignments_log.append(suspicious_misalignment)
      
        
       
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
        tif_output = TIFF.open(os.path.join(OUTPUT_ALIGNED_PATH, moving_image_filename.replace('.tif', '_ALIGNED.tif')), mode='w')
        # Note the compression parameter
        tif_output.write_image(corrected_image, compression='lzw')
        # flushes data to disk
        del tif_output 

        # Move to next slice
        fixed_image = moving_image
        fixed_image_filename = moving_image_filename


    # Apply the same shift to the other channels(Assuming the Microscope is done with position first imaging method)
    for chl in channel_dict:
        if chl != MORPHOLOGY_CHANNEL:
            cur_channel_stack = channel_dict[chl]
            other_channel_log = "Applying shift to other channels [experiment: %s, well: %s, channel: %s]" % (cur_channel_stack[0][2], cur_channel_stack[0][3], chl)
            print other_channel_log
            current_experiment_well_log.append(other_channel_log)

            fixed_image = None
            moving_image = None
            for ix in range(num_of_timepoints-1):
                # Write first time point image as fixed image
                if ix == 0:
                    
                    fixed_image = TIFF.open(cur_channel_stack[ix][0], mode='r')
                    fixed_image = fixed_image.read_image()
                    fixed_image_filename = os.path.basename(cur_channel_stack[ix][0])
                    tif_output = TIFF.open(os.path.join(OUTPUT_ALIGNED_PATH, fixed_image_filename.replace('.tif', '_ALIGNED.tif')), mode='w')
                    # Note the compression parameter
                    tif_output.write_image(fixed_image, compression='lzw')
                    # flushes data to disk
                    del tif_output 

                
                moving_image = TIFF.open(cur_channel_stack[ix+1][0], mode='r')
                moving_image = moving_image.read_image()
                
                moving_image_filename = os.path.basename(cur_channel_stack[ix+1][0])
                bit_depth = moving_image.dtype
                # print fixed_image_filename
                # print moving_image_filename
               
                # Shift back to fixed. Note the transform usage is opposite to normal. For example this shift from T1 to T0(fixed target) is [x, y],
                # parameter in transform.warp should be reversed as [-x, -y]
                tform = transform.SimilarityTransform(translation=(-shift_list[ix][1], -shift_list[ix][0]))
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
                tif_output = TIFF.open(os.path.join(OUTPUT_ALIGNED_PATH, moving_image_filename.replace('.tif', '_ALIGNED.tif')), mode='w')
                # Note the compression parameter
                tif_output.write_image(corrected_image, compression='lzw')
                # flushes data to disk
                del tif_output 

                # Move to next slice
                fixed_image = moving_image
                fixed_image_filename = moving_image_filename
    # Return dict of current well log
    current_experiment_well_log.extend(suspicious_misalignments_log)
    return {(morphology_stack[0][2], morphology_stack[0][3][0], int(morphology_stack[0][3][1:])): current_experiment_well_log}        
                        

def cross_correlation():
    input_image_stack_list = get_image_stack_list()
    # Initialize workers pool
    workers_pool = multiprocessing.Pool(processes=NUMBER_OF_PROCESSORS)
    # Feed data to workers in parallel
    # 99999 is timeout, can be 1 or 99999 etc, used for KeyboardInterrupt multiprocessing
    map_results = workers_pool.map_async(register_stack, input_image_stack_list).get(99999)   
    for r in map_results:
        LOG_INFO.update(r) 
    workers_pool.close()
    workers_pool.join()
   

if __name__ == '__main__':
    # # --- Command line test ---
    # start_time = datetime.datetime.utcnow()
    # INPUT_MONTAGED_PATH = '/Users/guangzhili/GladStone/AutoAlignment/data/BioP7asynA/MontagedImages'
    # OUTPUT_ALIGNED_PATH = '/Users/guangzhili/GladStone/AutoAlignment/data/BioP7asynA/AlignedCCG' 
    # MORPHOLOGY_CHANNEL = 'RFP-DFTrCy5'
    # VALID_WELLS = ['A10', 'B10', 'C10', 'D10', 'E10', 'F10', 'G10', 'H10']
    # # VALID_TIMEPOINTS = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']

    # try:     
    #     os.makedirs(OUTPUT_ALIGNED_PATH)
    # except OSError:
    #     if not os.path.isdir(OUTPUT_ALIGNED_PATH):
    #         raise
    
    # # Run alignment
    # cross_correlation()

    # end_time = datetime.datetime.utcnow()
    # print 'Alignment correction run time:', end_time-start_time





    # --- For Galaxy run ---
    start_time = datetime.datetime.utcnow()
    # Parser
    parser = argparse.ArgumentParser(
        description="Align images using DFT.")
    parser.add_argument("input_dict", 
        help="Load input variable dictionary.")
    parser.add_argument("input_path", 
        help="Folder path to input data.")
    parser.add_argument("output_path",
        help="Folder path to ouput results.")
    parser.add_argument("output_dict", 
        help="Write variable dictionary.")
    args = parser.parse_args()

    # Load path dict
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))

    # Initialize parameters
    INPUT_MONTAGED_PATH = args.input_path
    OUTPUT_ALIGNED_PATH = args.output_path
    MORPHOLOGY_CHANNEL = var_dict["MorphologyChannel"]
    VALID_WELLS = var_dict['Wells']
    VALID_TIMEPOINTS = var_dict['TimePoints']
    CHANNEL_SET = set()
    outfile = args.output_dict
     
    # Create output folder
    try:     
        os.makedirs(OUTPUT_ALIGNED_PATH)
    except OSError:
        if not os.path.isdir(OUTPUT_ALIGNED_PATH):
            raise

    # Run alignment
    cross_correlation()

    # Print Total process time
    end_time = datetime.datetime.utcnow()
    print 'Alignment correction run time:', end_time-start_time


    # Output for user
    print 'Montaged images were aligned.'
    print 'Output was written to:'
    print OUTPUT_ALIGNED_PATH
    print 'Check out ResultLog.txt for detail log.'

    # For select_analysis_module input, set var_dict['OutputPath']
    var_dict["OutputPath"] = OUTPUT_ALIGNED_PATH

    # Save dict to file
    with open(outfile, 'wb') as ofile: 
        pickle.dump(var_dict, ofile)     


    # Output console log info to file
    with open(os.path.join(OUTPUT_ALIGNED_PATH, 'ResultLog.txt'), 'wb') as logfile: 
        log_values = [LOG_INFO[ewkey] for ewkey in sorted(LOG_INFO)]
        for log in log_values:
            logfile.write(log[0]+'\n')

        for t in range(1, len(VALID_TIMEPOINTS)+len(CHANNEL_SET)-1):
            for log in log_values:
                logfile.write(log[t]+'\n')
        print '\n\n-------------------\n'
        for log in log_values:
            if len(log) > len(VALID_TIMEPOINTS)+len(CHANNEL_SET)-1:
                for t in range(len(VALID_TIMEPOINTS)+len(CHANNEL_SET)-1, len(log)):
                    print log[t]+'\n'
                    logfile.write(log[t]+'\n')            
        












