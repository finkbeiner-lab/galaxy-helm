'''
Functions dealing with setting up configuration parameters.
Main function collects tokens from filenames in given path,
parses the tokens and writes the parameters to a dictionary.
'''

import utils, sys, os, argparse
import pickle, pprint, shutil, datetime
from utils import numericalSort

def make_results_folders(input_path, output_path):

    '''Generate folder hierarchy for each output step.'''

    bg_corrected_path = os.path.join(output_path, 'BackgroundCorrected') #background_removal
    montaged_path = os.path.join(output_path, 'MontagedImages') #montage
    aligned_path = os.path.join(output_path, 'AlignedImages') #alignment
    cropped_path = os.path.join(output_path, 'CroppedImages') #shift_crop
    results = os.path.join(output_path, 'OverlaysTablesResults') #overlay_tracks and extract_cell_info
    cell_masks = os.path.join(output_path, 'CellMasks') # segmentation
    qc_path = os.path.join(output_path, 'QualityControl') #segmentation visualization
    stacking_scratch_path = os.path.join(output_path, 'StackingTemp')

    output_dir_names = ['BackgroundCorrected', 'MontagedImages',
         'AlignedImages', 'CroppedImages', 'QualityControl',
         'OverlaysTablesResults', 'CellMasks', 'StackingTemp']
    output_subdirs = [bg_corrected_path, montaged_path,
        aligned_path, cropped_path, qc_path, results, cell_masks]

    utils.create_folder_hierarchy(output_subdirs, output_path)

    var_dict = {'RawImageData': input_path}
    for output_dir_name, output_subdir in zip(
        output_dir_names, output_subdirs):

        var_dict[output_dir_name] = output_subdir

    return var_dict

def get_exp_params_general(var_dict, all_files, morph_channel):
    '''
    Using filenames in input directory,
    collect experiment parameters (wells, timepoints, channels),
    add them to var_dict.

    Currently supported naming schemes
    Robo3: PIDdate_ExptName_Timepoint_Hours_Well_MontageNumber_Channel.tif
    Robo4 epi: PIDdate_ExptName_Timepoint_Hours_Well_MontageNumber_Channel_FilterDet_Camera.tif
    Robo4 confocal: PIDdate_ExptName_Timepoint_Hours_Well_MontageNumber_FilterDet1_FilterDet2_[FilterDet3]_Channel_DepthIndex_DepthIncrement_Camera.tif
    Robo0: PIDdate_ExptName_Timepoint_Hours-BurstIndex_Well_MontageNumber_Channel_TimeIncrement_DepthIndex_DepthIncrement.tif
    '''

    num_tokens = len(os.path.basename(all_files[0]).split('_'))
    print 'Number of tokens in file name:', num_tokens
    print 'Expected number of tokens is:', var_dict['NumberTokens']

    var_dict['TimePoints'] = utils.get_timepoints(all_files)
    var_dict['Wells'] = utils.get_wells(all_files)
    var_dict['Channels'] = utils.get_channels(
        all_files, var_dict['RoboNumber'], light_path=var_dict['ImagingMode'])
    var_dict['MorphologyChannel'] = utils.get_ref_channel(morph_channel, var_dict['Channels'])
    var_dict['PlateID'] = utils.get_plate_id(all_files)
    var_dict['Bursts'] = utils.get_burst_iter(all_files)
    var_dict['BurstIDs'] = utils.get_bursts(all_files)
    var_dict['Depths'] = utils.get_depths(all_files, var_dict['RoboNumber'])
    if 'ZMAX' not in var_dict['Depths'] and 'ZAVG' not in var_dict['Depths']:
        var_dict['Depths'] = [int(zdepth) for zdepth in var_dict['Depths']]

    return var_dict

def main():
    '''Point of entry.'''

    # Argument parsing
    parser = argparse.ArgumentParser(description="Process cell data.")
    parser.add_argument("input_path",
        help="Folder path to input data.")
    parser.add_argument("output_path",
        help="Folder path to ouput results.")
    parser.add_argument("dir_structure", help="Type of directory structure.")
    parser.add_argument("robo_num",
        type=int,
        help="Robo number")
    parser.add_argument("imaging_mode",
        help="Light path (epi or confocal).")
    parser.add_argument("morph_channel",
        help="A unique string corresponding to morphology channel.")
    parser.add_argument("num_cols",
        type=int,
        help="Number of horizontal images in montage.")
    parser.add_argument("num_rows",
        type=int,
        help="Number of vertical images in montage.")
    parser.add_argument("pixel_overlap",
        type=int,
        help="Number of pixels to overlap during stitching.")
    parser.add_argument("outfile",
        help="Name of output dictionary.")
    parser.add_argument("--chosen_wells", "-cw",
        dest = "chosen_wells", default = '',
        help="Folder path to input data.")
    parser.add_argument("--chosen_timepoints", "-ct",
        dest = "chosen_timepoints", default = '',
        help="Folder path to input data.")
    args = parser.parse_args()

    # Set up I/O parameters
    input_path = args.input_path
    output_path = args.output_path
    dir_structure = args.dir_structure
    robo_num = args.robo_num
    imaging_mode = args.imaging_mode
    morph_channel = args.morph_channel
    outfile = args.outfile

    # Confirm given folders exist
    if not os.path.exists(input_path):
        print 'Confirm the given path for data exists.'
    assert os.path.exists(input_path), 'Confirm the given path for data exists.'
    if not os.path.exists(output_path):
        print 'Confirm the given path for results exists.'
    assert os.path.exists(output_path), 'Confirm the given path for results exists.'
    assert input_path != output_path, 'With new well subdirectory requirement, output destination must be different than input.'

    # Confirm that morphology channel is given
    assert morph_channel != '', 'Confirm you have provided a morphology channel.'

    # Set up dictionary parameters
    var_dict = make_results_folders(input_path, output_path)
    var_dict['Resolution'] = -1 #0 is 8-bit, -1 is 16-bit
    var_dict['NumberHorizontalImages'] = args.num_cols
    var_dict['NumberVerticalImages'] = args.num_rows
    var_dict['ImagePixelOverlap'] = args.pixel_overlap
    var_dict['DirStructure'] = args.dir_structure
    var_dict['RoboNumber'] = args.robo_num
    var_dict['ImagingMode'] = args.imaging_mode


    start_time = datetime.datetime.utcnow()

    # all_files = utils.get_all_files(input_path)
    all_files = utils.get_all_files_all_subdir(input_path, verbose=False)
    var_dict['AnalyzedFiles'] = all_files
    assert len(all_files) > 0, 'No files to process.'
    assert robo_num == 3 or robo_num == 4 or robo_num == 0, 'No analysis avialable for that robo number.'
    if robo_num == 3:
        var_dict['NumberTokens'] = 7
    if robo_num == 4:
        var_dict['NumberTokens'] = 13
    if robo_num == 0:
        var_dict['NumberTokens'] = 10
    var_dict = get_exp_params_general(var_dict, all_files, morph_channel)

    # Get parameters for saving json metadata at the end
    var_dict['GalaxyOutputPath'] = args.output_path
    var_dict['ExperimentName'] = os.path.basename(all_files[0]).split('_')[1]

    # Handle processing specified wells
    user_chosen_wells = args.chosen_wells
    if user_chosen_wells !='':
        user_chosen_wells = utils.get_iter_from_user(user_chosen_wells, 'wells')
        print 'Initial wells', var_dict["Wells"]
        var_dict["Wells"] = user_chosen_wells
        print 'Selected wells', var_dict["Wells"]

    # Handle processing specified timepoints
    user_chosen_timepoints = args.chosen_timepoints
    if user_chosen_timepoints !='':
        user_chosen_timepoints = utils.get_iter_from_user(user_chosen_timepoints, 'timepoints')
        print 'Initial timepoints', var_dict["TimePoints"]
        var_dict["TimePoints"] = user_chosen_timepoints
        print 'Selected timepoints', var_dict["TimePoints"]

    # Print status of dict:
    print 'Summary of', var_dict['RawImageData'], 'data.'
    print 'Using Robo', var_dict['RoboNumber'], 'and', var_dict['MorphologyChannel'], 'morphology channel.'
    summary_features = ['AnalyzedFiles', 'TimePoints', 'Wells', 'Channels',
        'MorphologyChannel', 'Bursts', 'BurstIDs', 'Depths']
    for feature in var_dict.keys():
        if feature in summary_features:
            if feature == 'MorphologyChannel':
                print feature, ':', var_dict[feature]
            elif feature == 'AnalyzedFiles':
                print feature, ':', len(var_dict[feature])
            else:
                print feature, '-', 'Total', len(var_dict[feature]), ':', var_dict[feature]

    end_time = datetime.datetime.utcnow()
    print 'Configuration run time:', end_time-start_time

    # ----Output for user and save dict----------
    print 'ImagePipeline will process raw data in:'
    print input_path
    print 'Results will be written to:'
    print output_path

    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    # outfile = os.rename('var_dict.p', outfile)
    outfile = shutil.move('var_dict.p', outfile)
    utils.save_user_args_to_csv(args, output_path, 'create_folders')

if __name__ == "__main__":
    main()
