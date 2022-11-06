'''
Runs individual analysis modules.
'''
import sys, os, utils, pickle, datetime, argparse, pprint, shutil
from background_removal import background_removal
from montage import montage
from utils import order_wells_correctly
from create_folders import get_exp_params_robo2_robo3
from create_folders import get_exp_params_robo4
from alignment import alignment
from shift_crop import shift_crop
from segmentation import segmentation
from tracking import tracking
from overlay_tracks import overlay_tracks
from extract_cell_info import extract_cell_info

def module_to_function_dict(var_dict, module):
    '''
    Take parameters collected in variabe dictionary.
    Instantiate and use as arguments for individual modules.
    '''

    # Paths
    input_path = var_dict["InputPath"]
    output_path = var_dict["OutputPath"]
    qc_dest_path = var_dict["QualityControl"]
    try: path_to_masks = var_dict["CellMasks"]
    except KeyError: path_to_masks = ''

    module_dict = {
        "BgCorrection": (background_removal, (var_dict, input_path, output_path, qc_dest_path)),
        "Montage": (montage, (var_dict, input_path, output_path)),
        "AlignIJ": (alignment, (var_dict, input_path, output_path, qc_dest_path)),
        "CropBorders": (shift_crop, (var_dict, input_path, output_path)),
        "Segment": (segmentation, (var_dict, input_path, output_path, qc_dest_path)),
        "Track": (tracking, (var_dict, input_path, output_path)),
        "Overlay": (overlay_tracks, (var_dict, input_path, output_path, path_to_masks)),
        "Extract": (extract_cell_info, (var_dict, input_path, output_path)),
        }

    return module_dict

def overwrite_io_paths(var_dict, output_path):
    '''
    If var_dict is given as input, the data input_path 
    for this module will be set from var_dict['OutputPath'] 
    of the previous step. The new var_dict['OutputPath'] 
    will be set as output_path.'''

    # print 'Initial input (passed var_dict)', var_dict['InputPath']
    # print 'Initial output (passed var_dict)', var_dict["OutputPath"]

    input_path = var_dict["OutputPath"]
    var_dict['InputPath']  = input_path
    var_dict["OutputPath"] = output_path

    # print 'Final input (new var_dict)', var_dict['InputPath']
    # print 'Final output (new var_dict)', var_dict["OutputPath"]

    return var_dict

if __name__ == '__main__':

   #----Argument parsing--------
    parser = argparse.ArgumentParser(description="Process iPSC data.")
    parser.add_argument("--input_path", "-ip", 
        dest = "input_path", default = '', 
        help="Folder path to input data.")
    parser.add_argument("output_path",
        help="Folder path to ouput results.")
    parser.add_argument("morph_channel",
        help="A unique string corresponding to morphology channel.")
    parser.add_argument("robo_num",
        type=int,
        help="Robo number")
    parser.add_argument("light_path",
        help="Light path (epi or confocal).")
    parser.add_argument("num_cols",
        type=int, 
        help="Number of horizontal images in montage.")
    parser.add_argument("num_rows",
        type=int,
        help="Number of vertical images in montage.")
    parser.add_argument("pixel_overlap",
        type=int,
        help="Number of pixels to overlap during stitching.")
    parser.add_argument("module",
        help="Module.py to execute.")
    parser.add_argument("outfile",
        help="Name of output dictionary.")
    parser.add_argument("--min_cell",
        dest="min_cell", type=int, default=50,
        help="Minimum feature size considered as cell.")
    parser.add_argument("--max_cell",
        dest="max_cell", type=int, default=2500,
        help="Maximum feature size considered as cell.")
    parser.add_argument("--threshold_percent", "-tp",
        dest="threshold_percent", type=float, default=0.1,
        help="Threshold value as a percent of maximum intensity.")
    parser.add_argument("--var_dict", "-vd",
        dest="var_dict", default='',
        help="Dictionary of variables from previous step.")
    parser.add_argument("--chosen_wells", "-cw", 
        dest = "chosen_wells", default = '', 
        help="Folder path to input data.") 
    parser.add_argument("--chosen_timepoints", 
        dest = "chosen_timepoints", default = '', 
        help="Folder path to input data.")   
    parser.add_argument("--channel_token",
        dest = "chosen_channel_token", default = '', 
        help="Token with unique channel (index from zero).") 
    args = parser.parse_args()

    # ----Load parameters------------------------
    module = args.module
    input_path  = args.input_path
    output_path = args.output_path
    morph_tag = args.morph_channel
    light_path = args.light_path
    infile = args.var_dict

    # ----Handle module type------
    print module, 'result.'

    # Handle modules that require dict
    if module == 'Extract' or module == 'Overlay':
        print 'Results from tracking output must be passed in.'
        assert args.var_dict != '', module+' module requires input data from Tracking step.'
        # var_dict = pickle.load(open(infile, 'rb'))

    if args.var_dict != '':
        print '----Read in:------------'
        var_dict = pickle.load(open(infile, 'rb'))
        if module == 'Extract' or module == 'Overlay':
            var_dict['InputPath'] = args.input_path
            var_dict['OutputPath'] = args.output_path
        else:
            overwrite_io_paths(var_dict, output_path)
    else:
        print '----Generated:----------'
        files_to_analyze = utils.get_all_files(input_path)
        print 'The files:', len(files_to_analyze)
        print  input_path
        assert len(files_to_analyze) > 0, 'Warning: Input path has no files.'
        var_dict = {
            'ImagePixelOverlap': args.pixel_overlap,
            'InputPath': args.input_path,
            'OutputPath': args.output_path,
            'QualityControl': os.path.join(output_path, 'QualityControl'),
            'MaxCellSize': args.max_cell,
            'MinCellSize': args.min_cell,
            'NumberHorizontalImages': args.num_cols,
            'NumberVerticalImages': args.num_rows,
            'Resolution': -1,
            'RoboNumber': args.robo_num,
            'IntensityThreshold': args.threshold_percent
        }
        if args.robo_num == 3 or args.robo_num == 2:
            var_dict = get_exp_params_robo2_robo3(var_dict, files_to_analyze, morph_tag)
        elif args.robo_num == 4:
            get_exp_params_robo4(var_dict, files_to_analyze, morph_tag, light_path)
        else:
            print 'Not supporting this robo yet:', args.robo_num

    # Handle processing specified wells
    user_chosen_wells = args.chosen_wells
    if user_chosen_wells !='':
        user_chosen_wells = utils.get_iter_from_user(user_chosen_wells, 'wells')
        print 'Initial wells:'
        print var_dict["Wells"]
        var_dict["Wells"] = user_chosen_wells
        print 'Selected wells:'
        print var_dict["Wells"]

    # Handle processing specified timepoints
    user_chosen_timepoints = args.chosen_timepoints
    if user_chosen_timepoints !='':
        user_chosen_timepoints = utils.get_iter_from_user(user_chosen_timepoints, 'timepoints')
        print 'Initial timepoints:'
        print var_dict["TimePoints"]
        var_dict["TimePoints"] = user_chosen_timepoints
        print 'Selected timepoints:' 
        print var_dict["TimePoints"]

    # Handle processing specified channel token, overwriting get_channels.
    user_chosen_channel_token = args.chosen_channel_token
    files_to_analyze = utils.get_all_files(input_path)
    if user_chosen_channel_token !='':
        user_chosen_channel_token = utils.get_channels_from_user(
            files_to_analyze, user_chosen_channel_token)
        print 'Channels found:'
        print var_dict["Channels"]
        var_dict["Channels"] = user_chosen_channel_token
        print 'Selected channels:' 
        print var_dict["Channels"]

    # Handle quality control folders
    if module == "AlignIJ" or module == "BgCorrection" or module == "Segment":
        if not os.path.exists(var_dict["QualityControl"]):
            os.makedirs(var_dict["QualityControl"])

    if module == "Segment" or module == "Track":
        var_dict["CellMasks"] = args.output_path

    # Testing
    print var_dict

    print 'The *', module, '* module was called.'
    module_dict = module_to_function_dict(var_dict, module)

    # ----Run module-----------------------------
    start_time = datetime.datetime.utcnow()

    # Confirm given folders exist.
    assert os.path.exists(var_dict['InputPath']), 'Confirm the given path for data input exists.'
    assert os.path.exists(var_dict['OutputPath']), 'Confirm the given path for data output exists.'

    # print 'Module_dict value for:', module, '=', module_dict[module]
    run_module = module_dict[module][0]
    run_args = module_dict[module][1]
    run_module(*run_args)

    end_time = datetime.datetime.utcnow()
    print 'Module run time:', end_time-start_time

    # ----Output for user and save dict----------
    print 'Images were processed from:'
    print var_dict['InputPath']
    print 'Output was written to:'
    print var_dict["OutputPath"]

    outfile = args.outfile
    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    # outfile = os.rename('var_dict.p', outfile)
    outfile = shutil.move('var_dict.p', outfile)
