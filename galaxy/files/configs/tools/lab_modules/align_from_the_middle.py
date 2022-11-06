import os, sys, utils, argparse
from alignment import alignment
from create_folders import order_wells_correctly
import datetime, pickle, shutil, pprint


def get_starting_wells(all_input_files):
    '''
    Take input path and return all represented wells.
    '''
    input_wells = set([os.path.basename(fname).split('_')[4]
        for fname in all_input_files
        if os.path.basename(fname).split('_')[4] != 'FIDUCIARY'])
    input_wells = list(input_wells)
    input_wells.sort(key=order_wells_correctly)

    return input_wells

def get_processed_wells(all_output_files):
    '''
    Take output path and return all represented wells.
    '''
    processed_wells = set([os.path.basename(fname).split('_')[4]
        for fname in all_output_files
        if os.path.basename(fname).split('_')[4] != 'FIDUCIARY'])
    # print 'Processed wells:', processed_wells
    processed_wells = list(processed_wells)
    # print 'Processed wells:', processed_wells
    processed_wells.sort(key=order_wells_correctly)
    print 'Processed wells:', processed_wells

    return processed_wells

def get_unaligned_wells(all_input_files, all_output_files):
    '''
    Take the input and output path.
    Compare filelists and return unprocessed wells.
    '''
    input_wells = get_starting_wells(all_input_files)
    processed_wells = get_processed_wells(all_output_files)

    # Sorted lists: get index of last processed well
    ind = len(processed_wells) #len(processed_wells)-1 to repeat broken well
    remaining_wells = input_wells[ind:]
    print 'Unprocessed wells:', remaining_wells

    return remaining_wells

def remove_broken_well(all_input_files, all_output_files):
    '''
    Remove the broken well from further analysis steps
    by pop from var_dict["Wells"].

    TODO: Take the original wells here, not the input wells.
    Is there a situation when these wouldn't be equivalent?
    '''
    input_wells = get_starting_wells(all_input_files)
    processed_wells = get_processed_wells(all_output_files)

    broken_well = input_wells.pop(
        input_wells.index(processed_wells[-1]))
    print 'Removed broken well from analysis:', broken_well
    print 'Corrected after broken well pop:', input_wells

    return input_wells


if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Align from the middle.")
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
    original_wells = var_dict["Wells"]

    # ----Initialize parameters--------
    path_montaged_images = args.input_path
    all_input_files = utils.make_filelist(path_montaged_images, '')
    path_aligned_images = args.output_path
    all_output_files = utils.make_filelist(path_aligned_images, '')
    intermediate_align_images = os.path.join(path_aligned_images, 'IntermediateAlignment')
    if not os.path.exists(intermediate_align_images):
            os.makedirs(intermediate_align_images)
    outfile = args.output_dict

    # ----Confirm given folders exist--
    assert os.path.exists(path_montaged_images), 'Confirm the given path for data exists.'
    assert os.path.exists(path_aligned_images), 'Confirm the given path for results exists.'

    # ----Run alignment on remaining wells-------
    start_time = datetime.datetime.utcnow()

    remaining_wells = get_unaligned_wells(
        all_input_files, all_output_files)
    var_dict['Wells'] = remaining_wells
    alignment(
        var_dict, path_montaged_images, 
        path_aligned_images, 
        intermediate_align_images)
    var_dict["Wells"] = remove_broken_well(
        all_input_files, all_output_files)

    end_time = datetime.datetime.utcnow()
    print 'Alignment correction run time:', end_time-start_time

    # ----Output for user and save dict----
    print 'Remaining montaged images were aligned.'
    print 'Output was written to:'
    print path_aligned_images

    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    outfile = shutil.move('var_dict.p', outfile)
