import os, sys, argparse, subprocess
import glob, shutil

def make_selector(iterator='', well='', timepoint='', channel='', frame='', verbose=False):
    'Constructor of selector string to glob appropriate files.'
    
    burst=''
    depth=''

    if iterator=='TimeBursts':
        burst = frame
    elif iterator=='ZDepths':
        depth = frame
    else:
        burst=''
        depth=''

    selector = timepoint+'_*'+burst+'_*'+well+'_'+'*'+'_'+channel+'*'+depth
    if verbose==True:
        print 'Depth:', depth, 'Burst', burst, 'Frame', frame
        print 'Set selector:', selector
    return selector

def get_iter_from_user(comma_list_without_spaces, iter_value, verbose=False):
    '''
    Takes list from user and returns a set to overwrite 'found' params for var_dict.

    # Add string.uppercase later.
    '''
    user_chosen_iter = comma_list_without_spaces.split(',')
    user_range = [user_iter for user_iter in user_chosen_iter if len(user_iter.split('-'))>1]
    all_letters = map(chr, range(65, 91))
    if len(user_range) > 0:
        for iter_range in user_range:
            user_chosen_iter.remove(iter_range)
            end_values = iter_range.split('-')
            start_letter = all_letters.index(end_values[0][0])
            end_letter = all_letters.index(end_values[1][0])
            letter_range = all_letters[start_letter:end_letter+1]
            number_start = end_values[0][1:]
            number_end = end_values[1][1:]
            num_range = range(int(number_start), int(number_end)+1)
            complete_range = []
            for letter in letter_range:
                for num in num_range:
                    complete_range.append(letter+str(num))
            user_chosen_iter.append(end_values[0])
            user_chosen_iter.extend(complete_range)
            user_chosen_iter.append(end_values[1])
    user_chosen_iter = list(set(user_chosen_iter))
    if verbose==True:
        print 'Your selected '+iter_value+':', user_chosen_iter
    return user_chosen_iter

def make_filelist(path, identifier):
    '''
    Takes a directory and a string identifier.
    Returns a list of files.
    '''
    filelist = glob.glob(os.path.join(path, '*'+identifier+'*'))
    return filelist

def make_stack_magically(out_path):
    '''
    Calls image magick to stack individual images into a stack.
    '''

    in_selector = os.path.join(out_path, 'qc_vis_*.tif')
    out_writer = os.path.join(out_path, 'QC_vis.tif')

    magic_command = ['convert', in_selector, out_writer]
    p = subprocess.Popen(magic_command, stderr=subprocess.PIPE)
    p.wait()


def make_qc_images(well, time, in_path, out_path, verbose=False):
    '''
    Calls image magick to concatenate individual images into one matrix.
    '''

    in_selector = os.path.join(
        in_path, '*'+make_selector(well=well, timepoint=time)+'.tif')
    out_writer = os.path.join(
        out_path, 'qc_vis_'+time+'_'+well+'.tif')

    if verbose:
        print 'In selector:', in_selector
        print 'Output file:', out_writer

    magic_command = ['convert', '+append', '-auto-level', '-resize', '800x800', in_selector, out_writer]
    # magic_command = ['montage', '-tile', 'x1', '-geometry', '800x800', in_selector, out_writer]
    
    if verbose:
        print 'The magic command:', magic_command

    p = subprocess.Popen(magic_command, stderr=subprocess.PIPE)
    p.wait()


if __name__ == '__main__':
    
    # Argument parsing
    parser = argparse.ArgumentParser(description="MagickQC.")
    parser.add_argument("input_path", 
        help="Folder path to input data.")
    parser.add_argument("output_path",
        help="Folder path to ouput results.")
    parser.add_argument("wells",
        help="Folder path to input data.") 
    parser.add_argument("timepoints",  
        help="Folder path to input data.")
    parser.add_argument("outfile",
        help="Name of output dictionary.")    
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    # Confirm given folders exist
    if not os.path.exists(input_path):
        print 'Confirm the given path for data exists.'
    assert os.path.exists(input_path), 'Confirm the given path for data exists.'
    if not os.path.exists(output_path):
        print 'Confirm the given path for results exists.'
    assert os.path.exists(output_path), 'Confirm the given path for results exists.'

    user_chosen_wells = args.wells
    user_chosen_wells = get_iter_from_user(user_chosen_wells, 'wells')
    user_chosen_timepoints = args.timepoints
    user_chosen_timepoints = get_iter_from_user(user_chosen_timepoints, 'timepoints')

    for well in user_chosen_wells:
        for time in user_chosen_timepoints:

            make_qc_images(well, time, input_path, output_path)

    # This works, but removes well and time ID from file
    # make_stack_magically(output_path)
    # qc_files = make_filelist(output_path, 'qc_vis_*.tif')

    # for qc_file in qc_files:
    #     os.remove(qc_file)


