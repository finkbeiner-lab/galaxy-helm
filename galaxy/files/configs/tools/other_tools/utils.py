'''
Common functions used by other programs in processing neuron images.
'''

import re
import os
import glob
import numpy as np
import cv2
import pickle
import subprocess, datetime
import string, shutil

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    '''
    Turns a string (containing integers) into a list of elements,
    split on the numbers.  The strings containing integers are
    transformed into integers, so that the array is properly
    sortable.  Useful as a key function for sorted().
    '''
    # splits on numbers, e.g., 'ho-1-22-333' ->
    #     ['ho-', '1', '-', '22', '-', '333']
    parts = numbers.split(value)
    # integerizes and replaces all the strings containing numbers
    # (which is every other element), e.g., ['ho-', 1, '-', 22, '-', 333]
    parts[1::2] = map(int, parts[1::2])
    return parts

def make_filelist(path, identifier):
    '''
    Takes a directory and a string identifier.
    Returns a list of files.
    '''
    filelist = sorted(
        glob.glob(os.path.join(path, '*'+identifier+'*')), key=numericalSort)
    return filelist

def extract_file_name(filename_path):
    '''Parses out file name from long path.'''
    img_file_name = os.path.basename(filename_path)
    img_name = os.path.splitext(img_file_name)
    return img_name[0]

def make_file_name(path, image_name, ext='.tif'):
    '''Generates file name'''
    return os.path.join(path, image_name+ext)

def modify_file_name(filename_path, identifier):
    '''Changes a file name by appending info.'''
    path = os.path.dirname(filename_path)
    original_name = extract_file_name(filename_path)
    new_file_name = os.path.join(path, original_name+identifier+'.tif')
    return new_file_name

def save_obj(object_name, destination_path, obj_save_name):
    '''
    Saves object called object_name to destination_path
    with filename 'obj_save_name', type(obj_save_name) = string.
    '''
    pickle.dump(object_name, open(os.path.join(
        destination_path, obj_save_name+'.p'), 'wb'))

def load_obj(source_path, obj_save_name):
    '''
    Reads object with filename obj_save_name, from the source_path.
    '''
    return pickle.load(open(os.path.join(
        source_path, obj_save_name+'.p'), 'rb'))

def find_ind_of_filename(image_path, image_filename):
    '''
    Gets an index associated with complete filename.
    Ex. image_filename = 'Shh_slide1_slice1_c1.tif'
    Generates filelist based on identifier.
    Returns index of that file.
    '''
    image_name = extract_file_name(image_filename)
    print image_name
    ch_id = image_name[len(image_name)-2:]
    one_channel_filelist = make_filelist(image_path, ch_id)
    if os.path.join(image_path, image_filename) not in one_channel_filelist:
        print 'Image', image_filename, 'is not in this experiment.'
        print ' Please correct options.py.'
        ind = False
    # one_channel_filelist = make_filelist(image_path, 'c1')
    if os.path.join(image_path, image_filename) in one_channel_filelist:
        ind = one_channel_filelist.index(os.path.join(image_path, image_filename))
    return ind

def return_image_indices(bad_image_list, image_path):
    '''
    Takes a list of bad image strings and prints their indices.
    Ex. bad_image_list = ['Shh_slide1_slice12_c1.tif', 'Shh_slide9_slice9_c1.tif']
    '''
    bad_img_indices = [
        find_ind_of_filename(image_path, image)
        for image in bad_image_list
        if type(find_ind_of_filename(image_path, image)) == int]
    return bad_img_indices

def assign_or_make_dir_path(base_path, directory):
    '''
    Creates a new directory with name 'directory', within the base_path.
    '''
    if not os.path.exists(base_path+directory):
        os.makedirs(base_path+directory)

def create_folder_hierarchy(output_subdirs, base_path):
    '''Creates folder hierarchy. Takes a list of paths.'''
    for directory in output_subdirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            continue
        if len(os.listdir(directory)) == 0:
            continue

def draw_1mm_scale_bar(image, num_pixels_in_1mm):
    '''
    Add 1mm scale bar to image.
    '''
    num_rows, num_cols = image.shape[0:2] #(y,x)
    cv2.line(image, (int(num_cols*.05), int(num_rows*.95)), (
        int(num_cols*.05)+int(num_pixels_in_1mm), int(num_rows*.95)), (
            255, 255, 255), int(num_cols*.005))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, '1 mm', (int(num_cols*.05), int(
        num_rows*.93)), font, 1, (255, 255, 255), 2, cv2.CV_AA)

def align_renamer(well, path_aligned_images, path_montaged_images):
    '''
    For IJ alignment output where names cannot be saved in stacking.
    Takes names in filelist.
    Uses output from previous step to generate correct names.
    '''
    rename_list = make_filelist(path_aligned_images, well)
    previous_list = make_filelist(path_montaged_images, well)
    prev_green = [fname for fname in previous_list if 'FITC' in fname]
    prev_red = [fname for fname in previous_list if 'RFP' in fname]
    rename_green = [fname for fname in rename_list if 'C2' in fname]
    rename_red = [fname for fname in rename_list if 'C1' in fname]
    for rname, pname in zip(rename_green, prev_green):
        rname_new = os.path.join(
            path_aligned_images, extract_file_name(
                pname)+'_ALIGNED.tif')
        os.rename(rname, rname_new)
    for rname, pname in zip(rename_red, prev_red):
        rname_new = os.path.join(
            path_aligned_images, extract_file_name(
                pname)+'_ALIGNED.tif')
        os.rename(rname, rname_new)

def zerone_normalizer(image):
    '''
    Normalizes matrix to have values between some min and some max.
    This is exactly equivalent to cv2.equalizeHist(image) if min and max are 0 and 255
    '''
    copy_image = image.copy()
    #set scale
    new_img_min, new_img_max = 0, 240
    zero_one_norm = (copy_image - image.min())*(
        (new_img_max-new_img_min) / (image.max() - image.min()) )+new_img_min
    return zero_one_norm

def give_error_exit(error_string):
    print '---------------'
    print error_string
    print '---------------'
    sys.exit(error_string)

def collapse_stack_to_image(file_list, collapse_type='max_proj'):
    '''
    Takes each image in file_list.
    Returns a maximum z-projection or average.
    '''
    start_time = datetime.datetime.utcnow()
    if collapse_type == 'max_proj':
        img_list = [cv2.imread(img_pointer, -1) for img_pointer in file_list]
        max_img = np.max(np.array(img_list), axis=0)
       
        end_time = datetime.datetime.utcnow()
        print 'Numpy collapse run time:', end_time-start_time
        return max_img

    if collapse_type == 'avg_proj':
        img_list = [cv2.imread(img_pointer, -1) for img_pointer in file_list]
        avg_img = np.average(np.array(img_list), axis=0)

        end_time = datetime.datetime.utcnow()
        print 'Numpy collapse run time:', end_time-start_time
        return avg_img

# ----Magick---------------------------
def collapse_stack_magically(output_file_name, selector, collapse_type='max_proj'):
    '''
    Takes a selector string for specific files.
    Returns a maximum z-projection or average.
    Uses imagemagick. Faster.
    '''
    start_time = datetime.datetime.utcnow()
    if collapse_type == 'max_proj':
        magic_command = ['convert', '-maximum', selector, output_file_name]

    if collapse_type == 'avg_proj':
        magic_command = ['convert', '-average', selector, output_file_name]

    p = subprocess.Popen(magic_command, stderr=subprocess.PIPE)
    p.wait()
    end_time = datetime.datetime.utcnow()
    print 'Magic collapse run time:', end_time-start_time


def split_stack_magically(stack_selector, output_filenames):
    '''
    Calls image magick split stack into individual images.
    
    @Usage
    Output should be specified with %d to number. 
        Ex. output_filenames = single%d.tif
    Stack is selected with * args.
        Ex. stack_selector = *selector*
    '''
    start_time = datetime.datetime.utcnow()
    
    magic_command = ['convert', stack_selector, output_filenames]
    p = subprocess.Popen(magic_command, stderr=subprocess.PIPE)
    p.wait()

    end_time = datetime.datetime.utcnow()
    print 'Magic stack images run time:', end_time-start_time

def make_stack_magically(selector, output_filename, verbose=False):
    '''
    Calls image magick to stack individual images into a stack.
    '''
    start_time = datetime.datetime.utcnow()

    magic_command = ['convert', selector, output_filename]
    p = subprocess.Popen(magic_command, stderr=subprocess.PIPE)
    p.wait()

    end_time = datetime.datetime.utcnow()
    if verbose:
        print 'Magic stack images run time:', end_time-start_time

# def check_if_stack(img_pointer, size_cut):
#     '''Use image size to check if image is stack.'''
#     img = cv2.imread(img_pointer, -1)
#     if img img.size() < size_cut
#         is_stack = False
#     else:
#         is_stack = True
#     return is_stack

def make_unstacking_folder(path):
    '''Make folder to recieve unstacked singles.'''
    directory = os.path.join(path, 'Unstacked_Collector')
    if not os.path.exists(directory):
        os.makedirs(directory)

def unstack_stacks(stack_pointer):
    '''
    Take any stack image and write it out to single files.
    '''
    path = os.path.dirname(stack_pointer)
    directory = os.path.join(path, 'Unstacked_Collector')
    output_filenames = os.path.join(directory, 'unstacked-%d.tif')
    split_stack_magically(stack_pointer, output_filenames)

def kill_unstacking_folder(path):
    '''Remove folder holding temporary single files.'''
    directory = os.path.join(path, 'Unstacked_Collector')
    if os.path.exists(directory):
        shutil.rmtree(directory)

def create_cleanup_unstacked(path):
    '''
    Take path, loop through files in path. 
    If files are stacks, unstack them and save into new directory.
    With each iteration, overwrite unstacked files.
    Clean up the new directory with single files.
    '''
    stack_files = make_filelist(path, 'PID')
    make_unstacking_folder(path)
    for stack in stack_files:
        unstack_stacks(stack)
    kill_unstacking_folder(path)


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

def set_iterator(var_dict):
    '''
    Evaluated file parameters to decide if bursts or depths are captured.
    '''
    if len(var_dict['Bursts'])>0:
        iter_list = var_dict['BurstIDs']
        iterator = 'TimeBursts'
    elif len(var_dict["Depths"])>0:
        iter_list = var_dict["Depths"]
        iterator = 'ZDepths'
    else:
        iter_list = []
        iterator = None

    iter_list.sort(key=numericalSort)
    return iterator, iter_list

def order_wells_correctly(value):
    '''
    Lets me sort list to follow A1, A2, A3, A11, A12....
    Instead of A1, A11, A12...
    '''
    return (value[0], int(value[1:]))


def get_all_files(input_path, verbose=False):
    '''
    Takes all PID image files in input path. Removes fiduciary image files.
    '''
    all_files = make_filelist(input_path, 'PID')
    all_files = [afile for afile in all_files if 'FIDUCIARY' not in afile]
    all_files = [afile for afile in all_files if '.tif' in afile]
    if verbose==True:
        print 'Number of image files:', len(all_files)
    return all_files

def get_wells(all_files, verbose=False):
    '''
    Use appropriate well token to collect the ordered set of wells.
    '''
    wells = []
    for one_file in all_files:
        well = os.path.basename(one_file).split('_')[4]
        wells.append(well)
    wells = list(set(wells))
    wells.sort(key=order_wells_correctly)
    if verbose==True:
        print 'Wells:', wells
    return wells

def get_timepoints(all_files, verbose=False):
    '''
    Use appropriate timepoint token to collect the ordered set of timepoints.
    '''
    timepoints = []
    for one_file in all_files:
        time = os.path.basename(one_file).split('_')[2]
        timepoints.append(time)
    timepoints = sorted(list(set(timepoints)))
    timepoints.sort(key=order_wells_correctly)
    if verbose==True:
        print 'Timepoints:', timepoints
    return timepoints

def get_channels(all_files, robonum, light_path, verbose=False):
    '''
    Use appropriate channel token for robo to collect the represented channels.
    Return list of channels.
    '''
    robonum = int(robonum)
    channels = []
    for one_file in all_files: 
        if robonum == 2 or robonum == 3:
            channel = os.path.basename(one_file).split('_')[6].split('.')[0]
        elif robonum == 4:
            if light_path == 'epi':
                channel = os.path.basename(one_file).split('_')[6].split('.')[0]
            elif light_path == 'confocal':
                # channel = os.path.basename(one_file).split('_')[9].split('.')[0]
                channel = os.path.basename(one_file).split('_')[-4].split('.')[0]
            else:
                assert light_path == 'epi' or light_path == 'confocal', 'No treatment for this light path yet.'
        else: 
            assert robonum == 2 or robonum == 3 or robonum == 4, 'No treatment for this robo yet'
        
        if 'Brightfield' not in channel:
            channels.append(channel)

    channels = list(set(channels))
    if verbose==True:
        print 'Channels:', channels
    return channels

def get_channels_from_user(all_files, channel_token, verbose=False):
    '''
    Use appropriate channel token for robo to collect the represented channels.
    Return list of channels.
    '''

    channel_token = int(channel_token)
    channels = []
    for one_file in all_files: 
        channel = os.path.basename(one_file).split('_')[channel_token].split('.')[0]
        if 'Brightfield' not in channel:
            channels.append(channel)

    channels = list(set(channels))
    if verbose==True:
        print 'Channels:', channels
    return channels

def get_ref_channel(morph_channel, channels, verbose=False):
    '''
    Take list of channels and substring for morphology channel.
    Return the complete morphology channel reference.
    '''
    morphology_channel = [ch for ch in channels if morph_channel in ch][0]
    if verbose==True:
        print 'Morphology channel:', morphology_channel
    return morphology_channel

def get_plate_id(all_files, verbose=False):
    '''
    Use appropriate plate ID tokens to return plate id of first image.
    '''
    first_data_file = os.path.basename(all_files[0])
    plateID_tokens = first_data_file.split('_')[0:2]
    plateID = '_'.join(plateID_tokens)
    if verbose==True:
        print 'PlateID:', plateID
    return plateID

def get_bursts(all_files, verbose=False):
    '''
    Use appropriate burst tokens to return a list of bursts.
    '''
    burstIDs = list(set(
        [os.path.basename(fname).split('_')[3] for fname in all_files]))
    burstIDs.sort(key=numericalSort)
    if verbose == True:
        print 'Burst IDs:', burstIDs
    return burstIDs

def get_burst_iter(all_files, verbose=False):
    '''
    Use appropriate burst iterator tokens to return a list of burst iterators.
    '''
    burstIDs = get_bursts(all_files, verbose=False)
    burst_frames = []
    for burst in burstIDs:
        try:  burst_frames.append('-'+burst.split('-')[1])
        except IndexError: continue
    burst_frames = list(set(burst_frames))
    burst_frames.sort(key=numericalSort)
    if verbose==True:
        print 'Burst frames:', burst_frames
    return burst_frames

def get_depths(all_files, verbose=False):
    '''
    Use appropriate depth tokens to return a list of depths.
    '''
    # Depths are currently stored as stacks
    depths = list(set([int(
        os.path.basename(fname).split('_')[-3]) for fname in all_files]))
    if all([el==1 for el in depths]):
        depths = []
    else:
        depths.sort()#key=numericalSort)
    if verbose==True:
        print 'Depths:', depths
    return depths

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
    user_chosen_iter.sort(key=numericalSort)
    if verbose==True:
        print 'Your selected '+iter_value+':', user_chosen_iter
    return user_chosen_iter


def overwrite_io_paths(var_dict, output_path, verbose=False):
    '''
    If var_dict is given as input, the data input_path 
    for this module will be set from var_dict['OutputPath'] 
    of the previous step. The new var_dict['OutputPath'] 
    will be set as output_path.'''

    if verbose==True:
        print 'Initial input (passed var_dict)', var_dict['InputPath']
        print 'Initial output (passed var_dict)', var_dict["OutputPath"]

    input_path = var_dict["OutputPath"]
    var_dict['InputPath']  = input_path
    var_dict["OutputPath"] = output_path

    if verbose==True:
        print 'Final input (new var_dict)', var_dict['InputPath']
        print 'Final output (new var_dict)', var_dict["OutputPath"]

    return var_dict


