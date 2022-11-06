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
import string

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

def find_stack_size(path, identifier):
    '''
    From list of files in folder specified by path.
    Return number of files with the identifier string.
    '''
    num_images = len(glob.glob(os.path.join(path, '*'+identifier+'*')))
    print 'Number of images:', num_images
    return num_images

def find_max_dimensions(filelist):
    '''
    Finds the largest number of rows and largest number of columns
    for all images in a list of files.
    '''
    max_rows = max([cv2.imread(filename, 0).shape[0] for filename in filelist])
    max_cols = max([cv2.imread(filename, 0).shape[1] for filename in filelist])
    return (max_rows, max_cols)

def find_image_canvas(aligned_path, unprocessed_data_path, identifier, label):
    '''
    Check if maximum row, column dimensions are in the directory.
    If so, read in. Otherwise, calculate them and write out.
    '''
    max_dim_entry = make_filelist(aligned_path, identifier)
    if len(max_dim_entry) > 0:
        max_dim = load_obj(
            aligned_path,
            extract_file_name(max_dim_entry[0]))
    else:
        a_channel_list = make_filelist(unprocessed_data_path, 'c1')
        max_dim = find_max_dimensions(a_channel_list)
        print max_dim
        save_obj(max_dim, aligned_path, label+'_max_dim')

    print 'Found max image dimensions (height, width):', max_dim
    return max_dim

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

def cnum_to_tnum_renamer(filename_cnum, character_string):
    '''Generates new name to save thresholded files.'''
    original_name = extract_file_name(filename_cnum)
    filename_tnum = ''.join((
        original_name[0:len(original_name)-2],
        character_string,
        original_name[len(original_name)-1]))
    return filename_tnum

def channel_free_name_extractor(filename_cnum):
    '''Generates new name to save thresholded files.'''
    original_name = extract_file_name(filename_cnum)
    channel_free_name = original_name[0:len(original_name)-3]
    return channel_free_name

def get_filelists(path, channel_list):
    '''
    Generate channel-keyed dictionary of filename lists grouped by channel.

    Usage:
    path = '/Users/masha/Dropbox (MIT)/unison/Projects/ \
                ucsf_brain_imaging/brain_images/DLX'
    channel_list is a list of strings.
    example: ['c1', 'c2', 'c3', 'c4', 't1', 't2', 'coloc', 'c1c2']
    '''
    channel_dictionary = {}
    for channel_name in channel_list:
        filelist = make_filelist(path, channel_name)
        if len(filelist) == 0:
            continue
        channel_dictionary[channel_name] = filelist
    return channel_dictionary

#--------------------stuck on this---------------------------------------
def compare_filename_lists(channel_dictionary):
    '''Checks that all lists are equivalent in length and filename.'''
    #create a reference
    first_filelist = channel_dictionary.values()[0]
    ref_filelist = [channel_free_name_extractor(first_filelist[ind])
        for ind in range(len(first_filelist))]
    #check all against reference
    for chnl, filelist in channel_dictionary.items():

        if len(filelist) != len(first_filelist):
            print 'Channel', chnl, '/ c1', 'has ', \
                len(filelist), '/', len(first_filelist), 'files.'
            print 'Check length of '+chnl+ '.'
            print 'Below are the files that do not match.'
            print 'This list is short if a file is deleted.'
            print 'This list is long if slides were imaged \
                with different number of channels.'
            print '\n'.join(set(filelist) ^ set(first_filelist))
        comp_filelist = [channel_free_name_extractor(filelist[ind])
            for ind in range(len(first_filelist))]
        assert comp_filelist == ref_filelist, 'Check that '+chnl+' files match.'

#------------------------------------------------------------------------

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

def resizer(img, factor=0.25):
    ''''
    Takes each image in stach.
    Saves it to smaller image.
    For quickly viewing registration result.
    '''
    height, width = img.shape
    small_width, small_height = int(factor * width), int(factor * height)
    small_img = cv2.resize(img, (
        small_width, small_height), interpolation=cv2.INTER_CUBIC)
#
    return small_img

def width_resizer(img, target_width=100):
    ''''
    Takes each image in stach.
    Saves it to smaller image.
    For quickly viewing registration result.
    '''
    height, width = img.shape[0:2]
    factor = float(target_width)/width
    small_width, small_height = int(factor * width), int(factor * height)
    small_img = cv2.resize(img, (
        small_width, small_height), interpolation=cv2.INTER_CUBIC)

    return small_img

def height_resizer(img, target_height=100):
    ''''
    Takes each image in stach.
    Saves it to smaller image.
    For quickly viewing registration result.
    '''
    height, width = img.shape[0:2]
    factor = float(target_height)/height
    small_width, small_height = int(factor * width), int(factor * height)
    small_img = cv2.resize(img, (
        small_width, small_height), interpolation=cv2.INTER_CUBIC)

    return small_img

def show_wait_close(display_string, image):
    '''
    Shows image, with window name = display_string.
    Closes window upon keystroke.
    '''
    cv2.imshow(display_string, resizer(cv2.equalizeHist(image), 0.1))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_stack(path, identifier):
    '''
    Takes channel identifier string and path.
    Sequentially shows all images in path corresponding to channel.
    '''
    filelist = make_filelist(path, identifier)
    for ind in range(len(filelist)):
        image = cv2.imread(filelist[ind], 0)
        font = cv2.FONT_HERSHEY_PLAIN
        details = extract_file_name(filelist[ind])
        cv2.putText(image, details, (10, 20), font, 1, 200, 2, cv2.CV_AA)
        cv2.imshow(extract_file_name(
            filelist[ind]), resizer(cv2.equalizeHist(image), 0.1))
        cv2.waitKey(0)
    cv2.destroyAllWindows()

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

def get_image_parameters(renamed_unaligned_path, aligned_path, label):
    '''
    Collects relevant stack parameters.
    Returns dimensions and number of channels and images.
    '''
    num_images = find_stack_size(renamed_unaligned_path, 'c3')
    max_dim = find_image_canvas(
        aligned_path, renamed_unaligned_path, 'dim', label)
    num_channels = len(get_filelists(
        renamed_unaligned_path, ['c1', 'c2', 'c3', 'c4']).keys())
    #HD video, scale high resolution to write video
    scale_factor = 1800./max_dim[1]
    sc_max_dim = (int(max_dim[0]*scale_factor), int(max_dim[1]*scale_factor))
    return num_images, max_dim, num_channels, scale_factor, sc_max_dim

def draw_1mm_scale_bar(image, num_pixels_in_1mm):
    '''
    Add 5mm scale bar to image.
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

    selector = timepoint+'_*'+burst+'_*'+well+'_*'+channel+'*'+depth
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
        depths.sort(key=numericalSort)
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


