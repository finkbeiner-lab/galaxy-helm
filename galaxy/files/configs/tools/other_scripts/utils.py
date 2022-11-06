'''
Common functions used by other programs in processing brain images.
'''

import re
import os
import glob
import numpy as np
import cv2
import pickle

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

def make_file_name(path, image_name, ext='.tiff'):
    '''Generates file name'''
    return os.path.join(path, image_name+ext)

def modify_file_name(filename_path, identifier):
    '''Changes a file name by appending info.'''
    path = os.path.dirname(filename_path)
    original_name = extract_file_name(filename_path)
    new_file_name = os.path.join(path, original_name+identifier+'.tiff')
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
    Ex. image_filename = 'Shh_slide1_slice1_c1.tiff'
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
    Ex. bad_image_list = ['Shh_slide1_slice12_c1.tiff', 'Shh_slide9_slice9_c1.tiff']
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

def create_folder_hierarchy(output_subdirs, label, base_path):
    base_path = os.path.join(base_path, label)
    '''Creates folder hierarchy. Takes a list of paths.'''
    for directory in output_subdirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            continue
        if len(os.listdir(directory)) == 0:
            continue

def read_coords_save_tab_delim(path, label, color, num_channels):
    '''Load coordinates back in and save them out to tab-delimited file.'''
    if num_channels < 4:
        channel_color_mapper = {
            't1': 'red',
            't2': 'green',
            't3': 'blue',
            'l3': 'yellow',
            }

    if num_channels == 4:
        channel_color_mapper = {
            't1': 'magenta',
            't2': 'red',
            't3': 'green',
            't4': 'blue',
            'l3': 'yellow',
            }

    color_text = channel_color_mapper[color]
    coord_file = glob.glob(os.path.join(
        path, '*'+color_text+'*'+'_coordinates.npz'+'*'))
    # print 'coord_file', coord_file
    coordinates = np.load(coord_file[0])
    # print 'coordinates.files', coordinates.files
    x, y, z = coordinates['x'], coordinates['y'], coordinates['z']
    print os.path.join(path, label+'_'+color_text+'_coords.txt')
    with open(os.path.join(path, label+'_'+color_text+'_coords.txt'), 'w') as txt_file:
        txt_file.writelines('\t'.join(
            ['wavex_'+color_text, 'wavey_'+color_text, 'wavez_'+color_text])+'\n')
        txt_file.writelines('\t'.join(
            [str(x[i]), str(y[i]), str(z[i])])+'\n' for i in range(x.size))
    txt_file.close()

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

