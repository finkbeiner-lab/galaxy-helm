import utils, os, sys, datetime, pprint
import cv2, shutil, numpy as np

def preprocess_stacks(input_path):
    '''
    Takes images in tiff stack files and saves them to individual images.
    The stack is moved to a folder inside the same directory.
    '''

    # Make folder inside input path
    directory = 'Z-stacks'
    if not os.path.exists(os.path.join(input_path, directory)):
        os.makedirs(os.path.join(input_path, directory))

    for channel in var_dict["Channels"]:
        print channel

        selector = utils.make_selector(channel=channel)
        print selector
        well_stack_list = utils.make_filelist(input_path, selector)
        pprint.pprint([os.path.basename(stack_pointer) for stack_pointer in well_stack_list])
        
        for stack_pointer in well_stack_list:
            stack_name = os.path.basename(stack_pointer)
            print stack_name

            if channel == var_dict['MorphologyChannel']:
                # Overwrite original stack to max projection
                utils.collapse_stack_magically(stack_pointer, stack_pointer)
                continue

            num_depths = stack_name.split('_')[-3]
            num_depths_str = '_'+str(num_depths)+'_'
            
            output_stack_name = stack_name[0:stack_name.index(
                num_depths_str)+1]+'%d'+stack_name[stack_name.index(
                    num_depths_str)+2:]
            output_filename = os.path.join(input_path, output_stack_name)

            utils.split_stack_magically(stack_pointer, output_filename)

            # Move file after unpacking
            shutil.move(stack_pointer, os.path.join(
                input_path, directory, stack_name))
            print 'Stacked file was moved to:',
            print os.path.join(
                input_path, directory, stack_name)

        print 'Completed processing channel,', channel

def preprocess_expt(var_dict):
    '''
    Point of entry.
    '''
    # Select an iterator, if needed
    if len(var_dict['Bursts'])>0:
        iter_list = var_dict['Bursts']
        iterator = 'bursts'
    elif len(var_dict["Depths"])>0:
        iter_list = var_dict["Depths"] 
        iterator = 'depths'
    else:
        iter_list = []
        iterator = None

    print 'Iterator was set to:', iterator
    if iterator == 'depths':
        preprocess_stacks(var_dict["RawImageData"])


if __name__ == '__main__':

    var_dict = {}
    var_dict['MorphologyChannel'] = '607'
    var_dict['Channels'] = ['607', '525']
    var_dict['Bursts'] = []
    var_dict['Depths'] = [1, 5]
    var_dict["RawImageData"] = '/home/mariyabarch/Desktop/MagickOutput'

    preprocess_expt(var_dict)