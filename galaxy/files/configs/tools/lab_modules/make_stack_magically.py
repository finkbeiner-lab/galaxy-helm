#!/usr/bin/env python
"""
Call ImageJ to open a sequence of images, creates and saves a stack.
"""
import sys, os
import pickle, datetime
import utils, shutil
from create_folders import get_exp_params_robo2_robo3
from create_folders import get_exp_params_robo4


def make_magick_stack_and_save(images_path, write_path, robo_num, light_path):
    '''
    Main point of entry.
    '''

    all_files = utils.get_all_files(images_path)
    wells = utils.get_wells(all_files)
    channels = utils.get_channels(all_files, robo_num, light_path)

    for well in wells:
        for channel in channels:
            selector = utils.make_selector(well=well, channel=channel)
            # print 'Actual selector:', selector
            files_to_stack = utils.make_filelist(images_path, selector)
            selector = os.path.join(images_path, '*'+selector+'*')
            # print 'Created selector:', selector
            output_filename = os.path.basename(files_to_stack[0])
            output_filename = output_filename[:-4]+'_TSTACK.tif'
            # print 'Will be saved as:', output_filename
            utils.make_stack_magically(selector, output_filename)


if __name__ == '__main__':

    # ----Initialize parameters--------
    images_path = sys.argv[1]
    write_path = sys.argv[2]
    robo_num = sys.argv[3]
    light_path = sys.argv[4]
    outfile = sys.argv[5]

    if not os.path.exists(images_path):
        sys.exit('Input path does not exist.')
    if not os.path.exists(write_path):
        sys.exit('Output path does not exist')

    # ----Run stacking---------------------------
    start_time = datetime.datetime.utcnow()

    make_magick_stack_and_save(images_path, write_path, robo_num, light_path)

    end_time = datetime.datetime.utcnow()
    print 'Make stack run time:', end_time-start_time
    # ----Output for user and save dict----
    print 'Images in', images_path, 'were stacked.'
    print 'Output was written to:', write_path

    # pickle.dump(var_dict, open('var_dict.p', 'wb'))
    # outfile = shutil.move('var_dict.p', outfile)

