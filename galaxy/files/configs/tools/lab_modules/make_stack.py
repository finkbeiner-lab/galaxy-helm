#!/usr/bin/env python
"""
Call ImageJ to open a sequence of images, creates and saves a stack.
"""
import sys, os, subprocess, tempfile
import pickle, datetime
import utils, shutil
from select_analysis_module import get_var_dict

assert sys.version_info[:2] >= (2, 4)

if 'darwin' in sys.platform:
    FIJI = "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"
elif 'linux' in sys.platform:
    FIJI = "/usr/local/bin/Fiji.app/ImageJ-linux64"
else:
    raise RuntimeError('System path for FIJI was not set on this OS.')


def run_ij_commands(ij_code):
    '''Set up and i/o and run functions in ImageJ.'''

    macro_fd, macro_filename = tempfile.mkstemp()
    os.write(macro_fd, ij_code)
    os.close(macro_fd)
    print 'Tempfile:', macro_filename
    fiji_command = [FIJI, "--headless", "-macro", macro_filename]
    print "Running", fiji_command
    #p = subprocess.call(fiji_command, stderr=subprocess.STDOUT)
    p = subprocess.Popen(fiji_command, stderr=subprocess.PIPE)
    p.wait()

def make_ij_stack_save_code(images_path, write_path, selector):
    '''
    Imports image sequence and saves stack.
    '''
    well_channel = selector.split('_')
    well_channel_ID = '_'.join([well_channel[0], well_channel[2]])

    ij_code = '''
        print("Starting IJ commands...");
        run("Image Sequence...", "open='''+images_path+''' file='''+selector+''' sort");

        //The title for a stack is obtained from source folder
        img_info = getTitle();
        img_title = img_info+"_'''+well_channel_ID+'''";

        ext = endsWith(img_title, ".tif");
        if (ext != 0) {
            s = lastIndexOf(img_title, '.');
            img_title = substring(img_title, 0, s);
            }

        stack_name = img_title+"_STACK";
        print(stack_name);
        saveAs("Tiff", "'''+write_path+'''/"+stack_name+".tif");

        run("Close All");
        eval("script", "System.exit(0);");
        '''

    return ij_code

def make_stack_and_save(var_dict, images_path, write_path):
    '''
    Main point of entry.
    '''

    for well in var_dict['Wells']:
        for channel in var_dict['Channels']:
            selector = well+'_1_'+channel

            print 'Selection criteria', selector

            ij_code = make_ij_stack_save_code(
                images_path, write_path, selector)
            run_ij_commands(ij_code)


if __name__ == '__main__':

    # ----Initialize parameters--------
    images_path = sys.argv[1]
    write_path = sys.argv[2]
    outfile = sys.argv[3]

    if not os.path.exists(images_path):
        sys.exit('Input path does not exist.')
    if not os.path.exists(write_path):
        sys.exit('Output path does not exist')

    analysis_files = utils.make_filelist(images_path, 'PID')
    var_dict = get_var_dict(analysis_files, 'RFP') #channel not important

    # ----Run stacking---------------------------
    start_time = datetime.datetime.utcnow()

    make_stack_and_save(var_dict, images_path, write_path)

    end_time = datetime.datetime.utcnow()
    print 'Make stack run time:', end_time-start_time
    # ----Output for user and save dict----
    print 'Images in', images_path, 'were stacked.'
    print 'Output was written to:', write_path

    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    outfile = shutil.move('var_dict.p', outfile)

