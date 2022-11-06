#!/usr/bin/env python
"""
Takes all timepoints associated with a well and aligns them relative to one another.
Each alignment shift generates borders along the edge.
The borders are used to calculate shift for each subsequent channel.

Notes:
All available timepoints in input folder will be aligned.
Well-channel selector is generated.
In burst case, only initial timepoints should be aligned.
Stolen coordinates for corresponding files should apply to bursts.
"""
import sys, os, subprocess, tempfile
import pickle, datetime, argparse
import utils, shutil, cv2, pprint
# import save_stdout_to_vectors as tvect

assert sys.version_info[:2] >= ( 2, 4 )

if 'darwin' in sys.platform:
    FIJI = "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"
elif 'linux' in sys.platform:
    FIJI = "/usr/local/bin/Fiji.app/ImageJ-linux64"
else:
    raise RuntimeError('System path for FIJI was not set on this OS.')


def run_ij_commands(ij_code, verbose=False):
    '''Set up and i/o and run functions in ImageJ.'''

    macro_fd, macro_filename = tempfile.mkstemp()
    os.write(macro_fd, ij_code)
    os.close(macro_fd)
    fiji_command = ["sudo", FIJI, "--headless", "-macro", macro_filename]
    if verbose:
        print 'Tempfile:', macro_filename
        print "Running", fiji_command
    #p = subprocess.call(fiji_command, stderr=subprocess.STDOUT)
    p = subprocess.Popen(fiji_command, stderr=subprocess.PIPE)
    p.wait()

def stack_images_align_save_indv(images_path, write_path, well, var_dict, verbose=False):
    '''
    Loops through all timepoints associated with a well in images_path.
    Opens the files, creates a stack.
    Runs image registration on them.
    Converts stack to individual images.
    Saves individual images to write_path.
    If you are processing select timepoints, the images_path should reflect this.
    Selector can also now be set with a regex.
    Example: (.*[T][1][_].*[E][1][_].*[RFP].*)
    selector = '(.*['+well[0]+']['+well[1:]+'][_].*['+morph_channel[0:3]+'].*)'

    @Notes:
    Important: IJ does not handle titles with more than 60 characters
    img_title = getTitle() <--Does not get complete filename, 60 char limit
    Use instead a substring of getImageInfo
    getImageInfo holds the complete title in the first line
    Can avoid this with jython, but intentionlly sticking with macro language for now.
    '''

    morph_channel = var_dict["MorphologyChannel"]
    # selector = well+'_1_'+morph_channel
    selector = utils.make_selector(well=well, channel=morph_channel)

    files_to_align = utils.make_filelist(images_path, selector)

    # Note that timepoint specification here would not enter StackReg call
    # var_times = [tp+'_' for tp in var_dict['TimePoints']]
    # files_to_align = [tfile for tfile in files_to_align if any(tp in tfile for tp in var_times)]

    if len(files_to_align)>1:
        selector_start = os.path.basename(files_to_align[0]).find(well+'_')
        selector_end = os.path.basename(files_to_align[0]).find(morph_channel)+len(morph_channel)
        selector = os.path.basename(files_to_align[0])[selector_start:selector_end]

    if verbose:
        print 'File selector:', selector
        print 'Files to be aligned:'
        pprint.pprint([os.path.basename(fname) for fname in files_to_align])

    if len(files_to_align) <= 1:
        return len(files_to_align)


    ij_code ='''
        run("Image Sequence...", "open='''+images_path+''' file='''+selector+''' sort");

        //print("Beginning alignment...");
        run("StackReg", "transformation=Translation");
        //run("StackReg", "transformation=Rigid Body");
        //print("Completed alignment!");
        run("Stack to Images");

        for (i=0; i<nImages; i++) {

            selectImage(i+1);
            img_info = getImageInfo();
            img_title = substring(img_info, 0, indexOf(img_info, "\\n"));

            ext = endsWith(img_title, ".tif");
            if (ext != 0) {
                s = lastIndexOf(img_title, '.');
                img_title = substring(img_title, 0, s);
            }
            aligned_name = img_title+"_AL";
            saveAs("Tiff", "'''+write_path+'''/"+aligned_name+".tif");
        }
        run("Close All");
        eval("script", "System.exit(0);");
        '''

    return ij_code

def find_xshift(image, rows, cols):
    '''
    Takes image pointer to alignment image.
    Returns x shift_crop coordinates.
    '''

    left_border = 0
    right_border = cols
    xshift = 0

    # Find left non-border row (+x shift)
    for col_ind in range(cols):
        first_col = image[:, col_ind]
        if all([x == 0 for x in first_col]): #alignment shifts asigned value 0
            left_border = col_ind
        else:
            break
    # Find right non-border row (-x shift)
    for col_ind in reversed(range(cols)):
        last_col = image[:, col_ind]
        if all([x == 0 for x in last_col]): #alignment shifts asigned value 0
            right_border = col_ind
        else:
            break

    #  This will break for masks:
    assert left_border == 0 or right_border == cols, 'X shift conflict.'

    if left_border > 0:
        xshift = left_border
    elif right_border < cols:
        xshift = -(cols - right_border)

    return left_border, right_border, xshift

def find_crop_cols(xshift, target_cols):
    '''
    Get the left and right of x cropping boundaries.
    '''

    image_cols = target_cols + abs(xshift)

    if xshift < 0: # Left shift (-x)
        col_start = image_cols - target_cols
        col_end = image_cols
    else:  # Right shift (+x)
        col_start = 0
        col_end = target_cols

    assert col_end - col_start == target_cols

    return col_start, col_end

def find_yshift(image, rows, cols):
    '''
    Takes image pointer to alignment image.
    Returns y shift_crop coordinates.
    '''

    top_border = 0
    bottom_border = rows
    yshift = 0

    # Find top non-border row (+y shift)
    for row_ind in range(rows):
        first_row = image[row_ind, :]
        if all([y == 0 for y in first_row]): #alignment shifts asigned value 0
            top_border = row_ind
        else:
            break
    # Find bottom non-border row (-y shift)
    for row_ind in reversed(range(rows)):
        last_row = image[row_ind, :]
        if all([y == 0 for y in last_row]): #alignment shifts asigned value 0
            bottom_border = row_ind
        else:
            break

    # This will break for masks:
    assert top_border == 0 or bottom_border == rows, 'Y shift conflict.'

    if top_border > 0:
        yshift = top_border
    elif bottom_border < cols:
        yshift = -(rows - bottom_border)

    return top_border, bottom_border, yshift

def find_crop_rows(yshift, target_rows):
    '''
    Get the top and bottom of y cropping boundaries.
    '''

    image_rows = target_rows + abs(yshift)

    if yshift < 0: # Up shift (-y)
        row_start = image_rows - target_rows
        row_end = image_rows
    else:  # Down shift (+y)
        row_start = 0
        row_end = target_rows

    assert row_end - row_start == target_rows

    return row_start, row_end


def add_shift_border_to_image(border_images, all_border_images, write_path, var_dict, verbose=False):
    '''
    Create border of pixel value = 0 for each row/column image is shifted for alignment.
    '''

    for border_img in border_images:

        time_point = [tp for tp in var_dict['TimePoints'] if tp+'_' in border_img][0]
        if verbose:
            print 'Time point:', time_point
            print 'Current Border-stealing image:'
            pprint.pprint(os.path.basename(border_img))

        image = cv2.imread(border_img, -1)
        rows, cols = image.shape[0:2]

        left_border, right_border, xshift = find_xshift(image, rows, cols)
        col_start, col_end = find_crop_cols(xshift, cols)
        top_border, bottom_border, yshift = find_yshift(image, rows, cols)
        row_start, row_end = find_crop_rows(yshift, rows)

        for channel_list in all_border_images:

            if verbose:
                print 'Channel list:'
                pprint.pprint([os.path.basename(ch_el) for ch_el in channel_list])

            # Excludes timepoints (T10, T11, ... ) not selected by user from coordinate stealing.
            # Note that all available timepoints were aligned either way.
            time_list = [chfile for chfile in channel_list if time_point+'_' in chfile]

            if verbose:
                print 'Number of files with', time_point, 'coords:', len(time_list)
                pprint.pprint([os.path.basename(timg) for timg in time_list])

            for frame in time_list:

                channel_img = cv2.imread(frame, -1)
                channel_img_name = utils.extract_file_name(frame)
                aligned_img_name = utils.make_file_name(
                    write_path, channel_img_name+'_AL')
                shifted_img = cv2.copyMakeBorder(channel_img,
                    top_border, rows-bottom_border,
                    left_border, cols-right_border,
                    cv2.BORDER_CONSTANT, value=200)
                bordereded_to_size = shifted_img[row_start:row_end, col_start:col_end]

                cv2.imwrite(aligned_img_name, bordereded_to_size)

                # if display:
                #     orig_small = 50*utils.width_resizer(channel_img, 300)
                #     shifted_small = 50*utils.width_resizer(shifted_img, 300)
                #     bordered_small = 50*utils.width_resizer(bordereded_to_size, 300)
                #     cv2.imshow('original_image', orig_small)
                #     cv2.waitKey(0)
                #     cv2.imshow('shifted_imgd', shifted_small)
                #     cv2.waitKey(0)
                #     cv2.imshow('bordered_images', bordered_small)
                #     cv2.waitKey(0)
                #     cv2.destroyAllWindows()

def alignment(var_dict, path_montaged_images, path_aligned_images, intermediate_align_images, verbose=False):
    '''
    Main point of entry. Executes all steps associated with alignment.

    border_images is a list of files from IntermediateAlignment folder after StackReg alignment.
    all_border_images is a list of files from MontagedImages folder that need to have StackReg output applied.
    '''
    morph_channel = var_dict["MorphologyChannel"]
    for well in var_dict['Wells']:
        # Launch StackReg alignment from ImageJ
        print 'Running IJ stack_reg on well', well
        ij_code = stack_images_align_save_indv(
            path_montaged_images, intermediate_align_images, well, var_dict, verbose=False)
        # Exit if 'Stack' cannot be made, no images to align.
        if ij_code <= 1:
            print ij_code
            continue
        run_ij_commands(ij_code)

        # Get coordinates and apply to all channels
        selection_criterion = utils.make_selector(well=well, channel=morph_channel)
        border_images = utils.make_filelist(
            intermediate_align_images, selection_criterion)
        # Filtering list of images to select timepoints in config file.
        var_times = [tp+'_' for tp in var_dict['TimePoints']]
        border_images = [tfile for tfile in border_images if any(tp in tfile for tp in var_times)]

        if verbose:
            print 'These are the border images:', len(border_images)
            pprint.pprint([os.path.basename(bimg) for bimg in border_images])

        all_border_images = []
        for channel in var_dict['Channels']:
            select_criterion = utils.make_selector(well=well, channel=channel)
            select_images = utils.make_filelist(
                path_montaged_images, select_criterion)
            all_border_images.append(select_images)

        if verbose:
            print 'Selected:', select_criterion, well, channel
            print 'Number of channels:', len(all_border_images)
            print 'Number of images per channel:', max([len(all_border_images[n]) for n in range(len(all_border_images))])
            print 'Images to use stolen coordinates:'
            for blist in range(len(all_border_images)):
                print 'Channel images associated with each selection:', blist
                pprint.pprint([os.path.basename(bimg) for bimg in all_border_images[blist]])

        add_shift_border_to_image(
            border_images, all_border_images, path_aligned_images, var_dict)

    # For select_analysis_module input, set var_dict['OutputPath']
    var_dict["OutputPath"] = path_aligned_images

if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Align images via TurboReg.")
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

    # ----Initialize parameters--------
    path_montaged_images = args.input_path
    path_aligned_images = args.output_path
    outfile = args.output_dict
    intermediate_align_images = os.path.join(path_aligned_images, 'IntermediateAlignment')
    if not os.path.exists(intermediate_align_images):
            os.makedirs(intermediate_align_images)

    # ----Confirm given folders exist--
    assert os.path.exists(path_montaged_images), 'Confirm the given path for data exists.'
    assert os.path.exists(path_aligned_images), 'Confirm the given path for results exists.'

    # ----Run alignment--------------------------
    start_time = datetime.datetime.utcnow()

    alignment(var_dict, path_montaged_images, path_aligned_images, intermediate_align_images)

    end_time = datetime.datetime.utcnow()
    print 'Alignment correction run time:', end_time-start_time
    # ----Output for user and save dict----
    print 'Montaged images were aligned.'
    print 'Output was written to:'
    print path_aligned_images

    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    # outfile = os.rename('var_dict.p', outfile)
    outfile = shutil.move('var_dict.p', outfile)
    utils.save_user_args_to_csv(args, path_aligned_images, 'alignment')
