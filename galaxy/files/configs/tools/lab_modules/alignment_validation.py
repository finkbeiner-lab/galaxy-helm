import argparse, os, sys, shutil
import cv2, datetime, glob, re, pprint
import numpy as np


# Calculate shift
def find_xshift(image, rows, cols, res):
    '''
    Takes image pointer to alignment image.
    Returns x shift_crop coordinates.
    '''

    left_border = 0
    right_border = cols
    xshift = 0

    if res == 0: #8bit
        bg_val = 0
    elif res == -1: #16bit
        bg_val = 200
    else:
        print 'No bit depth detected.'

    # Find left non-border row (+x shift)
    for col_ind in range(cols):
        first_col = image[:, col_ind]
        if all([x == bg_val for x in first_col]): #alignment shifts asigned value 0
            left_border = col_ind
        else:
            break
    # Find right non-border row (-x shift)
    for col_ind in reversed(range(cols)):
        last_col = image[:, col_ind]
        if all([x == bg_val for x in last_col]): #alignment shifts asigned value 0
            right_border = col_ind
        else:
            break

    #  This will break for masks:
    assert left_border == 0 or right_border == cols, 'X shift conflict.'

    if left_border > 0:
        xshift = left_border
    elif right_border < cols:
        xshift = -(cols - right_border)

    return xshift

def find_yshift(image, rows, cols, res):
    '''
    Takes aligned image.
    Returns y shift_crop coordinates.
    '''

    top_border = 0
    bottom_border = rows
    yshift = 0

    if res == 0: #8bit
        bg_val = 0
    if res == -1: #16bit
        bg_val = 200

    # Find top non-border row (+y shift)
    for row_ind in range(rows):
        first_row = image[row_ind, :]
        if all([y == bg_val for y in first_row]): #alignment shifts asigned value 0 for 8bit and 200 for 16bit
            top_border = row_ind
        else:
            break
    # Find bottom non-border row (-y shift)
    for row_ind in reversed(range(rows)):
        last_row = image[row_ind, :]
        if all([y == bg_val for y in last_row]): #alignment shifts asigned value 0 for 8bit and 200 for 16bit
            bottom_border = row_ind
        else:
            break

    # This will break for masks:
    assert top_border == 0 or bottom_border == rows, 'Y shift conflict.'

    if top_border > 0:
        yshift = top_border
    elif bottom_border < cols:
        yshift = -(rows - bottom_border)

    return yshift

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

def get_all_files(input_path, channel, verbose=False):
    '''
    Takes all PID image files in input path. Removes fiduciary image files.
    '''
    all_files = make_filelist(input_path, channel)
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

def get_iter_from_user(comma_list_without_spaces, verbose=False):
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
        print 'Your selected wells:', user_chosen_iter
    return user_chosen_iter

def order_wells_correctly(value):
    '''
    Lets me sort list to follow A1, A2, A3, A11, A12....
    Instead of A1, A11, A12...
    '''
    return (value[0], int(value[1:]))


def split_stack_magically(stack_selector, output_filenames, verbose=False):
    '''
    Calls image magick split stack into individual images.

    @Usage
    Stack is selected with * args.
        Ex. stack_selector = *selector*
    Output should be specified with %d to number.
        Ex. output_filenames = single%d.tif
    '''
    start_time = datetime.datetime.utcnow()

    magic_command = ['convert', stack_selector, output_filenames]
    p = subprocess.Popen(magic_command, stderr=subprocess.PIPE)
    p.wait()

    end_time = datetime.datetime.utcnow()
    if verbose:
        print 'Magic stack images run time:', end_time-start_time

def make_magick_io(source_path, output_path, selector):


    return stack_selector, output_filenames

def get_image_shift(img_pointer):

    res = 0
    img = cv2.imread(img_pointer, res)
    # print 'Image mean, max, min:', round(img.mean(),2), img.max(), img.min()
    rows, cols = img.shape[0:2]
    diagonal = (np.sqrt(rows*rows + cols*cols))
    xshift = find_xshift(img, rows, cols, res)
    yshift = find_yshift(img, rows, cols, res)
    xy_shift =  np.sqrt(xshift*xshift + yshift*yshift)
    print 'xy_shift, diagonal', xy_shift, '\t', int(diagonal)

    return xy_shift, diagonal

def get_cumulative_shifts(well_files):

    cum_xyshift = 0
    cum_diagonal = 0

    for img_pointer in well_files:
        # print os.path.basename(img_pointer)
        # print '----'
        xy_shift, diagonal = get_image_shift(img_pointer)
        cum_xyshift = cum_xyshift + xy_shift
        cum_diagonal = cum_diagonal + diagonal
    # cum_shift = np.sqrt(cum_xshift*cum_xshift + cum_yshift*cum_yshift)
    percent_shift = round((float(cum_xyshift)/cum_diagonal)*100, 2)

    return percent_shift


def get_wells_shifts(input_path, channel, wells):

    all_files = get_all_files(input_path, channel)
    # print 'All files:'
    # pprint.pprint([os.path.basename(f) for f in all_files])
    if len(all_files) == 0:
        print 'No files found.'
    if wells == '':
        wells = get_wells(all_files)
    else:
        wells = get_iter_from_user(wells, verbose=True)
    shifts = []
    for well in wells:
        well_files = [well_file for well_file in all_files if well+'_' in well_file]
        # if len(well_files)==1 and 'STACK_ALIGNED' in well_files[0]:
        # print 'Well files:'
        # pprint.pprint([os.path.basename(wellfile) for wellfile in well_files])
        well_shift = get_cumulative_shifts(well_files)
        shifts.append(well_shift)
        print 'Well:', well, 'Shift:', well_shift
    shift_sorted_wells = sorted(zip(wells, shifts), key=lambda tup: tup[1])

    return shift_sorted_wells


def filter_bad_wells(shift_sorted_wells, cutoff):
    '''
    Returns list of wells that are above threshold value of reasonable shift.
    '''

    bad_wells = [tup for tup in shift_sorted_wells if tup[1] > cutoff]

    return bad_wells


def alignment_validation(input_path, channel, wells, outfile):
    '''Main point of entry. Returns wells that have poor alignment.'''
    shift_sorted_wells = get_wells_shifts(input_path, channel, wells)
    print 'shift_sorted_wells', shift_sorted_wells
    bad_wells = filter_bad_wells(shift_sorted_wells, 0)
    outf = open(outfile,'w')
    outf.write('\t'.join(['Well', 'Percent_Shift']))
    outf.write('\n')
    for well_shift in shift_sorted_wells:
        outf.write('\t'.join([str(well_shift[0]), str(well_shift[1])]))
        outf.write('\n')
    outf.close
    return bad_wells


if __name__ == '__main__':
    # Collect user inputs from GUI
    parser = argparse.ArgumentParser(description="Alignment Validation.")

    parser.add_argument("aligned_input_path",
         help="Folder path to ouput results.")
    parser.add_argument("morph_channel",
        help="A unique string corresponding to morphology channel.")
    parser.add_argument("wells",
        help="Wells to analyze.")
    # parser.add_argument("validation-metric",
    #     help="Set True if csv file should be generated.")
    parser.add_argument("outfile",
        help="Name of output dictionary.")
    args = parser.parse_args()

    # Set variables based on user inputs
    aligned_images_path = args.aligned_input_path
    morphology_channel = args.morph_channel
    wells = args.wells
    outfile = args.outfile

    # ----Run alignment validation----------
    start_time = datetime.datetime.utcnow()

    bad_wells = alignment_validation(aligned_images_path, morphology_channel, wells, outfile)

    end_time = datetime.datetime.utcnow()
    print 'Alignment validation run time:', end_time-start_time

    # ----Output for user and save dict----
    print 'The following wells may have poor alignment:'
    print [bad_well[0] for bad_well in bad_wells]

