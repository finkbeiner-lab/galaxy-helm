import utils, os, sys, cv2, argparse
import numpy as np
import pickle, datetime, pprint, shutil

'''
Dealing with alignment and cropping based on coordinates.
TODO: all of the channels should be handled simultaneously based on morphology.
'''

def accumulated_coordinates(transl_crdn):
    '''Accumulate total translation at each timepoint.'''
    acc_crdn = []
    for ind in range(len(transl_crdn)):
        x, y = zip(*transl_crdn)
        acc_x = sum(x[0:ind+1])
        acc_y = sum(y[0:ind+1])
        acc_crdn.append((acc_x, acc_y))
    return acc_crdn

def translate_each_image_add_border(transl_crdn, transl_files, write_path, resolution):
    '''
    Translate each image by specified number of pixels
    by adding gray border.
    '''

    acc_crds = accumulated_coordinates(transl_crdn)
    # print transl_crdn
    # print acc_crds

    for acc_crd, trnsl_file in zip(acc_crds, transl_files):

        orig_img = cv2.imread(trnsl_file, resolution)
        orig_name = utils.extract_file_name(trnsl_file)
        bordered_img_name = utils.make_file_name(
            write_path, orig_name+'_SHIFTED')

        rows, cols = orig_img.shape[0:2]
        xshift = acc_crd[0]
        yshift = acc_crd[1]
        print 'xshift', xshift, 'yshift', yshift

        # initiate
        top_border = 0; bottom_border = 0
        left_border = 0; right_border = 0
        col_start = 0; col_end = cols
        row_start = 0; row_end = rows

        # move down   +y
        if yshift > 0:
            top_border = yshift
        # move up     -y
        if yshift < 0:
            bottom_border = np.absolute(yshift)
            row_start = rows - (rows - np.absolute(yshift))
            row_end = rows + bottom_border
        # move right  +x
        if xshift > 0:
            left_border = xshift
        # move left   -x
        if xshift < 0:
            right_border = np.absolute(xshift)
            col_start =  cols - (cols - np.absolute(xshift))
            col_end = cols + right_border

        grey_val = 30000 if resolution==-1 else 127 if resolution==0 else 250
        # border takes: top, bottom, left, right
        shifted_img = cv2.copyMakeBorder(orig_img,
            top_border, bottom_border,
            left_border, right_border,
            cv2.BORDER_CONSTANT, value=grey_val)
        bordereded_to_size = shifted_img[row_start:row_end, col_start:col_end]

        # print 'Original:', orig_img.shape
        # print 'Shifted:', shifted_img.shape
        # print 'Cropped:', bordereded_to_size.shape

        # cv2.imshow('orig_img', utils.width_resizer(orig_img, 500))
        # cv2.waitKey(0)
        # cv2.imshow('shifted_img', utils.width_resizer(shifted_img, 500))
        # cv2.waitKey(0)
        # cv2.imshow('cropped_shifted_img', utils.width_resizer(bordereded_to_size, 500))
        # cv2.waitKey(0)
        cv2.imwrite(bordered_img_name, bordereded_to_size)
    cv2.destroyAllWindows()

def crop_from_coordinates(transl_crdn, border_images, write_path, resolution):
    '''
    Cropping v2: Using translation coordinates to crop out border
    and set common matrix.
    '''

    acc_crds = accumulated_coordinates(transl_crdn)

    left_holder = 0; right_holder = 0; top_holder = 0; bottom_holder = 0
    xshifts, yshifts = zip(*transl_crdn)
    if max(xshifts) > 0:
        left_holder = max(xshifts)
    if min(xshifts) < 0:
        right_holder = np.absolute(min(xshifts))
    if max(yshifts) > 0:
        top_holder = max(yshifts)
    if min(yshifts) < 0:
        bottom_holder = np.absolute(min(yshifts))

    for border_img in border_images:

        orig_img = cv2.imread(border_img, resolution)
        orig_name = utils.extract_file_name(border_img)
        cropped_img_name = utils.make_file_name(
            write_path, orig_name+'_CR')

        rows, cols = orig_img.shape[0:2]
        cropped_img = orig_img[top_holder:rows-bottom_holder, left_holder:cols-right_holder]
        # print 'Version2:', orig_img.shape, cropped_img.shape
        # cv2.imshow('original', utils.width_resizer(img, 300)); cv2.waitKey(0)
        # cv2.imshow('cropped', utils.width_resizer(cropped_img, 300)); cv2.waitKey(0)
        cv2.imwrite(cropped_img_name, cropped_img)

    cv2.destroyAllWindows()


# def crop_from_borders(border_images, write_path, resolution, all_border_images):
def crop_from_borders(border_images, resolution):
    '''
    Cropping v1: Using borders to crop out border and set common matrix.
    The border_images hold all the timepoints, depths, and bursts for one well (morphology channel)
    The most conservative (farthest from edge) crop borders are selected.
    
    TODO: some more testing, not working in some cases:
        1. Final images are smaller
        2. Dimensions do not match the coordinate based translation
            This might be okay given the algorithm. Nothing should be lost.
    '''

    sample_img = cv2.imread(border_images[0], resolution)
    rows, cols = sample_img.shape[0:2]

    left_holder = 0; right_holder = cols; top_holder = 0; bottom_holder = rows
    lholder = [0]; rholder = [cols]; tholder = [0]; bholder = [rows]

    for i, border_img in enumerate(border_images):

        img = cv2.imread(border_img, resolution)
        rows, cols = img.shape[0:2]

        for rf in range(rows):
            # print 'Row:', rf
            row_one = img[rf, :]
            if all([y == row_one[0] for y in row_one]):
                # top_holder = rf
                tholder.append(rf)
                # print 'top_holder', top_holder
            if not all([y == row_one[0] for y in row_one]):
                break
        for rb in reversed(range(rows)):
            # print 'Row:', rb
            row_last = img[rb, :]
            if all([y == row_last[0] for y in row_last]):
                # bottom_holder = rb
                bholder.append(rb)
                # print 'bottom_holder', bottom_holder
            if not all([y == row_last[0] for y in row_last]):
                break
        for cf in range(cols):
            # print 'Column:', cf
            col_one = img[:, cf]
            if all([x == col_one[0] for x in col_one]):
                # left_holder = cf
                lholder.append(cf)
                # print 'left_holder', left_holder
            if not all([x == col_one[0] for x in col_one]):
                break
        for cb in reversed(range(cols)):
            # print 'Column:', cb
            col_last = img[:, cb]
            if all([x == col_last[0] for x in col_last]):
                # right_holder = cb
                rholder.append(cb)
                # print 'right_holder', right_holder
            if not all([x == col_last[0] for x in col_last]):
                break

    left_holder = max(lholder); right_holder = min(rholder)
    top_holder = max(tholder); bottom_holder = min(bholder)
    # print 'top_holder', top_holder
    # print 'bottom_holder', bottom_holder
    # print 'left_holder', left_holder
    # print 'right_holder', right_holder
    edges = {
        'left_holder': left_holder, 
        'right_holder': right_holder, 
        'top_holder': top_holder, 
        'bottom_holder': bottom_holder}
    return edges


def apply_max_crop(write_path, resolution, all_border_images, edges):
    '''
    These borders are applied to all images in the well (to all channels).
    '''
    top_holder = edges['top_holder']
    bottom_holder = edges['bottom_holder']
    left_holder = edges['left_holder']
    right_holder = edges['right_holder']

    for border_img in all_border_images:
        # print os.path.basename(border_img)
        img = cv2.imread(border_img, resolution)
        orig_name = utils.extract_file_name(border_img)
        cropped_img_name = utils.make_file_name(
            write_path, orig_name+'_CR')

        cropped_img = img[top_holder:bottom_holder, left_holder:right_holder]
        # print 'Version1: Image', i, img.shape, cropped_img.shape
        # cv2.imshow('original', utils.width_resizer(img, 300)); cv2.waitKey(0)
        # cv2.imshow('cropped', utils.width_resizer(cropped_img, 300))
        # cv2.waitKey(0)
        cv2.imwrite(cropped_img_name, cropped_img)

    cv2.destroyAllWindows()

def shift_crop(var_dict, path_aligned_images, path_shift_cropped_images):
    '''
    Main point of entry.
    '''

    coords_exist = 'Translation_coordinates_xy' in var_dict.keys()
    resolution = -1
    morph_channel = var_dict['MorphologyChannel']

    for well in var_dict['Wells']:

        # If coordinates were generated during alignment,
        # translate images and create borders
        if coords_exist:
            for channel in var_dict['Channels']:
                selection_criterion = utils.make_selector(well=well, channel=channel)

                # Get translation coordinates
                transl_crdn = var_dict['Translation_coordinates_xy'][well]

                # Translate, add border, crop
                transl_files = utils.make_filelist(
                    path_aligned_images, selection_criterion)
                # # This condition has implications
                # if len(transl_files) == 0:
                #     continue
                translate_each_image_add_border(
                    transl_crdn, transl_files,
                    path_shift_cropped_images, resolution)
                border_images = utils.make_filelist(
                    path_shift_cropped_images, selection_criterion)
                # # This condition has implications
                # if len(border_images) == 0:
                #     continue
                crop_from_coordinates(
                    transl_crdn, border_images,
                    path_shift_cropped_images, resolution)

        # If borders were generated during alignment, start here
        else:
            selection_criterion = utils.make_selector(
                well=well, channel=morph_channel)
            border_images = utils.make_filelist(
                path_aligned_images, selection_criterion)
            # # This has implications:
            # if len(border_images) == 0:
            #     continue
            # pprint.pprint([os.path.basename(
            #     border_image) for border_image in border_images])

            # all_border_images = []
            # for channel in var_dict['Channels']:
            #     # select_criterion = well+'_*'+channel
            #     select_criterion = utils.make_selector(
            #         well=well, channel=channel)
            #     select_images = utils.make_filelist(
            #         path_aligned_images, select_criterion)
            #     all_border_images.append(select_images)

            # print 'Number of channels', len(all_border_images)

            # # crop_from_borders(
            # #     border_images, path_shift_cropped_images,
            # #     resolution, all_border_images)

            # edges = crop_from_borders(
            #     border_images, resolution)
            # apply_max_crop(
            #     path_shift_cropped_images, resolution, all_border_images, edges)
           
            select_criterion = utils.make_selector(well=well)
            select_images = utils.make_filelist(
                path_aligned_images, select_criterion)
            # # This has implications:
            # if len(select_images) == 0:
            #     continue 
            edges = crop_from_borders(
                border_images, resolution)
            apply_max_crop(
                path_shift_cropped_images, resolution, select_images, edges)

    # For select_analysis_module input, set var_dict['OutputPath']
    var_dict["OutputPath"] = path_shift_cropped_images


if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Crop images to common matrix.")
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

    # ----Initialize parameters------------------
    path_aligned_images = args.input_path
    path_shift_cropped_images = args.output_path
    outfile = args.output_dict
    resolution = var_dict['Resolution']
    morph_channel = var_dict['MorphologyChannel']

    # ----Confirm given folders exist--
    assert os.path.exists(path_aligned_images), 'Confirm the given path for data exists.'
    assert os.path.exists(path_shift_cropped_images), 'Confirm the given path for results exists.'

    # ----Shift and crop-------------------------
    start_time = datetime.datetime.utcnow()
    coords_exist = 'Translation_coordinates_xy' in var_dict.keys()

    shift_crop(var_dict, path_aligned_images, path_shift_cropped_images)

    end_time = datetime.datetime.utcnow()
    print 'Crop run time:', end_time-start_time
    # ----Output for user and save dict----------
    print 'Aligned images were cropped.'
    print 'Output was written to:'
    print path_shift_cropped_images

    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    # outfile = os.rename('var_dict.p', outfile)
    outfile = shutil.move('var_dict.p', outfile)

