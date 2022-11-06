'''
Segments each image and keeps all contours.
Selects contours that are cells based on selection criteria.
Saves image of all found contours and selected contours for QC.
'''

import cv2, utils, sys, os, argparse
import numpy as np
import pickle, datetime, shutil

def find_cells(img, percent_int_thresh=0.1, img_is_mask=False):
    '''
    Resets depth of image to 'uint8'.
    Finds contours in given image.
    '''
    if img_is_mask == True:
        contours, hierarchy = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print 'Number of masked cells', len(contours)
        return contours

    # if img.max() < 50:
    #     factor = int(50./img.max())
    #     img = img*factor
    ret, mask = cv2.threshold(
        img, int(img.max()*percent_int_thresh), img.max(), cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_cells_sd(img_pointer, num_sd=4):
    '''
    Resets depth of image to 'uint8'.
    Finds contours in given image.
    '''
    img = cv2.imread(img_pointer, -1)
    thresh_val = img.mean() + num_sd*img.std()
    indices_below_threshold = img < thresh_val
    img[indices_below_threshold] = 0
    img[-indices_below_threshold] = 255

    img = np.array(img, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def find_cells_percent(img_pointer, percent_int_thresh=0.1, img_is_mask=False):
    '''
    Resets depth of image to 'uint8'.
    Finds contours in given image.
    '''
    img = cv2.imread(img_pointer, -1)
    thresh_val = percent_int_thresh * img.max()
    indices_below_threshold = img < thresh_val
    img[indices_below_threshold] = 0
    img[-indices_below_threshold] = 255

    img = np.array(img, dtype=np.uint8)
    contours, hierarchy = cv2.findContours(
        img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def filter_contours(contours, small=50, large=2500, ecn=.1, verbose=True):
    '''
    Filters contours based on size and eccentricity.
    Will learn other important parameters.
    '''

    contours_kept = []
    for cnt in contours:
        if len(cnt)>5 and cv2.contourArea(cnt)>small and cv2.contourArea(cnt)<large:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            ecc = np.sqrt(1-((MA)**2/(ma)**2))
            if ecc >= ecn:
                contours_kept.append(cnt)

    if verbose:
        print 'Kept', len(contours_kept), \
            '/', len(contours), 'contours.'

    return contours_kept

def show_kept_cells(img_pointer, contours, contours_kept, write_path, max_val=2**10):
    '''
    Generate small image centered around cell of interest.
    '''
    # Original image
    orig_img = cv2.imread(img_pointer, -1)
    orig_img = cv2.normalize(orig_img, alpha=0, beta=2**16, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    orig_img = np.array(orig_img, dtype=np.uint8)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)

    # Naming
    orig_name = utils.extract_file_name(img_pointer)
    img_name = utils.make_file_name(write_path, orig_name+'_KEPTCELLS')
    well_name = os.path.basename(img_pointer).split('_')[4]
    img_name = utils.reroute_imgpntr_to_wells(img_name, well_name)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.drawContours(orig_img, contours, -1, (0, 255, 0), 5)
    cv2.putText(orig_img, 'All contours', (20, 120), font, 4, (0, 255, 0), 5, cv2.CV_AA)

    cv2.drawContours(orig_img, contours_kept, -1, (255, 0, 0), 5)
    cv2.putText(orig_img, 'Selected cell contours', (20, 220), font, 4, (255, 0, 0), 5, cv2.CV_AA)

    cv2.imwrite(img_name, orig_img)


def make_cell_mask(img_pointer, contours_kept, write_path, resolution):
    '''
    Draw kept cells onto image.
    '''
    # Handling 8bit vs 16bit
    d_type = np.uint16 if resolution == -1 else np.uint8 if resolution == 0 else None
    assert d_type == np.uint16 or d_type == np.uint8, 'Unsupported intensity depth.'

    img = cv2.imread(img_pointer, resolution)
    mask = np.zeros(img.shape[0:2], dtype=np.uint8)
    orig_name = utils.extract_file_name(img_pointer)
    img_name = utils.make_file_name(write_path, orig_name+'_CELLMASK')
    well_name = os.path.basename(img_pointer).split('_')[4]
    img_name = utils.reroute_imgpntr_to_wells(img_name, well_name)

    cv2.drawContours(mask, contours_kept, -1, 255, -1)
    # cv2.drawContours(mask, contours_kept, -1, 255, 5)
    # Consided adding black border for better separation
    # cv2.drawContours(mask, contours_kept, -1, 0, 1)

    cv2.imwrite(img_name, mask)

def segmentation(var_dict, path_to_images, write_masks_path, write_qc_path):
    '''
    Main point of entry.
    '''
    if var_dict['ThresholdType'] in ["sd_from_mean","intensity_percent"]:
        resolution = -1
    else:
        resolution = 0

    for well in var_dict['Wells']:
        print 'Well', well
        selector = utils.make_selector(
            well=well, channel=var_dict['MorphologyChannel'])
        # images_list = utils.make_filelist(path_to_images, selector)
        images_list = utils.make_filelist_wells(path_to_images, selector)
        for img_pointer in images_list:

            if var_dict['ThresholdType'] == 'intensity_percent_legacy':
                img = cv2.imread(img_pointer, resolution)
                contours = find_cells(
                    img, percent_int_thresh=var_dict['IntensityThreshold'])
            elif var_dict['ThresholdType'] == 'sd_from_mean':
                contours = find_cells_sd(
                    img_pointer, num_sd=var_dict['IntensityThreshold'])
            elif var_dict['ThresholdType'] == 'intensity_percent':
                contours = find_cells_percent(
                    img_pointer, percent_int_thresh=var_dict['IntensityThreshold'])
            else:
                assert False, "Thresholding type was not provided and no contours were identified."

            contours_kept = filter_contours(
                contours, small=var_dict["MinCellSize"],
                large=var_dict["MaxCellSize"],
                ecn=var_dict['Eccentricity'])
            make_cell_mask(
                img_pointer, contours_kept,
                write_masks_path, resolution)
            show_kept_cells(
                img_pointer, contours, contours_kept, write_qc_path)

    # For select_analysis_module input, set var_dict['OutputPath']
    var_dict["OutputPath"] = write_masks_path

if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Segment cell bodies and create masks.")
    parser.add_argument("input_dict",
        help="Load input variable dictionary")
    parser.add_argument("input_path",
        help="Folder path to input data.")
    parser.add_argument("output_path",
        help="Folder path to ouput results.")
    parser.add_argument("qc_path",
        help="Folder path to qc of results.")
    parser.add_argument("threshold_type",
        help="Method for setting threhsold.")
    parser.add_argument("output_dict",
        help="Write variable dictionary.")
    parser.add_argument("--min_cell",
        dest="min_cell", type=int, default=50,
        help="Minimum feature size considered as cell.")
    parser.add_argument("--max_cell",
        dest="max_cell", type=int, default=2500,
        help="Maximum feature size considered as cell.")
    parser.add_argument("--eccentricity", "-ec",
        dest="eccentricity", type=float, default=0.1,
        help="Threshold value as a percent of maximum intensity.")
    parser.add_argument("--intensity_threshold", "-t",
        dest="intensity_threshold", type=float, default=0.1,
        help="If using percent intensity threshold, enter percent of pixels to assign to background. Example: 0.1 of maximum intensity (10 percent). If using SD-based threshold, use threshold and histogram tools in FIJI to calculate number of standard deviations from the mean tequired to threshold images accurately. Enter value as a multiple of standard deviations. Example: 3.5")
    args = parser.parse_args()

    # ----Load path dict-------------------------
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))

    # ----Initialize parameters------------------
    path_to_images = args.input_path
    write_masks_path = args.output_path
    write_qc_path = args.qc_path
    threshold_type = args.threshold_type
    morphology_channel = var_dict['MorphologyChannel']
    var_dict["MinCellSize"] = int(args.min_cell)
    var_dict["MaxCellSize"] = int(args.max_cell)
    var_dict['Eccentricity'] = float(args.eccentricity)
    var_dict['IntensityThreshold'] = float(args.intensity_threshold)
    var_dict['ThresholdType'] = threshold_type
    outfile = args.output_dict
    resolution = 0#-1

    assert var_dict["MinCellSize"] < var_dict["MaxCellSize"], 'Confirm that min size is smaller than max size.'

    print "Minimum object size set to:", var_dict["MinCellSize"]
    print "Maximum object size set to:", var_dict["MaxCellSize"]
    print "Eccentricity set to:", var_dict["Eccentricity"]
    print "Background threshold set to:", float(var_dict['IntensityThreshold'])

    # ----Confirm given folders exist--
    assert os.path.exists(path_to_images), 'Confirm the given path for data exists.'
    assert os.path.exists(write_masks_path), 'Confirm the given path for mask output exists.'
    assert os.path.exists(write_qc_path), 'Confirm the given path for qc output exists.'

    # ----Run segmentation-----------------------
    start_time = datetime.datetime.utcnow()

    segmentation(var_dict, path_to_images, write_masks_path, write_qc_path)

    end_time = datetime.datetime.utcnow()
    print 'Segmentation run time:', end_time-start_time
    # ----Output for user and save dict----------
    print 'Selected images were segmented.'
    print 'Output was written to:'
    print args.qc_path

    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    # outfile = os.rename('var_dict.p', outfile)
    outfile = shutil.move('var_dict.p', outfile)
    utils.save_user_args_to_csv(args, write_masks_path, 'segmentation')

