'''
Segments each image and keeps all contours.
Selects contours that are cells based on selection criteria.
Saves image of all found contours and selected contours for QC.
'''

import cv2, utils, os, argparse
import numpy as np
import pickle, datetime, shutil, math

def find_cells(img, percent_int_thresh=0.1, img_is_mask=False):
    '''
    Resets depth of image to 'uint8'.
    Finds contours in given image.
    '''
    if img_is_mask == True:
        contours, hierarchy = cv2.findContours(
            img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print('Number of masked cells', len(contours))
        return contours
    ret, mask = cv2.threshold(
        img, int(img.max()*percent_int_thresh), img.max(), cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def find_cells_sd(img_pointer, num_sd=4):
    '''
    Resets depth of image to 'uint8'.
    Finds contours in given image within a given standard deviation
    '''
    img = cv2.imread(img_pointer, -1)
    if var_dict['ThresholdType'] == 'sd_from_mean_dataset':
        thresh_val = dataset_mean + num_sd*dataset_sd
    else:
        thresh_val = img.mean() + num_sd*img.std()
    indices_below_threshold = img < thresh_val
    img[indices_below_threshold] = 0
    img[~indices_below_threshold] = 255

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
    img[~indices_below_threshold] = 255

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
        print ('Kept', len(contours_kept), \
            '/', len(contours), 'contours.')

    return contours_kept

def show_kept_cells(img_pointer, contours, contours_kept, write_path, max_val=2**10):
    '''
    Generate small image centered around cell of interest.
    '''
    # Original image
    orig_img = cv2.imread(img_pointer, -1)
    orig_img = cv2.normalize(orig_img, None, alpha=0, beta=2**16, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    orig_img = np.array(orig_img, dtype=np.uint8)
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_GRAY2BGR)

    # Naming
    orig_name = utils.extract_file_name(img_pointer)
    img_name = utils.make_file_name(write_path, orig_name+'_KEPTCELLS')
    well_name = os.path.basename(img_pointer).split('_')[4]
    img_name = utils.reroute_imgpntr_to_wells(img_name, well_name)

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.drawContours(orig_img, contours, -1, (0, 255, 0), 5)
    cv2.putText(orig_img, 'All contours', (20, 120), font, 4, (0, 255, 0), 5, cv2.LINE_AA)

    cv2.drawContours(orig_img, contours_kept, -1, (255, 0, 0), 5)
    cv2.putText(orig_img, 'Selected cell contours', (20, 220), font, 4, (255, 0, 0), 5, cv2.LINE_AA)

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
    cv2.imwrite(img_name, mask)

def get_plate_thresh(img_pointer, dataset_mean, dataset_sd):
    img = cv2.imread(img_pointer, -1)
    img_mean = img.mean()
    img_sd = img.std()
    dataset_mean.append(img_mean)
    dataset_sd.append(img_sd)

    return dataset_mean, dataset_sd


def segmentation(var_dict, path_to_images, write_masks_path, write_qc_path):
    '''
    Main point of entry.
    '''
    if var_dict['ThresholdType'] in ["sd_from_mean_image","sd_from_mean_dataset","intensity_percent"]:
        resolution = -1
    else:
        resolution = 0

    if var_dict['ThresholdType'] == 'sd_from_mean_dataset':
        img_list = utils.get_all_files_all_subdir(path_to_images)
        global dataset_mean
        global dataset_sd
        dataset_mean = []
        dataset_sd = []
        for img_pointer in img_list:
            dataset_mean, dataset_sd = get_plate_thresh(img_pointer, dataset_mean, dataset_sd)
        dataset_mean = sum(dataset_mean)/len(dataset_mean)
        k = len(dataset_sd)
        dataset_sd = math.sqrt(sum([x**2 for x in dataset_sd])/k)
        print('Calculated dataset mean: %.1f' % dataset_mean)
        print('Calculated dataset standard deviation: %.1f' % dataset_sd)

    for well in var_dict['Wells']:
        print ('Well', well)
        selector = utils.make_selector(
            well=well, channel=var_dict['MorphologyChannel'])
        images_list = utils.make_filelist_wells(path_to_images, selector)

        for img_pointer in images_list:
            if var_dict['ThresholdType'] == 'intensity_percent_legacy':
                img = cv2.imread(img_pointer, resolution)
                contours = find_cells(
                    img, percent_int_thresh=var_dict['IntensityThreshold'])
            elif 'sd_from_mean' in var_dict['ThresholdType']:
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
    parser.add_argument("--input_image_path",
        help="Folder path to input data.", default = '')
    parser.add_argument("--output_results_path",
        help="Folder path to ouput results.", default = '')
    parser.add_argument("--qc_path",
        help="Folder path to qc of results.", default = '')
    parser.add_argument("threshold_type",
        help="Method for setting threshold.")
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
    path_to_images = utils.get_path(args.input_image_path, var_dict['GalaxyOutputPath'], 'AlignedImages')
    print ('Input path: %s' % path_to_images)

    write_masks_path = utils.get_path(args.output_results_path, var_dict['GalaxyOutputPath'], 'CellMasks')
    print ('Cell Masks output path: %s' % write_masks_path)

    write_qc_path = utils.get_path(args.qc_path, var_dict['GalaxyOutputPath'], 'QualityControl')
    print ('QC output path: %s' % write_qc_path)

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

    print ("Threshold type: %s" % threshold_type)
    print ("Minimum object size set to:", var_dict["MinCellSize"])
    print ("Maximum object size set to:", var_dict["MaxCellSize"])
    print ("Eccentricity set to:", var_dict["Eccentricity"])
    print ("Background threshold set to:", float(var_dict['IntensityThreshold']))

    # ----Confirm given folders exist--
    assert os.path.exists(path_to_images), 'Confirm the path for data exists (%s)' % path_to_images
    assert os.path.exists(write_masks_path), 'Confirm the path for mask output exists (%s)' % write_masks_path
    assert os.path.exists(write_qc_path), 'Confirm the path for qc output exists (%s)' % write_qc_path

    # ----Run segmentation-----------------------
    start_time = datetime.datetime.utcnow()

    segmentation(var_dict, path_to_images, write_masks_path, write_qc_path)

    end_time = datetime.datetime.utcnow()
    print ('Segmentation run time:', end_time-start_time)
    # ----Output for user and save dict----------

    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    outfile = shutil.move('var_dict.p', outfile)
    timestamp = utils.update_timestring()
    utils.save_user_args_to_csv(args, write_masks_path, 'segmentation'+'_'+timestamp)


