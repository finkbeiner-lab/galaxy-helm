'''
Segments each image and keeps all contours. 
  This step is intensity-based.
Selects contours that are objects of interest based on selection criteria.
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
            img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        print 'Number of masked cells', len(contours)
        return contours

    # if img.max() < 50:
    #     factor = int(50./img.max())
    #     img = img*factor
    ret, mask = cv2.threshold(
        img, int(img.max()*percent_int_thresh), img.max(), cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def filter_contours(contours, small=50, large=2500, ecn=.1, verbose=True):
    '''
    Filters contours based on size and eccentricity.
    Will learn other important parameters.
    '''

    contours_kept = []
    for cnt in contours:
        if len(cnt) > 5 and cv2.contourArea(cnt) > small \
            and cv2.contourArea(cnt) < large:
            (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
            ecc = np.sqrt(1-((MA)**2/(ma)**2))
            if ecc > ecn:
                contours_kept.append(cnt)

    if verbose:
        print 'Kept', len(contours_kept), \
            '/', len(contours), 'contours.'

    return contours_kept

def zerone_normalizer(image):
    '''
    Normalizes matrix to have values between some min and some max.
    This is exactly equivalent to cv2.equalizeHist(image) if min and max are 0 and 255
    '''
    copy_image = image.copy()
    #set scale
    new_img_min, new_img_max = 0, 240
    zero_one_norm = (copy_image - image.min()) * (
        (new_img_max - new_img_min) / (image.max() - image.min())) + new_img_min
    return zero_one_norm

def show_kept_cells(img_pointer, contours, contours_kept, write_path):
    '''
    Draw kept cells onto image.
    '''

    img = 50*cv2.imread(img_pointer, 0)
    # factor = int(255./img.max())
    # img = factor*img
    # img = zerone_normalizer(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    orig_name = utils.extract_file_name(img_pointer)
    img_name = utils.make_file_name(write_path, orig_name+'_KEPTCELLS')

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.drawContours(img, contours, -1, (255, 255, 0), 2)
    cv2.putText(img, 'All contours', (20, 120), font, 4, (255, 255, 0), 5, cv2.CV_AA)
    # cv2.imshow('img', utils.width_resizer(img, 500))
    # cv2.waitKey(0)

    cv2.drawContours(img, contours_kept, -1, (255, 0, 0), 5)
    cv2.putText(img, 'Selected cell contours', (20, 220), font, 4, (255, 0, 0), 5, cv2.CV_AA)
    # cv2.imshow('img', utils.width_resizer(img, 500))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    cv2.imwrite(img_name, img)

def make_cell_mask(img_pointer, contours, contours_kept, write_path, resolution):
    '''
    Draw kept cells onto image.
    '''
    # Handling 8bit vs 16bit
    d_type = np.uint16 if resolution == -1 else np.uint8 if resolution == 0 else None
    assert d_type == np.uint16 or d_type == np.uint8, 'Unsupported intensity depth.'

    img = cv2.imread(img_pointer, resolution)
    mask = np.zeros(img.shape[0:2], dtype=d_type)
    orig_name = utils.extract_file_name(img_pointer)
    img_name = utils.make_file_name(write_path, orig_name+'_CELLMASK')

    cv2.drawContours(mask, contours_kept, -1, 255, -1)
    cv2.drawContours(mask, contours_kept, -1, 255, 5)

    cv2.imwrite(img_name, mask)

def segmentation(var_dict, path_to_images, write_masks_path, write_qc_path):
    '''
    Main point of entry.
    '''
    resolution = 0

    for well in var_dict['Wells']:
        print 'Well', well
        selector = utils.make_selector(
            well=well, channel=var_dict['MorphologyChannel'])
        images_list = utils.make_filelist(path_to_images, selector)
        for img_pointer in images_list:
            img = cv2.imread(img_pointer, resolution)
            contours = find_cells(img, var_dict['IntensityThreshold'])
            contours_kept = filter_contours(
                contours, 
                small=var_dict["MinCellSize"], 
                large=var_dict["MaxCellSize"], 
                ecn=var_dict['Eccentricity'])
            make_cell_mask(
                img_pointer, contours, contours_kept,
                write_masks_path, resolution)
            show_kept_cells(
                img_pointer, contours, contours_kept, write_qc_path)  
    cv2.destroyAllWindows()

    # For select_analysis_module input, set var_dict['OutputPath']
    var_dict["OutputPath"] = write_masks_path   

if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Segment objects and create masks.")
    parser.add_argument("input_dict", 
        help="Load input variable dictionary")
    parser.add_argument("input_path", 
        help="Folder path to input data.")
    parser.add_argument("output_path",
        help="Folder path to ouput results.")
    parser.add_argument("qc_path",
        help="Folder path to qc of results.")
    parser.add_argument("output_dict", 
        help="Write variable dictionary.")
    parser.add_argument("--min_feature",
        dest="min_feature", type=int, default=50,
        help="Minimum feature size considered as cell.")
    parser.add_argument("--max_feature",
        dest="max_feature", type=int, default=2500,
        help="Maximum feature size considered as cell.")
    parser.add_argument("--eccentricity", "-ec",
        dest="eccentricity", type=float, default=0.1,
        help="Threshold value as a percent of maximum intensity.")
    parser.add_argument("--threshold_percent", "-tp",
        dest="threshold_percent", type=float, default=0.1,
        help="Threshold value as a percent of maximum intensity.")
    args = parser.parse_args()

    # ----Load path dict-------------------------
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))

    # ----Initialize parameters------------------
    path_to_images = args.input_path
    write_masks_path = args.output_path
    write_qc_path = args.qc_path

    morphology_channel = var_dict['MorphologyChannel']
    var_dict["MinCellSize"] = int(args.min_feature)
    var_dict["MaxCellSize"] = int(args.max_feature)
    var_dict['Eccentricity'] = float(args.eccentricity)
    var_dict['IntensityThreshold'] = float(args.threshold_percent)
    outfile = args.output_dict
    resolution = 0

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
    print 'Images were segmented.'
    print 'Output was written to:'
    print write_masks_path

    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    # outfile = os.rename('var_dict.p', outfile)
    outfile = shutil.move('var_dict.p', outfile)

