'''
Takes dictionary from tracking output. Randomly selects 20 cell objects.
For each cell object, gets coordinates and associated 250x250 image
Read image in as 8bit and equalize hists for each color.
Print image name, and evaluated parameter+value (mean: mean_value) on mask image.
RGB merge cell panel and add to cell image list.
Stacks each five images horizontally and then four horizontal panels vertically.

@Usage
base_path sets directory for experiment's processed data.
the original images will be collected from base_path/MontagedImages/
the overlays for (use csv) will be collected from base_path/OverlaysTablesResults/
write_path sets directory for output
channel naming scheme is currently RFP, DAPI, FITC (3) or Green, Cyan, Red (4)
these represent morphology, positive (neuron/live), negative (glia/dead)
Cell object info can come from dictoinary or csv.

base path examples
  base_path = '/media/robodata/Robo3Images/MariyaLiveStainPlate9/Processed/'
  base_path = '/media/robodata/Gaia/Robo4/MariyaPlates/MariyaPlate8ICC/Processed/'
write path examples
  write_path = '/media/robodata/Mariya/RandomCells/FilteredContoursT1toT6andRowARowB'
  write_path = '/media/robodata/Mariya/RandomCells/ICC/'
robo3 or robo4 channel naming reference
Use dict (True) or csv (False)
'''

import cv2, utils, sys, os, argparse
import numpy as np
import scipy.stats as stat
import pickle, collections
import pprint, datetime, shutil
import random, glob
import pandas as pd

class Cell(object):
    '''
    A class that makes cells from contours.
    '''
    def __init__(self, cnt, ch_images=None):
        self.cnt = cnt
        self.all_ch_int_stats = None

        if ch_images:
            self.collect_all_ch_intensities(ch_images)

    def __repr__(self):
        return "Cell instance (%s center)" % str(self.get_center()[0])

    # ----Contours-------------------------------
    def get_center(self):
        '''Returns centroid of contour.'''
        (x,y), radius = cv2.minEnclosingCircle(self.cnt)
        center = (int(x), int(y))
        return center

    def calculate_cnt_parameters(self):
        '''Extracts all cell-relevant parameters.'''
        cell_params = {}
        area_cnt = cv2.contourArea(self.cnt)
        cell_params['BlobArea'] = area_cnt

        perimeter = cv2.arcLength(self.cnt, True)
        cell_params['BlobPerimeter'] = perimeter

        center, radius = cv2.minEnclosingCircle(self.cnt)
        cell_params['Radius'] = radius

        (ex, ey), (MA, ma), angle = cv2.fitEllipse(self.cnt)
        cell_params['BlobCentroidX'] = ex
        cell_params['BlobCentroidY'] = ey
        ecc = np.sqrt(1-((MA)**2/(ma)**2))
        cell_params['BlobCircularity'] = ecc

        hull = cv2.convexHull(self.cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area_cnt)/hull_area
        cell_params['Spread'] = solidity

        convexity = cv2.isContourConvex(self.cnt)
        cell_params['Convexity'] = convexity

        return cell_params

    # ----Intensities----------------------------
    def find_cnt_int_dist(self, img):
        '''
        Finds pixels associated with contour.
        Returns intensity parameters.
        This is one of the required parameters to instnatiate a Cell_obj.
        '''

        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [self.cnt], 0, 256, -1)
        cnt_ints = img[np.nonzero(mask)]

        cell_int_stats = {}
        # Intensity params
        cell_int_stats['PixelIntensityMinimum'] = cnt_ints.min()
        cell_int_stats['PixelIntensityMaximum'] = cnt_ints.max()
        cell_int_stats['PixelIntensityMean'] = cnt_ints.mean()
        cell_int_stats['PixelIntensityStdDev'] = cnt_ints.std()
        cell_int_stats['PixelIntensityVariance'] = cnt_ints.var()

        (q1, q5, q10, q25, q50, q75, q90, q95, q99) = np.percentile(
            cnt_ints, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        cell_int_stats['PixelIntensity1Percentile'] = q1
        cell_int_stats['PixelIntensity5Percentile'] = q5
        cell_int_stats['PixelIntensity10Percentile'] = q10
        cell_int_stats['PixelIntensity25Percentile'] = q25
        cell_int_stats['PixelIntensity50Percentile'] = q50
        cell_int_stats['PixelIntensity75Percentile'] = q75
        cell_int_stats['PixelIntensity90Percentile'] = q90
        cell_int_stats['PixelIntensity95Percentile'] = q95
        cell_int_stats['PixelIntensity99Percentile'] = q99
        cell_int_stats['PixelIntensityInterquartileRange'] = q75-q25

        cell_int_stats['PixelIntensitySkewness'] = stat.skew(cnt_ints)
        cell_int_stats['PixelIntensityKurtosis'] = stat.kurtosis(cnt_ints)

        return cell_int_stats


def find_cells(img, percent_int_thresh=0.1):
    '''
    Resets depth of image to 'uint8'.
    Finds contours in given image.
    '''

    ret, mask = cv2.threshold(
        img, int(img.max()*percent_int_thresh), img.max(), cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    kept_contours = filter_contours(contours)

    return kept_contours

def filter_contours(contours, small=50, large=2500, ecn=.1, verbose=True):
    '''
    Filters contours based on size and eccentricity.
    Will learn other important parameters.
    '''

    contours_kept = []
    for cnt in contours:
        if len(cnt) > 5 and cv2.contourArea(cnt) > small \
            and cv2.contourArea(cnt) < large:
            (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
            ecc = np.sqrt(1-((MA)**2/(ma)**2))
            if ecc > ecn:
                contours_kept.append(cnt)

    if verbose:
        print 'Kept', len(contours_kept), \
            '/', len(contours), 'contours.'

    return contours_kept

def zerone_normalizer(image, new_img_min=0, new_img_max=200):
    '''
    Normalizes matrix to have values between some min and some max.
    This is exactly equivalent to cv2.equalizeHist(image) if min and max are 0 and 255
    '''
    copy_image = image.copy()
    #set scale
    zero_one_norm = (copy_image - image.min())*(
        (new_img_max-new_img_min) / (image.max() - image.min()) )+new_img_min
    return zero_one_norm

def get_random_cell_object_dict(var_dict, path, morph_string, blue_string, green_string):
    '''
    Get random cell object and relevant image pointer.
    '''

    # Obtaining the cell: (cell_id, cell_obj)
    well_ind = random.randint(0,len(var_dict['Wells'])-1)
    well = var_dict['Wells'][well_ind]
    time_ind = random.randint(1,len(var_dict['TimePoints'])-1)
    time = var_dict['TimePoints'][time_ind]
    cell_ind = random.randint(0,len(var_dict['TrackedCells'][well][time])-1)
    cell = var_dict['TrackedCells'][well][time][cell_ind]

    # Obtaining the images: blue, green, red, overlay
    assoc_images = glob.glob(os.path.join(path, '*'+time+'_*'+well+'_*.tif'))
    pprint.pprint([os.path.basename(assoc_image) for assoc_image in assoc_images])
    if len(assoc_images)<3:
        print "This happened!!!"
        print well, time, 
        print os.path.join(path, '*'+time+'_*'+well+'_*.tif')
        cell, bg_images = get_random_cell_object_dict(var_dict, path, morph_string, blue_string, green_string)
        return cell, bg_images

    blue_img = [img for img in assoc_images if blue_string in img]
    green_img = [img for img in assoc_images if green_string in img]
    morph_img = [img for img in assoc_images if morph_string in img] #cell_img
    bg_images = (blue_img[0], green_img[0], morph_img[0])

    return cell, bg_images

def get_random_cell_object_csv(
    var_dict, path, base_path, morph_string, blue_string, green_string):
    '''
    Get random cell object and relevant image pointer.
    '''

    # Obtaining the cell: row in csv
    rows, cols = var_dict.shape
    cell_row = random.randint(0,rows-1)
    cell_details = var_dict.ix[cell_row]

    # Obtaining the images: blue, green, red, overlay
    assoc_images = glob.glob(os.path.join(
        path, '*T'+str(cell_details.Timepoint)+'_*'+str(cell_details.Sci_WellID)+'_*.tif'))
    ol_path = os.path.join(base_path, overlays_folder)
    assoc_morph = glob.glob(os.path.join(
        ol_path, '*T'+str(cell_details.Timepoint)+'_*'+str(cell_details.Sci_WellID)+'_*.tif'))
    mask_img = [img for img in assoc_morph if morph_string in img] #mask
    morph_img = [img for img in assoc_images if morph_string in img] #cell_img
    blue_img = [img for img in assoc_images if blue_string in img]
    green_img = [img for img in assoc_images if green_string in img]
    if len(morph_img)<1 or len(blue_img)<1 or len(green_img)<1 or len(mask_img)<0:
        print 'Did not find an image.'
        get_random_cell_object_csv(
            var_dict, path, base_path, morph_string, blue_string, green_string)
    else:    
        bg_images = (blue_img[0], green_img[0], morph_img[0], mask_img[0])

    print 'Row index:', cell_row
    print cell_details[[1,5,6,11,14]]
    print morph_img[0]

    return cell_details, bg_images

def make_small_img(img_pointer, xy_coords, max_val=250):
    '''
    Generate small image centered around cell of interest.
    '''
    x_left, x_right, y_top, y_bottom = xy_coords
    orig_img = cv2.imread(img_pointer, 0)
    small_img = np.zeros((250, 250), np.uint8)
    small_img[0:y_bottom-y_top,0:x_right-x_left] = orig_img[y_top:y_bottom, x_left:x_right]
    small_img = zerone_normalizer(small_img, new_img_max=max_val)
    small_img = small_img.astype(np.uint8)
    return small_img

def make_small_mask(mask_pointer, xy_coords, cellx, celly):
    '''
    Generate small overlay image centered around cell of interest.
    '''
    x_left, x_right, y_top, y_bottom = xy_coords
    orig_img = cv2.imread(mask_pointer, -1)*0.3
    cv2.circle(orig_img, (cellx,celly), 2, 255, -1)
    small_img = np.zeros((250, 250, 3), np.uint8)
    small_img[0:y_bottom-y_top,0:x_right-x_left, 0] = orig_img[y_top:y_bottom, x_left:x_right]
    small_img[0:y_bottom-y_top,0:x_right-x_left, 1] = orig_img[y_top:y_bottom, x_left:x_right]
    small_img = small_img.astype(np.uint8)

    return small_img

def make_small_cnt_img(xy_coords, rows, cols, cell_obj):
    '''
    Generate small image with cell contour drawn for cell of interest.
    '''
    x_left, x_right, y_top, y_bottom = xy_coords
    orig_img = np.zeros((rows, cols), np.uint8)
    cv2.drawContours(orig_img, [cell_obj.cnt], 0, 255, 1)
    mask_img = np.zeros((250,250,3), np.uint8)
    mask_img[0:y_bottom-y_top,0:x_right-x_left, 2] = orig_img[y_top:y_bottom, x_left:x_right]
    mask_img[0:y_bottom-y_top,0:x_right-x_left, 1] = orig_img[y_top:y_bottom, x_left:x_right]

    return mask_img

def find_xy_ends(bimg_pointer, cellx, celly, verbose=False):
    '''
    Returns x and y range to crop image around cell center.
    '''
    # Define image parameters
    im = cv2.imread(bimg_pointer, 0)
    rows, cols = im.shape

    # Get image coordinates around cell center
    x_left = int(cellx)-125
    if x_left < 0:
        x_left = 0
    x_right = int(cellx)+125
    if x_right > cols:
        x_right = cols
    y_top = int(celly)-125
    if y_top < 0:
        y_top = 0
    y_bottom = int(celly)+125
    if y_bottom > rows:
        y_bottom = rows

    if verbose:
        print 'X borders:', x_left, cellx, x_right
        print 'Y borders:', y_top, celly, y_bottom
    return (x_left, x_right, y_top, y_bottom), rows, cols

def get_all_images_and_params(cell, bg_images, eval_string):
    '''
    For a cell object, get the relevant cropped cell image.
    Return BGR image of all channels and printed text corresponding to cell.
    '''
    # Get images and cell object
    if len(bg_images) == 3:
        bimg_pointer, gimg_pointer, morph_img_pointer = bg_images
        mask_img_pointer = None
    else:
        bimg_pointer, gimg_pointer, morph_img_pointer, mask_img_pointer = bg_images

    # Get cell center and index
    if type(cell) == tuple:
        cell_ind, cell_obj = cell
        (cellx, celly) = cell_obj.get_center()
        #blue_img = cv2.imread(bimg_pointer, 0)
        #green_img = cv2.imread(gimg_pointer, 0)
        #blue_ints = cell_obj.find_cnt_int_dist(blue_img)
        #green_ints = cell_obj.find_cnt_int_dist(green_img)
    else:
        cell_ind = cell.ObjectLabelsFound
        (cellx, celly) = int(cell.BlobCentroidX), int(cell.BlobCentroidY)

    # Define image parameters
    xy_coords, rows, cols = find_xy_ends(bimg_pointer, cellx, celly)

    # Blue and green image handling
    blue_img = make_small_img(bimg_pointer, xy_coords)
    green_img = make_small_img(gimg_pointer, xy_coords)

    if not mask_img_pointer: 
        # Mask and morphology image handling with dict
        morph_img = make_small_img(morph_img_pointer, xy_coords, max_val=240)
        mask_img = make_small_cnt_img(xy_coords, rows, cols, cell_obj)
        text_img = print_on_image(mask_img, cell, cell_ind, eval_string, bimg_pointer, (0, 255, 255))
    else: 
        # Mask and morphology image handling with csv
        mask_img = make_small_mask(mask_img_pointer, xy_coords, cellx, celly)
        morph_img = make_small_img(morph_img_pointer, xy_coords, max_val=240)
        text_img = print_on_image(mask_img, cell, cell_ind, eval_string, bimg_pointer, (0, 255, 255))

    # Returning BGR image with text
    image_panel = cv2.merge((blue_img, green_img, morph_img))
    image_panel = cv2.add(image_panel, text_img)

    return image_panel

def print_on_image(img_print, cell, cell_ind, eval_string, bimg_pointer, color):
    '''
    Prints cell and image details onto given image.
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_print, '_'.join(os.path.basename(bimg_pointer).split('_')[0:2]), (10, 10),
        font, .35, color, 1, cv2.CV_AA)
    cv2.putText(img_print, '_'.join(os.path.basename(bimg_pointer).split('_')[2:]), (10, 25),
        font, .35, color, 1, cv2.CV_AA)
    cv2.putText(img_print, 'Cell: '+str(cell_ind), (10, 40),
        font, .35, color, 1, cv2.CV_AA)
    if len(cell)!=2:
        cv2.putText(img_print, 'Row index: '+str(cell.name), (10, 250-30),
            font, .35, color, 1, cv2.CV_AA)
        # cv2.putText(img_print, eval_string, (10, 250-15),
        #     font, .35, color, 1, cv2.CV_AA)
        # cv2.putText(img_print, 'Live: '+str(cell.is_live), (10, 250-30),
        #     font, .35, color, 1, cv2.CV_AA)
    return img_print

def show_num_cells(var_dict, images_path, base_path, num_cells, eval_string, txt_f, write_img_pointer, csv=False):
    '''
    Generates an image matrix of random cell images.
    '''

    panel_holder = []
    for i in range(num_cells):
        print '--------Panel iteration:', i
        if csv:
            # Cell is object info, bg images are the blue and green images
            cell, bg_images = get_random_cell_object_csv(
                var_dict, images_path, base_path, morph_ch, type1_ch, type2_ch)
        else:
            cell, bg_images = get_random_cell_object_dict(
                var_dict, images_path, morph_ch, type1_ch, type2_ch)

        txt_f.write(','.join([str(cell.name), "", str(cell.name+1), write_img_pointer]))
        txt_f.write('\n')

        panel_holder.append(get_all_images_and_params(cell, bg_images, eval_string))

    cell_panels = np.vstack((
        np.hstack(tuple(panel_holder[0:5])),
        np.hstack(tuple(panel_holder[5:10])),
        np.hstack(tuple(panel_holder[10:15])),
        np.hstack(tuple(panel_holder[15:20])),
        ))

    return cell_panels

if __name__ == '__main__':
    print '........Started program.'
    # ----Parse args-----------------------------
    parser = argparse.ArgumentParser(
        description="Show random cells to label/evaluate.")
    parser.add_argument("base_path", 
        help="/media/../Processed/")
    parser.add_argument("write_path", 
        help="Write path from root.")
    parser.add_argument("channel_combo",
        help="Are the channels RFP,FITC,DAPI (3) or Green,Cyan,Red (4).")
    parser.add_argument("dict_in", 
        help="True is using dictionary as input. False if using csv as input.")
    args = parser.parse_args()


    base_path = args.base_path
    write_path = args.write_path
    robo_ch = args.channel_combo
    dict_in = args.dict_in == str(True)

    # ----Parameters-----------------------------
    images_path = os.path.join(base_path,'MontagedImages')
    overlays_folder = 'OverlaysTablesResults'
    num_cells = 20
    verbose = False

    if robo_ch == str(3):
        morph_ch = 'RFP'
        type1_ch = 'DAPI'
        type2_ch = 'FITC'
    if robo_ch == str(4):
        morph_ch = 'Green'
        type1_ch = 'Cyan'
        type2_ch = 'Red'

    # CSV output
    txt_f = open(os.path.join(write_path, 'manual_cell_labels.csv'), 'w')
    txt_f.write(','.join(["PandasRow", "GroundTruth", "RRow", "FilePath"]))
    txt_f.write('\n')

    # ----Load and execute-----------------------
    csv_name = 'cell_data_refactored.csv'
    csv_pointer = os.path.join(base_path, overlays_folder, csv_name)
    print csv_name

    if dict_in:
        infile = '/home/mariyabarch/Desktop/dictracked'
        var_dict = pickle.load(open(infile, 'rb'))
        print '....Unpickled...........'
        for k in range(5):
            write_img_pointer = os.path.join(
                write_path, 'random_cells_dict'+str(k)+'.tif')
            image_panel = show_num_cells(
                var_dict, images_path, base_path, num_cells, 
                "", txt_f, write_img_pointer)
            if verbose:
                print image_panel.shape
            cv2.imwrite(write_img_pointer, image_panel)

    else:
        var_dict = pd.read_csv(csv_pointer)
        print '....Read in csv.........'
        for k in range(1,2):
            write_img_pointer = os.path.join(
                write_path, 'random_cells_csv'+str(k)+'.tif')

            if verbose:
                print var_dict.shape
            image_panel = show_num_cells(
                var_dict, images_path, base_path, num_cells, 
                "", txt_f, write_img_pointer, csv=True)
            cv2.imwrite(write_img_pointer, image_panel)

    txt_f.close()

# ----------dkp----------
# eval_strings = ['Blue Mean/SD > Green Mean/SD', 'Blue Mean > Green Mean',
#     'Blue Max-Min > Green Max-Min', 'Blue Kurtosis > Green Kurtosis',
#     'Blue Int50Percent > Green Int50Percent', 'Blue Max > Green Max',
#     'Blue SD > Green SD', 'Blue (Max-Min)/SD > Green (Max-Min)/SD']

# csv_ids = ['mean-sd', 'mean', 
#     'range', 'kurt', 
#     'int50', 'max', 
#     'sd', 'range-sd']

# for evaluator, csv_id in zip(eval_strings, csv_ids):

    # csv_name = 'cell_data_refactored_labeled_'+csv_id+'.csv'


# Testing one:
    # # var_dict = pd.read_csv(csv_pointer)
    # infile = '/home/mariyabarch/Desktop/dictracked'
    # var_dict = pickle.load(open(infile, 'rb'))
    # cell, bg_images = get_random_cell_object_dict(
    #     var_dict, images_path, 'RFP', 'DAPI', 'FITC')
    # # cell, bg_images = get_random_cell_object_csv(
    # #     var_dict, images_path, base_path, 'RFP', 'DAPI', 'FITC')
    # image_panel = get_all_images_and_params(cell, bg_images)
    # cv2.imwrite(os.path.join(write_path, 'random_cell_csv.tif'), image_panel)

# def get_random_cell_object_csv(var_dict, path, csv_pointer, morph_string, blue_string, green_string):
#     '''
#     Get random cell object and relevant image pointer.
#     '''

#     rows, cols = var_dict.shape
#     cell_row = random.randint(0,rows-1)
#     cell_details = var_dict.ix[cell_row]

#     # obtaining the images - (blue_img, green_img)
#     assoc_images = glob.glob(os.path.join(
#         path, '*'+str(cell_details.Timepoint)+'_*'+str(cell_details.Sci_WellID)+'_*.tif'))
#     ol_path = os.path.dirname(csv_pointer)
#     assoc_morph = glob.glob(os.path.join(
#         ol_path, '*'+str(cell_details.Timepoint)+'_*'+str(cell_details.Sci_WellID)+'_*.tif'))
#     morph_img = [img for img in assoc_morph if morph_string in img]
#     blue_img = [img for img in assoc_images if blue_string in img]
#     green_img = [img for img in assoc_images if green_string in img]
#     bg_images = (blue_img[0], green_img[0], morph_img[0])

#     # r = int(cell_details.Radius)+10
#     # x, y = int(cell_details.BlobCentroidX), int(cell_details.BlobCentroidY)
#     # # print 'Rows range', y-r, 'to', y+r, 'for x:', y
#     # # print 'Cols range', x-r, 'to', x+r, 'for y:', x
#     # # print 'The image:', morph_img[0]
#     # cnt_img = cv2.imread(morph_img[0], 0)#[y-r:y+r, x-r:x+r]
#     # contours = find_cells(cnt_img)
#     # print 'Number of items in contours:', len(contours)
#     # len_cnts = [len(cnt) for cnt in contours]
#     # main_cnt = contours[len_cnts.index(max(len_cnts))]
#     # for cnt in contours:
#     #     # print 'Length of this contour', len(cnt)
#     #     # main_cnt = [0]
#     #     (cnt_x, cnt_y), (MA, ma), angle = cv2.fitEllipse(cnt)
#     #     # print 'Comparing X', int(cnt_x), x
#     #     # print 'Comparing Y', int(cnt_y), y
#     #     e = 2
#     #     if int(cnt_x) < x+e and int(cnt_x) > x-e and int(cnt_y) < y+e and int(cnt_y) > y-e:
#     #         main_cnt = cnt
#     #         # print 'Comparing X', int(cnt_x), x
#     #         # print 'Comparing Y', int(cnt_y), y
#     #         print 'Found a match'
#     #         break


#     # if len(contours)>1:
#     #     len_cnts = [len(cnt) for cnt in contours]
#     #     main_cnt = contours[len_cnts.index(max(len_cnts))]
#     # else:
#     #     main_cnt = contours[0]

#     # cell = cell_details.ObjectLabelsFound, Cell(main_cnt), cell_details.is_live
#     # cell = cell_details

#     return cell_details, bg_images
