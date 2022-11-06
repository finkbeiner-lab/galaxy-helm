'''
From csv or tracked dictionary, show 10 (2x5) cells with specified label in specified column.
From csv or tracked dictionary, show a specific cell.
Takes spreadsheet or dictionary, cell ID as follows: well-time-cell, 
name of column or key to print or select, value for 'select' column.
Will figure out channels if not specified. If more than three, break and request three. 

@Requirements
csv file must contain labels and at least one column named Vitality or Phenotype that has True values.

@Alert
If PP csv is used as the input, the times in the files don't match the spreadsheet.
Setting csv_source flag to PP will look for files with T string T-1.
Otherwise, the time point entered for the specific cell should match the csv.

@Usage
input_path - Path to images (typically AlignedImages)
write_path - Directory where cell image or panel will be written.
channel_token - The index where channels should be read
use_dict - Specifies if source is tracked dictionary (True) or labeled csv file (False)
source_file_pointer - the labeled csv file or the tracked dictionary
evaluation_metric - CSV column you would like to visualize/check: Vitality or Phenotype
--eval_metric_bool -  Default is to show you True examples, change to False to see dead or not_neuron
--show_specific_cell - Set to True to show one cell defined by well, time integer, objectid integer (A1,0,16)
--channel_list - String with channels separated by commas
--csv_source - PP for pipeline pilot or G for galaxy (addressing the initial time point discrepency)
--specific_cell_list - w1,t1,id1-w2,t2,id2 to specify cells 

Example:
python /media/mbarch/imaging-code/galaxy/tools/validation/show_cells.py /media/robodata/Mariya/BiogenScreenUnstack/BioPrePlate13B/AlignedImages/ /media/robodata/Mariya/ShowCells/ 6 False /media/robodata/Mariya/BiogenScreenUnstack/BioPrePlate13B/BioPrePlate13B_HCA-R-L.csv Vitality --show_specific_cell True --csv_source PP --specific_cell_list E2,5,31-H2,0,60

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

    def get_center(self):
        '''Returns centroid of contour.'''
        (x,y), radius = cv2.minEnclosingCircle(self.cnt)
        center = (int(x), int(y))
        return center

# ----Decipher the channels and files------------
def find_channels(all_files, channel_token):
    '''
    Find the representative channels for the list of files. 
    Return list of channels represented.
    '''
    all_channels = []
    for one_file in all_files:
        channel = os.path.basename(one_file).split('_')[channel_token].split('.')[0]
        if 'Brightfield' not in channel:
            all_channels.append(channel)
    all_channels = list(set(all_channels))
    return all_channels

def find_channels_from_user(all_channels, user_channels):
    '''
    Take a string of channels from user.
    Return list of channels represented in files.
    '''
    list_of_user_channels = user_channels.split(',')
    real_user_channels = []
    for user_channel in list_of_user_channels:
        for real_channel in all_channels:
            if user_channel in real_channel:
                real_user_channels.append(real_channel)
    return real_user_channels

def get_cell_files(all_files, user_well, user_time):
    '''
    Filters file list.
    Returns files with specified well and time.
    '''
    user_well = user_well+'_'
    user_time = 'T'+str(user_time)+'_'
    well_time_files = [one_file for one_file in all_files 
        if user_well in one_file and user_time in one_file]
    print 'The selected files:'
    pprint.pprint(well_time_files)
    return well_time_files

# ----Specific cell support----------------------
def get_specific_cell_from_csv(cell_data_csv, well, time, cellID, source):
    "Get specific cell object from csv."
    cell_details = cell_data_csv[
        (cell_data_csv['Sci_WellID']==str(well)) & 
        (cell_data_csv['Timepoint']==int(time)) & 
        (cell_data_csv['ObjectLabelsFound']==int(cellID))]
    if source == 'PP':
        print cell_details[[1,8,11,46,47]]
    else:
        print cell_details[[1,29,32,33,34]]
        # print cell_details[[1,16,19,168,169]]
    assert len(cell_details.index)==1, 'Pick a different cell.'+str(len(cell_details.index))
    return cell_details

def get_specific_cell_from_dict(cell_data_dict, well, time, cellID):
    "Get specific cell object from dict."
    time = 'T'+str(time)
    cellID = int(cellID)-1
    cell_obj = cell_data_dict['TrackedCells'][well][time][cellID]
    # also assert to pick a different cell
    return cell_obj

# ----Random cell support------------------------
def get_random_cell_from_csv(cell_data_csv, source):
    "Get random cell object from csv."
    # Obtaining the cell: row in csv
    rows, cols = cell_data_csv.shape
    rand_int = random.randint(0,rows-1)
    cell_row = cell_data_csv.index.tolist()[rand_int]
    print 'cell_row', cell_row
    cell_details = cell_data_csv.ix[cell_row]
    print 'Row index:', cell_row
    if source == 'PP':
        print cell_details[[1,8,11,46,47]]
    else:
        print cell_details[[1,16,19,168,169]]
    return cell_details

def get_random_cell_from_dict(cell_data_dict):
    "Get random cell object from dict."
    # Obtaining the cell: (cell_id, cell_obj)
    well_ind = random.randint(0,len(cell_data_dict['Wells'])-1)
    well = cell_data_dict['Wells'][well_ind]
    time_ind = random.randint(1,len(cell_data_dict['TimePoints'])-1)
    time = cell_data_dict['TimePoints'][time_ind]
    cell_ind = random.randint(0,len(cell_data_dict['TrackedCells'][well][time])-1)
    cell_obj = cell_data_dict['TrackedCells'][well][time][cell_ind]
    return cell_obj, well, time

# ----Get cell specific cropping info------------
def get_cell_center_index(cell_details, use_dict):
    '''Take either pandas cell object or dictionary object.
    Return cell index and center coordinates.'''
    # if type(cell_details) == tuple:
    if use_dict:
        cell_ind, cell_obj = cell_details
        (cellx, celly) = cell_obj.get_center()
    else:
        cell_ind = cell_details.ObjectLabelsFound
        (cellx, celly) = int(cell_details.BlobCentroidX), int(cell_details.BlobCentroidY)
    return cell_ind, cellx, celly

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

# ----One of these will be executed--------------
def make_small_cnt_img(xy_coords, rows, cols, cell_obj):
    '''
    Generate small image with cell contour drawn for cell of interest.
    '''
    x_left, x_right, y_top, y_bottom = xy_coords
    orig_img = np.zeros((rows, cols), np.uint8)
    cv2.drawContours(orig_img, [cell_obj[1].cnt], 0, 255, 1)
    mask_img = np.zeros((250,250,3), np.uint8)
    mask_img[0:y_bottom-y_top,0:x_right-x_left, 2] = orig_img[y_top:y_bottom, x_left:x_right]
    mask_img[0:y_bottom-y_top,0:x_right-x_left, 1] = orig_img[y_top:y_bottom, x_left:x_right]
    mask_img[0:y_bottom-y_top,0:x_right-x_left, 0] = orig_img[y_top:y_bottom, x_left:x_right]
    return mask_img

def make_small_circle_img(xy_coords, rows, cols, cellx, celly):
    '''
    Generate small overlay dot centered on the cell of interest.
    '''
    x_left, x_right, y_top, y_bottom = xy_coords
    orig_img = np.zeros((rows, cols), np.uint8)
    cv2.circle(orig_img, (cellx,celly), 2, 200, -1)
    small_img = np.zeros((250, 250, 3), np.uint8)
    small_img[0:y_bottom-y_top,0:x_right-x_left, 0] = 0#orig_img[y_top:y_bottom, x_left:x_right]
    small_img[0:y_bottom-y_top,0:x_right-x_left, 1] = orig_img[y_top:y_bottom, x_left:x_right]
    small_img[0:y_bottom-y_top,0:x_right-x_left, 2] = 0#orig_img[y_top:y_bottom, x_left:x_right]
    # small_img = small_img.astype(np.uint8)
    return small_img

# --->...and layered with this img---------------
def make_small_img(img_pointer, xy_coords, max_val=2**10):
    '''
    Generate small image centered around cell of interest.
    '''
    x_left, x_right, y_top, y_bottom = xy_coords
    orig_img = cv2.imread(img_pointer, -1)
    orig_img = cv2.normalize(orig_img, alpha=0, beta=max_val, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    orig_img = np.array(orig_img, dtype=np.uint8)
    small_img = np.zeros((250, 250), np.uint8)
    small_img[0:y_bottom-y_top,0:x_right-x_left] = orig_img[y_top:y_bottom, x_left:x_right]
    return small_img

# ----Other--------------------------------------
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

def print_on_image(img_print, cell, cell_ind, eval_metric, img_pointer, use_dict, specific_cell):
    '''
    Prints cell and image details onto given image.
    '''
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (0, 200, 0)
    cv2.putText(img_print, '_'.join(os.path.basename(img_pointer).split('_')[0:2]), 
        (10, 10), font, .35, color, 1, cv2.CV_AA)
    cv2.putText(img_print, '_'.join(os.path.basename(img_pointer).split('_')[2:-1]), 
        (10, 25), font, .35, color, 1, cv2.CV_AA)
    if not use_dict:
        
        # TODO: Need to straighten this out... why objects are returned differently.
        if specific_cell:
            cell_ind = str(cell_ind.tolist()[0])
            # cell_vit = str(cell.Vitality.tolist()[0])
            # cell_ph = str(cell.Phenotype.tolist()[0])
        else:
            cell_ind = str(cell_ind)
            cell_vit = str(cell.Vitality)
            cell_ph = str(cell.Phenotype)
            
        cv2.putText(img_print, 'Cell: '+cell_ind, (10, 40),
            font, .35, color, 1, cv2.CV_AA)
        if eval_metric=='Vitality':
            cv2.putText(img_print, eval_metric, (10, 250-30),
                font, .35, color, 1, cv2.CV_AA)
            cv2.putText(img_print, 'Live: '+cell_vit, (10, 250-15),
                font, .35, color, 1, cv2.CV_AA)
        if eval_metric=='Phenotype':
            cv2.putText(img_print, eval_metric, (10, 250-30),
                font, .35, color, 1, cv2.CV_AA)
            cv2.putText(img_print, 'Neuron: '+cell_ph, (10, 250-15),
                font, .35, color, 1, cv2.CV_AA) 
        if eval_metric=='IDChanges':
            cv2.putText(img_print, eval_metric, (10, 250-30),
                font, .35, color, 1, cv2.CV_AA)
            cv2.putText(img_print, 'CuratedID: '+str(cell.ObjectLabelsFound.tolist()[0]), (10, 250-15),
                font, .35, color, 1, cv2.CV_AA) 
    else:
        cv2.putText(img_print, 'Cell: '+str(cell_ind), (10, 40),
            font, .35, color, 1, cv2.CV_AA)   
    return img_print

if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Show cells or cell.")
    parser.add_argument("input_path", 
        help="Folder path to input data.")
    parser.add_argument("write_path", 
        help="Write path from root.")
    parser.add_argument("channel_token", 
        help="Positional token where channel is encoded.")
    parser.add_argument("use_dict",
        help="True if using dictionary as input. False if using csv as input.")
    parser.add_argument("all_cells_pointer", 
        help="Path to dictionary or csv containing all cell data.")
    parser.add_argument("metric", 
        help="Choose value to print on panel: 'Vitality' or 'Phenotype'.")
    parser.add_argument("--show_specific_cell", "-cell",
        default='False',
        help="True if checking one cell. False if collecting multiple random cells.")
    parser.add_argument("--channel_list", "-chl",
        help="Maximum three channels, without spaces, separated by commas.")
    parser.add_argument("--eval_metric_bool", "-emb",
        default='True',
        help="By default, will choose random positive examples to show. Set to False if you want to see positive examples.")
    parser.add_argument("--csv_source",
        default='G',
        help="By default, will use Galaxy time scheme starting from 0. Set to PP for off by one correction.")
    parser.add_argument("--specific_cell_list",
        default='',
        help="Cells should be specified: 'w1,t1,id1-w2,t2,id2-w3,t3,id3-w4,t4,id4'. By default, no specific cells will be iterated.")
    args = parser.parse_args()
    
    # ----Initialize parameters------------------
    read_path = args.input_path
    write_path = args.write_path
    channel_token = int(args.channel_token)
    specific_cell = args.show_specific_cell=='True'
    use_dict = args.use_dict=='True'
    user_channels = args.channel_list
    source_file_pointer = args.all_cells_pointer
    eval_metric = args.metric
    eval_metric_true = args.eval_metric_bool=='True'
    csv_source = args.csv_source
    list_of_cells = args.specific_cell_list


    # ----Handle cell list-----------------------
    list_of_cells = list_of_cells.split('-')
    list_of_cells = [tuple(well_time_id.split(',')) for well_time_id in list_of_cells]

    # ----Get channels and files-----------------
    all_files = sorted(glob.glob(os.path.join(read_path, '*.tif')))
    real_channels = find_channels(all_files, channel_token)
    if user_channels != None:
        real_user_channels = find_channels_from_user(real_channels, user_channels)

    # Specific cell handling
    if specific_cell:
        for (user_well,user_time,user_cellid) in list_of_cells:

            if csv_source=='PP':
                # Get files and coordinates
                bimg_pntr, rimg_pntr = get_cell_files(all_files, user_well, str(int(user_time)-1))
            else: 
                bimg_pntr, rimg_pntr = get_cell_files(all_files, user_well, user_time)

            if use_dict:
                cell_data_dict = pickle.load(open(source_file_pointer, 'rb'))
                print 'Imported dictionary ...'
                cell_obj = get_specific_cell_from_dict(
                    cell_data_dict, str(user_well), int(user_time), int(user_cellid))

            else:
                cell_data_csv = pd.read_csv(source_file_pointer)
                print 'Imported csv with (rows,columns)', cell_data_csv.shape
                 # Keep data based on eval_metric
                cell_data_csv = cell_data_csv[(cell_data_csv['MeasurementTag']=='Confocal-GFP16')]
                print 'Filtered by:', eval_metric_true, cell_data_csv.shape
                print '******', user_time, '******'
                cell_obj = get_specific_cell_from_csv(
                    cell_data_csv, user_well, user_time, user_cellid, csv_source)

            cell_ind, cellx, celly = get_cell_center_index(cell_obj, use_dict)
            xy_coords, rows, cols = find_xy_ends(bimg_pntr, cellx, celly)

            # Make the image panels
            blue_img = make_small_img(bimg_pntr, xy_coords, max_val=255)
            red_img = make_small_img(rimg_pntr, xy_coords, max_val=2**12)

            if use_dict:
                ref_img = make_small_cnt_img(
                    xy_coords, rows, cols, cell_obj)
            else:
                ref_img = make_small_circle_img(
                    xy_coords, rows, cols, cellx, celly)
            
            text_img = print_on_image(
                ref_img, cell_obj, cell_ind, eval_metric, bimg_pntr, use_dict, True)

            # Returning BGR image with text
            image_panel = cv2.merge((blue_img, np.zeros((250, 250), np.uint8), red_img))
            image_panel = cv2.add(image_panel, text_img)

            write_img_pointer = os.path.join(
                    write_path, 'specific_cell_T'+user_time+'_'+user_well+'_N'+user_cellid+'.png')
            print write_img_pointer
            cv2.imwrite(write_img_pointer, image_panel)


    # Random cells handling
    else:
        # get a random cell and ultimately...
        # make a panel of representative cells
        cell_data_csv = pd.read_csv(source_file_pointer)
        print 'Imported csv with (rows,columns)', cell_data_csv.shape
        # Keep data based on eval_metric
        cell_data_csv = cell_data_csv[(cell_data_csv[eval_metric]==eval_metric_true)]
        print 'Filtered by:', eval_metric_true, cell_data_csv.shape
    

        all_img_panels = []
        for i in range(0,5):
            cell_obj = get_random_cell_from_csv(cell_data_csv, csv_source)
            cell_time = str(cell_obj.Timepoint)
            cell_well = str(cell_obj.Sci_WellID)

            bimg_pntr, rimg_pntr = get_cell_files(all_files, cell_well, cell_time)
            cell_ind, cellx, celly = get_cell_center_index(cell_obj, False)
            xy_coords, rows, cols = find_xy_ends(bimg_pntr, cellx, celly)

            # Make the image panels
            blue_img = make_small_img(bimg_pntr, xy_coords)
            red_img = make_small_img(rimg_pntr, xy_coords, max_val=2**12)
            ref_img = make_small_circle_img(xy_coords, rows, cols, cellx, celly)
            text_img = print_on_image(
                ref_img, cell_obj, cell_ind, eval_metric, bimg_pntr, False, False)

            # Returning BGR image with text
            image_panel = cv2.merge((blue_img, np.zeros((250, 250), np.uint8), red_img))
            image_panel = cv2.add(image_panel, text_img)
            all_img_panels.append(image_panel)
            print len(all_img_panels)

        large_img_panel = np.hstack(tuple(all_img_panels))
        write_img_pointer = os.path.join(
                write_path, 'five_random_cells.tif')
        print write_img_pointer
        cv2.imwrite(write_img_pointer, large_img_panel)

