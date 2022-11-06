'''
File takes dictionary output from tracking of cell somas (masks).
Generates individual cell images from given paths to
raw data and traced neurites.
'''

from __future__ import absolute_import, division, print_function
import pickle
import os
import glob
import pprint
import numpy as np
import cv2
import matplotlib.pyplot as plt

IMG_DIM = 128
NUM_EDGE_CELLS = 0
NUM_EMPTY_CELLS = 0
TOTAL_CELLS = 0


def make_filelist(path, identifier, verbose=False):
    '''
    Takes a directory and a string identifier.
    Returns a list of files.
    '''
    filelist = sorted(glob.glob(os.path.join(path, '*' + identifier + '*')))

    if verbose:
        print("Selection filter:")
        print(os.path.join(path, '*' + identifier + '*'))
        pprint.pprint([os.path.basename(fel) for fel in filelist])

    return filelist


def make_filelist_wells(path, identifier, verbose=False):
    '''
    Takes a directory and a string identifier.
    Returns a list of files within the input path
    and all the files one subdirectory inside the initial path.
    '''
    folders = [x[0] for x in os.walk(path)]
    filelist = []
    for folder in folders:
        if verbose:
            print('Checking for files in:', os.path.basename(folder))
        #if len(os.path.basename(folder))>0 and len(os.path.basename(folder))<4:
        well_files = sorted(
            glob.glob(os.path.join(folder, '*' + identifier +
                                   '*')))  #, key=numericalSort)
        if verbose:
            print('Found the following files:')
            pprint.pprint([os.path.basename(wf) for wf in well_files])
        filelist.append(well_files)
        ffilelist = [item for sublist in filelist for item in sublist]
    if verbose:
        print("Selection filter:")
        print(os.path.join(path, '*' + identifier + '*'))
        pprint.pprint([os.path.basename(fel) for fel in ffilelist])
        pprint.pprint([fel for fel in ffilelist])
    return ffilelist


class Cell(object):
    '''
    A class that makes cells from contours.
    '''
    def __init__(self, cnt):
        self.cnt = cnt

    def __repr__(self):
        return "Cell instance (%s center, %s area)" % (str(
            self.get_circle()[0]), str(cv2.contourArea(self.cnt)))

    # ----Contours-------------------------------
    def get_circle(self):
        '''Returns centroid of contour.'''
        center, radius = cv2.minEnclosingCircle(self.cnt)
        center, (_, _), _ = cv2.fitEllipse(self.cnt)
        return center, radius

    def get_cell_square(self, img, dim=128, show=False):
        '''Returns either a dim x dim square centered at cell centroid
        or a patch of zeros dim x dim if a cell intersects an edge.'''
        rows, cols = img.shape
        (x, y), _ = self.get_circle()
        x = int(x)
        y = int(y)
        if y + (dim // 2) > rows or y - (dim // 2) < 0 or x + (
                dim // 2) > cols or x - (dim // 2) < 0:
            print('Exluding cell:', self)
            cnt_patch = 120 + np.zeros((dim, dim), np.uint16)
        else:
            cnt_patch = img[y - (dim // 2):y + (dim // 2),
                            x - (dim // 2):x + (dim // 2)]
        if show:
            plt.subplot(121), plt.imshow(cnt_patch,
                                         'gray'), plt.title('cnt_patch')
            plt.subplot(122), plt.imshow(
                cnt_patch.flatten().reshape((dim // 2), (dim // 2)),
                'gray'), plt.title('re_shaped')
            plt.show()
        return cnt_patch.flatten()

    def is_edge_cell(self, cell_pixels):
        '''Sets edge flag to True when cells are at edges and do not
        return complete set of pixels within image confines.'''
        edge = sum(cell_pixels == 120) == cell_pixels.size
        # edge = np.all(cell_pixels == 120)
        # if edge:
        if edge:
            print('~~~~~~The cell is at an edge:', edge,
                  np.all(cell_pixels == 120), sum(cell_pixels == 120),
                  cell_pixels.size)
        return edge

    def is_empty_trace(self, cell_pixels):
        '''Sets empty flag to True if the cell was not annotated
        in the corresponding traced image.'''
        empty = sum(sum(cell_pixels == 0)) == cell_pixels.size
        if empty:
            print('~!~~!~The cell patch/trace is empty:', empty,
                  sum(sum(cell_pixels == 0)), cell_pixels.size)
        return empty


def get_file_groups(image_filelist, token_position_list, token_annot_list):
    '''Take filelist and lists of annotations and positions and returns
    dict with keys of each combination of relevant tokens and values of
    image lists corresponding to each.'''

    output = {}
    for image_filename in image_filelist:
        token_key = get_relevant_token_keys(image_filename,
                                            token_position_list,
                                            token_annot_list)
        if token_key not in output:
            output[token_key] = []
        output[token_key].append(image_filename)
    return output


def get_relevant_token_keys(image_filename, token_position_list,
                            token_annot_list):
    '''Take filename and relevant token position and annotation
    list and generate a key.
    Time token is 2 in all naming conventions.
    Well token is 4 in all naming conventions.

    @Usage
    Example for grouping all channels for montaged images
    of a particular well and time point:
    image_filename = PID20180706_070418PINK1ParkinSurvival1_T5_60-0_A1_0_RFP-DFTrCy5_MN.tif\
    token_position_list = [2,4]
    token_annot_list = ["",""] --> name_list included for cases
    where token value field is ambiguous and needs to be prepended with annotation
    (ex. montage panel and hours).'''

    image_fn_tokens = os.path.basename(image_filename).split('_')
    token_keys = []
    for token_pos, token_name in zip(token_position_list, token_annot_list):
        token_keys.append(token_name + image_fn_tokens[token_pos])
    # return ','.join(sorted(token_keys))
    return ','.join(token_keys)


def make_cell_files_from_keylist(tracked_dict_file,
                                 path_to_raw_images,
                                 verbose=True):
    '''Write cell patches from files.'''
    var_dict = pickle.load(open(tracked_dict_file, 'rb'))
    well_filelist = make_filelist_wells(path_to_raw_images, '.tif')
    grouped_files_dict = get_file_groups(well_filelist, [2, 4, 6],
                                         ['', '', ''])
    # To collect all cells need all wells and time points and iterators
    # Single channel here, ultimately should read in each channel file
    for token_key_group in grouped_files_dict.keys():
        if verbose:
            print('token_key_group', token_key_group)
        time, well, _ = token_key_group.split(',')
        cells = var_dict['TrackedCells'][well][time]
        image_file = grouped_files_dict[token_key_group]
        if verbose:
            print('List of files collected:', image_file)
        # assert len(image_file)==1, 'More than one image specified by selector.'
        img = cv2.imread(image_file[0], -1)
        get_cell_pixels_write_cell_files(img, image_file[0], cells)


def make_cell_files_from_raw_traced(soma_tracked_dict,
                                    list_input_paths,
                                    verbose=True):
    '''Use this function to join lists of files
    in directory and group them by relevant key.'''
    var_dict = pickle.load(open(soma_tracked_dict, 'rb'))
    image_filelist = []
    for path in list_input_paths:
        image_filelist += make_filelist_wells(path, '.tif')

    grouped_files_dict = get_file_groups(
        image_filelist=image_filelist,
        token_position_list=[0, 1, 2, 3, 4, 5, 6],
        token_annot_list=['', '', '', 'H', '', 'P', ''])

    for token_key_group in grouped_files_dict.keys():
        if verbose:
            print('Token_key_group', token_key_group)
        _, _, time, _, well, _, _ = token_key_group.split(',')
        cells = var_dict['TrackedCells'][well][time]
        key_grouped_image_files = grouped_files_dict[token_key_group]
        if verbose:
            print('List of files collected:',
                  len(key_grouped_image_files) == 2,
                  len(key_grouped_image_files), key_grouped_image_files)
        if len(key_grouped_image_files) == 2:
            img_name_raw = key_grouped_image_files[0]
            img_name_traced = key_grouped_image_files[1]
            img_raw = cv2.imread(img_name_raw, -1)
            img_traced = cv2.imread(img_name_traced, -1)
            # check_for_empty_trace(cell_pixels)
            get_cell_pixels_write_cell_files(img_raw, img_name_raw, cells)
            get_cell_pixels_write_cell_files(img_traced, img_name_traced,
                                             cells)


def make_cell_csv_img_names(image_filename, cell_inst, verbose=False):
    '''Take image name and output paths,
    generate a new filename for csv and png
    to include appended cell-identifier token.'''

    cell_output_path = os.path.join(os.path.dirname(image_filename), 'cells')
    if not os.path.exists(cell_output_path):
        os.makedirs(cell_output_path)
    full_image_name = os.path.basename(image_filename)
    img_name = os.path.splitext(full_image_name)[0]
    cell_img_name = os.path.join(
        cell_output_path,
        '_'.join([img_name, make_cell_identifier(cell_inst) + '.png']))
    cell_csv_name = os.path.join(
        cell_output_path,
        '_'.join([img_name, make_cell_identifier(cell_inst) + '.csv']))
    if verbose:
        print('Input image : ', image_filename)
        print('Output csv  : ', cell_csv_name)
        print('Output cell : ', cell_img_name)
    return cell_img_name, cell_csv_name


def check_for_edge_cell(cell_pixels):
    '''Cell class (tracking_extra) checks for edge cells.
     If a cell is at an edge, all pixels are set to 120+0.
     This function evaluates if all pixels are 120.
     Returns True for edge cells, False for non-edge cells.'''

    edge_cell = sum(cell_pixels == 120) == cell_pixels.size

    if edge_cell:
        print('This is an edge cell:', edge_cell, sum(cell_pixels == 120),
              cell_pixels.size)
    return edge_cell


def check_for_empty_cell(cell_pixels):
    '''Cell class (tracking_extra) checks for edge cells.
     If a cell is at an edge, all pixels are set to 0.
     This function evaluates if all pixels are 0.
     Returns True for empty cell patches, False for non-empty.'''

    empty_cell = sum(cell_pixels == 0) == cell_pixels.size
    if empty_cell:
        print('This is an empty cell:', empty_cell, sum(cell_pixels == 120),
              cell_pixels.size)
    return empty_cell


def check_for_empty_trace(tr_tif_img, cells, cell_pixels):
    '''Cell class (tracking_extra) checks for empty images.
     If a cell trace is empty, all pixels are 0.
     This function evaluates if all pixels are 0.
     Returns True for non-traced cells, False for traced cells.
     Trace of neurite is not alreay in dict.'''

    for cell_inst in cells:
        cell_pixels = cell_inst[1].get_cell_square(tr_tif_img,
                                                   dim=IMG_DIM,
                                                   show=False)
    empty_cell = sum(cell_pixels == 0) == cell_pixels.size
    global NUM_EMPTY_CELLS
    if empty_cell:
        NUM_EMPTY_CELLS += 1
    return empty_cell


def get_cell_pixels_write_cell_files(orig_tif_img,
                                     image_filename,
                                     cells,
                                     out_type='img'):
    ''''Write out cell pixels to image or csv if cell is not at edge
    and trace is not empty. The raw images will not have empty images,
    but the traces will. This mismatch in output lengths should be
    addressed with the file matching function in ds_neurites_ae.py'''

    global TOTAL_CELLS
    global NUM_EDGE_CELLS
    global NUM_EMPTY_CELLS

    # Get pixels and name string for each cell
    for cell_inst in cells:

        TOTAL_CELLS += 1
        cell_pixels = cell_inst[1].get_cell_square(orig_tif_img,
                                                   dim=IMG_DIM,
                                                   show=False)
        cell_pixels_str = [
            str(cell_pixels_item) for cell_pixels_item in cell_pixels
        ]
        cell_img_name, cell_csv_name = make_cell_csv_img_names(
            image_filename, cell_inst)

        # Evaluate edginess and emptiness
        # is_edge = cell_inst[1].is_edge_cell(cell_pixels)
        is_edge = check_for_edge_cell(cell_pixels)
        if is_edge:
            NUM_EDGE_CELLS += 1
        # is_empty = cell_inst[1].is_empty_trace(cell_pixels)
        is_empty = check_for_empty_cell(cell_pixels)
        if is_empty:
            NUM_EMPTY_CELLS += 1

        # Discard edge cells and empty cells
        if not is_edge and not is_empty:
            if out_type == 'csv':
                cell_txt = open(cell_csv_name, 'w')
                cell_txt.write(','.join(cell_pixels_str))
                cell_txt.close()
            else:
                # half_max_pixels = (cell_pixels - HALF_MAX_IMG)/HALF_MAX_IMG
                # print('means', cell_pixels.mean(), half_max_pixels.mean() )
                cell_pixels = cell_pixels.reshape((IMG_DIM, IMG_DIM))
                cv2.imwrite(cell_img_name, cell_pixels)
        else:
            print('Image name:', os.path.basename(image_filename))
            print('Cell excluded (edge/empty):', is_edge, '/', is_empty,
                  cell_inst)


def make_cell_identifier(cell_inst):
    '''Generate string cellID-x-y string that identifies cell. '''
    cell_identifier = '-'.join([
        'cell' + str(cell_inst[0]),
        'x' + str(round(cell_inst[1].get_circle()[0][0], 2)),
        'y' + str(round(cell_inst[1].get_circle()[0][1], 2)),
        'area' + str(round(cv2.contourArea(cell_inst[1].cnt)))
    ])
    return cell_identifier


if __name__ == '__main__':
    VERBOSE = False

    # Inputs
    # infile = '/finkbeiner/imaging/smb-robodata/Minnie/ImagesForMariya/MSN/tracking_cellmasks_curated.data'
    infile = '/finkbeiner/imaging/smb-robodata/Minnie/ImagesForMariya/MSN/tracking_cellmasks.data'
    raw_img_path = '/finkbeiner/imaging/smb-robodata/Minnie/ImagesForMariya/MSN/MontagedImages'
    # traced_img_path = '/finkbeiner/imaging/smb-robodata/Minnie/ImagesForMariya/MSN/Output'
    traced_img_path = '/finkbeiner/imaging/smb-robodata/Minnie/ImagesForMariya/MSN/CellMasks'

    # infile = '/data/home/mariyabarch/Dropbox/random_files/io_tfdata/Galaxy20-[Track_All_T9_Cells].data'
    # raw_img_path = '/finkbeiner/imaging/smb-robodata/mbarch/neurites/MontagedImages/T9/'
    # traced_img_path = '/finkbeiner/imaging/smb-robodata/mbarch/neurites/CellMasksNeurites/'
    # raw_image_filelist = make_filelist(raw_img_path, '.png')
    # traced_image_filelist = make_filelist(traced_img_path, '.png')

    # Small image generation
    # make_cell_files_from_keylist(infile, '/finkbeiner/imaging/smb-robodata/mbarch/neurites/MontagedImages/')
    make_cell_files_from_raw_traced(
        soma_tracked_dict=infile,
        list_input_paths=[raw_img_path, traced_img_path])
    print(
        'Number of edge cells excluded/empty/total cells:',
        NUM_EDGE_CELLS,
        '/',
        NUM_EMPTY_CELLS,
        '/',
        TOTAL_CELLS,
    )
