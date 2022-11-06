'''
Takes images at any stage of processing, finds cells,
and extracts their parameters to a table.
'''

import cv2, utils, sys, os, collections
import numpy as np
import pickle, datetime, pprint, shutil
from extract_cell_info import extract_cell_info
from select_analysis_module import get_var_dict
from tracking import Cell


# ----Segmentation handling----------------------
def find_cells(img):
    '''
    Finds contours in given image.
    '''
    if img.max() < 50:
        factor = int(50./img.max())
        img = img*factor

    ret, mask = cv2.threshold(img, int(img.max()/10.), img.max(), cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours

def filter_contours(contours, small, large, ecn, verbose=True):
    '''
    Currently filters contours based on size and eccentricity.
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

def show_kept_cells(img_pointer, contours, contours_kept, write_path):
    '''
    Draw kept cells onto image.
    '''

    img = 50*cv2.imread(img_pointer, 0)

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
    cv2.destroyAllWindows()

    cv2.imwrite(img_name, img)

# ----Organizing the cell objects in dictionary--
def time_id_cell_dict(time_id, time_dictionary, kept_contours):
    '''
    Finds contours and filters for cells.
    Then adds cells to time_dictionary.
    '''

    time_dictionary[time_id] = []
    for cnt in kept_contours:
        time_dictionary[time_id].append(
            ['n', Cell(cnt)])

    return time_dictionary

def populate_cell_ind(time_dictionary, time_id, number_of_cells):
    '''
    Number the cells for each time point.
    Consider continuing the numbers instead of restarting.
    '''
    assert len(time_dictionary.keys()) > 0, 'No time point data given.'

    for ind, cell_record in enumerate(time_dictionary[time_id]):
        cell_record[0] = ind + sum(number_of_cells[0:-1])

    return time_dictionary

def sort_cell_info_by_index(time_dictionary, time_id):
    '''
    Takes arrays for each timepoint (key) and sorts on index of tuple.
    '''

    cell_inds = [cell[0] for cell in time_dictionary[time_id]]
    cell_objs = [cell[1] for cell in time_dictionary[time_id]]
    inds_and_objs = zip(cell_inds, cell_objs)

    sorted_cell_objs = sorted(inds_and_objs, key=lambda pair: pair[0])
    time_dictionary[time_id] = sorted_cell_objs

    return time_dictionary
# -----------------------------------------------

def segmentation_and_ordering(var_dict, path_to_images, write_qc_path):
    '''
    Carries out segmentation an ordering based on filtered contours.
    '''
    # Initialize parameters
    resolution = 0
    small = 10
    large = 4500
    ecc = .01

    var_dict['TrackedCells'] = {}
    for well in var_dict['Wells']:

        images_list = utils.make_filelist(
            path_to_images, well+'_*'+var_dict['MorphologyChannel'])
        time_dictionary = collections.OrderedDict()
        time_list = var_dict["TimePoints"]

        number_of_cells = [0]
        for img_pointer, time_id in zip(images_list, time_list):
            img = cv2.imread(img_pointer, resolution)
            contours = find_cells(img)
            contours_kept = filter_contours(contours, small, large, ecc)
            show_kept_cells(
                img_pointer, contours, contours_kept, write_qc_path)

            time_dictionary = time_id_cell_dict(
                time_id, time_dictionary, contours_kept)
            number_of_cells.append(len(time_dictionary[time_id]))

            time_dictionary= populate_cell_ind(time_dictionary, time_id, number_of_cells)

            var_dict['TrackedCells'][well] = time_dictionary

        # print 'Number of cells:', number_of_cells
        # print 'Whole dictionary:'
        # pprint.pprint(time_dictionary.items())

        print '------------------------'
        print 'Initial number of cells:', number_of_cells[1]
        print 'Final number of cells:', sum(number_of_cells)
        print '------------------------'


def main():
    '''
    Main point of entry.
    '''
    # Initialize parameters
    path_to_images = sys.argv[1]
    write_qc_path = sys.argv[2]
    analysis_files = utils.make_filelist(path_to_images, 'PID')
    morph_channel = sys.argv[3]
    var_dict = get_var_dict(analysis_files, morph_channel)
    var_dict["ImageDataPath"] = sys.argv[1]
    var_dict["OutputWritePath"] = sys.argv[2]
    pprint.pprint(var_dict.keys())

    # Run segmentation to extraction steps
    start_time = datetime.datetime.utcnow()
    segmentation_and_ordering(var_dict, path_to_images, write_qc_path)
    extract_cell_info(var_dict, path_to_images, write_qc_path)
    end_time = datetime.datetime.utcnow()
    print 'Segmentation->Extraction run time:', end_time-start_time

    # Output for user and save dict
    print 'Images in', path_to_images, 'were stacked.'
    print 'Output was written to:', write_qc_path
    outfile = sys.argv[4]
    orig_outfile = os.path.join(write_qc_path, 'cell_data.csv')
    new_outfile = os.path.join(write_qc_path, 'cell_data2.csv')
    shutil.copyfile(orig_outfile, new_outfile)
    outfile = shutil.move(new_outfile, outfile)

if __name__ == '__main__':
    main()
