'''
Counts number of neurite-synapse intersections for thresholded neuromuscular junction images.

@Usage
Program takes a path to images and number specifying the channel that labels junctions. 
The images in this path must binary images, thresholded to keep the relevant regions. 
Returns a mask of objects (labeled junctions) that have intersections with objects in neuron image.
Returns a csv file of the number of intersections for each junction object.  
'''

import os, sys, cv2, pprint, argparse
import numpy as np
import collections, datetime, glob

def get_j_and_n_contours(junction_file_pointer, neuron_file_pointer, verbose=True):
    '''
    Take paths for two binary masks and return their contours. 
    '''

    nimg = cv2.imread(neuron_file_pointer, 0)
    jimg = cv2.imread(junction_file_pointer, 0)

    if verbose:
        print 'Jimg', jimg.shape
        print 'Nimg', nimg.shape

    # Collect contours from thresholded junctions image
    contours_n, hierarchy = cv2.findContours(
        nimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Collect contours from thresholded neuron image
    contours_j, hierarchy = cv2.findContours(
        jimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if verbose:
        print 'Found junctions:', len(contours_j), 
        print 'Found neurons:', len(contours_n)

    return contours_j, contours_n

def count_intersections(junction_file_pointer, contours_j, contours_n, info_dict):
    '''
    Take contours for neurons and junctions
    '''

    # Initiate image and dictionary to collect meta data
    img_dimensions = cv2.imread(junction_file_pointer, 0).shape
    mask_counter = np.zeros(img_dimensions, np.uint16)
    index_intersection_holder = collections.OrderedDict()

    font = cv2.FONT_HERSHEY_SIMPLEX
    cnt_ind = 1
    # Test each junction contour against every point in neuron contour
    for contour_id, cntj in enumerate(contours_j):
        for cntn in contours_n:
            
            counter_intersection = 0
            for point_neuron_contour in cntn:
                # If overlap = true: Add to dictionary with identifier
                point_tuple =  point_neuron_contour[0][0], point_neuron_contour[0][1]
                if cv2.pointPolygonTest(cntj, point_tuple, False) == 0:
                    counter_intersection += 1
                else:
                    continue
             # Encode the mask with contour ID
             # Print contour ID and number of intersections onto masks
            if counter_intersection > 0:
                index_intersection_holder[contour_id] = counter_intersection
                # cv2.drawContours(mask_counter, [cntj], 0, contour_id, -1)
                cv2.drawContours(mask_counter, [cntj], 0, 2000, 2)
                if counter_intersection == 2:
                    cv2.drawContours(mask_counter, [cntj], 0, 2000, 3)
                (x,y), radius = cv2.minEnclosingCircle(cntj)
                int_center = int(x), int(y)
                cv2.putText(mask_counter, str(contour_id), 
                    (int(x), int(y)), font, 2, 2000, 2, cv2.CV_AA)
                cv2.putText(mask_counter, str(counter_intersection), 
                    (int(x), int(y)), font, 0.5, 1500, 1, cv2.CV_AA)
            else:
                continue

    info_dict[junction_file_pointer] = index_intersection_holder
    return info_dict, mask_counter

def extract_info(info_dict, write_path):
    '''
    Take dictionary entries and print to spreadsheet.
    '''    
    
    txt_f = open(os.path.join(write_path, 'intersection_data.csv'), 'w')
    headers = ['Filename', 'ObjectID', 'NumberIntersections']
    txt_f.write(','.join(headers))
    txt_f.write('\n')

    for j_filename in info_dict.keys():
        for contour_id, num_intersection in info_dict[j_filename].items():
            values = [os.path.basename(j_filename), str(contour_id), str(num_intersection)]
            txt_f.write(','.join(values))
            txt_f.write('\n')

    txt_f.close()

def make_filelist(path, identifier, verbose=False):
    '''
    Takes a directory and a string identifier.
    Returns a list of files.
    '''

    filelist = sorted(
        glob.glob(os.path.join(path, '*'+identifier+'*')))
    if verbose==True:
        print 'Number of image files:', len(filelist)
        print 'Complete file list:'
        pprint.pprint([os.path.basename(el) for el in filelist])

    return filelist

def get_red_green_pointers(j_channel, red_green_list_pair, verbose=False):
    '''
    Takes channel corresponding to junctions and pair list. 
    Returns pointers to neuron and junction files.
    '''

    jnxn_ch_str = str(j_channel)+'.tif'

    junction_file_pointer = [img_name for img_name in red_green_list_pair if jnxn_ch_str not in img_name]
    neuron_file_pointer = [img_name for img_name in red_green_list_pair if jnxn_ch_str in img_name]
    
    assert len(junction_file_pointer), 'Junction file identifier should be more unique than [-10:-5]'
    assert len(neuron_file_pointer), 'Neuron file identifier should be more unique than [-10:-5]'
    if verbose:
        print neuron_file_pointer, '<- Neuron file'
        print junction_file_pointer, '<- Junction file'

    return junction_file_pointer[0], neuron_file_pointer[0]


def get_list_thresh_file_pairs(image_file_list, verbose=False):
    '''
    Takes a list of files in folder.
    Returns a list of file pairs (one for each channel).
    '''

    unique_parts = set([imfile[-10:-5] for imfile in image_file_list])
    list_of_pairs = []

    for unique_part in unique_parts:
        pair_list = [imfile for imfile in image_file_list if unique_part in imfile]
        list_of_pairs.append(pair_list)
    
    if verbose:
        print 'Unique parts found:', unique_parts
        print 'Paired list:'
        pprint.pprint(list_of_pairs)

    return list_of_pairs

def extract_file_name(filename_path):
    '''
    Parses out file name from long path.
    '''

    img_file_name = os.path.basename(filename_path)
    img_name = os.path.splitext(img_file_name)

    return img_name[0]

if __name__ == '__main__':

    # ----Parse arguments--------------
    parser = argparse.ArgumentParser(
        description="Find number of intersections per junction.")
    parser.add_argument("input_path", 
        help="Folder path to thresholded binary images data.")
    parser.add_argument("color", 
        type=int, help="Junction channel. Ex. 02")
    args = parser.parse_args()

    # ----I/O--------------------------
    j_channel = args.color
    ipath = args.input_path
    wpath = os.path.join(ipath, 'Ouput')
    if not os.path.exists(wpath):
        os.makedirs(wpath)

    # ----Process data-----------------
    start_time = datetime.datetime.utcnow()
    # Get list of green and red thresholded files
    # Result will be as good as the thresholding allows
    image_file_list = make_filelist(ipath, '.tif')
    image_pair_list = get_list_thresh_file_pairs(image_file_list)
    info_dict = collections.OrderedDict()

    # Find intersections for each file pair
    for file_pair in image_pair_list:
        junction_file_pointer, neuron_file_pointer = get_red_green_pointers(
            j_channel, file_pair)
        contours_j, contours_n = get_j_and_n_contours(
            junction_file_pointer, neuron_file_pointer)
        info_dict[junction_file_pointer] = {}
        info_dict, mask_counter = count_intersections(
            junction_file_pointer, contours_j, contours_n, info_dict)
        new_fname = extract_file_name(junction_file_pointer)+'_int.tif'
        print new_fname
        cv2.imwrite(os.path.join(wpath, new_fname), mask_counter)

    extract_info(info_dict, wpath)
    
    end_time = datetime.datetime.utcnow()
    print 'Run time:', end_time-start_time
