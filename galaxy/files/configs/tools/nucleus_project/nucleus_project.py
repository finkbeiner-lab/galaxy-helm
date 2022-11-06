import os
import cv2
import numpy as np
import math
from copy import deepcopy
from operator import itemgetter
import csv
import argparse
from PIL import Image
from collections import Counter


# Becareful of the paths, if path contains () or [], they need to escape. So better avoid using path name contains ()

# TEST_FOLDER = "/Users/guangzhili/GladStone/Nucleus_projct/test_sample_imgs/"
# CELLS_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/test_sample_imgs/PID20150904_PGPSTest_T1_8_A7_MONTAGE_RFP-DFTrCy5.tif"
# NUCLEUSES_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/test_sample_imgs/PID20150904_PGPSTest_T1_8_A7_MONTAGE_DAPI.tif"
# CYTOPLASMS_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/test_sample_imgs/PID20150904_PGPSTest_T1_8_A7_MONTAGE_FITC-DFTrCy5.tif"

# CELLS_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/source_images/PGPSTest(7)/Montaged/RFP-DFTrCy5_red"
# NUCLEUSES_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/source_images/PGPSTest(7)/Montaged/DAPI_blue"
# CYTOPLASMS_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/source_images/PGPSTest(7)/Montaged/FITC-DFTrCy5_green"

# CELLS_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/source_images/PGPSTest(7)/Aligned/RFP-DFTrCy5_red"
# NUCLEUSES_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/source_images/PGPSTest(7)/Aligned/DAPI_blue"
# CYTOPLASMS_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/source_images/PGPSTest(7)/Aligned/FITC-DFTrCy5_green"

# CELLS_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/source_images/PGPSTest(8)/Aligned/RFP-DFTrCy5"
# NUCLEUSES_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/source_images/PGPSTest(8)/Aligned/DAPI"
# CYTOPLASMS_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/source_images/PGPSTest(8)/Aligned/FITC-DFTrCy5"

# CELLS_PATH = "/Users/guangzhili/mntpoint/_Piyush Goyal/Nanosyn/TDP43ConcTest/Aligned/RFP-DFTrCy5"
# NUCLEUSES_PATH = "/Users/guangzhili/mntpoint/_Piyush Goyal/Nanosyn/TDP43ConcTest/Aligned/DAPI"
# CYTOPLASMS_PATH = "/Users/guangzhili/mntpoint/_Piyush Goyal/Nanosyn/TDP43ConcTest/Aligned/FITC-DFTrCy5"


CELLS_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/source_images/TDP43ConcTest/Aligned/RFP-DFTrCy5"
NUCLEUSES_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/source_images/TDP43ConcTest/Aligned/DAPI"
CYTOPLASMS_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/source_images/TDP43ConcTest/Aligned/FITC-DFTrCy5"


# ENCODED_MASKS_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/test_sample_encoded_masks"
# INITIAL_CELL_ENCODED_MASKS_PATH = os.path.join(ENCODED_MASKS_PATH, "initialcell_encoded_masks")
# CELL_ENCODED_MASKS_PATH = os.path.join(ENCODED_MASKS_PATH, "cell_encoded_masks")
# NUCLEUS_ENCODED_MASKS_PATH = os.path.join(ENCODED_MASKS_PATH, "nucleus_encoded_masks")
# CYTOPLASM_ENCODED_MASKS_PATH = os.path.join(ENCODED_MASKS_PATH, "cytoplasm_encoded_masks")


# CELLS_PATH = "/Volumes/Analysis_Drive/Aligned/RFP-DFTrCy5"
# NUCLEUSES_PATH = "/Volumes/Analysis_Drive/Aligned/DAPI"
# CYTOPLASMS_PATH = "/Volumes/Analysis_Drive/Aligned/FITC-DFTrCy5"


ENCODED_MASKS_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/test_sample_encoded_masks"
INITIAL_CELL_ENCODED_MASKS_PATH = os.path.join(ENCODED_MASKS_PATH, "initialcell_encoded_masks")
CELL_ENCODED_MASKS_PATH = os.path.join(ENCODED_MASKS_PATH, "cell_encoded_masks")
NUCLEUS_ENCODED_MASKS_PATH = os.path.join(ENCODED_MASKS_PATH, "nucleus_encoded_masks")
CYTOPLASM_ENCODED_MASKS_PATH = os.path.join(ENCODED_MASKS_PATH, "cytoplasm_encoded_masks")



# CELL_ENCODED_MASKS_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/result_encoded_masks_250_500.0_60_10_0.25/cell_encoded_masks"
# NUCLEUS_ENCODED_MASKS_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/result_encoded_masks_250_500.0_60_10_0.25/nucleus_encoded_masks"
# CYTOPLASM_ENCODED_MASKS_PATH = "/Users/guangzhili/GladStone/Nucleus_projct/result_encoded_masks_250_500.0_60_10_0.25/cytoplasm_encoded_masks"


RESULT_CSV = "/Users/guangzhili/GladStone/Nucleus_projct/test_sample_encoded_masks/result.csv"
# CELL_THRESHOLD_INTENSITY = 200
# CELL_MIN_AREA = 500
# CELL_MAX_AREA = 50000
# CELL_MIN_CIRCULARITY = 0.0001
# CELL_MAX_CIRCULARITY = 0.25

# NUCLEUS_THRESHOLD_INTENSITY = 23
# NUCLEUS_MIN_AREA = 50
# NUCLEUS_MAX_AREA = 8000
# NUCLEUS_MIN_CIRCULARITY = 0.19
# NUCLEUS_MAX_CIRCULARITY = 0.99 


CELL_THRESHOLD_INTENSITY = 100
CELL_MIN_AREA = 350
CELL_MAX_AREA = 50000
CELL_MIN_CIRCULARITY = 0.001
CELL_MAX_CIRCULARITY = 0.25

NUCLEUS_THRESHOLD_INTENSITY = 320
NUCLEUS_MIN_AREA = 30
NUCLEUS_MAX_AREA = 8000
NUCLEUS_MIN_CIRCULARITY = 0.19
NUCLEUS_MAX_CIRCULARITY = 0.99 

 
START_TIMEPOINT = 'T0'

def get_threshold_image(orig_img, threshold_intensity):
    '''
    Threshold image binarize/split the image on user defined intensity value. 
    to apply properly contour algorithm we need convert to binary
    Get threshold image first can greatly reduce contours fould, i.e. get rid of noises.
    Note that the result threshold image may have black pixels inside the cell object. 
    In order to only get the outside shape of an  object, we use cv2.findContours later
    '''
    # source images are 16 bits
    img = orig_img.copy()
    img[img>threshold_intensity] = 65535
    img[img<=threshold_intensity] = 0  
    # cv2.findContours only support 8-bit single-channel image
    # thus convert <type 'numpy.uint16'> to numpy.uint8
    img = img.astype(np.uint8, copy=False) 
    # cv2.imwrite(os.path.join(test_folder, threshold_img_output), img)
    return img

def cal_circularity(blob_contour):
    blob_perimeter = cv2.arcLength(blob_contour, True) 
    blob_area = cv2.contourArea(blob_contour)
                      
    # Deal with ZeroDivisionError: float division by zero in python 2 in case blob_perimeter == 0
    if blob_perimeter:
        blob_circularity = 4*math.pi*blob_area/float((blob_perimeter**2))  
    else:
        blob_circularity = 0   
    return blob_circularity    


def get_contours(threshold_image, min_area, max_area, min_circularity, max_circularity):
    # Find the contours in the threshold image, cv2.findContours only support 8-bit single-channel image
    # Source: an 8-bit single-channel image. Non-zero pixels are treated as 1's. Zero pixels remain 0's, 
    # so the image is treated as binary 
    (cnts, _) = cv2.findContours(threshold_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print "Fould %d objects." %(len(cnts))

    # Filter out area in threshold range
    cnts = filter(lambda x: min_area<cv2.contourArea(x)<max_area, cnts)
    print "%d objects after area filter." %len(cnts)
    
    # if threshold_circularity_inequality == 'morethan':
    #     return cnts
    # A circle has a circularity of 1, circularity of a square is 0.785, and so on.
    # Filter out cell/nucleus out of range
    cnts = filter(lambda x: min_circularity<cal_circularity(x)<max_circularity, cnts)
    print "%d objects after circularity filter." % len(cnts)

    return cnts

def get_encoded_mask_from_contours(mask_shape, contours):    
    # Use 16 bits to contain more objects in the masks
    # Generate black background for drawing first
    encoded_mask = np.zeros(mask_shape, np.uint16)     
    # Encode contour id as intensity value
    for i in range(len(contours)):
        cv2.drawContours(encoded_mask, contours, i, i+1, thickness=cv2.cv.CV_FILLED)   
    return encoded_mask


def get_unique_ids(encoded_mask):
    ''' Get all the unique object ids'''
    ids = np.unique(encoded_mask)
    # Exclude value 0
    ids = ids[ids != 0]
    return ids

def get_image_names_tokens(images_path):
    img_names = [name for name in os.listdir(images_path) if name.endswith('.tif')]
    img_names_tokens = []
    for im in img_names:
        # Extract tokens from the file name
        # FileName TEXT, PID TEXT, ExperimentName TEXT, TimePoint TEXT, NumOfHours INT, WellID TEXT, Channel TEXT
        # e.g. PID20150904_PGPSTest_T1_8_A7_MONTAGE_FITC-DFTrCy5.tif
        name_tokens = im.split('_')
        pid_token = name_tokens[0]
        experiment_name_token = name_tokens[1]
        timepoint_token = name_tokens[2]
        num_of_hours_token = int(name_tokens[3])
        well_id_token = name_tokens[4]
        channel_token = name_tokens[6].split('.')[0]
        img_names_tokens.append([im, pid_token, experiment_name_token, timepoint_token, num_of_hours_token, well_id_token, channel_token])
    # We only care about the object ids exist from T0(or beginning) according to the experiment purpose, all the objects starts existing later then T0 will be get rid of
    # Sort the list by number of hours in order to find the objects exist in T0 first in loop, order by(experiment, well, channel, hours)
    img_names_tokens = sorted(img_names_tokens, key=itemgetter(2, 5, 6, 4))
    return img_names_tokens

def get_imagestack_names_tokens(images_path):
    img_names = [name for name in os.listdir(images_path) if name.endswith('.tif')]
    img_names_tokens = []
    for im in img_names:
        # Extract tokens from the file name
        # FileName TEXT, PID TEXT, ExperimentName TEXT, TimePoint TEXT, NumOfHours INT, WellID TEXT, Channel TEXT
        # separate image
        # e.g. PID20150904_PGPSTest_T1_8_A7_MONTAGE_FITC-DFTrCy5.tif
        # stack image
        # e.g. PID20150904_PGPSTest_STACK_ALIGNED_A7_MONTAGE_FITC-DFTrCy5.tif
        name_tokens = im.split('_')
        pid_token = name_tokens[0]
        experiment_name_token = name_tokens[1]
        # timepoint_token = START_TIMEPOINT # '1' instead of 'T1' for simple comparison

        well_id_token = name_tokens[4]
        channel_token = name_tokens[6].split('.')[0]
        img_names_tokens.append([im, pid_token, experiment_name_token, well_id_token, channel_token])
    # We only care about the object ids exist from T0(or beginning) according to the experiment purpose, all the objects starts existing later than T0 will be get rid of
    # Sort the list by number of hours in order to find the objects exist in T0 first in loop, order by(experiment, well, channel)
    img_names_tokens = sorted(img_names_tokens, key=itemgetter(2, 3, 4))
    return img_names_tokens

def run_nucleus_project(cells_path, nucleuses_path, cytoplasms_path, initial_cell_encoded_masks_path, cell_encoded_masks_path, nucleus_encoded_masks_path, cytoplasm_encoded_masks_path, start_timepoint, cell_threshold_intensity, cell_min_area, cell_max_area, cell_min_circularity, cell_max_circularity, nucleus_threshold_intensity, nucleus_min_area, nucleus_max_area, nucleus_min_circularity, nucleus_max_circularity, result_csv):
    # Separated timepoints source images
    # PID20150904_PGPSTest_T1_8_A7_MONTAGE_RFP-DFTrCy5.tif  Cell
    # PID20150904_PGPSTest_T2_16_A7_MONTAGE_RFP-DFTrCy5.tif
    # PID20150904_PGPSTest_T1_8_A7_MONTAGE_DAPI.tif  Nucleus
    # PID20150904_PGPSTest_T1_8_A7_MONTAGE_FITC-DFTrCy5.tif  Cytoplasm

    # STACK source images
    # PID20150904_PGPSTest_STACK_ALIGNED_A7_MONTAGE_RFP-DFTrCy5.tif  Cell
    # PID20150904_PGPSTest_STACK_ALIGNED_A7_MONTAGE_DAPI.tif  Nucleus
    # PID20150904_PGPSTest_STACK_ALIGNED_A7_MONTAGE_FITC-DFTrCy5.tif  Cytoplasm
    cell_names_tokens = get_imagestack_names_tokens(cells_path)
    # To copy a list you can use list(a) or a[:]. In both cases a new object is created.
    # These two methods, however, have limitations with collections of mutable objects as inner objects keep 
    # their references intact: If you want a full copy of your objects you need copy.deepcopy
    # http://stackoverflow.com/questions/8744113/python-list-by-value-not-by-reference
    # [im, pid_token, experiment_name_token, well_id_token, channel_token]
    # nucleus_names_tokens = deepcopy(cell_names_tokens)
    # for i in xrange(len(nucleus_names_tokens)):
    #     # Note that str.replace will NOT change str itself
    #     nucleus_names_tokens[i][0] = nucleus_names_tokens[i][0].replace("RFP-DFTrCy5", "DAPI")
    #     nucleus_names_tokens[i][4] = nucleus_names_tokens[i][4].replace("RFP-DFTrCy5", "DAPI")
    # cytoplasm_names_tokens = deepcopy(cell_names_tokens)
    # for i in xrange(len(cytoplasm_names_tokens)):
    #     cytoplasm_names_tokens[i][0] = cytoplasm_names_tokens[i][0].replace("RFP-DFTrCy5", "FITC-DFTrCy5")
    #     cytoplasm_names_tokens[i][4] = cytoplasm_names_tokens[i][4].replace("RFP-DFTrCy5", "FITC-DFTrCy5")    

    nucleus_names_tokens = get_imagestack_names_tokens(nucleuses_path)
    cytoplasm_names_tokens = get_imagestack_names_tokens(cytoplasms_path)
    


    # Keep track of cell and nucleus encoded masks at timepoints
    # e.g. {(experiment0, well0): [(encoded_mask_cell_path_0, encoded_mask_nucleus_path_0), ...]}
    encoded_masks_for_experiment_well = {}
    # result csv file header
    col_names = ["PID", "ExperimentName", "WellID", "ObjectID", "TimePoint", "CellMeanIntensity", "NucleusMeanIntensity", "CytoplasmMeanIntensity"]                     
    with open(result_csv, 'wb') as csvfile:
        # It's delimiter and not delimeter
        writer = csv.writer(csvfile, delimiter=',')
        # Write header
        writer.writerow(col_names)
        # Loop through eath file (experiment, well, numofhours)
        for i in xrange(len(cell_names_tokens)):

            # tokens format [im, pid_token, experiment_name_token, well_id_token, channel_token]
            # Get corresponding tokens
            cell_filename = cell_names_tokens[i][0]
            nucleus_filename = nucleus_names_tokens[i][0]
            cytoplasm_filename = cytoplasm_names_tokens[i][0]
            pid = cell_names_tokens[i][1]
            experiment_name = cell_names_tokens[i][2]
            # Track initial timepoint/numberofhours 
            timepoint = start_timepoint 


            # timepoint = cell_names_tokens[i][3]
            # num_of_hours = cell_names_tokens[i][4] # int
            well_id = cell_names_tokens[i][3]
            cell_channel = cell_names_tokens[i][4]
            nucleus_channel = nucleus_names_tokens[i][4]  
            cytoplasm_channel = cytoplasm_names_tokens[i][4]  
            # if well_id != 'F7':
            #     continue
            
            cell_path = os.path.abspath(os.path.join(cells_path, cell_filename))    
            nucleus_path = os.path.abspath(os.path.join(nucleuses_path, nucleus_filename))  
            cytoplasm_path = os.path.abspath(os.path.join(cytoplasms_path, cytoplasm_filename))  
            cur_experiment_well = (experiment_name, well_id)
            
            cell_stack_image = Image.open(cell_path)
            nucleus_stack_image = Image.open(nucleus_path)
            cytoplasm_stack_image = Image.open(cytoplasm_path)

            
            # Count how many images in current Multi-image stack TIFF file
            slice_idx = 0
            while True:
                try:
                    # Seeks to the given frame in this sequence file. If you seek beyond the end of the sequence, 
                    # the method raises an EOFError exception. When a sequence file is opened, 
                    # the library automatically seeks to frame 0.
                    # Note that in the current version of the library, 
                    # most sequence formats only allows you to seek to the next frame.
                    cell_stack_image.seek(slice_idx)
                    # Convert to opencv numpy.ndarray representation
                    cell_image = np.asarray(cell_stack_image)
                    print "Start processing: (%s, %s, %s)" %(experiment_name, well_id, timepoint)

                    # Get Cell encoded mask.
                    # objects are valid only when both cell and nucleus exist, so current encoded objects not necessary valid, will get rid of the invalid ones and write to file later after nucleus caculations
                    # cell_image = cv2.imread(cell_path, cv2.CV_LOAD_IMAGE_UNCHANGED)
                    threshold_image_cell = get_threshold_image(cell_image, cell_threshold_intensity)
                    print "Extracting cell contours:"
                    contours_cell = get_contours(threshold_image_cell, cell_min_area, cell_max_area, cell_min_circularity, cell_max_circularity)

                    # if timepoint == 'T2':

                    #     # contours_cell_test = get_contours(threshold_image_cell, 0, 10000000000, 0, 1)
                    #     contours_cell_test = get_contours(threshold_image_cell, 0, 100000, 0.001, 0.35)

                    #     # 500 10000 0.0 0.3
                    #     es = np.zeros(threshold_image_cell.shape, np.uint16)  
                    #     for i in range(len(contours_cell_test)):
                    #         if cv2.contourArea(contours_cell_test[i])>10000:
                    #             print "what"
                    #             print cv2.contourArea(contours_cell_test[i])
                    #         cv2.drawContours(es, contours_cell_test, i, i+1, thickness=cv2.cv.CV_FILLED)  
                    #     firstout = os.path.join(initial_cell_encoded_masks_path, "testfirst.tif")
                    #     cv2.imwrite(firstout, es)  

                    # encoded_mask_cell = get_encoded_mask_from_contours(threshold_image_cell.shape, contours_cell)


                    # test_mask = np.zeros(threshold_image_cell.shape, np.uint16)  

                    # Use 16 bits to contain more objects in the masks
                    # Generate black background for drawing first
                    encoded_mask_cell = np.zeros(threshold_image_cell.shape, np.uint16)  
                    # Dict to get corresponding contour with intensity value
                    intensity_cellcontours_relation = {}
                    if timepoint == START_TIMEPOINT:               
                        # Encode contour id as intensity value
                        for i in range(len(contours_cell)):

                            cv2.drawContours(encoded_mask_cell, contours_cell, i, i+1, thickness=cv2.cv.CV_FILLED)   
                            intensity_cellcontours_relation[i+1] = [contours_cell[i]]
                    else:
                        # If it is not initial timepoint, only draw contours center exist previous 
                        # Retrieve previous timepoint mask
                        mark_encoded_mask_cell = cv2.imread(encoded_masks_for_experiment_well[cur_experiment_well][-1][0], cv2.CV_LOAD_IMAGE_UNCHANGED)
                         
                        # if timepoint == 'T3':
                        #     es = np.zeros(threshold_image_cell.shape, np.uint16)  
                        #     for i in range(len(contours_cell)):
                        #         cv2.drawContours(es, contours_cell, i, i+1, thickness=cv2.cv.CV_FILLED)  
                        #     outf = os.path.join(initial_cell_encoded_masks_path, "test.tif")
                        #     cv2.imwrite(outf, es)  

                        for i in range(len(contours_cell)):
                            # Find if overlapping, do not care the ones start later than initial timepoint
                            # Find center first
                            (x,y),radius = cv2.minEnclosingCircle(contours_cell[i])
                            blob_centroid_x, blob_cnetroid_y = (int(x),int(y))
                            # Note to convert <type 'numpy.uint16'> to int
                            cell_centroid_intensity = int(mark_encoded_mask_cell[blob_cnetroid_y][blob_centroid_x])
                      
                            if cell_centroid_intensity != 0: 
                                cv2.drawContours(encoded_mask_cell, contours_cell, i, cell_centroid_intensity, thickness=cv2.cv.CV_FILLED)
                                # Note that, it is possible when the cell is dying, it separate into multiple parts contours, 
                                # and they have the same intensity. That is why we relate list of contours instead of single 
                                # contour to intensity
                                if intensity_cellcontours_relation.get(cell_centroid_intensity, None):
                                    intensity_cellcontours_relation[cell_centroid_intensity].append(contours_cell[i])
                                else:    
                                    intensity_cellcontours_relation[cell_centroid_intensity] = [contours_cell[i]]
                            else:
                                # Center may NOT be sufficient, since the cell shift or alignment problem, we check if there is any overlap instead
                                # make a mask for current single one cell contour
                                contour_mask = np.zeros(mark_encoded_mask_cell.shape, np.uint8)  
                                cv2.drawContours(contour_mask, contours_cell, i, 255, thickness=cv2.cv.CV_FILLED)
                                # Get all the coordinates of points inside the contour
                                contour_coordinates = np.transpose(np.nonzero(contour_mask))

                                # Check if there is any overlap to previous timepoint mask
                                # Note that it is possible current cell overlaps multiple cells of previous timepoints
                                # Thus we just pick the intensity of largest overlapping previous cell
                                overlap_intensities = map(lambda x: int(mark_encoded_mask_cell[x[0]][x[1]]), contour_coordinates)
                                # Filter out all zeros
                                overlap_intensities = filter(lambda x: x!=0, overlap_intensities)
                                # Count the most common nonzero elements in the list, i.e. the largest area overlap
                                if overlap_intensities != []:
                                    cell_intensity = Counter(overlap_intensities).most_common(1)[0][0]
                                    cv2.drawContours(encoded_mask_cell, contours_cell, i, cell_intensity, thickness=cv2.cv.CV_FILLED)
                                    # Note that, it is possible when the cell is dying, it separate into multiple parts contours, 
                                    # and they have the same intensity. Later we want to get rid of them, thus need to 
                                    # keep track of all of them in list 
                                    if intensity_cellcontours_relation.get(cell_intensity, None):
                                        intensity_cellcontours_relation[cell_intensity].append(contours_cell[i])
                                    else:    
                                        intensity_cellcontours_relation[cell_intensity] = [contours_cell[i]] 

                                else:
                                    # print "No overlapping"
                                    pass

                                
                                    
                                 

                            
                    # Write the cell encode mask to file. Note that this is the initial cell encoded mask, and (may) contains cells that do not have nucleus 
                    initial_encoded_mask_cell_outputfile = os.path.join(initial_cell_encoded_masks_path, "%s_%s_%s_%s_encoded_mask_%s-initial.tif" %(pid, experiment_name, well_id, timepoint, cell_channel))
                    cv2.imwrite(initial_encoded_mask_cell_outputfile, encoded_mask_cell)  

                   



                    # Get Nucleus encoded mask
                  
                    nucleus_stack_image.seek(slice_idx)
                    nucleus_image = np.asarray(nucleus_stack_image)

                    # We only care about the nucleus inside the cells
                    # Python: cv2.bitwise_and(src1, src2[, dst[, mask]]) -> dst
                    # mask - optional operation mask, 8-bit single channel array, that specifies elements of the output array to be changed.
                    mask_cell = encoded_mask_cell.copy()
                    mask_cell[mask_cell!=0] = 255
                    mask_cell = mask_cell.astype(np.uint8, copy=False)

                    
                    nucleus_within_cell_image = cv2.bitwise_and(nucleus_image, nucleus_image, mask=mask_cell)

             
                    # cv2.imwrite(os.path.join(test_folder, "nucleus_within_cell.tif"), nucleus_within_cell_image)
                    threshold_image_nucleus = get_threshold_image(nucleus_within_cell_image, nucleus_threshold_intensity)
                    print "Extracting nucleus contours:"
                    contours_nucleus = get_contours(threshold_image_nucleus, nucleus_min_area, nucleus_max_area, nucleus_min_circularity, nucleus_max_circularity)
                    
                    # if timepoint == 'T2':

                    #     es = np.zeros(threshold_image_cell.shape, np.uint16)  
                    #     for i in range(len(contours_nucleus)):
                    #         cv2.drawContours(es, contours_nucleus, i, i+1, thickness=cv2.cv.CV_FILLED)  
                    #     outf = os.path.join(initial_cell_encoded_masks_path, "test.tif")
                    #     cv2.imwrite(outf, nucleus_within_cell_image)  


                    # Initail encoded_mask_nucleus as black background. Use 16 bits to contain more objects in the masks
                    encoded_mask_nucleus = np.zeros(nucleus_within_cell_image.shape, np.uint16)  
                    # There has to be only one nucleus inside single cell, so use dict to keep track of list of nucleus contours related to specific intensity value
                    intensity_nucleuscontours_relation = {}    
                    valid_object_ids = []
                    valid_contours_nucleus = []

                    for i in range(len(contours_nucleus)):
                        (x,y),radius = cv2.minEnclosingCircle(contours_nucleus[i])
                        blob_centroid_x, blob_cnetroid_y = (int(x),int(y))  # Almost always the same compared to compute_center function, sometimes 1 pixel diff
                        # Note the different order of x and y coordinates
                        nucleus_centroid_intensity = int(encoded_mask_cell[blob_cnetroid_y][blob_centroid_x])
                        if nucleus_centroid_intensity != 0: # If center at edge, do not count
                            if intensity_nucleuscontours_relation.get(nucleus_centroid_intensity, None):
                                intensity_nucleuscontours_relation[nucleus_centroid_intensity].append(contours_nucleus[i])
                            else:
                                intensity_nucleuscontours_relation[nucleus_centroid_intensity] = [contours_nucleus[i]]    
                        

                    # Encode nucleus contour id as intensity value which comes from cell. i.e. nucleus have same id with containing cell
                    # Only the largest contour inside cell counts as nucleus
                    for k, v in intensity_nucleuscontours_relation.iteritems():
                         
                        if len(v) == 1:
                            blob_cnt = v[0]
                            # blob_area = cv2.contourArea(blob_cnt)
                        else:
                            # Find the only one with maximum area
                            print "more than one nucleus contour inside cell."
                            blob_cnt, blob_area = None, None
                            for c in v:
                                c_area = cv2.contourArea(c) 
                                if blob_area is None or c_area > blob_area:
                                    blob_cnt = c
                                    blob_area = c_area   
                            # Leave only one largest nucleus contour relates to identical intensity
                            intensity_nucleuscontours_relation[k] = [blob_cnt]
                          
                        cv2.drawContours(encoded_mask_nucleus, [blob_cnt], 0, k, thickness=cv2.cv.CV_FILLED)
                        valid_object_ids.append(k)
                        valid_contours_nucleus.append(blob_cnt)
                           
                                   

                    encoded_mask_nucleus_outputfile = os.path.join(nucleus_encoded_masks_path, "%s_%s_%s_%s_encoded_mask_%s.tif" %(pid, experiment_name, well_id, timepoint, nucleus_channel))
                    cv2.imwrite(encoded_mask_nucleus_outputfile, encoded_mask_nucleus)  


                    # Sort object ids for better print
                    valid_object_ids.sort()
                    print "%d valid objects for (%s, %s, %s): \n%s" %(len(valid_object_ids), experiment_name, well_id, timepoint, valid_object_ids)
                   

                    


                    # Now for encoded_mask_cell, get rid of cells without nucleuses
                    # Make sure only the valid nucleus and cell objects are left
                    for intensity_id in intensity_cellcontours_relation:
                        # Get rid of invalid cells
                        if intensity_id not in valid_object_ids:
                            cv2.drawContours(encoded_mask_cell, intensity_cellcontours_relation[intensity_id], -1, 0, thickness=cv2.cv.CV_FILLED)
                        # Get rid of invalid cells with valid intensity
                        # For example, dying cell may elapses into multiple cells, or other cell may intercept. 
                        # Thus they have the same intensity, but some contours are invalid without nucleus
                        elif len(intensity_cellcontours_relation[intensity_id])>1:
                            cellcontours_of_intensityid = intensity_cellcontours_relation[intensity_id]
                            temp_background = np.zeros(encoded_mask_cell.shape, np.uint16)  
                            for idx in range(len(cellcontours_of_intensityid)):
                                cv2.drawContours(temp_background, [cellcontours_of_intensityid[idx]], 0, idx+1, thickness=cv2.cv.CV_FILLED)
                            # Find which cell contour is the nucleus contour located
                            # Note intensity_nucleuscontours_relation[intensity_id] is a list of contours too
                            (x,y),radius = cv2.minEnclosingCircle(intensity_nucleuscontours_relation[intensity_id][0])
                            blob_centroid_x, blob_cnetroid_y = (int(x),int(y))
                            # Note to convert <type 'numpy.uint16'> to int
                            valid_cellcontour_intensity = int(temp_background[blob_cnetroid_y][blob_centroid_x]) 
                            valid_cellcontour = cellcontours_of_intensityid[valid_cellcontour_intensity-1]
                            # Now black out the invalid cell contours
                            for idx in range(len(cellcontours_of_intensityid)):
                                if idx+1 != valid_cellcontour_intensity:
                                    cv2.drawContours(encoded_mask_cell, [cellcontours_of_intensityid[idx]], 0, 0, thickness=cv2.cv.CV_FILLED)
                            # Leave only one cell contour relates to identical intensity
                            intensity_cellcontours_relation[intensity_id] = [valid_cellcontour]
                    
                    # Now since invalid cells are already mask out, we can write it to file
                    encoded_mask_cell_outputfile = os.path.join(cell_encoded_masks_path, "%s_%s_%s_%s_encoded_mask_%s.tif" %(pid, experiment_name, well_id, timepoint, cell_channel))
                    cv2.imwrite(encoded_mask_cell_outputfile, encoded_mask_cell)  


                    # e.g. {(experiment1, well1): (encoded_mask_cell_path, encoded_mask_nucleus_path)}
                  
                    if timepoint == START_TIMEPOINT:     
                        encoded_masks_for_experiment_well[cur_experiment_well] = [(encoded_mask_cell_outputfile, encoded_mask_nucleus_outputfile)]
                    else:
                        encoded_masks_for_experiment_well[cur_experiment_well].append((encoded_mask_cell_outputfile, encoded_mask_nucleus_outputfile))  


                    # Get Cytoplasm encoded mask
                    encoded_mask_cytoplasm = encoded_mask_cell.copy()
                    cv2.drawContours(encoded_mask_cytoplasm, valid_contours_nucleus, -1, 0, thickness= cv2.cv.CV_FILLED)

                   
                    encoded_mask_cytoplasm_outputfile = os.path.join(cytoplasm_encoded_masks_path, "%s_%s_%s_%s_encoded_mask_%s.tif" %(pid, experiment_name, well_id, timepoint, cytoplasm_channel))
                    cv2.imwrite(encoded_mask_cytoplasm_outputfile, encoded_mask_cytoplasm)  


                    # print "cell areas:"
                    # cell_areas = []
                    # for k, v in intensity_cellcontours_relation.iteritems():
                    #     cell_areas.append(cv2.contourArea(v[0]))
                    # print "min:%s, max:%s" %(min(cell_areas), max(cell_areas))
                    # print "nucleus areas"
                    # nucleus_areas = []
                    # for k, v in intensity_nucleuscontours_relation.iteritems():
                    #     nucleus_areas.append(cv2.contourArea(v[0]))
                    # print "min:%s, max:%s" %(min(nucleus_areas), max(nucleus_areas))

                        
                    # Start cell, nucleus and cytoplasm ROI mean intensities calculation ON CYTOPLASM IMAGE!!     
                    # cytoplasm_image = cv2.imread(cytoplasm_path, cv2.CV_LOAD_IMAGE_UNCHANGED)
                    cytoplasm_stack_image.seek(slice_idx)
                    cytoplasm_image = np.asarray(cytoplasm_stack_image)


                    # First get roi obejcts with full intensities inside ready
                    # Cell ROI with full inside intensities ON CYTOPLASM IMAGE 
                    cell_objects = cv2.bitwise_and(cytoplasm_image, cytoplasm_image, mask=mask_cell)

                    # Nucleus ROI with full inside intensities ON CYTOPLASM IMAGE 
                    # Get binary mask from the encoded mask for bitwise_and mask usage
                    mask_nucleus = encoded_mask_nucleus.copy()
                    mask_nucleus[mask_nucleus!=0] = 255
                    mask_nucleus = mask_nucleus.astype(np.uint8, copy=False)
                    nucleus_objects = cv2.bitwise_and(cytoplasm_image, cytoplasm_image, mask=mask_nucleus)

                    # Cytoplasm ROI with full inside intensities ON CYTOPLASM IMAGE 
                    # Get binary mask from the encoded mask for bitwise_and mask usage
                    mask_cytoplasm = encoded_mask_cytoplasm.copy()
                    mask_cytoplasm[mask_cytoplasm!=0] = 255 
                    # numpy astype conversion from 16 bits to 8 bits works like these: 255 -> 255, 256 -> 0, 257 -> 1
                    mask_cytoplasm = mask_cytoplasm.astype(np.uint8, copy=False)
                    cytoplasm_objects = cv2.bitwise_and(cytoplasm_image, cytoplasm_image, mask=mask_cytoplasm)

                    for i in valid_object_ids:
                        print "object_id: %d" % i
                        # Intensities, include 3 region, nucleus, cell, and area between nucleus and cell(i.e. Cytoplasm)
                        # Get mean intensity of cell
                        cell_object_mask = encoded_mask_cell.copy()
                        cell_object_mask[cell_object_mask!=i] = 0
                        # Get rid of pixels outside of current only one cell object, since we do not include those pixels in mean intensity calculation
                        # np.shape becomes 1D from 2D. shape like (236,)
                        cell_roi_cnt_intensities = cell_objects[np.nonzero(cell_object_mask)]
                        cell_roi_pixel_intensity_mean = int(np.average(cell_roi_cnt_intensities)) # convert from numpy.float64 to int
                        print "cell_intensity_mean:     ", cell_roi_pixel_intensity_mean

                        # Get mean intensity of nucleus
                        nucleus_object_mask = encoded_mask_nucleus.copy()
                        nucleus_object_mask[nucleus_object_mask!=i] = 0
                        # Get rid of pixels outside of current only one nucleus object, since we do not include those pixels in mean intensity calculation
                        # np.shape becomes 1D from 2D. shape like (236,)
                        nucleus_roi_cnt_intensities = nucleus_objects[np.nonzero(nucleus_object_mask)]

                        nucleus_roi_pixel_intensity_mean = int(np.average(nucleus_roi_cnt_intensities)) # convert from numpy.float64 to float
                        print "nucleus_intensity_mean:  ", nucleus_roi_pixel_intensity_mean

                        # Get mean intesity of cytoplasm
                        cytoplasm_object_mask = encoded_mask_cytoplasm.copy()
                        cytoplasm_object_mask[cytoplasm_object_mask!=i] = 0
                        # Get rid of pixels outside of current only one cytoplasm object, since we do not include those pixels in mean intensity calculation
                        # np.shape becomes 1D from 2D. shape like (236,)
                        cytoplasm_object_cnt_intensities = cytoplasm_objects[np.nonzero(cytoplasm_object_mask)]
                        # Careful! It is possible that the segmentation make nucleus object expand the whold area of cell, thus make area of cytoplasm to be 0
                        if len(cytoplasm_object_cnt_intensities) != 0:
                            cytoplasm_object_pixel_intensity_mean = int(np.average(cytoplasm_object_cnt_intensities)) # convert from numpy.float64 to float
                        else:
                            cytoplasm_object_pixel_intensity_mean = 0    
                        print "cytoplasm_intensity_mean:", cytoplasm_object_pixel_intensity_mean

                        # col_names = ["PID", "ExperimentName", "WellID", "ObjectID", "TimePoint", "CellMeanIntensity", "NucleusMeanIntensity", "CytoplasmMeanIntensity"]                 
                        writer.writerow([pid, experiment_name, well_id, i, timepoint, cell_roi_pixel_intensity_mean, nucleus_roi_pixel_intensity_mean, cytoplasm_object_pixel_intensity_mean])   
                    slice_idx += 1
                    timepoint = timepoint[0] + str(int(timepoint[1:])+1)
                except EOFError:
                    break          
if __name__ == '__main__':
    # Command line test
    # paths = [INITIAL_CELL_ENCODED_MASKS_PATH, CELL_ENCODED_MASKS_PATH, NUCLEUS_ENCODED_MASKS_PATH, CYTOPLASM_ENCODED_MASKS_PATH]
    # for p in paths:
    #     try:     
    #         os.makedirs(p)
    #     except OSError:
    #         if not os.path.isdir(p):
    #             raise
     

    # run_nucleus_project(CELLS_PATH, NUCLEUSES_PATH, CYTOPLASMS_PATH, INITIAL_CELL_ENCODED_MASKS_PATH, CELL_ENCODED_MASKS_PATH, NUCLEUS_ENCODED_MASKS_PATH, CYTOPLASM_ENCODED_MASKS_PATH, START_TIMEPOINT, CELL_THRESHOLD_INTENSITY, CELL_MIN_AREA, CELL_MAX_AREA, CELL_MIN_CIRCULARITY, CELL_MAX_CIRCULARITY, NUCLEUS_THRESHOLD_INTENSITY, NUCLEUS_MIN_AREA, NUCLEUS_MAX_AREA, NUCLEUS_MIN_CIRCULARITY, NUCLEUS_MAX_CIRCULARITY, RESULT_CSV)







    parser = argparse.ArgumentParser()
    parser.add_argument("cells_path")
    parser.add_argument("nucleuses_path")
    parser.add_argument("cytoplasms_path")
    parser.add_argument("encoded_masks_path")
    parser.add_argument("--start_timepoint")
    parser.add_argument("--cell_threshold_intensity")
    parser.add_argument("--cell_min_area")
    parser.add_argument("--cell_max_area")
    parser.add_argument("--cell_min_circularity")
    parser.add_argument("--cell_max_circularity")
    parser.add_argument("--nucleus_threshold_intensity")
    parser.add_argument("--nucleus_min_area")
    parser.add_argument("--nucleus_max_area")
    parser.add_argument("--nucleus_min_circularity")
    parser.add_argument("--nucleus_max_circularity")
    parser.add_argument("result_csv")
    args = parser.parse_args()
    

    cells_path = args.cells_path
    nucleuses_path = args.nucleuses_path
    cytoplasms_path = args.cytoplasms_path
    encoded_masks_path = args.encoded_masks_path
    initial_cell_encoded_masks_path = os.path.join(encoded_masks_path, "initialcell_encoded_masks")
    cell_encoded_masks_path = os.path.join(encoded_masks_path, "cell_encoded_masks")
    nucleus_encoded_masks_path = os.path.join(encoded_masks_path, "nucleus_encoded_masks")
    cytoplasm_encoded_masks_path = os.path.join(encoded_masks_path, "cytoplasm_encoded_masks")
  
    paths = [initial_cell_encoded_masks_path, cell_encoded_masks_path, nucleus_encoded_masks_path, cytoplasm_encoded_masks_path]
    try: 
        for p in paths:
            os.makedirs(p)
    except OSError:
        if not os.path.isdir(p):
            raise


    if args.start_timepoint:
        start_timepoint = args.start_timepoint
    else:
        start_timepoint = START_TIMEPOINT
    if args.cell_threshold_intensity:
        cell_threshold_intensity = int(args.cell_threshold_intensity)
    else:
        cell_threshold_intensity = CELL_THRESHOLD_INTENSITY
    if args.cell_min_area:
        cell_min_area = float(args.cell_min_area)  
    else:   
        cell_min_area = CELL_MIN_AREA    
    if args.cell_max_area:
        cell_max_area = float(args.cell_max_area)  
    else:   
        cell_max_area = CELL_MAX_AREA      
    if args.cell_min_circularity:
        cell_min_circularity = float(args.cell_min_circularity)
    else:
        cell_min_circularity = CELL_MIN_CIRCULARITY     
    if args.cell_max_circularity:
        cell_max_circularity = float(args.cell_max_circularity)
    else:
        cell_max_circularity = CELL_MAX_CIRCULARITY         
    if args.nucleus_threshold_intensity:
        nucleus_threshold_intensity = int(args.nucleus_threshold_intensity)    
    else:
        nucleus_threshold_intensity = NUCLEUS_THRESHOLD_INTENSITY  
    if args.nucleus_min_area:
        nucleus_min_area = float(args.nucleus_min_area)
    else:
        nucleus_min_area = NUCLEUS_MIN_AREA 
    if args.nucleus_max_area:
        nucleus_max_area = float(args.nucleus_max_area)
    else:
        nucleus_max_area = NUCLEUS_MAX_AREA     
    if args.nucleus_min_circularity:
        nucleus_min_circularity = float(args.nucleus_min_circularity)
    else:
        nucleus_min_circularity = NUCLEUS_MIN_CIRCULARITY    
    if args.nucleus_max_circularity:
        nucleus_max_circularity = float(args.nucleus_max_circularity)
    else:
        nucleus_max_circularity = NUCLEUS_MAX_CIRCULARITY   

    result_csv = args.result_csv           

    run_nucleus_project(cells_path, nucleuses_path, cytoplasms_path, initial_cell_encoded_masks_path, cell_encoded_masks_path, nucleus_encoded_masks_path, cytoplasm_encoded_masks_path, start_timepoint, cell_threshold_intensity, cell_min_area, cell_max_area, cell_min_circularity, cell_max_circularity, nucleus_threshold_intensity, nucleus_min_area, nucleus_max_area, nucleus_min_circularity, nucleus_max_circularity, result_csv)







