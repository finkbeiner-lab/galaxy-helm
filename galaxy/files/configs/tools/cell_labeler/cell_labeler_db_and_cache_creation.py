''' Note:
All the objects will be calculated and inserted into database. But only the objects ValidFromStartTimePoint will be cached and labeled and output to csv
'''


# In my Python 2 modules, I almost always import division from __future__, so that I can't get caught out by accidentally passing integers to a division operation I don't expect to truncate
from __future__ import division
import sqlite3
import sys
import ntpath
import os
import cv2
import numpy as np
import multiprocessing
from libtiff import TIFF
import copy
import time
import datetime
import math
from scipy import stats
import argparse
from operator import itemgetter
from collections import Counter
# import shutil
from subprocess import call

import warnings
# stats.skew throws nonsense runtimewarning on object 93 of image PID20150217_BioP7asynA_T0_0_F7_1_FITC-DFTrCy5_BG_MONTAGE_ALIGNED_CROPPED.tif
# while executing pixel_intensity_skewness = stats.skew(obj_img_cnt_intensities)
# Error message as folows:
# /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/scipy/stats/stats.py:993: RuntimeWarning:
# invalid value encountered in double_scalars
# vals = np.where(zero, 0, m3 / m2**1.5)
# In order to depress the error on history pane of Galaxy, suppress the RuntimeWarning message
warnings.filterwarnings("ignore", category=RuntimeWarning)

# In order to run in Galaxy, the path must be absolute
NUMBER_OF_PROCESSORS = multiprocessing.cpu_count()
TABLE_NAME = 'BioMedImages'
EXPERIMENT_NAME = ''
NAMING_TYPE = ''
MORPHOLOGY_CHANNEL = ''
ORIGIANL_IAMGES_PATH = ''
ENCODED_MASKS_PATH = ''
DB_AND_CACHE_PATH = ''
SQLITE_DB_TMP_FOLDER = '/scratch/sqlite_db_tmp'
DB_PATH = ''
CACHE_PATH = ''
SORTED_TIMEPOINT_SET_LIST = []
SORTED_CHANNEL_SET_LIST = []
CACHE_WIDTH = 300
CACHE_HEIGHT = 300
DTYPE = 'uint16'

def path_leaf(abs_path):
    ''' Extract file name from abslute path'''
    head, tail = ntpath.split(abs_path)
    return tail or ntpath.basename(head)

def get_contour(obj_mask):
    # Find the contours in the mark, cv2.findContours only support 8-bit single-channel image
    (cnts, _) = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Exclude duplicate object ids
    # If duplicate return None
    if len(cnts) == 1:
        return cnts[0]
    else:
        return None

def compute_center(cnt):
    m = cv2.moments(cnt)

    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return cx, cy


def get_image_tokens_list(original_images_path, naming_type):
    ''' Get image file token list
    Args:
      original_images_path: Input dir. each image file is Montaged time point separated.
      naming_type: Different image naming mode

    Time separated image name examples(4 naming types):
    Robo3:
      PID20150217_BioP7asynA_T0_0_A1_1_RFP-DFTrCy5_BG_MONTAGE_ALIGNED_CROPPED.tif
      PID20150904_PGPSTest_T1_8_A7_MONTAGE_RFP-DFTrCy5.tif
      PID20150217_BioP7asynA_T1_12-3_A10_1_RFP-DFTrCy5_MN.tif
    Robo4 epi:
      PID20160706_MerckMitoPlate23_T0_12_A11_1_Cyan_DAPI-FITC_525_1_0.0_ANDORZYLA120XELWD.tif
    Robo4 confocal:
      PID20160706_MerckMitoPlate23_T0_12_A11_1_488_561_Empty_525_1_0.0_AndorZyla120XELWD.tif
    Robo4 latest:
      PID20160706_MerckMitoPlate23_T0_0-0_A11_1_Epi-DAPI_0.0_0_1.0.tif

    '''
    stack_dict = {}


    # use os.walk() to recursively iterate through a directory and all its subdirectories
    image_paths = [os.path.join(root, name) for root, dirs, files in os.walk(original_images_path) for name in files if name.endswith('.tif')]


    # Robo3 naming
    # Example: PID20150217_BioP7asynA_T1_12-3_A10_1_RFP-DFTrCy5_MN.tif
    if naming_type == 'robo3':
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            name_tokens = image_name.split('_')
            pid_token = name_tokens[0]
            experiment_name_token = name_tokens[1]
            timepoint_token = name_tokens[2]

            # Check burst
            burst_idx_token = None
            if '-' in name_tokens[3]:
                numofhours_token, burst_idx_token = name_tokens[3].split('-')
                numofhours_token = float(numofhours_token)
                burst_idx_token = int(burst_idx_token)
            else:
                numofhours_token = int(name_tokens[3])
                burst_idx_token = None
            well_id_token = name_tokens[4]

            channel_token = name_tokens[6].replace('.tif', '')
            z_idx_token = None

            # Split well id token to make sorting easier
            # Well ID example: H12
            experiment_well_key = (experiment_name_token, well_id_token[0], int(well_id_token[1:]))

            if experiment_well_key in stack_dict:
                stack_dict[experiment_well_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token])
            else:
                stack_dict[experiment_well_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token]]
    # Robo4 epi naming
    # Example: PID20160706_MerckMitoPlate23_T0_12_A11_1_Cyan_DAPI-FITC_525_1_0.0_ANDORZYLA120XELWD.tif
    elif naming_type== 'robo4_epi':
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            name_tokens = image_name.split('_')
            pid_token = name_tokens[0]
            experiment_name_token = name_tokens[1]
            timepoint_token = name_tokens[2]

            # Check burst
            burst_idx_token = None
            if '-' in name_tokens[3]:
                numofhours_token, burst_idx_token = name_tokens[3].split('-')
                numofhours_token = float(numofhours_token)
                burst_idx_token = int(burst_idx_token)
            else:
                numofhours_token = int(name_tokens[3])
                burst_idx_token = None
            well_id_token = name_tokens[4]
            # Include all the channel tokens
            channel_token = '_'.join([name_tokens[6], name_tokens[7], name_tokens[8]])
            # CHANNEL_SET.add(channel_token)
            z_idx_token = int(name_tokens[9])

            # Split well id token to make sorting easier
            # Well ID example: H12
            experiment_well_key = (experiment_name_token, well_id_token[0], int(well_id_token[1:]))

            if experiment_well_key in stack_dict:
                stack_dict[experiment_well_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token])
            else:
                stack_dict[experiment_well_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token]]
    # Robo4 confocal naming
    # Example: PID20160706_MerckMitoPlate23_T0_12_A11_1_488_561_Empty_525_1_0.0_AndorZyla120XELWD.tif
    elif naming_type == 'robo4_confocal':
        for i in range(len(image_paths)):
            image_path = image_paths[i]
            image_name = os.path.basename(image_path)
            name_tokens = image_name.split('_')
            pid_token = name_tokens[0]
            experiment_name_token = name_tokens[1]
            timepoint_token = name_tokens[2]

            # Check burst
            burst_idx_token = None
            if '-' in name_tokens[3]:
                numofhours_token, burst_idx_token = name_tokens[3].split('-')
                numofhours_token = float(numofhours_token)
                burst_idx_token = int(burst_idx_token)
            else:
                numofhours_token = int(name_tokens[3])
                burst_idx_token = None
            well_id_token = name_tokens[4]

            # Find the Z-step marker position
            z_step_pos = None
            if i == 0:
                for idx, e in reversed(list(enumerate(name_tokens))):
                    if name_tokens[idx].isdigit():
                        continue
                    else:
                        try:
                            float(name_tokens[idx])
                            z_step_pos = idx
                        except ValueError:
                            continue
            # Include all the channel tokens
            channel_token =[]
            for c_i in range(6, z_step_pos-2+1):
                channel_token.append(name_tokens[c_i])
            channel_token = '_'.join(channel_token)
            z_idx_token = int(name_tokens[z_step_pos-1])

            # Split well id token to make sorting easier
            # Well ID example: H12
            experiment_well_key = (experiment_name_token, well_id_token[0], int(well_id_token[1:]))

            if experiment_well_key in stack_dict:
                stack_dict[experiment_well_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token])
            else:
                stack_dict[experiment_well_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token]]
    # Robo4 latest naming Robo0
    # Example: PID20160706_MerckMitoPlate23_T0_0-0_A11_1_Epi-DAPI_0.0_0_1.0.tif
    elif naming_type == 'robo0':
        for image_path in image_paths:
            image_name = os.path.basename(image_path)
            name_tokens = image_name.split('_')
            pid_token = name_tokens[0]
            experiment_name_token = name_tokens[1]
            timepoint_token = name_tokens[2]

            # Check burst
            burst_idx_token = None
            if '-' in name_tokens[3]:
                numofhours_token, burst_idx_token = name_tokens[3].split('-')
                numofhours_token = float(numofhours_token)
                burst_idx_token = int(burst_idx_token)
            else:
                numofhours_token = int(name_tokens[3])
                burst_idx_token = None
            well_id_token = name_tokens[4]

            channel_token = name_tokens[6]
            z_idx_token = int(name_tokens[8])

            # Split well id token to make sorting easier
            # Well ID example: H12
            experiment_well_key = (experiment_name_token, well_id_token[0], int(well_id_token[1:]))

            if experiment_well_key in stack_dict:
                stack_dict[experiment_well_key].append([image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token])
            else:
                stack_dict[experiment_well_key] = [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token]]
    else:
        print 'Unknowed RoboNumber!'
        sys.exit(0)

    return [stack_dict[ewkey] for ewkey in sorted(stack_dict)]







def create_biomed_db(db_path, table_name):
    conn = sqlite3.connect(db_path)
    with conn:
        cur = conn.cursor()
        cur.execute('DROP TABLE IF EXISTS {}'.format(table_name))
        # File name example: PID20150218_BioP7asynA_T2_24_G8_1_FITC-DFTrCy5_BG_MONTAGE_ALIGNED_CROPPED.tif
        # Keep in mind that SQLite uses a more general dynamic type system. In SQLite, the datatype of a value is associated with the value itself, not with its container. So be careful of type checking
        cur.execute('CREATE TABLE {} (FileName TEXT, ObjectID INT, PID TEXT, ExperimentName TEXT, WellID TEXT, Channel TEXT, TimePoint TEXT, NumOfHours REAL, ZIndex INT, BurstIndex INT, OriginalImagePath TEXT, EncodedMaskPath TEXT, ObjectCount INT, ObjectLabelsFound INT, MeasurementTag TEXT, BlobArea REAL, BlobPerimeter REAL, Radius REAL, BlobCentroidX INT, BlobCentroidY INT,  BlobCircularity REAL, Spread TEXT, Convexity INT, PixelIntensityMaximum INT, PixelIntensityMinimum INT, PixelIntensityMean REAL, PixelIntensityVariance REAL, PixelIntensityStdDev REAL, PixelIntensity1Percentile INT, PixelIntensity5Percentile INT, PixelIntensity10Percentile INT, PixelIntensity25Percentile INT, PixelIntensity50Percentile INT, PixelIntensity75Percentile INT, PixelIntensity90Percentile INT, PixelIntensity95Percentile INT, PixelIntensity99Percentile INT, PixelIntensitySkewness INT, PixelIntensityKurtosis INT, PixelIntensityInterquartileRange INT, ValidFromStartTimePoint INT, IsFirstImageOfTimepoint INT, DeadTimePoint TEXT, Phenotype INT, Live INT, Comment TEXT, PRIMARY KEY(FileName, ObjectID))'.format(table_name))


def insert_image_attributes_to_db(image_stack_experiment_well):
    ''' Worker process for single well
    args:
      image_stack_experiment_well: Each image stack is a list which contains all images in one well, e.g. [[image_path, pid_token, experiment_name_token, well_id_token, channel_token, timepoint_token, numofhours_token, z_idx_token, burst_idx_token], ...]
    returns:
      object attributes to insert to database
    '''
    # print multiprocessing.current_process(), image_stack_experiment_well[0][3]

    # Dictionary key by channel
    channel_dict = {}
    for tks in image_stack_experiment_well:
        if tks[4] in channel_dict:
            channel_dict[tks[4]].append(tks)
        else:
            channel_dict[tks[4]] = [tks]

    # Dictionary key by timepoint
    for ch in channel_dict:
        timepoint_dict = {}
        tks_list_in_channel = channel_dict[ch]
        for tks in tks_list_in_channel:
            if tks[5] in timepoint_dict:
                timepoint_dict[tks[5]].append(tks)
            else:
                timepoint_dict[tks[5]] = [tks]
        # Sort timepoint_dict by z_idx_token and burst_idx_token
        for t in timepoint_dict:
            # int(value or 0) will use 0 in the case when you provide any value that Python considers False, such as None, 0, [], "",
            timepoint_dict[t] = sorted(timepoint_dict[t], key=lambda x: (int(x[7] or 0), int(x[8] or 0)))
        channel_dict[ch] = timepoint_dict

    # Store each row in list in order to executemany at one time for each image
    values = []
    # Track the object ids start at start timepoint which will log in the column ValidFromStartTimePoint
    valid_object_ids_from_start_timepoint = None
    # Loop through channels then timepoints then z/burst
    for chl in channel_dict:
        # Timepoint dict for current specific channel
        channel_timepoint_dict = channel_dict[chl]

        # Sort Tx (e.g. T8) in order and loop
        sorted_channel_timepoint_keys = sorted(channel_timepoint_dict, key=lambda x: int(x[1:]))
        # Loop through all timepoints
        for idx, t in enumerate(sorted_channel_timepoint_keys):
            encoded_mask_filename_of_current_timepoint = None
            encoded_mask_path_of_current_timepoint = None
            encoded_mask_nparray_of_current_timepoint = None
            object_ids = None
            object_count = None

            # For all the z, bursts in current timepoint
            for ix, item in enumerate(channel_timepoint_dict[t]):
                # Get corresponding tokens
                original_image_path = item[0]
                pid = item[1]
                experiment_name = item[2]
                well_id = item[3]
                channel = item[4]
                timepoint = item[5]
                num_of_hours = item[6] # int
                z_idx = item[7]
                burst_idx = item[8]

                # Read in image as numpy
                # original_image_nparray = TIFF.open(original_image_path, mode='r').read_image()
                original_image_nparray = cv2.imread(original_image_path, -1)
                original_image_filename = os.path.basename(original_image_path)
                # Get the related encoded mask if it is the first image of current timepoint
                if ix == 0:
                    if chl == MORPHOLOGY_CHANNEL:
                        encoded_mask_filename_of_current_timepoint = original_image_filename.replace('.tif', '_CELLMASK_ENCODED.tif')
                    else:
                        encoded_mask_filename_of_current_timepoint = original_image_filename.replace('.tif', '_CELLMASK_ENCODED.tif').replace(chl, MORPHOLOGY_CHANNEL)
                    encoded_mask_path_of_current_timepoint = os.path.abspath(os.path.join(ENCODED_MASKS_PATH, encoded_mask_filename_of_current_timepoint))
                    if os.path.isfile(encoded_mask_path_of_current_timepoint):
                        # encoded_mask_nparray_of_current_timepoint = TIFF.open(encoded_mask_path_of_current_timepoint, mode='r').read_image()
                        encoded_mask_nparray_of_current_timepoint = cv2.imread(encoded_mask_path_of_current_timepoint, -1)
                        # Get all the unique object ids
                        object_ids = np.unique(encoded_mask_nparray_of_current_timepoint)
                        # Exclude value 0
                        object_ids = object_ids[object_ids!=0]
                        object_count = len(object_ids)
                        if valid_object_ids_from_start_timepoint is None and idx == 0:
                            valid_object_ids_from_start_timepoint = object_ids
                    else:
                        print 'Encoded mask for %s does not exist!' % original_image_filename
                        break
                for obj_id in object_ids:
                    '''
                    ObjectLabelsFound INT, MeasurementTag TEXT, BlobArea REAL, BlobPerimeter REAL, Radius REAL, BlobCentroidX INT, BlobCentroidY INT,  BlobCircularity REAL, Spread TEXT, Convexity INT, \
                    PixelIntensityMaximum INT, PixelIntensityMinimum INT, PixelIntensityMean REAL, PixelIntensityVariance REAL, PixelIntensityStdDev REAL, PixelIntensity1Percentile INT, PixelIntensity5Percentile INT, PixelIntensity10Percentile INT, PixelIntensity25Percentile INT, PixelIntensity50Percentile INT, PixelIntensity75Percentile INT, PixelIntensity90Percentile INT, PixelIntensity95Percentile INT, PixelIntensity99Percentile INT, PixelIntensitySkewness INT, PixelIntensityKurtosis INT, PixelIntensityInterquartileRange INT \
                    '''
                    # obj_id <type 'numpy.uint8'>, has to convert back to type 'int' first!
                    obj_id = int(obj_id)
                    obj_lables_fould = obj_id
                    measurement_tag = channel
                    # Make a copy of encoded mask for each object processing, so that the original one would not mess up for the next object
                    obj_mask = encoded_mask_nparray_of_current_timepoint.copy()
                    obj_mask[obj_mask != obj_id] = 0
                    obj_mask[obj_mask == obj_id] = 255
                    # Convert 16 bit object mask to 8 bit, since opencv only support 8 bit mask
                    # Be careful that there might be more than 8bit(256) objects, ie, obj_id can be more than 256, so make sure to do the excluding the other objects masking before this bit conversion
                    obj_mask = obj_mask.astype(np.uint8, copy=False) # copy=False to save memory
                    cnt = get_contour(obj_mask)
                    # If there are duplicate id for different cells/particles, skip
                    if cnt is None:
                        continue

                    blob_area = cv2.contourArea(cnt)
                    blob_perimeter = cv2.arcLength(cnt, True)

                    (x,y),radius = cv2.minEnclosingCircle(cnt)
                    blob_centroidX, blob_cnetroidY = (int(x),int(y))  # Almost always the same compared to compute_center function, sometimes 1 pixel diff
                    # Deal with ZeroDivisionError: float division by zero in python 2 in case blob_perimeter == 0
                    if blob_perimeter:
                        blob_circularity = 4*math.pi*blob_area/float((blob_perimeter**2))
                    else:
                        blob_circularity = 0

                    hull = cv2.convexHull(cnt)
                    hull_area = cv2.contourArea(hull)
                    if hull_area:
                        solidity = float(blob_area)/hull_area
                    else:
                        solidity = 0
                    spread = solidity

                    convexity =  cv2.isContourConvex(cnt)
                    if convexity:
                        convexity = 1
                    else:
                        convexity = 0
                    # Calculate the object intensity properties
                    # Get the object image with full intensity
                    # Note that we only care about the intensity inside the object, make sure to exclude the zero intensity points from calculation
                    obj_img = cv2.bitwise_and(original_image_nparray, original_image_nparray, mask=obj_mask)
                    # Careful! Use np.nonzero(obj_mask) instead of np.nonzero(obj_img)
                    obj_img_cnt_intensities = obj_img[np.nonzero(obj_mask)]

                    pixel_intensity_maximum = int(np.max(obj_img_cnt_intensities))   #  convert numpy.uint16 to int
                    pixel_intensity_minimum = int(np.min(obj_img_cnt_intensities))
                    # print pixel_intensity_maximum, pixel_intensity_minimum
                    # cv2.mean returns 4 elements tuple like (1632.808695652174, 0.0, 0.0, 0.0)
                    # Note the difference between cv2.mean(orig_img, mask=obj_mask)[0] and obj_img.mean()
                    # Method 1: No need to convert back to float
                    # pixel_intensity_mean = cv2.mean(orig_img, mask=obj_mask)[0]
                    # Method 2:
                    # pixel_intensity_mean = float(np.average(obj_img, weights=(obj_img>0))) # convert from numpy.float64 to float
                    # http://stackoverflow.com/questions/11084710/numpy-mean-with-condition, do not use weight here
                    pixel_intensity_mean = float(np.average(obj_img)) # convert from numpy.float64 to float




                    pixel_intensity_variance = float(np.var(obj_img_cnt_intensities))
                    pixel_intensity_stddev = float(np.std(obj_img_cnt_intensities))
                    pixel_intensity_1percentile = float(np.percentile(obj_img_cnt_intensities, 1))
                    pixel_intensity_5percentile = float(np.percentile(obj_img_cnt_intensities, 5))
                    pixel_intensity_10percentile = float(np.percentile(obj_img_cnt_intensities, 10))
                    pixel_intensity_25percentile = float(np.percentile(obj_img_cnt_intensities, 25))
                    pixel_intensity_50percentile = float(np.percentile(obj_img_cnt_intensities, 50))
                    pixel_intensity_75percentile = float(np.percentile(obj_img_cnt_intensities, 75))
                    pixel_intensity_90percentile = float(np.percentile(obj_img_cnt_intensities, 90))
                    pixel_intensity_95percentile = float(np.percentile(obj_img_cnt_intensities, 95))
                    pixel_intensity_99percentile = float(np.percentile(obj_img_cnt_intensities, 99))

                    pixel_intensity_skewness = stats.skew(obj_img_cnt_intensities)
                    pixel_intensity_kurtosis = stats.kurtosis(obj_img_cnt_intensities)
                    pixel_intensity_interquartilerange = pixel_intensity_75percentile - pixel_intensity_25percentile

                    if obj_id in valid_object_ids_from_start_timepoint:
                        valid_from_start_timepoint = 1
                    else:
                        valid_from_start_timepoint = 0
                    if ix == 0:
                        is_first_image_of_timepoint = 1
                    else:
                        is_first_image_of_timepoint = 0
                    dead_timepoint = None
                    phenotype = None
                    live = None
                    comment = None


                    '''
                    (FileName TEXT, ObjectID INT, PID TEXT, ExperimentName TEXT, WellID TEXT, Channel TEXT, TimePoint TEXT, NumOfHours REAL, ZIndex INT, BurstIndex INT, \ 10
                    OriginalImagePath TEXT, EncodedMaskPath TEXT, \ 2
                    ObjectCount INT, ObjectLabelsFound INT, MeasurementTag TEXT, BlobArea REAL, BlobPerimeter REAL, Radius REAL, BlobCentroidX INT, BlobCentroidY INT,  BlobCircularity REAL, Spread TEXT, Convexity INT, \ 11
                    PixelIntensityMaximum INT, PixelIntensityMinimum INT, PixelIntensityMean REAL, PixelIntensityVariance REAL, PixelIntensityStdDev REAL, PixelIntensity1Percentile INT, PixelIntensity5Percentile INT, PixelIntensity10Percentile INT, PixelIntensity25Percentile INT, PixelIntensity50Percentile INT, PixelIntensity75Percentile INT, PixelIntensity90Percentile INT, PixelIntensity95Percentile INT, PixelIntensity99Percentile INT, PixelIntensitySkewness INT, PixelIntensityKurtosis INT, PixelIntensityInterquartileRange INT, \ 17
                    ValidFromStartTimePoint INT, \1
                    FirstImageOfTimepoint INT, \1
                    DeadTimePoint TEXT, \ 1
                    Phenotype INT, Live INT, \ 2
                    PRIMARY KEY(FileName, ObjectID))
                    '''
                    # Append row

                    row_value = (original_image_filename, obj_id, pid, experiment_name, well_id, channel, timepoint, num_of_hours, z_idx, burst_idx, original_image_path, encoded_mask_path_of_current_timepoint, object_count, obj_lables_fould, measurement_tag, blob_area, blob_perimeter, radius, blob_centroidX, blob_cnetroidY, blob_circularity, spread, convexity, pixel_intensity_maximum, pixel_intensity_minimum, pixel_intensity_mean, pixel_intensity_variance, pixel_intensity_stddev, pixel_intensity_1percentile, pixel_intensity_5percentile, pixel_intensity_10percentile, pixel_intensity_25percentile, pixel_intensity_50percentile, pixel_intensity_75percentile, pixel_intensity_90percentile, pixel_intensity_95percentile, pixel_intensity_99percentile, pixel_intensity_skewness, pixel_intensity_kurtosis, pixel_intensity_interquartilerange, valid_from_start_timepoint, is_first_image_of_timepoint, dead_timepoint, phenotype, live, comment)
                    values.append(row_value)

    return values



def multiprocess_db_insert(input_image_stack_list):
    # The following problem only happens on Mac OSX.
    # Disable multithreading in OpenCV for main thread to avoid problems after fork
    # Otherwise any cv2 function call in worker process will hang!!
    # cv2.setNumThreads(0)

    # Initialize workers pool
    workers_pool = multiprocessing.Pool(processes=NUMBER_OF_PROCESSORS)

    # Use chunksize to speed up dispatching, and make sure chunksize is roundup integer
    chunk_size = int(math.ceil(len(input_image_stack_list)/float(NUMBER_OF_PROCESSORS)))

    # Feed data to workers in parallel
    # 99999 is timeout, can be 1 or 99999 etc, used for KeyboardInterrupt multiprocessing
    map_results = workers_pool.map_async(insert_image_attributes_to_db, input_image_stack_list, chunksize=chunk_size).get(99999)

    # Use pool.imap to avoid MemoryError, but KeyboardInterrupt not work
    # map_results = workers_pool.imap(insert_image_attributes_to_db, input_image_stack_list, chunksize=chunk_size)

    # # Single instance test
    # print insert_image_attributes_to_db(input_image_stack_list[0])


    # DB insertion
    conn = sqlite3.connect(DB_PATH)
    with conn:
        cur = conn.cursor()
        for r in map_results:
            cur.executemany('INSERT INTO {} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)'.format(TABLE_NAME), r)

    workers_pool.close()
    workers_pool.join()

def cache_object_stack(object_features_dict):
    ''' Worker process for single object
    args:
      object_feature_dict: {(timepoint, channel): object_features_tuple}
    returns:
      None
    '''

    # Tiff file for output
    tif_output = None
    cache_of_object = None
    pre_missing_cache_of_timepoint = []
    for t in SORTED_TIMEPOINT_SET_LIST:
        # Cache of each timepoint contains all the channels(original and all the masked channels)
        cache_of_timepoint = None
        for c in SORTED_CHANNEL_SET_LIST:
            object_features = object_features_dict[(t, c)]
            object_tag = object_features[0]

            # Only when image exists
            if object_features[1] is not None:
                object_id = object_features[1][0]
                experiment_name = object_features[1][1]
                well_id = object_features[1][2]
                channel = object_features[1][3]
                timepoint = object_features[1][4]
                num_of_hours = object_features[1][5]
                original_image_path = object_features[1][6]
                encoded_mask_path = object_features[1][7]
                blob_centroid_x = object_features[1][8]
                blob_centroid_y = object_features[1][9]

                # Get the output filename
                if tif_output is None and experiment_name is not None and well_id is not None and object_id is not None:
                    # Sring join only take one argument which is a list or tuple of words
                    output_filename = '_'.join(['cache', experiment_name, well_id, str(object_id), '_'.join(SORTED_CHANNEL_SET_LIST)]) + '.tif'
                    tif_output = TIFF.open(os.path.join(CACHE_PATH, output_filename), mode='w')

                # Calculate offset
                y0=int(blob_centroid_y-(CACHE_HEIGHT/2))
                y_needs_padding = False
                offset_y = 0 # add padding
                # Deal with underrage padding, will deal with overrage padding later
                if y0<0:
                    y_needs_padding = True
                    offset_y = -y0
                    y0=0

                x0=int(blob_centroid_x-(CACHE_WIDTH/2))
                x_needs_padding = False
                offset_x = 0 # add padding
                # Deal with underrage padding, will deal with overrage padding later
                if x0<0:
                    x_needs_padding = True
                    offset_x = -x0
                    x0=0


            # cache current channel in current timepoint
            cache_of_channel = None
            if object_tag == 'object_tracked':
                # Read the original and encoded mask images
                # image_of_channel = TIFF.open(original_image_path, mode='r').read_image()
                image_of_channel = cv2.imread(original_image_path, -1)
                # encoded_mask_of_channel = TIFF.open(encoded_mask_path, mode='r').read_image()
                encoded_mask_of_channel = cv2.imread(encoded_mask_path, -1)

                # Deal with overrage padding
                if image_of_channel.shape[0] < y0 + CACHE_HEIGHT - offset_y:
                    y_needs_padding = True
                if image_of_channel.shape[1] < x0 + CACHE_WIDTH - offset_x:
                    x_needs_padding = True

                # Crop ROI of image and encoded mask
                # Example: crop_img = img[200:400, 100:300] # Crop from x, y, w, h -> 100, 200, 300, 400
                # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
                cropped_image_of_channel = image_of_channel[y0:y0+CACHE_HEIGHT-offset_y, x0:x0+CACHE_WIDTH-offset_x]


                # Fill with zeros to keep dimension
                if y_needs_padding or x_needs_padding:
                    # Padding original image
                    image_zeros = np.zeros((CACHE_HEIGHT,CACHE_WIDTH), dtype=DTYPE)
                    image_zeros[offset_y:offset_y+cropped_image_of_channel.shape[0], offset_x:offset_x+cropped_image_of_channel.shape[1]] = cropped_image_of_channel
                    cropped_image_of_channel = image_zeros

                # If it is morphology channel, include the original image and masked image
                if c == MORPHOLOGY_CHANNEL:
                    cropped_object_mask_of_channel = encoded_mask_of_channel[y0:y0+CACHE_HEIGHT-offset_y, x0:x0+CACHE_WIDTH-offset_x]

                    if y_needs_padding or x_needs_padding:
                        # Padding encoded mask
                        encoded_mask_zeros = np.zeros((CACHE_HEIGHT,CACHE_WIDTH), dtype=DTYPE)
                        encoded_mask_zeros[offset_y:offset_y+cropped_object_mask_of_channel.shape[0], offset_x:offset_x+cropped_object_mask_of_channel.shape[1]] = cropped_object_mask_of_channel
                        cropped_object_mask_of_channel = encoded_mask_zeros

                    # Get the object mask
                    cropped_object_mask_of_channel[cropped_object_mask_of_channel!=object_id] = 0
                    # change mask to int 8 in place because cv2.bitwise only supports 8-bit single channel array
                    cropped_object_mask_of_channel = cropped_object_mask_of_channel.astype(np.uint8, copy=False)

                    # Take only specified region of the object
                    cropped_object_mask_of_channel = cv2.bitwise_and(cropped_image_of_channel, cropped_image_of_channel, mask=cropped_object_mask_of_channel)

                    # Concatenates
                    cache_of_channel = np.concatenate((cropped_image_of_channel, cropped_object_mask_of_channel), axis=1)
                else:
                    # cache_of_channel = cropped_object_mask_of_channel
                    cache_of_channel = cropped_image_of_channel



            elif object_tag == 'object_not_tracked':
                # Read the original and encoded mask images
                # image_of_channel = TIFF.open(original_image_path, mode='r').read_image()
                image_of_channel = cv2.imread(original_image_path, -1)
                cropped_image_of_channel = image_of_channel[y0:y0+CACHE_HEIGHT-offset_y, x0:x0+CACHE_WIDTH-offset_x]


                # Deal with overrage padding
                if image_of_channel.shape[0] < y0 + CACHE_HEIGHT - offset_y:
                    y_needs_padding = True
                if image_of_channel.shape[1] < x0 + CACHE_WIDTH - offset_x:
                    x_needs_padding = True

                if y_needs_padding or x_needs_padding:
                    # Padding original image
                    image_zeros = np.zeros((CACHE_HEIGHT,CACHE_WIDTH), dtype=DTYPE)
                    image_zeros[offset_y:offset_y+cropped_image_of_channel.shape[0], offset_x:offset_x+cropped_image_of_channel.shape[1]] = cropped_image_of_channel
                    cropped_image_of_channel = image_zeros

                # For morphology channel
                if c == MORPHOLOGY_CHANNEL:
                    # If it is not tracked, let the mask be all black
                    cropped_object_mask_of_channel = np.zeros((CACHE_HEIGHT,CACHE_WIDTH), dtype=DTYPE)
                    cache_of_channel = np.concatenate((cropped_image_of_channel, cropped_object_mask_of_channel), axis=1)
                # Show other channels' ROI anyway, even lost track
                else:
                    cache_of_channel = cropped_image_of_channel

            elif object_tag == 'image_not_exist':
                if c == MORPHOLOGY_CHANNEL:
                    cache_of_channel = np.zeros((CACHE_HEIGHT, CACHE_WIDTH*2), dtype=DTYPE)
                else:
                    cache_of_channel = np.zeros((CACHE_HEIGHT, CACHE_WIDTH), dtype=DTYPE)

            else:
                print 'Tag not exist'

            # Concatenate cache of channels in current timepoint
            if cache_of_timepoint is None:
                cache_of_timepoint = cache_of_channel
            else:
                cache_of_timepoint = np.concatenate((cache_of_timepoint, cache_of_channel), axis=1)


        # Write each timepoint slice. Note the compression parameter
        # None type check, because sometimes first timepoint/timepoints does not exist
        if tif_output is not None:
            if pre_missing_cache_of_timepoint:
                for pmt in pre_missing_cache_of_timepoint:
                    tif_output.write_image(pmt, compression='lzw')
            pre_missing_cache_of_timepoint = []
            tif_output.write_image(cache_of_timepoint, compression='lzw')
        else:
            pre_missing_cache_of_timepoint.append(cache_of_timepoint)

    # flushes data to disk
    del tif_output








def multiprocess_stack_cache():
    # Must have global keyword to modify global variables
    global SORTED_TIMEPOINT_SET_LIST
    global SORTED_CHANNEL_SET_LIST

    # list for multiprocessing
    objects_feature_dict = {}
    # Fetch all the rows for caching
    conn = sqlite3.connect(DB_PATH)
    with conn:
        cur = conn.cursor()
        cur.execute("SELECT DISTINCT TimePoint FROM {}".format(TABLE_NAME))
        timepoint_set_list = [record[0] for record in cur.fetchall()]
        timepoint_set_list = sorted(timepoint_set_list, key=lambda x: int(x[1:]))
        SORTED_TIMEPOINT_SET_LIST = timepoint_set_list
        cur.execute("SELECT DISTINCT Channel FROM {}".format(TABLE_NAME))
        channel_set_list = [record[0] for record in cur.fetchall()]
        channel_set_list = sorted(channel_set_list)
        channel_set_list.remove(MORPHOLOGY_CHANNEL)
        channel_set_list.insert(0, MORPHOLOGY_CHANNEL)
        SORTED_CHANNEL_SET_LIST = channel_set_list
        cur.execute("SELECT ObjectID, ExperimentName, WellID, Channel, TimePoint, NumOfHours, OriginalImagePath, EncodedMaskPath, BlobCentroidX, BlobCentroidY FROM {} WHERE ValidFromStartTimePoint = 1 AND IsFirstImageOfTimepoint = 1 ORDER BY ExperimentName, substr(WellID,1, 1), abs(substr(WellID, 2)), ObjectID, Channel, NumOfHours".format(TABLE_NAME))
        records = cur.fetchall()
        for record in records:
            object_id = record[0]
            experiment_name = record[1]
            well_id = record[2]
            channel = record[3]
            timepoint = record[4]

            experiment_well_object_key = (experiment_name, well_id[0], int(well_id[1:]), object_id)
            if experiment_well_object_key in objects_feature_dict:
                objects_feature_dict[experiment_well_object_key][(timepoint, channel)] = ('object_tracked', record)
            else:
                objects_feature_dict[experiment_well_object_key] = {(timepoint, channel): ('object_tracked', record)}
        # Get object_not_tracked and image_not_exist as well`
        for k in objects_feature_dict:
            for t_index, t in enumerate(SORTED_TIMEPOINT_SET_LIST):
                for c in SORTED_CHANNEL_SET_LIST:
                    if (t, c) not in objects_feature_dict[k]:
                        # Note the second parameter of cur.excute() should be tuple!
                        cur.execute("SELECT DISTINCT OriginalImagePath, EncodedMaskPath FROM {} WHERE ExperimentName = ? AND WellID = ?  AND Channel = ? AND TimePoint = ? AND IsFirstImageOfTimepoint = 1".format(TABLE_NAME), (k[0], k[1]+str(k[2]), c, t))
                        object_not_tracked_record = cur.fetchone()
                        if object_not_tracked_record is not None:
                            # If image of previous timepoint does not exist, mark the current one as 'image_not_exist' instead of 'object_not_tracked'
                            # Check (SORTED_TIMEPOINT_SET_LIST[t_index-1], c) to make sure no KeyError. Because if duplicate cell id in first time point which is skipped, it is possible first timepoint is missing at the first place.
                            if (SORTED_TIMEPOINT_SET_LIST[t_index-1], c) in objects_feature_dict[k] and objects_feature_dict[k][(SORTED_TIMEPOINT_SET_LIST[t_index-1], c)][0] != 'image_not_exist':
                                adjusted_object_not_tracked_record = [k[3], k[0], k[1]+str(k[2]), c, t, None, object_not_tracked_record[0], object_not_tracked_record[1], objects_feature_dict[k][(SORTED_TIMEPOINT_SET_LIST[t_index-1], c)][1][8], objects_feature_dict[k][(SORTED_TIMEPOINT_SET_LIST[t_index-1], c)][1][9]]
                                objects_feature_dict[k][(t, c)] = ('object_not_tracked', adjusted_object_not_tracked_record)
                            else:
                                # Even there is record of other cell's image in the database, there is no x and y coordinates from previous timepoint, thus we treat it as image_not_exist
                                objects_feature_dict[k][(t, c)] = ('image_not_exist', None)

                        else:
                            # When none of the other cells in the well are tracked at this timepoint, which means no record in the database. we treat it as image not exist
                            objects_feature_dict[k][(t, c)] = ('image_not_exist', None)
    objects_feature_list = [objects_feature_dict[ewokey] for ewokey in sorted(objects_feature_dict)]

    # The following problem only happens on Mac OSX.
    # Disable multithreading in OpenCV for main thread to avoid problems after fork
    # Otherwise any cv2 function call in worker process will hang!!
    # cv2.setNumThreads(0)

    # Initialize workers pool
    workers_pool = multiprocessing.Pool(processes=NUMBER_OF_PROCESSORS)

    # Use chunksize to speed up dispatching, and make sure chunksize is roundup integer
    chunk_size = int(math.ceil(len(objects_feature_list)/float(NUMBER_OF_PROCESSORS)))

    # Feed data to workers in parallel
    # 99999 is timeout, can be 1 or 99999 etc, used for KeyboardInterrupt multiprocessing
    map_results = workers_pool.map_async(cache_object_stack, objects_feature_list, chunksize=chunk_size).get(99999)

    # Use pool.imap to avoid MemoryError, but KeyboardInterrupt not work
    # map_results = workers_pool.imap(cache_object_stack, objects_feature_list, chunksize=chunk_size)

    # # Single instance test
    # print cache_object_stack(objects_feature_list[0])
    # for i in objects_feature_list:
    #     cache_object_stack(i)

    workers_pool.close()
    workers_pool.join()










def main():
    # If you want to simply access/read a global variable you just use its name. However to change its value you need to use the global keyword.
    # you need to use the global keyword to let the interpreter know that you refer to the global variable done,
    # otherwise it's going to create a different one who can only be read in the function.
    global EXPERIMENT_NAME
    global DB_PATH
    global CACHE_PATH

    # Get input
    input_image_stack_list = get_image_tokens_list(ORIGIANL_IAMGES_PATH, NAMING_TYPE)
    # get global experiment name
    EXPERIMENT_NAME = input_image_stack_list[0][0][2]

    # Get db and cache paths
    # Because SQLite has lock conflict with NFS, we store db file to local file system, then copy it back to NFS.
    DB_PATH = os.path.join(SQLITE_DB_TMP_FOLDER, 'db_of_'+EXPERIMENT_NAME+'.db')
    CACHE_PATH = os.path.join(DB_AND_CACHE_PATH, 'cache_of_'+EXPERIMENT_NAME)
    # Create cache directory
    try:
        os.makedirs(CACHE_PATH)
    except OSError:
        pass

    # Create DataBase structure without data
    create_biomed_db(DB_PATH, TABLE_NAME)

    # Insert calculated attributes to db using multiprocessing
    multiprocess_db_insert(input_image_stack_list)

    # Cache images for labeling
    multiprocess_stack_cache()

    # Move local db file back to NFS file system. shutil.move may have permission issue
    # shutil.move(DB_PATH, os.path.join(DB_AND_CACHE_PATH, 'db_of_'+EXPERIMENT_NAME+'.db'))
    call(['cp', DB_PATH, os.path.join(DB_AND_CACHE_PATH, 'db_of_'+EXPERIMENT_NAME+'.db')])
    call(['rm', DB_PATH])


if __name__ == '__main__':
    # Start timer
    start_time = datetime.datetime.utcnow()


    # # --- For terminal Test ---
    # NAMING_TYPE = 'robo3'
    # MORPHOLOGY_CHANNEL = 'RFP-DFTrCy5'
    # # ORIGIANL_IAMGES_PATH = '/Users/guangzhili/GladStone/warm_up_exercises/img_all_test/CroppedImages'
    # # ENCODED_MASKS_PATH = '/Users/guangzhili/GladStone/warm_up_exercises/img_all_test/CellMasks'
    # # DB_AND_CACHE_PATH = '/Users/guangzhili/GladStone/CellLabeler/CellLabelerTest'

    # ORIGIANL_IAMGES_PATH = '/media/robodata/Guangzhi/CellLabeler/RoboImagesForTest/BioP7asynA/CroppedImages_A1'
    # ENCODED_MASKS_PATH = '/media/robodata/Guangzhi/CellLabeler/RoboImagesForTest/BioP7asynA/CellMasks'
    # DB_AND_CACHE_PATH = '/media/robodata/Guangzhi/CellLabeler/RoboImagesForTest/BioP7asynA/db_and_cache'
    # main()





    # --- For Galaxy run ---
    parser = argparse.ArgumentParser()
    parser.add_argument('naming_type')
    parser.add_argument('morphology_channel')
    parser.add_argument("original_images_path")
    parser.add_argument("encoded_masks_path")
    parser.add_argument("side_length", type=int)
    parser.add_argument('db_and_cache_path')
    parser.add_argument("out_report")
    args = parser.parse_args()

    # Get args from argparse
    NAMING_TYPE = args.naming_type
    MORPHOLOGY_CHANNEL = args.morphology_channel
    ORIGIANL_IAMGES_PATH = args.original_images_path
    ENCODED_MASKS_PATH = args.encoded_masks_path
    CACHE_WIDTH = args.side_length
    CACHE_HEIGHT = args.side_length
    DB_AND_CACHE_PATH = args.db_and_cache_path
    out_report = args.out_report

    # Check if both input paths exist
    if not os.path.isdir(ORIGIANL_IAMGES_PATH):
        print ORIGIANL_IAMGES_PATH + ' does NOT exist!'
        sys.exit(1)
    if not os.path.isdir(ENCODED_MASKS_PATH):
        print ENCODED_MASKS_PATH + ' does NOT exist!'
        sys.exit(1)

    # Create output folder if not exist yet
    try:
        os.makedirs(DB_AND_CACHE_PATH)
    except OSError:
        if not os.path.isdir(DB_AND_CACHE_PATH):
            print 'Invalid path format!'
            sys.exit(1)

    # Run process
    main()

    # Write output text file to Galaxy
    with open(out_report, 'wb') as f:
        f.write("Created database and cache location: %s\n" % DB_AND_CACHE_PATH)





    # Print Total elapsed time
    end_time = datetime.datetime.utcnow()
    print 'Total elapsed time for Cell Labeler DB And Cache Creation:', end_time-start_time







