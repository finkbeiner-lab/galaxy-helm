# # ----Input/output considerations----
# Input experiment to process by passing two paths:
#   Encoded masks
#   Corresponding files with original intensities
# Parse the filename for tokens correctly
# Match mask/image file names
# Add to database
# Output a table can be queried by: 
#   Well
#   Time point
#   Experiment name
#   PID
#   Number of hours
#   Channel
#   Live/dead label 
#   Neuron/not neuron label

# # ----Database side------------------
# Scale up database to full well or full plate
#   Add all the tokens to fields:
#   Sci_PlateID, Sci_WellID, RowID, ColumnID, Timepoint
#   Add all object properties to fields: 
#   ObjectCount, ObjectLabelsFound, MeasurementTag, 
#   BlobArea, BlobPerimeter, Radius, BlobCentroidX, BlobCentroidY
#   BlobCircularity, Spread, Convexity
#   Add intensity info as well:
#   PixelIntensityMaximum, PixelIntensityMinimum, PixelIntensityMean, 
#   PixelIntensityVariance, PixelIntensityStdDev, PixelIntensity1Percentile, 
#   PixelIntensity5Percentile, PixelIntensity10Percentile, PixelIntensity25Percentile, 
#   PixelIntensity50Percentile, PixelIntensity75Percentile, PixelIntensity90Percentile, 
#   PixelIntensity95Percentile, PixelIntensity99Percentile, PixelIntensitySkewness, PixelIntensityKurtosis,
#   PixelIntensityInterquartileRange 

# In my Python 2 modules, I almost always import division from __future__, so that I can't get caught out by accidentally passing integers to a division operation I don't expect to truncate
from __future__ import division
import sqlite3 as lite
import sys
import ntpath
import os
import cv2
import numpy as np
import copy
import time
import math
from scipy import stats
import argparse
from operator import itemgetter
import warnings
# stats.skew throws nonsense runtimewarning on object 93 of image PID20150217_BioP7asynA_T0_0_F7_1_FITC-DFTrCy5_BG_MONTAGE_ALIGNED_CROPPED.tif
# while executing pixel_intensity_skewness = stats.skew(obj_img_nonzero)
# Error message as folows:
# /Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages/scipy/stats/stats.py:993: RuntimeWarning: 
# invalid value encountered in double_scalars
# vals = np.where(zero, 0, m3 / m2**1.5)
# In order to depress the error on history pane of Galaxy, suppress the RuntimeWarning message
warnings.filterwarnings("ignore", category=RuntimeWarning)

# In order to run in Galaxy, the path must be absolute
# DB_NAME = '/Users/guangzhili/GladStone/galaxy-dev/tools/database_tools/BioImagesDB/bio_images_all.db'   
# DB_NAME = '/Users/guangzhili/GladStone/bio_images_t.db'
# DB_NAME = '/Users/guangzhili/GladStone/bio_images_sss.db'
DB_NAME = '/media/robodata/Guangzhi/ImagesDB/bio_images_all.db'   
# TABLE_NAME = 'Images'
# IMAGES_PATH = "/Users/guangzhili/GladStone/img_all/CroppedImages"
# MASKS_PATH = "/Users/guangzhili/GladStone/img_all/CellMasks"



def create_images_db(db_name, table_name):
    conn = lite.connect(db_name)
    with conn:
        cur = conn.cursor()
        # cur.execute("DROP TABLE IF EXISTS " + table_name)
        # File name example: PID20150218_BioP7asynA_T2_24_G8_1_FITC-DFTrCy5_BG_MONTAGE_ALIGNED_CROPPED.tif
        # Keep in mind that SQLite uses a more general dynamic type system. In SQLite, the datatype of a value is associated with the value itself, not with its container. So be careful of type checking
        cur.execute("CREATE TABLE IF NOT EXISTS " + table_name + "(FileName TEXT, ObjectID INT, PID TEXT, ExperimentName TEXT, TimePoint TEXT, NumOfHours INT, WellID TEXT, Channel TEXT, \
            OriginalImgPath TEXT, EncodedMaskPath TEXT, \
            ObjectCount INT, ObjectLabelsFound INT, MeasurementTag TEXT, BlobArea REAL, BlobPerimeter REAL, Radius REAL, BlobCentroidX INT, BlobCentroidY INT,  BlobCircularity REAL, Spread TEXT, Convexity INT, \
            PixelIntensityMaximum INT, PixelIntensityMinimum INT, PixelIntensityMean REAL, PixelIntensityVariance REAL, PixelIntensityStdDev REAL, PixelIntensity1Percentile INT, PixelIntensity5Percentile INT, PixelIntensity10Percentile INT, PixelIntensity25Percentile INT, PixelIntensity50Percentile INT, PixelIntensity75Percentile INT, PixelIntensity90Percentile INT, PixelIntensity95Percentile INT, PixelIntensity99Percentile INT, PixelIntensitySkewness INT, PixelIntensityKurtosis INT, PixelIntensityInterquartileRange INT, \
            ValidFromT0 INT, \
            DeadTimePoint TEXT, \
            Phenotype INT, Live INT, \
            PRIMARY KEY(FileName, ObjectID))")
        




def path_leaf(abs_path):
    ''' Extract file name from abslute path'''
    head, tail = ntpath.split(abs_path)
    return tail or ntpath.basename(head)


def get_unique_ids(encoded_mask_abspath):
    ''' Get all the unique object ids'''
    img = cv2.imread(encoded_mask_abspath, cv2.CV_LOAD_IMAGE_UNCHANGED)
    ids = np.unique(img)
    # Exclude value 0
    ids = ids[ids != 0]
    return ids

def get_contours(thresh_img):
    # Find the contours in the mark, cv2.findContours only support 8-bit single-channel image
    (cnts, _) = cv2.findContours(thresh_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print "Fould %d objects." %(len(cnts))
    return cnts

def get_contour(obj_mask):
    # Be careful it might get zero area contour, and the contour we interested may not be the first one
    cnts = get_contours(obj_mask)
    cnt = None
    cnt_area = None
    # Only get the largest contour  
    for c in cnts:
        c_area = cv2.contourArea(c)
        if cnt is not None:
            if c_area > cnt_area:
                cnt = c
                cnt_area = c_area
        else:
            cnt = c
            cnt_area = c_area
                
         
    return cnt          

def compute_center(cnt):
    m = cv2.moments(cnt)

    cx = int(m["m10"] / m["m00"])
    cy = int(m["m01"] / m["m00"])
    return cx, cy


def insert_imgs_to_db(db_name, table_name, img_folder, encoded_mask_folder):

    img_names = [name for name in os.listdir(img_folder) if name.endswith('.tif')]
    i = 0
    

    # Connect to db
    conn = lite.connect(db_name)
    with conn:
        cur = conn.cursor()
        # Abandon the images that are already exist in database
        # First retrieve the exist file names from the database
        # Then compare them to the input file names
        imgs_in_db = []
        # Cannot use SQL parameters to be placeholders in SQL objects, so ? will not work for table_name, have to use {} and format
        # GROUP BY FileName to get distinct FileName in order
        for r in cur.execute('SELECT FileName FROM {} GROUP BY FileName'.format(table_name)):
            imgs_in_db.append(r[0])
        # img_names = [im for im in img_names if im not in imgs_in_db]
        img_names_tokens = []
        for im in img_names:
            # Exclude already exist images
            if im not in imgs_in_db:
                # Extract tokens from the file name
                # FileName TEXT, PID TEXT, ExperimentName TEXT, TimePoint TEXT, NumOfHours INT, WellID TEXT, Channel TEXT
                # e.g PID20150218_BioP7asynA_T2_24_G8_1_FITC-DFTrCy5_BG_MONTAGE_ALIGNED_CROPPED.tif
                name_tokens = im.split('_')
                pid_token = name_tokens[0]
                experiment_name_token = name_tokens[1]
                timepoint_token = name_tokens[2]
                num_of_hours_token = int(name_tokens[3])
                well_id_token = name_tokens[4]
                channel_token = name_tokens[6]
                img_names_tokens.append((im, pid_token, experiment_name_token, timepoint_token, num_of_hours_token, well_id_token, channel_token))
        # We only care about the object ids exist from T0 according to the experiment purpose, all the objects starts existing later then T0 will be get rid of
        # Sort the list by number of hours in order to find the objects exist in T0 first in loop, order by(experiment, well, channel, hours)
        img_names_tokens = sorted(img_names_tokens, key=itemgetter(2, 5, 6, 4))
                
        
        img_count = len(img_names_tokens)

        # Dict to keep track of all the valid object ids(which start showing at T0) for each (experiment and well)
        valid_objids = {}
        
        for img_tokens in img_names_tokens:
            
            # Start timer for current image processing 
            start = time.time()

            # Get corresponding tokens
            img_name = img_tokens[0]
            pid = img_tokens[1]
            experiment_name = img_tokens[2]
            timepoint = img_tokens[3]
            num_of_hours = img_tokens[4] # int
            well_id = img_tokens[5]
            channel = img_tokens[6]

            # Get the encoded mask name
            # e.g. PID20150217_BioP7asynA_T0_0_D10_1_RFP-DFTrCy5_BG_MONTAGE_ALIGNED_CROPPED_CELLMASK_ENCODED.tif
            # Mask only in RFP channel, and can apply to all other channels
            if 'RFP' in channel:
                en_mask_name = img_name.replace('.tif', '_CELLMASK_ENCODED.tif')
            else:
                en_mask_name = img_name.split('.')[0].replace(channel, 'RFP-DFTrCy5') + '_CELLMASK_ENCODED.tif' 

            # Check if the corresponding encoded mask file exist
            encoded_mask_abspath = os.path.abspath(os.path.join(encoded_mask_folder, en_mask_name))
            if os.path.isfile(encoded_mask_abspath):
                orig_img_abspath = os.path.abspath(os.path.join(img_folder, img_name))
                orig_img = cv2.imread(orig_img_abspath, cv2.CV_LOAD_IMAGE_UNCHANGED)
                encoded_mask = cv2.imread(encoded_mask_abspath, cv2.CV_LOAD_IMAGE_UNCHANGED)

                # Get all the object ids in the mask
                obj_ids = get_unique_ids(encoded_mask_abspath)
                # Only the object ids start at T0 will be useful, track and log in the column ValidFromT0
                valid_objids_for_cur_experiment_well_channel = None
                if num_of_hours == 0:
                    valid_objids[(experiment_name, well_id, channel)] = obj_ids
                    valid_objids_for_cur_experiment_well_channel = obj_ids
                else:
                    valid_objids_for_cur_experiment_well_channel = valid_objids.get((experiment_name, well_id, channel), None)
                    if valid_objids_for_cur_experiment_well_channel is None:
                        # Handling the possibility that user may NOT insert the whole experiment images at once. Like insert T0, T1, then T2,T3. In this case we have to query and retrieve the encodeMaskPath at T0 to get the valid ids first.
                        # If valid object ids not yet exist in the dict, retrieve the T0 encoded mask to get the valid ids
                        cur.execute("SELECT DISTINCT EncodedMaskPath FROM {} WHERE ExperimentName = ? AND WellID = ? AND TimePoint = 'T0' AND Channel = ?".format(table_name), (experiment_name, well_id, channel))
                        encoded_mask_path_t0 = cur.fetchone()[0]                        
                        valid_objids_for_cur_experiment_well_channel = get_unique_ids(encoded_mask_path_t0)
                        # obj_ids = [oid for oid in obj_ids if oid in valid_objids_for_cur_experiment_well_channel]
                        valid_objids[(experiment_name, well_id, channel)] = valid_objids_for_cur_experiment_well_channel



                # The number of valid objects at current (experiment, well, timepoint, channel)
                obj_count = len(obj_ids)
                
                # Store each row in list in order to executemany at one time for each image
                values = []
                # For current file constraining (experiment, well, timepoint, channel), insert all the valid objects
                for obj_id in obj_ids:
                    '''
                    ObjectCount INT, ObjectLabelsFound INT, MeasurementTag TEXT, BlobArea REAL, BlobPerimeter REAL, Radius REAL, BlobCentroidX INT, BlobCentroidY INT,  BlobCircularity REAL, Spread TEXT, Convexity INT, \
                    PixelIntensityMaximum INT, PixelIntensityMinimum INT, PixelIntensityMean REAL, PixelIntensityVariance REAL, PixelIntensityStdDev REAL, PixelIntensity1Percentile INT, PixelIntensity5Percentile INT, PixelIntensity10Percentile INT, PixelIntensity25Percentile INT, PixelIntensity50Percentile INT, PixelIntensity75Percentile INT, PixelIntensity90Percentile INT, PixelIntensity95Percentile INT, PixelIntensity99Percentile INT, PixelIntensitySkewness INT, PixelIntensityKurtosis INT, PixelIntensityInterquartileRange INT \
                    '''
                    # obj_id <type 'numpy.uint8'>, has to convert back to type 'int' first!
                    obj_id = int(obj_id)
                    obj_lables_fould = obj_id
                    measurement_tag = channel
                    # Make a copy of encoded mask for each object processing, so that the original one would not mess up for the next object
                    obj_mask = encoded_mask.copy() 
                    obj_mask[obj_mask != obj_id] = 0
                    obj_mask[obj_mask == obj_id] = 255
                    # Convert 16 bit object mask to 8 bit, since opencv only support 8 bit mask
                    # Be careful that there might be more than 8bit(256) objects, ie, obj_id can be more than 256, so make sure to do the excluding the other objects masking before this bit conversion
                    obj_mask = obj_mask.astype(np.uint8, copy=False) # copy=False to save memory
                    
                    cnt = get_contour(obj_mask)
                    blob_area = cv2.contourArea(cnt)
                    blob_perimeter = cv2.arcLength(cnt, True)
                    # print "img_name", img_name
                    # print "object_id", obj_id
                    # print "area:", blob_area
                    # print "blob_perimeter:", blob_perimeter
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
                    obj_img = cv2.bitwise_and(orig_img, orig_img, mask=obj_mask)
                    obj_img_nonzero = obj_img[np.nonzero(obj_img)]
                 
                    pixel_intensity_maximum = int(np.max(obj_img_nonzero))   #  convert numpy.uint16 to int
                    pixel_intensity_minimum = int(np.min(obj_img_nonzero))
                    # print pixel_intensity_maximum, pixel_intensity_minimum

                    # cv2.mean returns 4 elements tuple like (1632.808695652174, 0.0, 0.0, 0.0)
                    # Note the difference between cv2.mean(orig_img, mask=obj_mask)[0] and obj_img.mean()
                    # Method 1: No need to convert back to float
                    # pixel_intensity_mean = cv2.mean(orig_img, mask=obj_mask)[0]
                    # Method 2: 
                    pixel_intensity_mean = float(np.average(obj_img, weights=(obj_img>0))) # convert from numpy.float64 to float
                 


                    pixel_intensity_variance = float(np.var(obj_img_nonzero))
                    pixel_intensity_stddev = float(np.std(obj_img_nonzero))
                    pixel_intensity_1percentile = float(np.percentile(obj_img_nonzero, 1))
                    pixel_intensity_5percentile = float(np.percentile(obj_img_nonzero, 5))
                    pixel_intensity_10percentile = float(np.percentile(obj_img_nonzero, 10))
                    pixel_intensity_25percentile = float(np.percentile(obj_img_nonzero, 25))
                    pixel_intensity_50percentile = float(np.percentile(obj_img_nonzero, 50))
                    pixel_intensity_75percentile = float(np.percentile(obj_img_nonzero, 75))
                    pixel_intensity_90percentile = float(np.percentile(obj_img_nonzero, 90))
                    pixel_intensity_95percentile = float(np.percentile(obj_img_nonzero, 95))
                    pixel_intensity_99percentile = float(np.percentile(obj_img_nonzero, 99))

                    pixel_intensity_skewness = stats.skew(obj_img_nonzero)
                    pixel_intensity_kurtosis = stats.kurtosis(obj_img_nonzero)
                    pixel_intensity_interquartilerange = pixel_intensity_75percentile - pixel_intensity_25percentile

                    if obj_id in valid_objids_for_cur_experiment_well_channel:
                        valid_from_t0 = 1
                    else:
                        valid_from_t0 = 0    
                    dead_timepoint = None
                    phenotype = None
                    live = None
                                    
                     
                    '''
                    (FileName TEXT, ObjectID INT, PID TEXT, ExperimentName TEXT, TimePoint TEXT, NumOfHours INT, WellID TEXT, Channel TEXT, \ 8
                    OriginalImgPath TEXT, EncodedMaskPath TEXT, \ 2
                    ObjectCount INT, ObjectLabelsFound INT, MeasurementTag TEXT, BlobArea REAL, BlobPerimeter REAL, Radius REAL, BlobCentroidX INT, BlobCentroidY INT,  BlobCircularity REAL, Spread TEXT, Convexity INT, \ 11
                    PixelIntensityMaximum INT, PixelIntensityMinimum INT, PixelIntensityMean REAL, PixelIntensityVariance REAL, PixelIntensityStdDev REAL, PixelIntensity1Percentile INT, PixelIntensity5Percentile INT, PixelIntensity10Percentile INT, PixelIntensity25Percentile INT, PixelIntensity50Percentile INT, PixelIntensity75Percentile INT, PixelIntensity90Percentile INT, PixelIntensity95Percentile INT, PixelIntensity99Percentile INT, PixelIntensitySkewness INT, PixelIntensityKurtosis INT, PixelIntensityInterquartileRange INT, \ 17
                    ValidFromT0 INT, \1
                    DeadTimePoint TEXT, \ 1
                    Phenotype INT, Live INT, \ 2
                    PRIMARY KEY(FileName, ObjectID))
                    '''
                    # Append row

                    values.append((img_name, obj_id, pid, experiment_name, timepoint, num_of_hours, well_id, channel, orig_img_abspath, encoded_mask_abspath, obj_count, obj_lables_fould, measurement_tag, blob_area, blob_perimeter, radius, blob_centroidX, blob_cnetroidY, blob_circularity, spread, convexity, pixel_intensity_maximum, pixel_intensity_minimum, pixel_intensity_mean, pixel_intensity_variance, pixel_intensity_stddev, pixel_intensity_1percentile, pixel_intensity_5percentile, pixel_intensity_10percentile, pixel_intensity_25percentile, pixel_intensity_50percentile, pixel_intensity_75percentile, pixel_intensity_90percentile, pixel_intensity_95percentile, pixel_intensity_99percentile, pixel_intensity_skewness, pixel_intensity_kurtosis, pixel_intensity_interquartilerange, valid_from_t0, dead_timepoint, phenotype, live))
                        
                     
                # Insert multi lines [(filename, object)] into table for one image file
                cur.executemany("INSERT INTO " + table_name + " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values) 
                # Save (commit) the changes
                conn.commit()
             
                i += 1  
                end = time.time()
                elapse = end - start
                print '[Progress done: %d/%d] [Elapse time: %ss For Image:\n%s]' % (i, img_count, elapse, img_name)
                
            else:
                print "Encoded mask for %s does not exist!" % img_name
                continue    
            
if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("table_name")
    parser.add_argument("images_path")
    parser.add_argument("masks_path")
    parser.add_argument("out_report")
    args = parser.parse_args()
    create_images_db(DB_NAME, args.table_name)
    insert_imgs_to_db(DB_NAME, args.table_name, args.images_path, args.masks_path)
    with open(args.out_report, 'wb') as f:
        f.write("Created database file: %s\n" % DB_NAME)
        f.write("Table name: %s" % args.table_name)
    
    # create_images_db(DB_NAME, TABLE_NAME)
    # insert_imgs_to_db(DB_NAME, TABLE_NAME, IMAGES_PATH, MASKS_PATH)

    end_time = time.time()
    elapse = end_time - start_time
    print 'Total process time for database creation: %ss' % elapse
         
