import cv2
import numpy as np
import os
import sys
from numpy import array
from math import sqrt
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import datetime
import utils


def getCenterMass(img, thresh_val = 6):
    '''
    Return the center (x,y) for 30 largest objects found in an image.
    Default threshold is 6.
    '''
    ret,thresh = cv2.threshold(img,thresh_val,255,0)
    contours,hierarchy = cv2.findContours(thresh, 1, 2)
    centers = []
    areas = []
    for cnt in contours:
        moments = cv2.moments(cnt)                          # Calculate moments
        if moments['m00']!=0:
            cx = int(moments['m10']/moments['m00'])         # cx = M10/M00
            cy = int(moments['m01']/moments['m00'])         # cy = M01/M00
            centers.append([cx,cy])
            areas.append(cv2.contourArea(cnt))

    # Sort the array by area in descending order
    sorteddata = sorted(zip(areas, centers), key=lambda x: x[0], reverse=True)
    # Find the nth largest contour [n-1][1], in this case 2
    # Return an array of centers
    return [c[1] for c in sorteddata[0:30]]


def getTimeandCM(path_to_aligned_images, well, channel):
    '''
    Takes path to images, well, and channel. 
    Reads image and returns timepoint with list of object centers.
    '''
    images = []
    
    for f in os.listdir(path_to_aligned_images):
        if channel in f.lower():
            if well in f.lower() :
                path = os.path.join(path_to_aligned_images, f)
                # Read image and get the 30x objs' centers coords
                img = cv2.imread(path, 0)
                centroids = getCenterMass(img)
                # File name is split on _
                # 3rd token contains the time point
                tokens = f.split("_")
                images.append([tokens[2], centroids])
    return images


def calcPersistenceOfObjects(time_centers):
    '''
    Takes in a list of each timepoints's centers of 30 objects/well
    Return average time persistence, # groups, max time persistence, max coords for a well
    
    @Step1
    time_centers is a list containing sublists of times and object coordinates
    Example: [[[T0, [x1,y1],[x2,y2]],
               [T1, [x1,y1],[x2,y2]]]
    Read the time_centers list and insert each object's time, x-coord, y-coord
    into a data frame that has 3 fields: time, x-coord, y-coord

    @Step2
    Go thru each time_center and calculate the ROI for each object
    Query the data frame to find other objects in the specified ROI => groups of objs
    Create a dictionary that stores a unique identifier of the group of objs  as the key
    and group size as the value.

    '''
    # Step1
    index = -1
    df = pd.DataFrame(columns=['time','x_coord', 'y_coord'])
    # Get each time period's objects and insert into the data frame
    for time,time_center in enumerate(time_centers):
        # Look at the second element in each time center, which is a list of all the objects
        for obj in time_center[1]:
            index += 1
            df.loc[index] = [time,obj[0],obj[1]]

    # Step2
    groups = {}
    for time_center in time_centers:
        # Look at the second element in each time center, which is a list of all the objects
        for obj in time_center[1]:

            radius = 30
            # Calculate the min and max of x and y to establish the ROI- region of interest
            min_x = obj[0] - radius
            max_x = obj[0] + radius
            min_y = obj[1] - radius
            max_y = obj[1] + radius
            cond = (df['x_coord']>= min_x) & (df['x_coord']<= max_x) & (df['y_coord']>= min_y) & (df['y_coord']<= max_y)
            matches = df[cond].groupby("time")
            
            # To prevent having 2+ closeby objects from the same time period,
            # only include the smallest x,y coord object from each time (for standardization purposes)
            # to the group of objects that belong to the unique ROI
            obj_groups = matches['x_coord','y_coord'].min()
            smallest_coord = obj_groups.iloc[0]
            
            # Dictionary stores the first time period's (x,y coord) as the unique key
            # This prevents duplicate groups from being stored
            groups[int(smallest_coord['x_coord']),int(smallest_coord['y_coord'])] = obj_groups['x_coord'].count()
            #print obj_groups

    # Get the number of groups - which stores objects in each distinct ROI
    number_groups = 1.0 * len(groups)
    # Find the object with the max time persistence
    max_time = max(groups.values())
    # Get the coordinates for the group of objs that had the max time persistence
    max_coords = groups.keys()[groups.values().index(max_time)]
    # Avg time persistence of the well  = sum of persistence of all groups/# distinct groups
    avg_time = 1.0 *sum(groups.values())/ number_groups
    # Avg_time ranges from 1 to number of time periods in that well
    # Return 4 vars: average time persistence, # groups, max time persistence, max coords
    return avg_time, number_groups, max_time, max_coords

def getTimeandMeanIntensity(path_to_aligned_images, well, channel):
    '''
    Get each time point's mean intensity of a well
    '''
    images = []
    # traverse thru each well img file in the aliginment input path
    for f in os.listdir(path_to_aligned_images):
        if channel in f.lower():
            if well in f.lower() :
                path = os.path.join(path_to_aligned_images, f)
                #get the mean pixel intensity for that Image
                img = cv2.imread(path, 0)
                tokens = f.split("_")
                images.append([tokens[2], np.mean(img)])
    return images

def getImageArrays(path_to_aligned_images, well, channel):
    '''
    Get the pixels for each time period of a well.
    '''
    images = []
    for f in os.listdir(path_to_aligned_images):
        if channel in f.lower():
            if well in f.lower() :
                path = os.path.join(path_to_aligned_images, f)
                #read images as is (16 bit) and store in array
                img = cv2.imread(path, 0)
                tokens = f.split("_")
                images.append([tokens[2], img])
    return images


def getDistanceShifts(v1,v2):
    '''
    Take in 2 vectors of coords and calculate the distance of each row of v1
    with the corresponding row of v2
    Rows are time periods (t0,t7); Column contains coord (x,y)
    Return an array with each time period's calculated distance or shift
    '''

    shifts = []
    # Find the distance between the 2 coords (x1,y1), (x2,y2), using distance formula
    for i in range(len(v1)):
        sq_diff_x = (v1[i][0][0] - v2[i][0][0])**2
        sq_diff_y = (v1[i][0][1] - v2[i][0][1])**2
        shifts.append([round(sqrt(sq_diff_x + sq_diff_y),2)])
    return array(shifts)

def reportTimePeristence(morph, wells):
    '''
    Create a csv file with each well's average and max time persistence of objects
    Used for training (given)data
    '''
    # Set up the empty data frame to store all the results for the wells
    d = {}
    df = pd.DataFrame(d, index=wells, columns=['avg_persistence','number_unique_objs','max_time','max_coord'])

    #For each well, get the average peristence and the number of unique objects
    for well in wells:
        #print 'Well', well
        try:
            well_n = well +'_'
            # For each time period, get the Center coords
            # This will be used to track the 30 largest objects throughout each time period
            time_centers = getTimeandCM(path_to_aligned_images, well_n.lower(), morph)
            avg_p, no_unique,max_time,max_coord = calcPersistenceOfObjects(time_centers)
            # store 4 fields in the data frame
            df.loc[well] = [avg_p, no_unique,max_time,str(max_coord)]
        except Exception as e:
            # print "Error: Well:" + well+ " " + str(e)
            continue

    return df

# Return each well's mean Intensity measurement
def reportMeanIntensities(morph, wells):
    # Set up the empty data frame to store all the results for the wells
    d = {}
    df = pd.DataFrame(d, index=wells, columns=['shift_mean_intensity'])

    #For each well, get the average peristence and the number of unique objects
    for well in wells:
        try:
            well_n = well +'_'
            time_centers = getTimeandMeanIntensity(path_to_aligned_images, well_n.lower(), morph)
            # calculate the average mean intensity for the whole well
            # sum of each time period's mean intensity/#time periods
            total = 0
            for time in time_centers:
                total = total + time[1]

            mean_int = total / len(time_centers)
            df.loc[well] = [mean_int]
        except Exception as e:
            # print "Error: Well:" + well+ " " + str(e)
            continue
    return df

# Return each well's percent border shift measurement
def reportBorderShifts(morph, wells):
    # Set up the empty data frame to store all the results for the wells
    d = {}
    df = pd.DataFrame(d, index=wells, columns=['percent_total_misalignment'])

    for well in wells:
        #print 'Well', well
        try:
            well_n = well +'_'
            # Get image pixels
            images = getImageArrays(path_to_aligned_images, well_n.lower(), morph)

            # Next, calculate the row and col shifts for each image
            # +row is up, -col means img shift left
            time_centers = []
            img_shifts = []
            # get the row,col shift for t0-t7
            for image in images:
                # Get the dimension of the image
                # This will be the total possible coord of shifts [r,c]
                img_shifts.append([image[1].shape[0],image[1].shape[1]])

                row_shift = np.where(~image[1].any(axis=1))
                if len(row_shift[0])==0:
                    r = 0
                elif row_shift[0][0]==0:
                    r = -(max(row_shift[0]) +1)
                else:
                    r = image[1].shape[0] - min(row_shift[0])

                col_shift = np.where(~image[1].any(axis=0))
                if len(col_shift[0])==0:
                    c = 0
                elif col_shift[0][0]==0:
                    c= max(col_shift[0]) +1
                else:
                    c = - (image[1].shape[1] - min(col_shift[0]))
                time_centers.append([image[0],[[r,c]]])

            v1 = []
            v2 = []
            # The before vector has the structure t0,t0,t0... n times in order to find distance from t0, t1,.. tn (after vector)
            for i in range(len(time_centers)):
                #print time_centers[i][1]
                v1.append([[0,0]])
                v2.append(time_centers[i][1])

            # Calculate the successive shifts between each time period's (row,col) coords and (0,0)
            each_time_shift = getDistanceShifts(v1,v2)
            #print img_shifts

            # Get total possible error or shift using distance formula [0,0][row_dimension,col_dimension] for each image
            total_shifts = 0
            for sh in img_shifts:
                x_sqdif = sh[0]**2
                y_sqdif = sh[1]**2
                d = sqrt(x_sqdif + y_sqdif)
                total_shifts += d
            '''
            print "total shift ", total_shifts
            print "each time shift: "
            for t in each_time_shift:
                print t
            '''
            # Calculate the percent border shift => image shift/total possible image shift using the image's dimesions *100%
            df.loc[well] = (sum(each_time_shift)/total_shifts)*100
        except Exception as e:
            # print "Error: Well:" + well+ " " + str(e)
            continue

    return df

def evaluateWells(path_to_aligned_images, wells, print_csv=True, metric='border', morph='rfp', bs_thresh =1.8, ap_thresh =1.9, mp_thresh =6.0):
    '''
    Takes in a list of wells that need to be evaluated as Good or Bad
    Default Method uses the Border Shift metric to classify the wells
    Option to print csv of metadata from the classification of the wells
    Prints list of Bad wells.

    @Other
    Subset the dataframe by the chosen metric, default is border-shift
    This metric has been evaulated by statistical tests and is the most promising in
    classifying good and bad wells
    '''

    # Metric 1 is Percent Border Shift
    if metric == 'border':
        #% Border shift threshold = 1.8
        # get the data frame called df_all for each well's border shift
        df_all = reportBorderShifts(morph, wells)

        # if printing csv, calculate the other 2 metric to report in the csv
        if print_csv:
            df2 = reportTimePeristence(morph, wells)
            df3 = reportMeanIntensities(morph, wells)
            df_all = pd.concat([df_all, df2, df3], ignore_index=False, axis=1)

        df_good = df_all[df_all.percent_total_misalignment < bs_thresh]
        df_bad = df_all[df_all.percent_total_misalignment >= bs_thresh]
        print 'Good alignment:',
        # print "Good wells sorted by increasing shift. Best to worst:"
        df_g = df_good.sort(['percent_total_misalignment'], ascending=[1])
        print df_g.index.tolist()
        print 'Possibly misaligned:',
        # print "Bad wells sorted by decreasing shift. Worst to best:"
        df_b = df_bad.sort(['percent_total_misalignment'], ascending=[0])
        # Print the bad wells list to outfile or to screen
        print df_b.index.tolist()

    # Metric 2 is Average Time Persistence
    elif metric == 'avg_persistence':
        # Average Time Persistence threshold = 1.9
        df_all = reportTimePeristence(morph, wells)

        # if printing csv, calculate the other 2 metric to report in the csv
        if print_csv:
            df2 = reportBorderShifts(morph, wells)
            df3 = reportMeanIntensities(morph, wells)
            df_all = pd.concat([df_all, df2, df3], ignore_index=False, axis=1)

        df_good = df_all[df_all.avg_persistence >= ap_thresh]
        df_bad = df_all[df_all.avg_persistence < ap_thresh]
        print 'Good alignment:',
        # print "Good wells sorted by descending average cell persistence time. Best to worst:"
        df_g = df_good.sort(['avg_persistence'], ascending=[0])
        print df_g.index.tolist()
        print 'Possibly misaligned:',
        # print "Bad wells sorted by ascending average cell persistence time. Worst to best:"
        df_b = df_bad.sort(['avg_persistence'], ascending=[1])
        # Print the bad wells list to outfile or to screen
        print df_b.index.tolist()

    # Metric 3 is Max Time Persistence
    elif metric == 'max_persistence':
        # Max Time Persistence threshold = 6
        df_all = reportTimePeristence(morph, wells)

        # if printing csv, calculate the other 2 metric to report in the csv
        if print_csv:
            df2 = reportBorderShifts(morph, wells)
            df3 = reportMeanIntensities(morph, wells)
            df_all = pd.concat([df_all, df2, df3], ignore_index=False, axis=1)

        df_good = df_all[df_all.max_time >= mp_thresh]
        df_bad = df_all[df_all.max_time < mp_thresh]
        print 'Good alignment:',
        # print "Good wells sorted by descending maximum cell persistence time. Best to worst:"
        df_g = df_good.sort(['max_time'], ascending=[0])
        print df_g.index.tolist()
        # print "Bad wells sorted by ascending maximum cell persistence time. Worst to best:"
        print 'Possibly misaligned:',
        df_b = df_bad.sort(['max_time'], ascending=[1])
        # Print the bad wells list to outfile or to screen
        print df_b.index.tolist()

    if print_csv:
        df_g.to_csv(os.path.join(path_to_aligned_images, 'Good_Wells.csv'))
        df_b.to_csv(os.path.join(path_to_aligned_images, 'Bad_Wells.csv'))
        print "Created Good_Wells.csv and Bad_Wells.csv"

    return df_b.index.tolist(), df_g.index.tolist()

if __name__ == "__main__":

    # ----Collect inputs from GUI------
    parser = argparse.ArgumentParser(description="Alignment Validation.")

    parser.add_argument("path_to_aligned_images",
        help="Folder path where input images are stored.")

    parser.add_argument("morphology_channel",
        help="A unique string corresponding to morphology channel. Default is RFP.")

    parser.add_argument("metric",
        help="Enter 1 for Shift, 2 for Average persistence time, 3 for Max persistence time")

    parser.add_argument("csv_flag",
        help="Enter 1 if csv file should be generated.")

    parser.add_argument("outfile",
        help="Name of output dictionary.")

    parser.add_argument("--chosen_wells", "-cw",
        dest = "well_args", default = '',
        help="Specify wells separated by ,")

    args = parser.parse_args()

    # ----Set variables----------------
    path_to_aligned_images = args.path_to_aligned_images
    morphology_channel = (args.morphology_channel).lower() 
    metric = args.metric
    csv_flag = bool(args.csv_flag) 
    if args.well_args == '':
        wells = utils.get_iter_from_user('A1-H12', 'wells')
    else:
        wells = utils.get_iter_from_user(args.well_args, 'wells')
    outfile = args.outfile

    # ----Evaluate alignment-----------
    start_time = datetime.datetime.utcnow() 

    bad_wells, good_wells = evaluateWells(
        path_to_aligned_images, wells, csv_flag, metric, morphology_channel, wells)

    end_time = datetime.datetime.utcnow() 
    print 'Alignment validation run time:', end_time-start_time 
    print 'Alignment output written to:', path_to_aligned_images 

    # ----Write to output--------------
    outf = open(outfile,'w')
    outf.write('Misaligned wells sorted worst to best by '+metric+'.')
    outf.write('\n')
    outf.write(', '.join(bad_wells))
    outf.write('\n')
    outf.write('Aligned wells sorted best to worst by '+metric+'.')
    outf.write('\n')
    outf.write(', '.join(good_wells))
    outf.write('\n')
    outf.close

    # print 'CSV flag:', csv_flag
    # print 'Aligned images path:', path_to_aligned_images
    # print 'Morphology channel:', morphology_channel.upper() 
    # print 'Metric Used:', metric
    # print 'Wells:', wells
#/Volumes/Mac_mb/BioP7asynA/RFP-DFTrCy5-singles/
