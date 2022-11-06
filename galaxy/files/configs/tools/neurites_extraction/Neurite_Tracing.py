import os
import re
import pandas as pd
import fnmatch
import numpy as np
import cv2
import argparse
import time

def distance_formula(x1,x2):
    #distance formula using two pixels
    # x1[1] = y-value of the first coordinate
    # x2[0] = x-value of the second coordinate
    return(np.sqrt(np.square(x2[0] - x1[0]) + np.square(x2[1] - x1[1])))

#any file that has the extension '.tif' will be added to the file_list
def list_tif_files(folder):
    file_list = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.tif'):
                file_list.append(file)
    return(file_list)

#filters the list provided using the given pattern
#pattern defaults to a blank string (thus no filtering)
def filter_file_list(file_list, pattern = ''):
    match_list = []
    leftover_list = []
    for file in file_list:
        if fnmatch.fnmatch(file, pattern):
            match_list.append(file)
        else:
            leftover_list.append(file)
    return match_list, leftover_list

#creates a Pandas dataframe from the given list of filenames
def folder_summary(file_list):
    columns = ['timepoints', 'hour', 'well', 'channel']

    file_df = pd.DataFrame(columns = columns)
    i = 0
    for filename in file_list:
        if filename.endswith('.tif'):
            # regex_groups = re.search('(T[0-9])_([0-9]+).*([A-Z][0-9]{1,2}).*([R].*)\.', filename)
            # timepoints = regex_groups.group(1)
            # hour = regex_groups.group(2)
            # well = regex_groups.group(3)
            # channel = regex_groups.group(4)
            # file_df.loc[i] = [timepoints,hour,well,channel]
            # file_df.rename(index={i:filename},inplace=True)
            # i += 1
            filename_tokens = os.path.basename(filename).split('_')
            timepoints = filename_tokens[2]
            hour = filename_tokens[3]
            well = filename_tokens[4]
            channel = filename_tokens[6]
            file_df.loc[i] = [timepoints,hour,well,channel]
            file_df.rename(index={i:filename},inplace=True)
            i += 1
        #file_list = file_list.append({'well':filename, 'channel': filename, 'timepoint', filename}, ignore_index= True)

    file_df = file_df.sort_values(by = ['well','timepoints','hour','channel'])
    return file_df

# checks if the input directory exists; if it does not exist, it will create the directory.
def check_dir_exists(curr_dir):
    if not os.path.exists(curr_dir):
        os.makedirs(curr_dir)
    return()


#creates our neurite path and soma path using the 'path_to_masks' created from argparse
def create_paths(path_to_masks):
    neurite_path = os.path.join(path_to_masks, 'neurite_masks')
    soma_path = os.path.join(path_to_masks, 'soma_masks')
    return(neurite_path, soma_path)


#using the 2 paths (neurite_path, soma_path) create a list of filenames and using the filenames, create 3 dataframes (neurite, cellmask, encoded)
# each column of the data frame is a different indicator substring of the filename
# the columns can be used to differentiate files from each other for parsing later
def create_dfs(neurite_path, soma_path):
    #list of neurite image filenames
    neurite_list = list_tif_files(neurite_path)
    #dataframe of neurite filename features
    neurite_df = folder_summary(neurite_list)

    #list of soma image filenames
    soma_list = list_tif_files(soma_path)
    #separates the encoded and regular images into two separate lists
    encoded_list, cellmask_list = filter_file_list(soma_list, '*ENCODED*')
    cellmask_df = folder_summary(cellmask_list)
    encoded_df = folder_summary(encoded_list)

    return(neurite_df, cellmask_df, encoded_df)

#using the 3 dataframes created from 'create_dfs' we merge neurite with cellmask to find which trios (pair of 3 files) are missing filesself.
# this ensures that we do not try to use our counting functions on images that do not have all 3 images (neurite, cellmask, encoded) in our folder
def create_filtered_lists(neurite_df, cellmask_df, encoded_df):
    #merges the neurite dataframe and the cellmask dataframe
    #right merge, so any row of the left dataframe (neurite_df)...
    #that does not match with a row of the right dataframe (cellmask_df) will be filled with NaN values
    merge_df = neurite_df.merge(cellmask_df, how = 'outer', left_on = 'well', right_on = 'well')
    #numpy array of missing wells
    missing_wells_for_somas = merge_df.loc[merge_df['channel_x'].isnull()].well.values
    missing_wells_for_neurites = merge_df.loc[merge_df['channel_y'].isnull()].well.values

    #input: cellmask_list, encoded_list
    #
    #output: cellmask_filter_df as list
    # checks if each row's 'well' value matches with a value in the missing_cellmask_wells array.
    # returns the rows that do NOT match any of the values in missing_wells
    neurite_filter_df = neurite_df[~(neurite_df['well'].isin(missing_wells_for_neurites))]
    cellmask_filter_df = cellmask_df[~(cellmask_df['well'].isin(missing_wells_for_somas))]
    encoded_filter_df = encoded_df[~(encoded_df['well'].isin(missing_wells_for_somas))]
    cellmask_filter_list = cellmask_filter_df.index.tolist()
    neurite_filter_list = neurite_filter_df.index.tolist()
    encoded_filter_list = encoded_filter_df.index.tolist()
    return(neurite_filter_list, cellmask_filter_list, encoded_filter_list)

def create_contours_and_img(neurite_test_file,cellmask_test_file, encoded_test_file):
    #read in neurite image and find the contours of the neurites
    #present total number of pixels
    neurite_img = cv2.imread(neurite_test_file, 0)
    neurite_contours, _ = cv2.findContours(neurite_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    #read in cellmask image and find the contours of the neurites
    cellmask_img = cv2.imread(cellmask_test_file, 0)
    cellmask_contours, _ = cv2.findContours(cellmask_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    encoded_img = cv2.imread(encoded_test_file, -1)
    ratio = np.amax(encoded_img) / 256
    encoded_img_8bit = (encoded_img / ratio).astype('uint8')
    encoded_contours, _ = cv2.findContours(encoded_img_8bit, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return(neurite_contours, cellmask_contours, cellmask_img, encoded_contours, encoded_img)

def create_encoded_ids(encoded_img):
    #numpy array that represents the value that each cell takes. Length of array is the number of cells in curr image.
    encoded_ids = np.unique(encoded_img)[1:]
    return(encoded_ids)




# finds the left-most pixel (same for north, south, and east)
def find_extreme_points(cont):
    extLeft = tuple(cont[cont[:,:,0].argmin()][0])
    extRight = tuple(cont[cont[:,:,0].argmax()][0])
    extTop = tuple(cont[cont[:,:,1].argmin()][0])
    extBot = tuple(cont[cont[:,:,1].argmax()][0])
    return[extLeft, extTop, extRight, extBot]

#iterates through each pixel in the second contour to check how close it is to the current pixel of the first contour
def smallest_distance_contours_full(first_cont, second_cont):
    min_distance = float("inf")
    for first_pixel in first_cont:
        for second_pixel in second_cont:
            dist_formula = distance_formula(first_pixel[0], second_pixel[0])
            #print("Current distance: " + str(dist_formula))
            #print("Current radius: " + str(radius))
            if (dist_formula < min_distance):
                min_distance = dist_formula
                cellmask_pixel = first_pixel
                neurite_pixel = second_pixel
    return(min_distance, cellmask_pixel[0], neurite_pixel[0])


#Write a function that takes a threshold distance and returns True/False boolean if the contours are closer than given distance
def threshold_distance_full(threshold, first_cont, second_cont):
    min_distance, first_cont_closest_pixel, second_cont_closest_pixel = smallest_distance_contours_full(first_cont, second_cont)
    #returns TRUE if the min_distance calculated by smallest_distance_contours_radius is less than the threshold (radius in this case)
    return((min_distance < threshold), first_cont_closest_pixel, second_cont_closest_pixel)


#Write a function that calculates the smallest distance between two contours (takes: two contours; returns: smallest distance)
#uses the center and radius of one contour (cellmask in this project)
def smallest_distance_contours_radius(first_cont_center, second_cont):
    #iterates through each pixel in the second contour to check how close it is to the center of the first contour
    min_distance = float("inf")
    for pixel in second_cont:
        dist_formula = distance_formula(pixel[0], first_cont_center)
        #print("Current distance: " + str(dist_formula))
        #print("Current radius: " + str(radius))
        if (dist_formula < min_distance):
            min_distance = dist_formula
            neurite_pixel = pixel


    return(min_distance, neurite_pixel[0])


#Write a function that takes a threshold distance and returns True/False boolean if the contours are closer than given distance
def threshold_distance_radius(threshold, first_cont_center, second_cont):
    min_distance, second_cont_closest_pixel = smallest_distance_contours_radius(first_cont_center, second_cont)
    #returns TRUE if the min_distance calculated by smallest_distance_contours_radius is less than the threshold (radius in this case)
    return((min_distance < threshold), second_cont_closest_pixel)


#Write a function that takes a width and distance and joins two contours if they are within a threshold distance (by drawing a line of width number of pixels wide)
# NOT WORKING CORRECTLY
def draw_connecting_line(img, first_pixel, second_pixel):
    img = cv2.line(img,first_pixel,second_pixel,(255,0,0),1)
    return(img)

# Write a function that ignores contours that are smaller than a threshold size (takes, threshold size, returns, false)
def threshold_contour_size(contour, threshold_size):
    assert(type(threshold_size) is int), 'Threshold_size must be an integer value representing the area of the smallest cell tolerable.'
    return(cv2.contourArea(contour) > threshold_size)


def check_id_to_cell(encoded_img, curr_id, center):
    curr_xy_tuple = np.where(encoded_img == curr_id)
    x_array = curr_xy_tuple[1]
    y_array = curr_xy_tuple[0]
    zip_array = zip(x_array, y_array)
    xy_list = list(zip_array)

    #This line checks if the center for the current cell is inside the xy_list(pixels that have the current encoded_id value)
    #center[0] is the x-coordinate of the pixels
    #center[1] is the y-coordinate of the pixels
    if (center in xy_list) or \
    (center[0] >= min(x_array) and center[0] <= max(x_array) \
    and center[1] >= min(y_array) and center[1] <= max(y_array)):

        print("Set fil_id to %d" % (curr_id))
        return(True)

    return(False)


#input: cellmask_contours (list of lists that are filled with pixel coordinate values for each cell in the cellmask files)
#       neurite_contours (list of lists that are filled with pixel coordinate values for each cell in the neurite files)
#       cellmask_img (cellmask img to modify with lines, circles, neurites, etc.)

#output: associated_array (list of multiple 2-item lists...
#                               ... each 2-item list represents the number of neurite pixels associated with that cell ...
#                               ... and the number of neurites that overlap with that cell)
#        img (modified cellmask array/image that can be saved as a .tif)
def draw_associated_neurites(cellmask_contours, neurite_contours, cellmask_img, encoded_img, encoded_ids, threshold_cell_size,radius_modifier):
    associated_list = []
    encoded_copy = np.zeros_like(encoded_img)
    #outer loop is iterating through the cell contours (cells in our image)
    for curr_contour in cellmask_contours:
        #print("Cell: " + str(i + 1))
        if not threshold_contour_size(curr_contour, threshold_cell_size):
            print("Current cell is smaller than the minimum threshold so it will be ignored.")
            continue
        (x,y),radius = cv2.minEnclosingCircle(curr_contour)
        center = (int(x),int(y))
        list_center = list(center)
        threshold_radius = int(radius) * radius_modifier

        #iterates throuh each id and sets fil_id to the curr encoded_id if the boolean returns True
        # fil_id is used to determine which value to draw the current neurite with later on
        for curr_id in encoded_ids:
            if (check_id_to_cell(encoded_img, curr_id, center)):
                fil_id = curr_id
        #ellipse = cv2.fitEllipse(curr_contour)
        num_pixels_connect = 0
        num_neurites = 0

        #finds the north, south, east, and west-most pixels in the current cellmask contour
        #extreme_points = find_extreme_points(curr_contour)

        #the 5 lines below are for fitting a line over the current cellmask contour
        #rows,cols = cellmask_img.shape[:2]
        #[vx,vy,x,y] = cv2.fitLine(curr_contour, cv2.DIST_L2,0,0.01,0.01)
        #lefty = int((-x*vy/vx) + y)
        #righty = int(((cols-x)*vy/vx)+y)


        #UNCOMMENT THIS TO DRAW THE FITTED LINE OVER EACH CELL
        #cellmask_img = cv2.line(cellmask_img,(cols-1,righty),(0,lefty),(255,255,255),thickness = 1)


        #this loop is iterating through the neurite contours (each individual neurite)
        for curr_fil_contour in neurite_contours:
            distance_bool, second_cont_closest_pixel = threshold_distance_radius(threshold_radius, list_center, curr_fil_contour)
            #distance_bool_full, first_cont_closest_pixel_full , second_cont_closest_pixel_full = threshold_distance_full(20, curr_contour, curr_fil_contour)
            #if distance_bool_full:
                #print(first_cont_closest_pixel_full)
                #print(second_cont_closest_pixel_full)
            if distance_bool:
                #extreme_distances = [distance_formula(second_cont_closest_pixel,x) for x in extreme_points]
                #closest_extreme_point_index = extreme_distances.index(min(extreme_distances))
                #closest_extreme_point = extreme_points[closest_extreme_point_index]
                #print(closest_extreme_point)
                #print(second_cont_closest_pixel)


                num_pixels_connect += cv2.contourArea(curr_fil_contour)
                num_neurites += 1

                print("num_pixels_connect inside the loop: "+ str(num_pixels_connect))


                #draws the current neurite on the cellmask_img array. If a neurite is not witihin the threshold distance to any cellmask in the entire image then it will not be drawn
                cellmask_img = cv2.drawContours(cellmask_img, curr_fil_contour, -1, color=255, thickness= -1)



                #print("Current value of 'i' inside the loop: " + str(i))
                print("Drawing current neurite contour with the value of " + str(fil_id))
                #print("length of encoded_ids:" + str(len(encoded_ids)))
                encoded_copy = cv2.drawContours(encoded_copy, [curr_fil_contour], 0, color= int(fil_id), thickness= -1)


                #UNCOMMENT THIS TO USE THE CONNECTING LINE FUNCTION (currently uses extreme point and the neurite pixel returned from threshold_distances_radius)
                #cellmask_img = draw_connecting_line(cellmask_img, closest_extreme_point, tuple(second_cont_closest_pixel), 1)

                #UNCOMMENT THIS TO DRAW AN ELLIPSE ON THE CURRENT CELLMASK CONTOUR THAT IS WITHIN THE THRESHOLD DISTANCE
                #cellmask_img = cv2.ellipse(cellmask_img,ellipse,(255,255,255), 1)

                #draws a circle around each cell that is 'close' enough to the neurites
                cellmask_img = cv2.circle(cellmask_img,center,int(radius),color = 255,thickness= 1)
        #print("Current value of 'i' outside of loop: " + str(i))
        print("This is num_neurites: " + str(num_neurites))
        associated_list.append([num_pixels_connect ,num_neurites])
    associated_array = np.array(associated_list)
    qc_img = cellmask_img
    processed_img = encoded_copy
    return(associated_array, qc_img, processed_img)


#input: encoded_ids (numpy array that represents the value that each cell takes...
#                       ... Length of array is the number of cells in curr image.)
#       associated_array (list/array that was returned by the associated_neurites function)

#output: curr_dict (a dictionary where the key represents the value of the cell in the encoded_cellmask images...
#                  ... the value is a 2-item list that represents the number of neurite pixels associated with this cell...
#                  ... as well as the number of neurites close to this cell)
def create_neurite_dict(encoded_ids, associated_array):
    curr_dict = {}
    for curr_id, curr_cell in zip(encoded_ids, associated_array):
        curr_dict[curr_id] = curr_cell
    return(curr_dict)


#input: neurite_dict (dictionary created from create_neurite_dict function)
#       img_file (the name of the file currently being counted and analyzed)
#
#output: creates a csv with the file name and '_ASSOCIATED_NEURITE.csv' appended to the end
#       4 columns (Filename, CellID(pixel value from encoded_id/dictionary), Overlapping Pixels, Number of Neurites)
def create_csv(neurite_dict, img_file):
    output_df = pd.DataFrame.from_dict(neurite_dict, orient='index', columns = ['Overlapping Pixels', 'Number of Neurites'])
    #creates an array that is the filename of the current cellmask repeated for the number of cells in the image
    filename_arr = np.repeat(img_file,output_df.shape[0])
    output_df['Filename'] = filename_arr
    output_df = output_df.rename_axis('CellID').reset_index()
    cols = output_df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    output_df = output_df[cols]
    output_df.to_csv(os.path.splitext(img_file)[0] + '_ASSOCIATED_NEURITE.csv',index = False)


# uses all of the functions and logic to analyze all files in the folders
#   (filters on missing wells currently)
def count_all_wells(neurite_filter_list, cellmask_filter_list, encoded_filter_list, neurite_path, soma_path, threshold_cell_size, radius_modifier, qc_img_directory, processed_img_directory, csv_directory):
    for fil_file, cell_file, encoded_file in zip(neurite_filter_list, cellmask_filter_list, encoded_filter_list):
        print ("Current file: %s" % (cell_file))
        curr_time = time.time()
        neurite_test_file = os.path.join(neurite_path, fil_file)
        encoded_test_file = os.path.join(soma_path, encoded_file)
        cellmask_test_file = os.path.join(soma_path, cell_file)
        neurite_contours, cellmask_contours, cellmask_img, encoded_contours, encoded_img = create_contours_and_img(neurite_test_file, cellmask_test_file, encoded_test_file)
        encoded_ids = create_encoded_ids(encoded_img)
        associated_array, qc_img, processed_img = draw_associated_neurites(cellmask_contours, neurite_contours, cellmask_img, encoded_img, encoded_ids, threshold_cell_size, radius_modifier)
        write_csv_and_imgs(encoded_ids, associated_array, qc_img_directory, processed_img_directory, csv_directory, cell_file, fil_file, qc_img, processed_img)
        print("Finished current file after --- %s seconds ---" % (time.time() - curr_time))


def write_csv_and_imgs(encoded_ids, associated_array, qc_img_directory, processed_img_directory, csv_directory, cell_file, fil_file, qc_img, processed_img):
    neurite_dict = create_neurite_dict(encoded_ids, associated_array)

    #joins the QC path with the current filename for the cellmask
    qc_img_dir_file = os.path.join(qc_img_directory, cell_file)

    #joins the processed path with the current filename for the neurite mask
    processed_img_dir_file = os.path.join(processed_img_directory, fil_file)

    #joins the csv path with the current filename for the cellmask
    csv_dir_file = os.path.join(csv_directory, cell_file)
    create_csv(neurite_dict, csv_dir_file)

    processed_filename = os.path.splitext(processed_img_dir_file)[0] + '_PROCESSED.tif'
    qc_filename = os.path.splitext(qc_img_dir_file)[0] + '_QC.tif'
    cv2.imwrite(qc_filename, qc_img)
    cv2.imwrite(processed_filename, processed_img)

#Write a function that ignores contours that are smaller than a threshold size (takes, threshold size, returns, false)
#Write a function that based on threshold neurite size and smallest distance connects neurites that are close together and rejects intensities that are too small.
#Plug into your counter function.

start_time = time.time()
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Track cells from cell masks.")

    #change so that the user provides the absolute path to the desired folder (e.g. NeuriteTracing)
    parser.add_argument("path_to_neurite_masks", help="Folder path to binary neurite masks.")
    parser.add_argument("path_to_soma_masks", help="Folder path to binary and encoded soma masks.")
    parser.add_argument("output_path", help="Folder path to input data.")
    parser.add_argument("--min_cell", nargs='?', const=1,
        dest="min_cell", type=int,
        help="Minimum feature size considered as cell.")
    parser.add_argument("--radius", nargs='?', const= 1,
        dest="radius_modifier", type=float,
        help="The amount that we will multiply each cell's radius by in order to measure overlap of cells and neurites that are not directly on top of each other.")
    args = parser.parse_args()


    # add argument for variable size on circle radius (test if better than current implementation)

    # ----Initialize parameters------------------
    neurite_path = args.path_to_neurite_masks
    soma_path = args.path_to_soma_masks
    output_path = args.output_path
    min_cell_size = int(args.min_cell)
    radius_modifier = args.radius_modifier

    # ----Confirm given folders exist--
    assert os.path.exists(neurite_path), 'Confirm neurite_masks folder exists: '+neurite_path
    assert os.path.exists(soma_path), 'Confirm soma_masks folder exists: '+soma_path

    print("Finished parsing in --- %s seconds ---" % (time.time() - start_time))
    qc_img_directory = os.path.join(output_path, 'qc_images')
    processed_img_directory = os.path.join(output_path, 'processed_images')
    csv_directory = os.path.join(output_path, 'associated_neurite_csvs')
    check_dir_exists(qc_img_directory)
    check_dir_exists(processed_img_directory)
    check_dir_exists(csv_directory)

    neurite_df, cellmask_df,encoded_df = create_dfs(neurite_path, soma_path)
    neurite_filter_list, cellmask_filter_list, encoded_filter_list = create_filtered_lists(neurite_df, cellmask_df, encoded_df)

    print("Finished creating dataframes and ordered lists in --- %s seconds ---" % (time.time() - start_time))
    count_all_wells(neurite_filter_list, cellmask_filter_list, encoded_filter_list, neurite_path, soma_path, min_cell_size, radius_modifier, qc_img_directory, processed_img_directory, csv_directory)
    print("Finished all files in --- %s seconds ---" % (time.time() - start_time))
    print("%d CSVs output to '%s'." % (len(neurite_filter_list), csv_directory) )
    print("%d QC images output to '%s'." % (len(neurite_filter_list), qc_img_directory))
    print("%d Processed images output to '%s'." % (len(neurite_filter_list), processed_img_directory))
