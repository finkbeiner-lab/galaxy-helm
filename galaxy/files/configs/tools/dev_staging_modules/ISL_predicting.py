''' This interface sends Unix commands to run In-Silico Labeling TensorFlow program (Christiansen et al 2018). This script is to run prediction on multiple images.
'''

import subprocess
import argparse
import pickle
import os
import shutil
import sys
from datetime import datetime
import cv2
import numpy as np

sys.path.append('/home/sinadabiri/galaxy-neuron-analysis_old/galaxy/tools/dev_staging_modules')
import configure

global temp_directory, tmp_location, dataset_prediction, VALID_WELLS, VALID_TIMEPOINTS, dataset_eval_path

INPUT_PATH = ''
ROBO_NUMBER = None
IMAGING_MODE = ''


def image_feeder(dataset_prediction, valid_wells, valid_timepoints):
    temp_directory = os.path.join(output_path, 'temp_directory')
    if not os.path.exists(temp_directory):
        os.mkdir(temp_directory)

    print('The subfolders in dataset_prediction folder are: ', os.listdir(dataset_eval_path), '\n')
    print("VALID_WELLS: ", VALID_WELLS, '\n')
    print("VALID_TIMEPOINTS: ", VALID_TIMEPOINTS, '\n')
    print('The created Temp Directory is: ', temp_directory, '\n')

    loop_iterations_col = 0
    loop_iterations_row = 0
    step_size = 0
    row = 0
    col = 0
    # For our dataset the following loop should be: for entry in VALID_WELLS: instead.
    for entry in VALID_WELLS:
        # os.listdir(dataset_eval_path)

        dataset_location = os.path.join(dataset_eval_path, entry)
        tmp_location = os.path.join(temp_directory, entry)
        if not os.path.exists(tmp_location):
            os.mkdir(tmp_location)

        k = 0

        for img in os.listdir(dataset_location):

            path = str(os.path.join(dataset_location, img))
            if len(os.path.basename(img).split('_')) >= 2:
                time_point = os.path.basename(img).split('_')[2]
            else:
                time_point = ''
            # print("The time point is: ",time_point)
            row, col = cv2.imread(path, cv2.IMREAD_ANYDEPTH).shape
            image = [np.zeros((row, col), np.int16)] * len(os.listdir(dataset_location))
            print('input image dimensions are row x col: ', row, col)
            for tp in VALID_TIMEPOINTS:
                if (img.endswith('.tif') >= 0 and time_point == tp):
                    # and os.path.basename(img).split('_')[6].find('BRIGHTFIELD')>=0

                    image[k] = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
                    # cv2.IMREAD_ANYDEPTH
                    # if img.find('MAXPROJECT') > 0:
                    # image[k] = image[k] - np.mean(image[k])
                    # cv2.equalizeHist(image[k], dst= cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
                    tmp_location_tp = os.path.join(tmp_location, tp)
                    if not os.path.exists(tmp_location_tp):
                        os.mkdir(tmp_location_tp)

                    base = os.path.splitext(img)[0]
                    New_file_name = str(tmp_location_tp) + '/' + base + '.png'
                    image[k] = cv2.imwrite(New_file_name, image[k])
                    k += 1
                elif time_point == tp:
                    # and os.path.basename(img).split('_')[6].find('BRIGHTFIELD')>=0
                    tmp_location_tp = os.path.join(tmp_location, tp)
                    if not os.path.exists(tmp_location_tp):
                        os.mkdir(tmp_location_tp)
                    tmp_location_img = str(os.path.join(tmp_location_tp, img))
                    os.popen('cp ' + path + ' ' + tmp_location_img + ';')
        step_size = int(int(crop_size) / 2.95)
        loop_iterations_row = int(row / step_size) - 2
        loop_iterations_col = int(col / step_size) - 2
        print('image row, col loop iterations are : ', loop_iterations_row, loop_iterations_col)

    return temp_directory, row, col, loop_iterations_col, loop_iterations_row, step_size


def image_montage(row_start, column_start, current_tile_row, current_tile_col,
                  stitched_image, stitched_image_channel, current_tile,
                  output_path_eval_well_tp_tile_channel):
    current_tile_channel = cv2.imread(output_path_eval_well_tp_tile_channel, cv2.IMREAD_ANYDEPTH)
    # cv2.imshow("current_tile is: ", current_tile_DAPI)

    # Appending and then stitch the tiles
    # current_column.append(current_tile_array)
    # predict_input_rows.append(np.concatenate(predict_input_row, axis=2))

    print("the stitched_image is: ", stitched_image)
    row_start_int = int(row_start)
    column_start_int = int(column_start)
    print("the row_start_int is: ", row_start_int, "column_start_int is: ", column_start_int)
    # current_column[row_start_int:row_start_int+current_tile_row,
    #                 column_start_int:column_start_int+current_tile_col] += current_tile

    stitched_image[row_start_int:row_start_int + current_tile_row,
    column_start_int:column_start_int + current_tile_col] = current_tile
    stitched_image_channel[row_start_int:row_start_int + current_tile_row,
    column_start_int:column_start_int + current_tile_col] = current_tile_channel
    # .append(np.concatenate(current_tile, axis=2))
    # stitched_image.append
    print("stitched array is: ", stitched_image)
    # cv2.imshow("Stitched Image is: ", stitched_image)
    stitched_image_array = np.array(stitched_image, dtype=np.uint16)
    print("the stitched image size is: ", stitched_image_array.shape)

    return stitched_image, stitched_image_channel


def main():
    """ First the script makes sure the Bazel has been shutdown properly. Then it starts the bazel command with the
    following arguments:

    Args: crop_size: the image crop size the user chose the prediction to be done for. model_location: wheter the
    user wants to use the model that has been trained before in the program, or use their own trained model.
    output_path: The location where the folder (eval_eval_infer) containing the prediction image will be stored at.
    dataset_eval_path: The location where the images to be used for prediction are sotred at. infer_channels: The
    microscope inference channels. """

    global current_column_concat
    output_path_eval_well_tp_tile = ''
    output_path_eval_well_tp_tile_img = ''
    base_directory_path = 'cd ' + base_directory + '; '

    for w in VALID_WELLS:
        date_time = datetime.now().strftime("%m-%d-%Y_%H:%M")
        print("We are on well: ", w)

        dataset_eval_path_w = str(os.path.join(temp_directory, w))
        print("dataset_eval_path_w is: ", dataset_eval_path_w, '\n')
        print('\n', 'The temp_directory subfolders are: ', os.listdir(dataset_eval_path_w), '\n')

        for tp in VALID_TIMEPOINTS:

            dataset_eval_path_tp = str(os.path.join(dataset_eval_path_w, tp))

            print("Dataset Eval Path is: ", dataset_eval_path_tp, '\n')
            print("Inference channels are: ", infer_channels)
            row_start = '0'
            stitched_image = np.empty([loop_iterations_row * step_size, loop_iterations_col * step_size],
                                      dtype=np.uint16)
            # TODO: loop through valid channels and have one big matrix for each chanel.
            stitched_image_channels = {}
            for ch in VALID_CHANNELS:
                stitched_image_channels[ch] = np.empty(
                    [loop_iterations_row * step_size, loop_iterations_col * step_size],
                    dtype=np.uint16)

            print("loop_iteration_col is: ", loop_iterations_col)

            for row_tile in range(0, loop_iterations_row):
                # loop_iterations_row
                print("we are on row tile: ", row_tile)
                print("row_start is set to: ", row_start)
                column_start = '0'

                for col_tile in range(0, loop_iterations_col):
                    # loop_iterations_col
                    date_time = datetime.now().strftime("%m-%d-%Y_%H:%M")
                    print("we are on column: ", col_tile)

                    # Running Bazel for prediction. Note txt log files are also being created
                    # in case troubleshooting is needed.
                    log_path = os.path.join(output_path, 'output_logs')
                    if not os.path.exists(log_path):
                        os.mkdir(log_path)

                    print("Bazel Launching------------------------", '\n')
                    base_dir = 'export BASE_DIRECTORY=' + base_directory + '/isl;  '
                    baz_cmd = [base_directory_path + base_dir + 'bazel run isl:launch -- \
                    --alsologtostderr \
                    --base_directory $BASE_DIRECTORY \
                    --mode EVAL_EVAL \
                    --metric INFER_FULL \
                    --stitch_crop_size ' + crop_size + ' \
                    --row_start ' + row_start + ' \
                    --column_start ' + column_start + ' \
                    --output_path ' + output_path + ' \
                    --read_pngs \
                    --dataset_eval_directory ' + dataset_eval_path_tp + ' \
                    --infer_channel_whitelist ' + infer_channels + ' \
                    --restore_directory ' + model_location + ' \
                    --error_panels False \
                    --infer_simplify_error_panels \
                    > ' + log_path + '/predicting_output_' + mod + '_' + date_time + '_' +
                               crop_size + '_' + w + '_' + tp + '_' + row_start + '-' + column_start +
                               '_images.txt \
                    2> ' + log_path + '/predicting_error_' + mod + '_' + date_time + '_' +
                               crop_size + '_' + w + '_' + tp + '_' + row_start + '-' + column_start +
                               '_images.txt;']

                    print('The baz_cmd is now: ', baz_cmd)
                    process = subprocess.Popen(baz_cmd, shell=True, stdout=subprocess.PIPE)
                    process.wait()
                    print("Bazel Finished=====================================================", '\n')
                    # --restore_directory ' + model_location + ' \
                    output_path_eval_well = os.path.join(output_path, "eval_eval_infer/00984658_" + w)
                    if not os.path.exists(output_path_eval_well):
                        os.mkdir(output_path_eval_well)
                    output_path_eval_well_tp = os.path.join(output_path_eval_well, tp)
                    if not os.path.exists(output_path_eval_well_tp):
                        os.mkdir(output_path_eval_well_tp)
                    output_path_eval_well_tp_tile = os.path.join(output_path_eval_well_tp,
                                                                 str(row_start) + '_' + str(column_start))
                    if not os.path.exists(output_path_eval_well_tp_tile):
                        os.mkdir(output_path_eval_well_tp_tile)
                    output_path_eval_well_tp_tile_img = os.path.join(output_path_eval_well_tp_tile,
                                                                     str(row_start) + '_' + str(column_start) +
                                                                     "_input-post-model.png")
                    print("the output_path_eval_well_tp_tile_img is: ", output_path_eval_well_tp_tile_img)
                    current_tile = cv2.imread(output_path_eval_well_tp_tile_img, cv2.IMREAD_ANYDEPTH)
                    # cv2.imshow("current_tile is: ", current_tile)
                    # print("current_tile is: ", current_tile)
                    current_tile_array = np.array(current_tile)
                    current_tile_row, current_tile_col = current_tile_array.shape
                    # (126, 126)
                    # current_tile_array.shape
                    print("current_tile shape row, col: ", current_tile_row, current_tile_col)
                    # TODO: make a loop to go through all the channels.
                    for ch in VALID_CHANNELS:
                        channel_name = ch.split("_")[0]
                        output_path_eval_well_tp_tile_channel = os.path.join(output_path_eval_well_tp_tile,
                                                                             str(row_start)
                                                                             + "_" + str(column_start) +
                                                                             "_" + channel_name + "_predicted.png")
                        stitched_image, stitched_image_channels[ch] = \
                            image_montage(row_start, column_start,
                                          current_tile_row,
                                          current_tile_col, stitched_image,
                                          stitched_image_channels[ch],
                                          current_tile,
                                          output_path_eval_well_tp_tile_channel)
                    column_start = str(int(column_start) + step_size)
                    if int(column_start) <= (col - int(step_size)):
                        print("column_start is set to: ", column_start)
                    else:
                        break

                # axis=0 is column-wise and 1 is row-wise concatenating.
                # stitched_image_concatenated = np.concatenate(stitched_image_array, axis=1)
                # print ("the concatenated stitched image is: ", stitched_image_concatenated)
                # stitched_image_array.astype(np.uint16)

                stitched_image_name = output_path_eval_well_tp_tile + "_input-post-model_column.png"
                cv2.imwrite(stitched_image_name, stitched_image)
                for ch in VALID_CHANNELS:
                    stitched_image_name = output_path_eval_well_tp_tile + "_" + ch + "_stitched_image.png"
                    cv2.imwrite(stitched_image_name, stitched_image_channels[ch])
                # cv2.imshow("Column Input Image post model", stitched_image)
                # stitcher = cv2.Stitcher_create()
                # (status, stitched) = stitcher.stitch(current_column)
                # if status == 1:
                #     cv2.imwrite(current_column_name, stitched)
                #     cv2.imshow("Column Input Image post model", current_column)
                # else:
                #     print ("Didn't stitch.")

                row_start = str(int(row_start) + step_size)
                if int(row_start) <= (row - int(step_size)):
                    print("column_start is set to: ", row_start)
                else:
                    break

    # Here we delete the temp folder.
    print("temp_directory is going to be removed: ", temp_directory)
    cmd3 = ['rm -r ' + temp_directory + ';']
    process3 = subprocess.Popen(cmd3, shell=True, stdout=subprocess.PIPE)
    process3.wait()
    return


if __name__ == '__main__':
    # Receiving the variables from the XML script, parse them, initialize them, and verify the paths exist.

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(description="ISL Predicting.")
    parser.add_argument("infile", help="Load input variable dictionary")
    parser.add_argument("crop_size", help="Image Crop Size.")
    parser.add_argument("model_location", help="Model Location.")
    parser.add_argument("output_path", help="Output Image Folder location.")
    parser.add_argument("dataset_eval_path", help="Folder path to images directory.")
    parser.add_argument("infer_channels", help="Channel Inferences.")
    parser.add_argument("outfile", help="Name of output dictionary.")
    args = parser.parse_args()

    # ----Load path dict-------------------------
    infile = args.infile
    var_dict = pickle.load(open(infile, 'rb'))

    # ----Initialize parameters------------------
    crop_size = args.crop_size
    print("The crop size is: ", crop_size)
    base_directory = configure.base_directory
    model_location = args.model_location
    print("The model location is: ", model_location)
    output_path = args.output_path
    print("The output path is: ", output_path)
    dataset_eval_path = args.dataset_eval_path
    print("The brightfield images are located at: ", dataset_eval_path)
    infer_channels = args.infer_channels
    print("The inference channels are: ", infer_channels)
    outfile = args.outfile

    INPUT_PATH = args.dataset_eval_path

    VALID_WELLS = var_dict['Wells']
    VALID_TIMEPOINTS = var_dict['TimePoints']
    VALID_CHANNELS = str(infer_channels).split(',')
    print("the valid channels are: ", VALID_CHANNELS)

    if model_location != '':
        mod = 'ISL-Model'
    else:
        model_location = ''
        mod = 'Your-Model'

    temp_directory, row, col, loop_iterations_col, loop_iterations_row, step_size = \
        image_feeder(dataset_eval_path, VALID_WELLS, VALID_TIMEPOINTS)

    # ----Confirm given folders exist--
    if not os.path.exists(dataset_eval_path):
        print('Confirm the given path to input images (transmitted images used to generate prediction image) exists.')
        assert os.path.exists(dataset_eval_path), 'Path to input images used to generate prediction image is wrong.'
        if not os.path.exists(output_path):
            print('Confirm the given path to output of prediction for fluorescent and validation images exists.')

            assert os.path.abspath(output_path) != os.path.abspath(
                dataset_eval_path), 'Please provide a unique data path.'
            assert os.path.abspath(output_path) != os.path.abspath(
                model_location), 'Please provide a unique model path.'

            date_time = datetime.now().strftime("%m-%d-%Y_%H:%M")

            print('\n The Evaluation Directory is: ', dataset_eval_path)
            print('\n The Output Directory is: ', output_path)
            print('\n ')

    main()

    # ----Output for user and save dict----------

    # Save dict to file
    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    outfile = shutil.move('var_dict.p', outfile)
