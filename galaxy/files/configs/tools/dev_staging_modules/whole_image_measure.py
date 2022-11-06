'''
Takes whole image measurements and saves to csv file.
'''

import os, cv2, pickle, argparse
import utils
import numpy as np

def extract_mean(img_path, threshold, timepoint, well, channel):
    ''' Takes file pointer string as input and returns image mean and
    thresholded mean measurements as string '''

    img = cv2.imread(img_path, -1)
    img_mean = img.mean()
    img_mean_thresh = img[img >= threshold].mean()
    img_max = img.max()
    img_min = img.min()

    # binarize image by threshold
    img_binarized = np.where(img >= threshold, 1, 0)
    signal_area = np.count_nonzero(img_binarized)

    image_mean_string = ','.join([os.path.basename(img_path), well, timepoint, channel, str(img_mean), str(img_max), str(img_min), str(threshold),
                                  str(img_mean_thresh), str(signal_area), '\n'])
    return image_mean_string

def write_csv(var_dict, input_path, csv_file, threshold, usr_channel):
    ''' Loops through list of images based on wells and timepoints passed in the dictionary file, gets measurements as string via
    extract_mean function, then writes measurements string to CSV file '''

    DIR_STRUCTURE = var_dict['DirStructure']

    # get image paths from root folder or well subdirectories
    image_paths = ''
    if DIR_STRUCTURE == 'root_dir':
        image_paths = [os.path.join(input_path, name) for name in os.listdir(input_path) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
    elif DIR_STRUCTURE == 'sub_dir':
        # Only traverse root and immediate subdirectories for images
        relevant_dirs = [input_path] + [os.path.join(input_path, name) for name in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, name))]
        image_paths = [os.path.join(relevant_dir, name) for relevant_dir in relevant_dirs for name in os.listdir(relevant_dir) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
    else:
        raise Exception('Unknown Directory Structure!')

    # tokenize paths
    img_list_tokens = utils.tokenize_files(image_paths)

    # filter selected channel
    ch_token = utils.get_channel_token(var_dict['RoboNumber'])+1

    if usr_channel != '':
        img_list_tokens = [i for i in img_list_tokens if usr_channel in i[ch_token]]

    # loop through images
    for img in img_list_tokens:
        csv_file.write(extract_mean(img[0], threshold, img[3], img[5], img[ch_token]))

def main():
    ''' Main point of entry '''

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Whole Well Measurements")
    parser.add_argument("input_dict")
    parser.add_argument("threshold", default = '')
    parser.add_argument("input_path")
    parser.add_argument("output_path")
    parser.add_argument("--channel", default = '')
    parser.add_argument("output_dict")
    args = parser.parse_args()

    # read local variables dictionary file
    var_dict = pickle.load(open(args.input_dict, 'rb'))

    # define variables for minimum threshold and images path
    threshold = int(str.strip(args.threshold))
    input_path = str.strip(args.input_path)
    output_path = str.strip(args.output_path)

    assert threshold < 2**16 and threshold >= 0, 'Threshold must be between 0 and 65,535'
    assert os.path.exists(input_path), 'Confirm the input path exists'
    assert os.path.exists(output_path) == 1, 'Confirm the output path exists'

    if args.channel != '':
        usr_channel = utils.get_ref_channel(str.strip(args.channel), var_dict['Channels'])
        print('Channel: %s' % usr_channel)
    else:
        usr_channel = ''

    print 'Threshold:', threshold
    print 'Input Path:', input_path
    print 'Output Path:', output_path

    # write new file to contain image measurements
    results = open(os.path.join(output_path, 'well_data.csv'), 'w')
    results.write(','.join(['Filename', 'Sci_WellID', 'Timepoint', 'Channel', 'MeanPixelIntensity', 'MaxIntensity', 'MinIntensity', 'MinThreshold',
                            'ThresholdMeanPixelIntensity', 'SignalArea', '\n']))
    write_csv(var_dict, input_path, results, threshold, usr_channel)
    results.close()

    timestamp = utils.update_timestring()
    utils.save_user_args_to_csv(args, output_path, 'whole_image_measure'+'_'+timestamp)

if __name__ == '__main__':

    main()
