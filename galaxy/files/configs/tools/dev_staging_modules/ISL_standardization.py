""" This script is intended to be an image processing module for galaxy pipeline. The equation used
 is from ImageJ BigStitcher plugin site: https://imagej.net/BigStitcher_Flatfield_correction
 6/25/2019 """

import argparse
import pickle
import os
import cv2
import numpy as np
import utils
# from matplotlib import pyplot as plt

IMAGES_PATH = ''
Troubleshooting = False


def global_intensity_normalization(img, std_mean):
    """
    The global_intensity_normalization function normalizes the corrected image pixel values to fall between 0 and
    1. Then it shifts it's mean to 0.5 for brightfield images and 0.25 for fluorescent images. It also expands their
    standard deviation to 0.125 based on Christiansen et al. 2018 In-Silico Labeling paper.

    :param img: The flatfield corrected image.
    :param std_mean: The mean pixel intensity to standardize the image: 0.5 for Brightfield, 0.25 for florescent
    images.
    :return: The normalized corrected image (pixel values 0 to 1), standardized corrected image, and rescaled
    corrected image where pixel values below 0.02 and above 0.9 were clipped.
    """
    if (img.ravel().max() - img.ravel().min()) == 0:
        row, col = img.shape
        img_normalized = np.zeros((row, col), dtype=np.float)
    else:
        img_normalized = (img - img.ravel().min()) / (img.ravel().max() - img.ravel().min())
    # print(img_normalized)
    img_norm_mean = img_normalized.mean()
    img_norm_std = img_normalized.std()

    mean_correction_factor = np.round(np.abs(std_mean - img_norm_mean), 3)
    stdDev_correction_factor = np.round(img_norm_std / 0.125, 4)
    if Troubleshooting: print("the mean and stdDev correction factor are: ", mean_correction_factor,
                              stdDev_correction_factor)

    # img_standardized = (img_normalized - std_mean) / 0.125
    # img_standardized = img_standardized + std_mean - img_standardized.mean()

    img_standardized = (img_normalized - mean_correction_factor) * stdDev_correction_factor
    img_standardized = img_standardized + np.abs(std_mean - img_standardized.mean())
    img_standardized = img_standardized.clip(0, 1)
    img_standardized_mean = np.round(img_standardized.mean(), 2)
    img_standardized_std = np.round(img_standardized.std(), 3)

    if Troubleshooting: print("the mean and std dev of standardized image is: ", img_standardized_mean,
                              img_standardized_std)
    img_standardized_rescale = np.array(img_standardized * 65535, dtype=np.uint16)
    # print(img_standardized_rescale)
    # plt.imshow(img_standardized_rescale)
    # plt.show()
    return img_standardized_mean, img_standardized_std, img_standardized_rescale


def standardize(dataset_path, output_path, var_dict_local, valid_channel):
    """
    The flatfield_corrector function uses the median image as the bright image to correct images.

    :param dataset_path: the input path.
    :param output_path: the output path.
    :param var_dict_local: the variable dictionary from create folder module.
    :param valid_channel: The channel user chose to process.
    :return: The flatfield corrected image.
    """

    valid_wells = var_dict_local['Wells']
    valid_timePoints = var_dict_local['TimePoints']
    z_size = len(var_dict_local['Depths'])
    if z_size == 0:
        z_size = 1
    output_path_std = os.path.join(output_path, 'Standardized')
    utils.create_dir(output_path_std)

    for well in valid_wells:
        print("We are on well: ", well)
        output_path_std_well = os.path.join(output_path_std, well)
        utils.create_dir(output_path_std_well)
        print("The output path for the STD well is: ", output_path_std_well)

        dataset_path_w = os.path.join(dataset_path, well)
        img_file_list = utils.make_filelist(dataset_path_w, valid_channel)
        if Troubleshooting: print("img_file_list is: ", img_file_list)
        print("valid time points are: ", valid_timePoints)
        for tp in valid_timePoints:
            print("We are on time point: ", tp)
            standardized_matrix = []
            image_file_list_tp = [fn for fn in img_file_list if tp == fn.split('/')[-1].split('_')[2]]
            if Troubleshooting: print("The img_file_list_tp is: ", image_file_list_tp)
            if valid_channel != 'BRIGHTFIELD':
                std_mean = 0.25
            else:
                std_mean = 0.50
            for img_path in image_file_list_tp:

                if img_path.find(valid_channel) > 0:
                    # if Troubleshooting: print("The channel is: ", valid_channel)
                    if Troubleshooting: print("the std mean is: ", std_mean)
                    img = cv2.imread(img_path, -1)
                    img_array = np.array(img, dtype=np.float)
                    base = os.path.basename(img_path)
                    img_standardized_mean, img_standardized_std, img_standardized_rescale = \
                        global_intensity_normalization(img_array, std_mean)
                    standardized_matrix.append([base, img_standardized_mean, img_standardized_std])
                    new_file_name = os.path.join(str(output_path_std_well), base)
                    cv2.imwrite(new_file_name, img_standardized_rescale)
                    # if Troubleshooting and base.find('A3_11') > 0:
                    #     plt.figure("Flatfield Correction: " + valid_channel)
                    #     plt.subplot(221)
                    #     plt.title('Original Image')
                    #     plt.imshow(img_array, cmap='gray')
                    #     plt.subplot(222)
                    #     plt.title('Corrected Image')
                    #     plt.imshow(img_standardized, cmap='gray')
                    #     plt.subplot(223)
                    #     plt.title('Original Histogram')
                    #     img_array_1d = img_array.ravel()
                    #     plt.hist(img_array_1d, bins=256, range=(img_array_1d.min(), img_array_1d.max()),
                    #              density=True)
                    #     plt.yticks([0, 0.00025])
                    #     plt.subplot(224)
                    #     plt.title('Corrected Histogram')
                    #     img_corrected_1d = img_standardized.ravel()
                    #     plt.hist(img_corrected_1d, bins=256, range=(img_corrected_1d.min(), img_corrected_1d.max()),
                    #              density=True)
                    #     plt.yticks([0, 0.00025])
                    #     plt.show()
            print("-------------------------------------------------------------------------------")
            print("Image name", "\t\t\t\t\t\t\t\t\t\t", "Normalized Mean ", "Std Dev")
            print(standardized_matrix)

        print("The standardized images are in: ", output_path_std_well)

    return


def main():
    """
    Main runs the standardizer, where it calls a sub function called global_intensity_normalization, that outputs
    normalized image, standardized image and a rescaled image (0.02, 0.9) using the following arguments:

    Args:
    VAR_DICT_LOCAL: importing the VAR_DICT_LOCAL dictionary to use its arguments such as valid wells,
    time points in the loops.
    valid_channel: the channels the user wants to work with.
    DATASET_PATH: the location of input images.
    output_path: The location where the flatfield corrected images will be stored at.

    """

    print("Valid channels are: ", VALID_CHANNELS)

    for valid_channel in VALID_CHANNELS:
        print("We are on Channel: ", valid_channel)
        standardize(DATASET_PATH, OUTPUT_PATH, VAR_DICT_LOCAL, valid_channel)


if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(description="Flat-field Correction.")
    parser.add_argument("infile", help="Load input variable dictionary")
    parser.add_argument("images_path", help="Folder path to images directory.")
    parser.add_argument("output_path", help="Output for the corrected images Folder location.")
    parser.add_argument("channels", help="The channels chose to flat field correct.")
    parser.add_argument("outfile", help="Name of output dictionary.")
    args = parser.parse_args()

    # ----Load path dict-------------------------
    infile = args.infile
    VAR_DICT_LOCAL = pickle.load(open(infile, 'rb'))

    # ----Initialize parameters------------------
    DATASET_PATH = args.images_path
    print("The input dataset path is: ", DATASET_PATH)
    OUTPUT_PATH = args.output_path
    print("The output path is: ", OUTPUT_PATH)
    outfile = args.outfile
    pickle.dump(VAR_DICT_LOCAL, open(outfile, 'wb'))
    # outfile = shutil.move('var_dict.p', VAR_DICT_LOCAL)

    # VAR_DICT_LOCAL = var_dict
    CHANNELS = str(args.channels).replace(" ", "")
    print("The channels are: ", CHANNELS)
    VALID_CHANNELS = CHANNELS.split(',')
    # VAR_DICT_LOCAL['Channels']

    Z_SIZE = len(VAR_DICT_LOCAL['Depths'])
    if Z_SIZE == 0:
        Z_SIZE = 1

    # ----Confirm given folders exist--
    if not os.path.exists(DATASET_PATH):
        print('Confirm the given path to input images exists.')
        assert os.path.exists(DATASET_PATH), 'Confirm the given path for training images directory exists.'

    if not os.path.exists(OUTPUT_PATH):
        print('Confirm the given path to output directory exists.')
        assert os.path.abspath(OUTPUT_PATH) != os.path.abspath(DATASET_PATH), \
            'Please provide unique output path  (not model or data path).'

    main()

    # Save dict to file
    pickle.dump(VAR_DICT_LOCAL, open('var_dict.p', 'wb'))
