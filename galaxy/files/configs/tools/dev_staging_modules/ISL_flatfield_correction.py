""" This script is intended to be an image processing module for galaxy pipeline. The equation used
 is from ImageJ BigStitcher plugin site: https://imagej.net/BigStitcher_Flatfield_correction
 6/25/2019 """

import argparse
import pickle
import os
import cv2
import numpy as np
import utils
import warnings

# from matplotlib import pyplot as plt

IMAGES_PATH = ''
Troubleshooting = False


def bright_image_maker(image_file_list_tp):
    """
    The following function makes the bright image out of median pixel intensities in a tile-stack matrix.

    :param image_file_list_tp: list of all the images in a well, for a channel and for a specific time point
    :return: per pixel medial intensity in a stack of all the tiles in each well.
    """

    stack_matrix = stack_mat(image_file_list_tp)
    utils.create_dir(OUTPUT_PATH)
    median_image = median_image_calculate(stack_matrix)

    return median_image


def stack_mat(image_stack_list):
    """
    The stack_mat function uses a list of image paths to read the images' pixel values into a matrix.

    :param image_stack_list: The list of all images to be processed.
    :return: Matrix of all the images pixel vales (slice, x-dimension, y-dimension).
    """

    stack_matrix = []
    Troubleshooting_stack_mat = False
    for image_path in image_stack_list:

        slice_img = cv2.imread(image_path, -1)
        if Troubleshooting_stack_mat: print("slice_img is: ", slice_img)
        if slice_img is not None:
            stack_matrix.append(slice_img)

        else:
            raise Exception('%s is corrupted, please run Corrupted Images Detector to find out '
                            'all the corrupted images.' % (os.path.basename(image_path)))

    return stack_matrix


def median_image_calculate(stack_matrix):
    """
    The median_image_graph function uses a matrix of images' pixel values to calculate the median image.

    :param stack_matrix: Matrix of all the images pixel vales (slice, x-dimension, y-dimension). :return: display the
    median image using matplotlib plot function and calculate the  per pixel medial intensity in a stack of all the
    tiles in each well.
    """

    Troubleshooting_median_image_calculate = False
    print("The stack matrix native shape is: ", np.shape(stack_matrix))
    tiles, row, col = np.shape(stack_matrix)
    if tiles > 0:
        median_image = np.median(stack_matrix, axis=0)
        if Troubleshooting_median_image_calculate: print("the median image is: ", median_image)
    else:
        print("stack_matrix_array size error!")
        median_image = np.ones((row, col))

    # # histogram comparisons: pyplot.hist vs. numpy hist ploted using pyplot.plot
    #
    # median_image_1d = median_image.ravel()
    # print("median_image_1d is: ", median_image_1d)
    # plt.figure("Median Image Histogram: ")
    # plt.subplot(131)
    # plt.title('Matplotlib histogram')
    # plt.hist(median_image_1d, bins=256, range=(median_image_1d.min(), median_image_1d.max()),
    #          density=True)
    # plt.subplot(132)
    # plt.title('Normalized histogram')
    # median_image_normalized = (median_image_1d - median_image_1d.min()) / \
    #                           (median_image_1d.max() - median_image_1d.min())
    # plt.hist(median_image_normalized, bins=256, density=True)
    # # plt.yscale('log')
    # plt.subplot(133)
    # plt.title('Median Image')
    # plt.imshow(median_image, cmap='gray')
    # plt.show()

    return median_image


def dark_image_calculate(output_path, med_image, img_1_path):
    """

    :rtype: dark_img: np.uint16
    :param output_path: the output path user chose.
    :param med_image: The median image.
    :param img_1_path: File list of all images
    :return: dark image and the mean for standardized image
    """
    dark_img_list = []

    row, col = cv2.imread(img_1_path, -1).shape
    assert np.size(med_image), "Median image is missing."

    # /mnt/finkbeinernas/robodata
    dark_image_path = \
        utils.make_filelist_wells('/finkbeiner/imaging/smb-robodata/Sina/data_sample/dark_image/', '.tif')
    if not os.path.exists(dark_image_path[0]):
        print("The dark image missing, setting it to zero.")
        dark_img = np.zeros((row, col), dtype=np.uint16)
    else:
        for img_name in dark_image_path:
            dark_img_list.append(cv2.imread(img_name, -1))
        print("dark image list is: ", np.shape(dark_img_list))
        dark_img = np.median(dark_img_list, axis=0)

    dark_img_path_name = os.path.join(output_path, 'QualityControl', 'dark_image.tif')
    cv2.imwrite(dark_img_path_name, np.array(dark_img, dtype=np.uint16))

    return dark_img


def flatfield_corrector(dataset_path, output_path, var_dict_local, valid_channel):
    """
    The flatfield_corrector function uses the median image as the bright image to correct images.

    :param dataset_path: the input path.
    :param output_path: the output path.
    :param var_dict_local: the variable dictionary from create folder module.
    :param valid_channel: The channel user chose to process.
    :return: The flatfield corrected image.
    """

    img_corrected = []
    valid_wells = var_dict_local['Wells']
    valid_timePoints = var_dict_local['TimePoints']
    # z_size = np.sort(var_dict_local['Depths']).__str__()[1:-1].split()
    # print("z-sizes are: ", z_size)
    output_path_ffc = os.path.join(output_path, 'FlatfieldCorrected')
    utils.create_dir(output_path_ffc)

    for well in valid_wells:
        print("We are on well: ", well)
        output_path_ffc_well = os.path.join(output_path_ffc, well)
        utils.create_dir(output_path_ffc_well)
        print("The output path for the FFC well is: ", output_path_ffc_well)

        dataset_path_w = os.path.join(dataset_path, well)
        img_file_list = utils.make_filelist(dataset_path_w, valid_channel)
        if Troubleshooting: print("img_file_list is: ", img_file_list)
        for tp in valid_timePoints:
            print("We are on time point: ", tp)
            percent_change_matrix = []
            image_file_list_tp = [fn for fn in img_file_list if tp == fn.split('/')[-1].split('_')[2]]
            if Troubleshooting: print("The img_file_list_tp is: ", image_file_list_tp)
            # Tried to make a list of each z-plane separately for their median image and correction, but it didn't
            # improve FFC.
            # if BIRGHTFIELD channel then check z-step, else the tp file list
            # for zs in z_size:
            #     print("We are on z-step: ", zs)
            #     image_file_list_tp_zs = [fn for fn in image_file_list_tp if zs == fn.split('/')[-1].split('_')[8]]
            #     print("file z-step list is: ", image_file_list_tp_zs)
            assert image_file_list_tp != [], 'image file list at this time point is empty, please check input path.'
            img_1_path = image_file_list_tp[0]
            med_image = bright_image_maker(image_file_list_tp)
            if not utils.make_filelist(os.path.join(output_path, 'QualityControl'), '.tif')\
                    .__contains__('dark_image.tif'):
                dark_img = dark_image_calculate(output_path, med_image, img_1_path)
            else:
                print("dark image found!")
                dark_img_path = utils.make_filelist(os.path.join(output_path, 'QualityControl'),
                                                    '.tif').__contains__('dark_image.tif')[0]
                print("The dark image path is: ", dark_img_path)
                dark_img = cv2.imread(dark_img_path, -1)

            for img_path in image_file_list_tp:

                if img_path.find(valid_channel) > 0:
                    if Troubleshooting: print("The channel is: ", valid_channel)
                    img_array = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
                    # if Troubleshooting: print("\n the np.mean((med_image - dark_img) is: ",
                    #                           np.mean(med_image - dark_img))
                    # Conditional for divide by zero, don't divide

                    try:
                        img_corrected = (img_array - dark_img) * np.median(med_image - dark_img) / \
                                        (med_image - dark_img)
                    except ZeroDivisionError:
                        print("med_image - dark_img", med_image-dark_img)
                        img_corrected = (img_array - dark_img) * np.median(med_image - dark_img) / med_image
                    warnings.simplefilter("ignore")
                    percent_change = FFC_measure(img_array, img_corrected)
                    percent_change_matrix.append([os.path.basename(img_path), percent_change])
                    base = os.path.basename(img_path)
                    if Troubleshooting: print("the base of the file name is: ", base)
                    new_file_name = os.path.join(str(output_path_ffc_well), base)
                    img_corrected_clipped = img_corrected.clip(0.115, 65535)
                    img_corrected_converted = np.array(img_corrected_clipped, dtype=np.uint16)
                    if Troubleshooting: print("The corrected Image is: ", img_corrected_converted)
                    cv2.imwrite(new_file_name, img_corrected_converted)
                    # if Troubleshooting: ffc_histogram_plots(img_path, img_array, dark_img, med_image,
                    #                                         valid_channel, img_corrected_converted)
                else:
                    break
            print("-------------------------------------------------------------------------------")
            print("The percent change after correction is: ",
                  percent_change_matrix)
        print("The flat-field corrected images are in: ", output_path_ffc_well)

    return img_corrected_converted


def FFC_measure(img_array, img_corrected):
    """
    The FFC_measure function uses the image and corrected image brightest 1/2 and darkest 1/2 pixel values to
    measure percent change. In other words, the top 50% pixel intensities - bottom 50% pixel intensities compared to
    original image.

    :param img_array: The pixel values of the original image being corrected.
    :param img_corrected: The pixel values of the corrected image.
    :return: Percent change between the difference between the mean of the top 1/2 and bottom 1/2 pixel intensities
    in the original image, and the corrected image.
    """

    percent_change = 0
    img_corrected_1d = img_corrected.ravel()
    img_array_1d = img_array.ravel()
    img_array_pix_range = img_array_1d.max() - img_array_1d.min()
    img_corrected_1d_median = np.median(img_corrected_1d)
    img_array_1d_median = np.median(img_array_1d)

    if img_array_pix_range > 55:  # to avoid blank images.
        if img_corrected_1d_median > 1 and img_array_1d_median > 1:
            percent_change_img_corr = float((img_corrected_1d_median - img_array_1d_median)) / \
                                      img_array_1d_median * 100.0
            percent_change_img_corr = np.round(percent_change_img_corr, 0)
        else:
            print("Too few pixels to calculate percent change after FFC.")
            percent_change_img_corr = 0

        percent_change = np.max(abs(percent_change_img_corr))

    return percent_change


# def ffc_histogram_plots(img, img_array, d, b, valid_channel, img_corrected):
#     """
#
#     :param img: the path to the file name being processed.
#     :param img_array: pixel values of the image bing processed
#     :param d: dark image
#     :param b: bright image, here it is known as median image
#     :param valid_channel: the channel user chose
#     :param img_corrected: the corrected image
#     :return: plots of the image and their corresponding histograms as the image is corrected, normalized,
#     and standardized.
#     """
#
#     # plt.figure("Histograms for raw - dark: " + img)
#     # plt.subplot(231)
#     # plt.imshow(img_array, cmap='gray')
#     # plt.title('Original Image')
#     # plt.subplot(232)
#     # plt.imshow(d, cmap='gray')
#     # plt.title('dark image')
#     # raw_minus_dark = img_array - d
#     # plt.subplot(233)
#     # plt.imshow(raw_minus_dark, cmap='gray')
#     # plt.title('raw - dark')
#     # plt.subplot(234)
#     # plt.hist(img_array.ravel(), bins=256, density=True)
#     # plt.title('Original Image hist')
#     # plt.subplot(235)
#     # plt.hist(d.ravel(), bins=256, density=True)
#     # plt.title('dark Image(d) hist')
#     # plt.subplot(236)
#     # plt.hist(raw_minus_dark.ravel(), bins=256, density=True)
#     # plt.title('raw - dark hist')
#     #
#     # plt.figure("Histograms for ((R - D)* avg(B-D)): " + img)
#     # plt.subplot(231)
#     # plt.imshow(b, cmap='gray')
#     # plt.title('Bright Image')
#     # plt.subplot(232)
#     # plt.imshow(d, cmap='gray')
#     # plt.title('Dark image')
#     # if Troubleshooting: print("the shape of bright image is: ", b.shape)
#     # if Troubleshooting: print("the bright image is: ", b)
#     # if Troubleshooting: print("the shape of dark image is: ", d.shape)
#     # bright_minus_dark = b - d
#     # avg_bright_minus_dark = np.round(np.mean(bright_minus_dark))
#     # raw_minus_dark_times_avg = avg_bright_minus_dark * raw_minus_dark
#     # plt.subplot(233)
#     # plt.imshow(raw_minus_dark_times_avg, cmap='gray')
#     # plt.title('(r-d)* mean(bright - dark image)')
#     # plt.subplot(234)
#     # plt.hist(b.ravel(), bins=256, density=True)
#     # plt.title('Original Image hist')
#     # plt.subplot(235)
#     # plt.hist(d.ravel(), bins=256, density=True)
#     # plt.title('dark Image(d) hist')
#     # plt.subplot(236)
#     # plt.hist(raw_minus_dark_times_avg.ravel(), bins=256, density=True)
#     # plt.title('(r-d)* mean(bright - dark image)')
#     #
#     # plt.figure("Histograms for ((R - D)* avg(B-D))/ (B-D): " + img)
#     # plt.subplot(231)
#     # plt.imshow(raw_minus_dark_times_avg, cmap='gray')
#     # plt.title('(R - D)* avg(B-D)')
#     # plt.subplot(232)
#     # plt.imshow(bright_minus_dark, cmap='gray')
#     # plt.title('B-D')
#     # corrected_image = raw_minus_dark_times_avg / bright_minus_dark
#     # if Troubleshooting: print("the corrected_image is: ", corrected_image)
#     # plt.subplot(233)
#     # plt.imshow(corrected_image, cmap='gray')
#     # plt.title('[(R - D)* avg(B-D)] / (B-D)')
#     # plt.subplot(234)
#     # raw_minus_dark_times_avg_1d = raw_minus_dark_times_avg.ravel()
#     # plt.hist(raw_minus_dark_times_avg_1d, bins=256, density=True)
#     # plt.title('(R - D)* avg(B-D) hist')
#     # plt.subplot(235)
#     # bright_minus_dark_1d = bright_minus_dark.ravel()
#     # plt.hist(bright_minus_dark_1d, bins=256, density=True)
#     # plt.title('B-D hist')
#     #
#     # plt.subplot(236)
#     # corrected_image_1d = corrected_image.ravel()
#     # plt.hist(corrected_image_1d, bins=256, density=True)
#     # plt.title('[(R - D)* avg(B-D)] / (B-D) hist')
#     # plt.show()
#     if Troubleshooting and img.find('B2_6') > 0:
#         plt.figure("Flatfield Correction: " + valid_channel)
#         plt.subplot(221)
#         plt.title('Original Image')
#         plt.imshow(img_array, cmap='gray')
#         plt.subplot(222)
#         plt.title('Corrected Image')
#         plt.imshow(img_corrected, cmap='gray')
#         plt.subplot(223)
#         plt.title('Original Histogram')
#         img_array_1d = img_array.ravel()
#         plt.hist(img_array_1d, bins=256, range=(img_array_1d.min(), img_array_1d.max()),
#                  density=True)
#         plt.yticks([0, 0.00025])
#         plt.subplot(224)
#         plt.title('Corrected Histogram')
#         img_corrected_1d = img_corrected.ravel()
#         plt.hist(img_corrected_1d, bins=256, range=(img_corrected_1d.min(), img_corrected_1d.max()),
#                  density=True)
#         plt.yticks([0, 0.00025])
#         plt.show()
#
#     return


def main():
    """
    Main runs the bright_image_maker, where it calculates median pixels intensities of all the tiles in a matrix,
    and uses the median image then to flatfield correct the images in various channels using the following arguments:

    Args:
    VAR_DICT_LOCAL: importing the VAR_DICT_LOCAL dictionary to use its arguments such as valid wells,
    time points in the loops.
    valid_channel: the channels the user wants to work with.
    DATASET_PATH: the location of input images.
    output_path: The location where the flatfield corrected images will be stored at.
    median_image: the image that is the output of bright_image_maker function and input of flatfield_corrector
    function.
    """

    print("Valid channels are: ", VALID_CHANNELS)

    for valid_channel in VALID_CHANNELS:
        print("We are on Channel: ", valid_channel)
        flatfield_corrector(DATASET_PATH, OUTPUT_PATH, VAR_DICT_LOCAL, valid_channel)


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

    CHANNELS = str(args.channels).replace(" ", "")
    print("The channels are: ", CHANNELS)
    VALID_CHANNELS = CHANNELS.split(',')

    # Z_SIZE = np.sort(VAR_DICT_LOCAL['Depths']).__str__()[1:-1].split()
    # print("The z-size is: ", Z_SIZE)

    # ----Confirm given folders exist--
    if not os.path.exists(DATASET_PATH):
        print('Confirm the given path to input images exists.')
        assert os.path.exists(DATASET_PATH), 'Confirm the given path for training images directory exists.'

    if not os.path.exists(OUTPUT_PATH):
        print('Confirm the given path to output directory exists.')
        assert os.path.abspath(OUTPUT_PATH) != os.path.abspath(DATASET_PATH), \
            'Please provide unique output path(data path).'

    main()

    # Save dict to file
    pickle.dump(VAR_DICT_LOCAL, open('var_dict.p', 'wb'))
