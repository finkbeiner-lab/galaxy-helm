'''

'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

import argparse
import configparser
import glob
import os
import pprint

import cv2
import numpy as np
from scipy import stats
import tensorflow as tf

print("TensorFlow version: {}".format(tf.__version__))

# print("Eager execution: {}".format(tf.executing_eagerly()))


# I/O data setup functions
def make_filelist(path, identifier, verbose=False):
    '''
    Takes a directory and a string identifier.
    Returns a list of files.
    '''
    filelist = sorted(glob.glob(os.path.join(path, '*' + identifier + '*')))

    if verbose:
        print("Selection filter:")
        print(os.path.join(path, '*' + identifier + '*'))
        pprint.pprint([os.path.basename(fel) for fel in filelist])

    assert len(filelist) > 0, os.path.join(path, identifier)

    return filelist


def hash_split(elements, p=0.8):
    if p < 0 or p > 1: raise ValueError
    training, validation = [], []
    for element in elements:
        training.append(element) \
            if hash(element) % 10 < (p * 10) else validation.append(element)
    return training, validation


def rehash_data_structure(elements):
    rehashed_elements = []
    for element in elements:
        rehashed_elements.append(element)
    return rehashed_elements


def get_label_from_filename(image_filename, tokenWithLabel, verbose=False):
    'Take filename path and token, return label.'
    id_tokens = os.path.basename(image_filename).split('_')
    joined_id_tokens = '-'.join(
        [id_tokens[2], id_tokens[4], id_tokens[tokenWithLabel]])
    label = os.path.splitext(joined_id_tokens)[0]
    if verbose:
        print('Filename to generate label with:', image_filename)
        print('Token given:', tokenWithLabel)
        print('Label extracted:', label)
    # return image_filename, label
    # return label
    return os.path.basename(image_filename)


# Normalization functions
def get_per_pixel_half_max(image_filenames, expected_dtype, verbose=True):
    '''Take image filenames and return a per pixel maximum image_filelist
    divided by two.'''
    half_max_image = np.zeros((IMG_DIM, IMG_DIM), dtype=expected_dtype)
    for filestring in image_filenames:
        new_img = cv2.imread(filestring, -1)
        assert new_img.dtype == expected_dtype, '%s %s' % (new_img.dtype,
                                                           filestring)
        half_max_image = np.maximum(half_max_image, new_img)
        if expected_dtype == np.uint8:
            assert np.all(half_max_image < 256), filestring
    if verbose:
        print('Max image:')
        print(half_max_image[0:3, 0:3])
        print('Halved max_img at generation:', ((half_max_image // 2)).dtype,
              ((half_max_image[0:3, 0:3] // 2)))
        print(half_max_image.min(), half_max_image.max(),
              (half_max_image // 2).min(), (half_max_image // 2).max())
    return half_max_image // 2


def calculate_fraction_on_pixels(image_filenames, max_val=1):
    ''''Takes a list of files and returns a mean for all pixels
    added in annotation. This value is used to later weight loss
    and reward the model for learning the relatively few examples
    of positive annotation.'''
    running_total_pixels = 0
    running_sum_on_pixels = 0
    for filestring in image_filenames:
        new_img = cv2.imread(filestring, -1)
        running_total_pixels += new_img.size
        fraction_img_on = np.mean(new_img == max_val)
        running_sum_on_pixels += fraction_img_on * new_img.size
    return np.true_divide(running_sum_on_pixels, running_total_pixels)


def get_norm_output(image_filenames, expected_dtype, max_val=1):
    '''Get the parameters needed for normalization:
    fraction of annotated pixels and the half max image.'''
    directory = os.path.dirname(image_filenames[0])
    if os.path.exists(os.path.join(directory, 'half_max_img.npy')):
        # Read in as numpy array rather than image
        half_max_img = np.load(os.path.join(directory, 'half_max_img.npy'))
        # half_max_img = cv2.imread(
        #     os.path.join(directory, 'half_max_img.png'), -1)
        print('Saved half max image properties:', half_max_img.shape)
        print('Type of half_max_img at **read in**:', half_max_img.dtype)
        print('Min:', half_max_img.min(), 'Max:', half_max_img.max())
    else:
        half_max_img = get_per_pixel_half_max(image_filenames, expected_dtype)
        # Write as numpy array rather than image
        np.save(os.path.join(directory, 'half_max_img.npy'), half_max_img)
        # cv2.imwrite(os.path.join(directory, 'half_max_img.png'), half_max_img)
        print('Wrote half max image to: ',
              os.path.join(directory, 'half_max_img.png'))
        print('Type of half_max_img at **creation**:', half_max_img.dtype)
        print('Min:', half_max_img.min(), 'Max:', half_max_img.max())
    if os.path.exists(os.path.join(directory, 'fraction_on_txt.txt')):
        # Read from npy
        print('Fraction on value in file:',
              open(os.path.join(directory, 'fraction_on_txt.txt'), 'r').read())
        fraction_on = float(
            open(os.path.join(directory, 'fraction_on_txt.txt'), 'r').read())
    else:
        fraction_on = calculate_fraction_on_pixels(image_filenames,
                                                   max_val=max_val)
        print('*****Max value*****:', max_val, '--', 'Fraction on:',
              fraction_on)
        # Write as npy
        open(os.path.join(directory, 'fraction_on_txt.txt'),
             'w').write(str(fraction_on))
        print('Wrote fraction on output to:',
              os.path.join(directory, 'fraction_on_txt.txt'))
    return half_max_img, fraction_on


# Matching file lists functions
def get_relevant_token_keys(image_filename, token_position_list,
                            token_annot_list):
    '''Take filename and relevant token position and annotation list
    and generate a key.
    Time token is 2 in all naming conventions.
    Well token is 4 in all naming conventions.
    @Usage
    Example for grouping all channels for montaged images
    of a particular well and time point:
    image_filename =
    PID20180706_070418PINK1ParkinSurvival1_T5_60-0_A1_0_RFP-DFTrCy5_MN.tif
    token_position_list = [2,4]
    token_annot_list = ["",""]
        name_list included for cases where token value field is
        ambiguous and needs to be prepended with annotation
        (ex. montage panel and hours).'''

    image_fn_tokens = os.path.basename(image_filename).split('_')
    token_keys = []
    for token_pos, token_name in zip(token_position_list, token_annot_list):
        token_keys.append(token_name + image_fn_tokens[token_pos])
    # return ','.join(sorted(token_keys))
    return ','.join(token_keys)


def get_file_groups(image_filelist, token_position_list, token_annot_list):
    '''Take filelist and lists of annotations and positions and returns
    dict with keys of each combination of relevant tokens and values of
    image lists corresponding to each.'''

    output = {}
    for image_filename in image_filelist:
        token_key = get_relevant_token_keys(image_filename,
                                            token_position_list,
                                            token_annot_list)
        if token_key not in output:
            output[token_key] = []
        output[token_key].append(image_filename)
    return output


def make_filelists_equal(raw_image_filelist_in,
                         traced_image_filelist_in,
                         verbose=False):
    '''Take filelist and lists of annotations and positions and returns
    dict with keys of each combination of relevant tokens and values of
    image lists corresponding to each.'''

    print('Lengths of initial image files (raw, traced):',
          len(raw_image_filelist_in), len(traced_image_filelist_in))
    image_filelist = raw_image_filelist_in + traced_image_filelist_in
    raw_image_filelist, traced_image_filelist, labels = [], [], []
    token_file_dict = get_file_groups(image_filelist,
                                      [0, 1, 2, 3, 4, 5, 6, -1],
                                      ['', '', '', 'H', '', 'P', '', ''])

    for token_key in token_file_dict:
        if len(token_file_dict[token_key]) == 2:
            for filename in token_file_dict[token_key]:
                if MASK_STRING in os.path.basename(filename):
                    traced_image_filelist.append(filename)
                else:
                    raw_image_filelist.append(filename)
                    labels.append(get_label_from_filename(filename, -1))
    print('Lengths of final image files (raw, traced):',
          len(raw_image_filelist), len(traced_image_filelist))
    if verbose:
        print('Raw images collected:')
        pprint.pprint(raw_image_filelist)
        print('Traced images collected:')
        pprint.pprint(traced_image_filelist)
        print('Labels:')
        pprint.pprint(labels)
    assert len(raw_image_filelist) == len(
        traced_image_filelist), 'Image lists are not the same lengths.'
    assert len(raw_image_filelist) == len(
        labels), 'Image lists and labels are not the same lengths.'
    return raw_image_filelist, traced_image_filelist, labels


# Network layers functions
def make_conv_layer(input_layer,
                    num_filters,
                    xypool,
                    kernel_dim=5,
                    verbose=False):  #argument was x, pixels
    'Conv layer creation.'
    # Convolutional Layer and Pooling Layer
    conv = tf.layers.conv2d(
        inputs=input_layer,
        filters=num_filters,
        kernel_size=[kernel_dim, kernel_dim],
        padding="same",
        # padding="valid",
        activation=tf.nn.relu)  #this adds biases tf.zeros_initializer()
    pool = tf.layers.max_pooling2d(inputs=conv,
                                   pool_size=[xypool, xypool],
                                   strides=xypool)
    if verbose:
        print("Dimensions conv: ", conv.shape)
        print("Dimensions pool: ", pool.shape)
    return pool


def make_deconv_layer(enc_layer,
                      num_filters,
                      xypool,
                      kernel_dim=5,
                      verbose=False):
    'Deconv layer creation.'
    # Convolutional and unpooling layer
    dconv = tf.layers.conv2d(
        inputs=enc_layer,
        filters=num_filters,
        kernel_size=[kernel_dim, kernel_dim],
        padding='SAME',
        # padding='valid',
        use_bias=True,
        activation=tf.nn.relu)
    unpool = tf.layers.conv2d_transpose(
        dconv,
        filters=num_filters,
        kernel_size=[xypool, xypool],
        padding='SAME',
        # padding='valid',
        strides=xypool)
    if verbose:
        print("Dimensions dconv: ", dconv.shape)
        print("Dimensions unpool: ", unpool.shape)
    return unpool


def conv_encoder(x, dimension_list, given_kernel=5):
    '''
    Takes an input tensor and dimension list for all layers.
    Returns an encoded layer with the number of intermediate
    layers equal to the number of elements in the dimension_list.
    Dimension list takes the form, [(num_filters1, xypooling-factor1),
    (num_filters2, xypooling-factor2)...]
    '''
    input_layer = tf.reshape(
        x, [-1, IMG_DIM, IMG_DIM, 1])  # x is the features matrix

    current_layer = input_layer
    for num_filters, xypool in dimension_list:
        current_layer = make_conv_layer(current_layer,
                                        num_filters,
                                        xypool,
                                        kernel_dim=given_kernel,
                                        verbose=True)
    print('Final encoded layer dimensions:', current_layer.shape)
    return current_layer


def conv_decoder(x, dimension_list, given_kernel=5, output_binary=False):
    '''
    Takes an input (embeddings) tensor and dimension list for all layers.
    Returns a decoded layer with the number of intermediate layers equal
    to the number of elements in the dimension_list.
    Dimension list takes the form, [(num_filters1, xypooling-factor1),
    (num_filters2, xypooling-factor2)...]
    '''
    current_layer = x
    for (num_filters, xypool) in dimension_list:
        # if
        current_layer = make_deconv_layer(current_layer,
                                          num_filters,
                                          xypool,
                                          kernel_dim=given_kernel,
                                          verbose=True)
    if output_binary:
        current_layer = tf.nn.sigmoid(current_layer)
    print('Final decoded layer dimensions:', current_layer.shape)
    return current_layer


# Flatten convolution layer
def create_flatten_layer(layer):
    'Takes layer, returns flattened layer.'
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer


def return_weights(label_vector, fraction_on=1):
    'Function dealing with weighting of the sparse on pixels.'
    # label_max = 1
    # label_max = tf.clip_by_value(label_vector, clip_value_max=1)
    # num_on_pixels = tf.equal(label_vector, label_max)
    # print('num_on_pixels', num_on_pixels)
    # weighted_value = num_on_pixels/NUM_INPUT
    # print(weighted_value)
    label_vector = (label_vector + 1) // 2.
    return label_vector + fraction_on


def linear_norm_depth_range(np_array, depth):
    '''Take input array and stretch values
    to full depth range for visualization'''
    normed_array = (np_array -
                    np_array.min()) * (np.iinfo(depth).max /
                                       (np_array.max() - np_array.min())) + 0
    return normed_array


# Prediction output functions
def visualize_before_after(num_panels, start_index, rawimgs, predictedimgs,
                           tracedimgs, labels, global_step_res, rmse):
    '''
    Takes:
    A tensor with a holdout batch equal to batch size.
    num_panels: Number of panels on each side of output.
    A value of 2 gives a 2x2 (4 panels).
    start_index: The first index of otherwise
    sequential selection of rows (cells).
    rawimgs: The original image pixel values
    predictedimgs: result of decoded autoencoder (the predicted image)
    tracedimgs: The target image the autoencoder should learn
    img_dest: string containing file destination to save to
    labels: The cell ID for the original image, formatted as:
    time-well-cell#-x-y-area
    Returns:
    Plot of num_panels x num_panels of cells [original, predicted, target]
    Associated csv with the labels associated with each cell in the image plot.
    '''
    print('Evaluating image reconstruction:')
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors

    vis_out_path = os.path.join(MODEL_LOCATION, 'visualizations')
    if not os.path.exists(vis_out_path):
        os.makedirs(vis_out_path)
    loss_step = 'step{step_number}_loss{rmse}'.format(
        step_number=global_step_res, rmse=round(rmse, 4))

    label_txt_name = os.path.join(vis_out_path,
                                  'validation_' + loss_step + '.txt')
    # with open(label_txt_name, 'w') as label_txt:
    # label_txt.write('\n'.join(labels[start_index:num_panels*num_panels+1]))
    label_txt = open(label_txt_name, 'w')

    border = 0
    raw_images = np.zeros((IMG_DIM * num_panels + (num_panels - 1) * border,
                           IMG_DIM * num_panels + (num_panels - 1) * border))
    predictions = np.copy(raw_images)
    traced_images = np.copy(raw_images)

    if verbose:
        print(IMG_DIM, border, num_panels,
              IMG_DIM * num_panels + (num_panels - 1) * border)
        print('raw_images.shape', raw_images.shape, 'predictions.shape',
              predictions.shape, 'traced_images.shape', traced_images.shape)

    for i in range(num_panels):
        for j in range(num_panels):
            img_index = start_index + (i * num_panels + j)

            row_pixel_start = i * IMG_DIM + i * border
            row_pixel_end = (i + 1) * IMG_DIM + i * border
            col_pixel_start = j * IMG_DIM + j * border
            col_pixel_end = (j + 1) * IMG_DIM + j * border

            if SCALE_TYPE == 'm1to1':
                raw_scaled = (
                    rawimgs[img_index, :].reshape([IMG_DIM, IMG_DIM]) *
                    HALF_MAX_RAW_IMG) + HALF_MAX_RAW_IMG
                pred_scaled = (
                    predictedimgs[img_index, :].reshape([IMG_DIM, IMG_DIM]) *
                    HALF_MAX_TRACED_IMG) + HALF_MAX_TRACED_IMG
                traced_scaled = (
                    tracedimgs[img_index, :].reshape([IMG_DIM, IMG_DIM]) *
                    HALF_MAX_TRACED_IMG) + HALF_MAX_TRACED_IMG
            if SCALE_TYPE == '0to1':
                raw_scaled = (rawimgs[img_index, :].reshape(
                    [IMG_DIM, IMG_DIM]))
                pred_scaled = (predictedimgs[img_index, :].reshape(
                    [IMG_DIM, IMG_DIM]))
                traced_scaled = (tracedimgs[img_index, :].reshape(
                    [IMG_DIM, IMG_DIM]))
            else:
                assert SCALE_TYPE in ('m1to1', '0to1'), 'No scale type found.'

            raw_images[row_pixel_start:row_pixel_end,
                       col_pixel_start:col_pixel_end] = raw_scaled
            predictions[row_pixel_start:row_pixel_end,
                        col_pixel_start:col_pixel_end] = pred_scaled
            traced_images[row_pixel_start:row_pixel_end,
                          col_pixel_start:col_pixel_end] = traced_scaled

            print('Original cells shown:', labels[img_index])
            label_txt.write(labels[img_index] + '\n')

    label_txt.close()

    if verbose:
        print('raw min max', raw_images.min(), raw_images.max())
        print('predicted min max', predictions.min(), predictions.max())
        print('traced min max', traced_images.min(), traced_images.max())

    plt.subplot(131)
    plt.imshow(raw_images,
               origin="upper",
               cmap="gray",
               norm=colors.PowerNorm(gamma=1 / 3.))
    # Other visualization
    # norm=colors.PowerNorm(gamma=1 / 3.)
    # norm=colors.Normalize(vmax=0.25 * raw_images.max())
    # norm=colors.LogNorm()
    plt.title('Original Images')
    plt.tick_params(which='both',
                    bottom=False,
                    top=False,
                    left=False,
                    right=False,
                    labelleft=False,
                    labelbottom=False)

    plt.subplot(132)
    plt.imshow(predictions, origin="upper", cmap="gray")
    plt.title('Predicted Images')
    plt.tick_params(which='both',
                    bottom=False,
                    top=False,
                    left=False,
                    right=False,
                    labelleft=False,
                    labelbottom=False)

    plt.subplot(133)
    plt.imshow(traced_images, origin="upper", cmap="gray")
    plt.title('Target Images')
    plt.tick_params(which='both',
                    bottom=False,
                    top=False,
                    left=False,
                    right=False,
                    labelleft=False,
                    labelbottom=False)

    plt.savefig(os.path.join(vis_out_path, 'validation_' + loss_step + '.pdf'),
                dpi=999)
    plt.close()

    raw_images = cv2.equalizeHist(
        linear_norm_depth_range(raw_images, np.uint8).astype(np.uint8))
    predictions = linear_norm_depth_range(predictions,
                                          np.uint8).astype(np.uint8)
    traced_images = linear_norm_depth_range(traced_images,
                                            np.uint8).astype(np.uint8)
    if verbose:
        print('raw min max', raw_images.min(), raw_images.max())
        print('predicted min max', predictions.min(), predictions.max())
        print('traced min max', traced_images.min(), traced_images.max())

    # raw-traced overlay
    raw_traced_image = cv2.merge(
        (np.zeros(raw_images.shape, np.uint8), traced_images, raw_images))
    cv2.imwrite(
        os.path.join(vis_out_path, 'raw_traced_overlay' + loss_step + '.tif'),
        raw_traced_image)

    # traced-predicted overlay
    traced_predicted_image = cv2.merge(
        (predictions, traced_images, np.zeros(traced_images.shape, np.uint8)))
    cv2.imwrite(
        os.path.join(vis_out_path,
                     'traced_predicted_overlay' + loss_step + '.tif'),
        traced_predicted_image)

    # raw-predicted overlay
    raw_predicted_image = cv2.merge(
        (predictions, np.zeros(raw_images.shape, np.uint8), raw_images))
    cv2.imwrite(
        os.path.join(vis_out_path,
                     'raw_predicted_overlay' + loss_step + '.tif'),
        raw_predicted_image)

    print('Image written to:',
          os.path.join(vis_out_path, 'validation_' + loss_step + '.pdf'))


def preview_prediction(prediction_array, content_string=''):
    '''Preview the output of prediction before writing to disk.'''
    import matplotlib.pyplot as plt
    # if len(prediction_array) == 2:
    plt.imshow(prediction_array)
    # if len(prediction_array) == 3:
    # plt.imshow(cv2.cvtColor(prediction_array, cv2.COLOR_BGR2GRAY))
    # plt.title('Min: ' + str(round(prediction_array.min(), 2)) + ' Max: ' +
    #           str(round(prediction_array.max(), 2)))
    plt.title(content_string)
    # plt.clim(vmin=prediction_array.min(), vmax=prediction_array.max())
    plt.colorbar()
    plt.show()


def write_predicted_panel(predictedimgs, labels, output_path):
    '''
    Takes:
    Takes predicted image tile, generates name and writes tile to output path.
    predictedimgs: result of decoded autoencoder (the predicted image)
    output_path: string containing file destination to save to
    labels: The cell ID for the original image, formatted as: time-well-cell#-x-y-area
    Returns:
    Saved image tile in path
    '''
    print('Writing predicted tiles:')
    for img_index, label in enumerate(labels):

        # Generate name string
        label_tokens = os.path.splitext(label)[0].split('_')
        tile_name = '_'.join([
            '_'.join(label_tokens[0:6]), 'PREDICTED', label_tokens[-1]
        ]) + '.png'

        # Reset scale back to original range (part 1)
        if SCALE_TYPE == 'm1to1':
            HALF_MAX_TRACED_IMG = 127  # temporary test
            pred_img_scaled = (
                predictedimgs[img_index, :].reshape([IMG_DIM, IMG_DIM]) *
                HALF_MAX_TRACED_IMG) + HALF_MAX_TRACED_IMG
        if SCALE_TYPE == '0to1':
            pred_img_scaled = (predictedimgs[img_index, :].reshape(
                [IMG_DIM, IMG_DIM]) * np.iinfo(np.uint16).max) + 0

        if verbose and False:
            preview_prediction(pred_img_scaled,
                               content_string='Norm image corrected.')

        # Reset scale back to original range part 2
        # Things tried to set min log:
        if verbose:
            print('Min', pred_img_scaled.min(), 'Max', pred_img_scaled.max())
        # pred_img_scaled = (pred_img_scaled-pred_img_scaled.min())*(((2**8)-1)/(pred_img_scaled.max()-pred_img_scaled.min()))
        # pred_img_scaled = (pred_img_scaled - pred_img_scaled.min())
        # pred_img_scaled = (pred_img_scaled * np.iinfo(np.uint8).max).astype(np.uint8)
        pred_img_scaled = pred_img_scaled.astype(np.int16)
        if verbose:
            print('Min np.int16', pred_img_scaled.min(), 'Max np.int16',
                  pred_img_scaled.max())

        # Clip is not behaving how I expect in some cases.
        pred_img_scaled[pred_img_scaled < 0] = 0
        pred_img_scaled[pred_img_scaled > 2**8] = 2**8 - 1
        print('Min clipped', pred_img_scaled.min(), 'Max clipped',
              pred_img_scaled.max())
        if verbose and False:
            preview_prediction(pred_img_scaled,
                               content_string='After clipping.')

        # TODO threshold image
        pred_img_scaled = pred_img_scaled.astype(np.uint8)
        print('Min np.uint8', pred_img_scaled.min(), 'Max np.uint8',
              pred_img_scaled.max())
        assert pred_img_scaled.min() >= 0, 'Min value is out of range: ' + str(
            pred_img_scaled.min())
        assert pred_img_scaled.max(
        ) < 2**8, 'Max value is out of range: ' + str(pred_img_scaled.max())

        print(os.path.join(output_path, tile_name))
        if verbose and False:
            preview_prediction(pred_img_scaled)
        cv2.imwrite(os.path.join(output_path, tile_name), pred_img_scaled)


def write_predicted_panel_tests(predictedimgs, labels, output_path):
    '''
    Takes:
    Takes predicted image tile, generates name and writes tile to output path.
    predictedimgs: result of decoded autoencoder (the predicted image)
    output_path: string containing file destination to save to
    labels: The cell ID for the original image, formatted as: time-well-cell#-x-y-area
    Returns:
    Saved image tile in path
    '''
    print('Writing predicted tiles:')

    for img_index, label in enumerate(labels):
        label_tokens = os.path.splitext(label)[0].split('_')
        # tile_name = '_'.join(['_'.join(label_tokens[0:8]), 'PREDICTED', label_tokens[-1]])+'.png'
        tile_name = '_'.join([
            '_'.join(label_tokens[0:6]), 'PREDICTED', label_tokens[-1]
        ]) + '.png'
        pred_img_scaled = (predictedimgs[img_index, :].reshape(
            [IMG_DIM, IMG_DIM]) * HALF_MAX_TRACED_IMG) + HALF_MAX_TRACED_IMG

        print('Min', pred_img_scaled.min(), 'Max', pred_img_scaled.max())
        # pred_img_scaled = (pred_img_scaled - pred_img_scaled.min()) * ((
        #     (2**8) - 1) / (pred_img_scaled.max() - pred_img_scaled.min()))
        # pred_img_scaled = (pred_img_scaled - pred_img_scaled.min())
        # pred_img_scaled = (pred_img_scaled * np.iinfo(np.uint8).max).astype(np.uint8)
        print('Min np.int16', pred_img_scaled.min(), 'Max np.int16',
              pred_img_scaled.max())

        # Clip is not behaving how I expect in some cases: clipping manually
        pred_img_scaled[pred_img_scaled < 0] = 0
        pred_img_scaled[pred_img_scaled > np.iinfo(np.uint8).max] = np.iinfo(
            np.uint8).max - 1
        print('Min clipped', pred_img_scaled.min(), 'Max clipped',
              pred_img_scaled.max())

        pred_img_scaled = pred_img_scaled.astype(np.uint8)
        print('Min np.uint16', pred_img_scaled.min(), 'Max np.uint16',
              pred_img_scaled.max())
        assert pred_img_scaled.min() >= 0, 'Min value is out of range: ' + str(
            pred_img_scaled.min())
        assert pred_img_scaled.max() <= np.iinfo(
            np.uint8).max, 'Max value is out of range: ' + str(
                pred_img_scaled.max())

        print(os.path.join(output_path, tile_name))
        cv2.imwrite(os.path.join(output_path, tile_name), pred_img_scaled)


# Graph run time functions
def create_graph(training_image_target_labels_triplet,
                 validation_image_target_labels_triplet):
    ''''Graph creation.'''
    training_image_filenames, training_target_filenames, training_labels = training_image_target_labels_triplet
    validation_image_filenames, validation_target_filenames, validation_labels = validation_image_target_labels_triplet

    training_dataset = tf.data.Dataset.from_tensor_slices(
        (list(training_image_filenames), list(training_target_filenames),
         list(training_labels)))
    validation_dataset = tf.data.Dataset.from_tensor_slices(
        (list(validation_image_filenames), list(validation_target_filenames),
         list(validation_labels)))

    def _parse_function(transform_fn,
                        filename_raw_img,
                        filename_traced_img,
                        label,
                        verbose=True):
        'Mapping function for image preprocessing steps.'

        if not transform_fn:
            transform_fn = lambda x: x

        image_string_raw = tf.read_file(filename_raw_img)

        raw_decoded_image = tf.squeeze(  # Remove extraneous 1-channel dimension
            tf.cast(
                tf.image.decode_png(image_string_raw,
                                    dtype=tf.uint16,
                                    channels=1), tf.float64))

        if SCALE_TYPE == 'm1to1':
            hmri_tensor = tf.convert_to_tensor(HALF_MAX_RAW_IMG,
                                               dtype=tf.float64)
            raw_halfmaxed_image = transform_fn(
                tf.math.divide(
                    tf.math.subtract(raw_decoded_image, hmri_tensor),
                    hmri_tensor))
        if SCALE_TYPE == '0to1':
            min_tensor = tf.convert_to_tensor(np.zeros((IMG_DIM, IMG_DIM)),
                                              dtype=tf.float64)
            max_tensor = tf.convert_to_tensor(np.zeros(
                (IMG_DIM, IMG_DIM), dtype=np.uint16) + np.iinfo(np.uint16).max,
                                              dtype=tf.float64)
            raw_halfmaxed_image = transform_fn(
                tf.math.divide(tf.math.subtract(raw_decoded_image, min_tensor),
                               max_tensor))

        image_string_traced = tf.read_file(filename_traced_img)
        traced_decoded_image = tf.squeeze(  # Remove 1-channel dimension
            tf.cast(
                tf.image.decode_png(image_string_traced,
                                    dtype=tf.uint8,
                                    channels=1), tf.float64))

        if SCALE_TYPE == 'm1to1':
            hmti_tensor = tf.convert_to_tensor(HALF_MAX_TRACED_IMG,
                                               dtype=tf.float64)
            traced_halfmaxed_image = transform_fn(
                tf.math.divide(
                    tf.math.subtract(traced_decoded_image, hmti_tensor),
                    hmti_tensor))
        if SCALE_TYPE == '0to1':
            min_tensor = tf.convert_to_tensor(np.zeros((IMG_DIM, IMG_DIM),
                                                       dtype=np.uint16),
                                              dtype=tf.float64)
            max_tensor = tf.convert_to_tensor(np.zeros(
                (IMG_DIM, IMG_DIM), dtype=np.uint16) + np.iinfo(np.uint8).max,
                                              dtype=tf.float64)
            traced_halfmaxed_image = transform_fn(
                tf.math.divide(
                    tf.math.subtract(traced_decoded_image, min_tensor),
                    max_tensor))

        if verbose:
            print('raw_halfmaxed_image', tf.reduce_min(raw_halfmaxed_image),
                  tf.reduce_max(raw_halfmaxed_image))
            print('traced_halfmaxed_image',
                  tf.reduce_min(traced_halfmaxed_image),
                  tf.reduce_max(traced_halfmaxed_image))
            print('label', label)

        return tf.reshape(raw_halfmaxed_image,
                          shape=[-1]), tf.reshape(traced_halfmaxed_image,
                                                  shape=[-1]), label

    transforms = [None, tf.transpose]
    if AUG_DATA:
        transforms = [
            None,
            tf.image.flip_left_right,
            tf.image.flip_up_down,
            tf.transpose,
            lambda x: tf.image.flip_up_down(tf.image.flip_left_right(x)),
            lambda x: tf.image.random_contrast(
                x, lower=0.1, upper=0.6, seed=12),
            # lambda x: tf.image.random_brightness(x, max_delta=0.3, seed=12),
            # lambda x: tf.squeeze(tf.image.random_contrast(tf.expand_dims(x, 2), lower=0.1, upper=10, seed=12)),
            # lambda x: tf.image.random_brightness(x, max_delta=200, seed=12),
        ]

    training_datasets = [
        training_dataset.map(functools.partial(_parse_function, transform_fn))
        for transform_fn in transforms
    ]

    print('Number transformations:', len(training_datasets))
    training_dataset = reduce((lambda x, y: x.concatenate(y)),
                              training_datasets)
    print('Data type/shape:', training_dataset.output_types,
          training_dataset.output_shapes)

    shuffle_buffer = 200
    # training_dataset = training_dataset.repeat(100).batch(BATCH_SIZE)
    training_dataset = training_dataset.repeat(100).shuffle(
        buffer_size=shuffle_buffer).batch(BATCH_SIZE)
    # training_dataset = training_dataset.repeat(100).batch(BATCH_SIZE)
    validation_dataset = validation_dataset.map(
        functools.partial(_parse_function, None)).batch(BATCH_SIZE)

    iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                               training_dataset.output_shapes)
    RawImages, TracedImages, labels = iterator.get_next()

    # Filters and pooling
    if LAYERS == 2:
        # 128x1f ->  32x16f -> 8x32f -> [8x32f] ->  32x16f  -> 128x1f
        encoder_op = conv_encoder(RawImages, [(16, 4), (32, 4)],
                                  given_kernel=KERNEL)
        decoder_op = create_flatten_layer(
            conv_decoder(encoder_op, [(16, 4), (1, 4)],
                         given_kernel=KERNEL,
                         output_binary=SIG_END))
    if LAYERS == 3:
        # 128x1f -> 64x16f -> 32x32f -> 16x64f -> [16x64f] -> 16x64f -> 32x32f -> 64x16f -> 128x1f
        encoder_op = conv_encoder(RawImages, [(16, 2), (32, 2), (64, 2)],
                                  given_kernel=KERNEL)
        decoder_op = create_flatten_layer(
            conv_decoder(encoder_op, [(32, 2), (16, 2), (1, 2)],
                         given_kernel=KERNEL,
                         output_binary=SIG_END))
    if LAYERS == 4:
        # encoder_op = conv_encoder(RawImages, [(16,2), (16,1), (32,2), (64,2)], given_kernel=KERNEL)
        # decoder_op = create_flatten_layer(conv_decoder(encoder_op, [(32,2), (16,2), (16,1), (1,2)], given_kernel=KERNEL, output_binary=SIG_END))
        # 128x1f -> 64x8f -> 32x16f -> 16x32f -> [8x64f] -> 16x32f -> 32x16f -> 64x8f -> 128x1f
        encoder_op = conv_encoder(RawImages, [(8, 2), (16, 2), (32, 2),
                                              (64, 2)],
                                  given_kernel=KERNEL)
        decoder_op = create_flatten_layer(
            conv_decoder(encoder_op, [(32, 2), (16, 2), (8, 2), (1, 2)],
                         given_kernel=KERNEL,
                         output_binary=SIG_END))

    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = TracedImages

    # Define loss and optimizer, minimize the squared error
    weights = return_weights(y_true, fraction_on=FRACTION_ON)
    if not LOSSRMSE:
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,
                                                    logits=y_pred))
        # loss = tf.reduce_mean(tf.multiply(
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred), weights))
    else:
        loss = tf.reduce_mean(tf.multiply(tf.pow(y_true - y_pred, 2), weights))
        if not LOSS_WEIGHTING:
            loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
            # loss = tf.reduce_mean(y_true - y_pred)

    global_step = tf.train.create_global_step()
    optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(
        loss, global_step=global_step)

    tf.summary.scalar('Loss', loss)
    #TODO: add more summaries here
    summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(MODEL_LOCATION, 'train'))
    test_writer = tf.summary.FileWriter(os.path.join(MODEL_LOCATION, 'test'))
    # Merge all the summaries and write them out to MODEL directory
    merged = tf.summary.merge_all()

    # Now plug in specific datasets to the iterator, which will happen later
    # when invoked via session.run.
    training_init_op = iterator.make_initializer(training_dataset)
    validation_init_op = iterator.make_initializer(validation_dataset)

    return {
        'train_op': optimizer,
        'predictions': y_pred,
        'weights': weights,
        'loss': loss,
        'train_init': training_init_op,
        'validation_init': validation_init_op,
        'traced_target_images': y_true,
        'raw_input_images': RawImages,
        'cell_id_tokens': labels,
        'global_step': global_step,
        'summaries': summaries,
        'train_writer': train_writer,
        'test_writer': test_writer,
        'debug': {
            'input_shape': tf.shape(RawImages),
            'y_pred_shape': tf.shape(y_pred),
            'y_true_shape': tf.shape(y_true)
        }
    }


def train_graph(sess, graph, step):
    'Training of graph.'
    # Plug in the training dataset to the iterator (by initializing it):
    sess.run(graph['train_init'])
    while True:
        try:
            _, global_step_res, loss, summaries_res, true, pred, raws = sess.run(
                [
                    graph['train_op'], graph['global_step'], graph['loss'],
                    graph['summaries'], graph['traced_target_images'],
                    graph['predictions'], graph['raw_input_images']
                ])

            tf.logging.log_every_n(
                tf.logging.WARN,
                'Global training step: %i ~~ Autoencoder loss: %f' %
                (global_step_res, loss), 1)
            # tf.logging.log_every_n(
            #     tf.logging.WARN,
            #     ('---trueminmax:', true.min(), true.max(), '---predminmax:',
            #         pred.min(), pred.max(), '---rawsminmax:', raws.min(),
            #         raws.max()),
            #     1)

            graph['train_writer'].add_summary(summaries_res, global_step_res)
        except tf.errors.OutOfRangeError:
            break

        break


def predict_graph(sess, graph, step, vis=False, pred=False):
    'Inference from graph.'
    # Plug in the validation dataset to the iterator (by initializing it):
    sess.run(graph['validation_init'])
    # for step in [step]:
    while True:
        try:
            global_step_res, predictions, raw_images, traced_images, labels, rmse, summaries_res = sess.run(
                [
                    graph['global_step'], graph['predictions'],
                    graph['raw_input_images'], graph['traced_target_images'],
                    graph['cell_id_tokens'], graph['loss'], graph['summaries']
                ])
            # print('RMSE:', rmse)

            print('Global Step in Inference:', global_step_res, step)
            tf.logging.log_every_n(
                tf.logging.WARN,
                'Global inference step %i: RMSE: %f' % (global_step_res, rmse),
                1)

            graph['test_writer'].add_summary(summaries_res, global_step_res)
            if vis:  # and labels.shape[0] == BATCH_SIZE:
                visualize_before_after(
                    num_panels=NUM_VIS,
                    start_index=1,  #27
                    rawimgs=raw_images,
                    predictedimgs=predictions,
                    tracedimgs=traced_images,
                    labels=labels,
                    global_step_res=global_step_res,
                    rmse=rmse)
            if pred:
                write_predicted_panel(predictions, labels, PREDICTION_OUTPATH)

        except tf.errors.OutOfRangeError:
            break


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Model config parameters.")
    parser.add_argument(
        "config_file_path",
        help="Path and ini filename containing config details.")
    args = parser.parse_args()
    config_file_path = args.config_file_path
    config = configparser.ConfigParser()
    config.read(config_file_path)
    print(config.sections())

    verbose = config.getboolean('VerbosityAndModes', 'verbose')
    split_data = config.getboolean('VerbosityAndModes', 'split_data')
    vis_mode = config.getboolean('VerbosityAndModes', 'vis_mode')
    predict_mode = config.getboolean('VerbosityAndModes', 'predict_mode')

    # Model outputs
    MODEL_NAME = config['ModelOutputs']['ModelName']
    MODEL_LOCATION = os.path.join(config['ModelOutputs']['ModelLocation'],
                                  MODEL_NAME)
    PREDICTION_OUTPATH = config['ModelOutputs']['PredictionOutputPath']

    # Training and display parameters
    LOSS_WEIGHTING = config.getboolean('TrainingDisplayParameters',
                                       'LossWeighting',
                                       fallback=True)
    LOSSRMSE = config.getboolean('TrainingDisplayParameters',
                                 'LossRMSE',
                                 fallback=True)
    LEARNING_RATE = config.getfloat('TrainingDisplayParameters',
                                    'LearningRate',
                                    fallback=0.0001)
    NUM_STEPS = config.getint('TrainingDisplayParameters',
                              'NumberSteps',
                              fallback=18000)
    BATCH_SIZE = config.getint('TrainingDisplayParameters',
                               'BatchSize',
                               fallback=500)
    IMG_DIM = config.getint('TrainingDisplayParameters',
                            'ImageDimensions',
                            fallback=128)
    KERNEL = config.getint('TrainingDisplayParameters',
                           'KernelDimensions',
                           fallback=3)
    SIG_END = config.getboolean('TrainingDisplayParameters',
                                'SigmoidToBinarizeAtEnd',
                                fallback=False)
    LAYERS = config.getint('TrainingDisplayParameters',
                           'NumberLayers',
                           fallback=3)
    TILES = config['SomaOrAllTiles']['TilesSource']
    MASK_STRING = config['SomaOrAllTiles']['MaskString']
    DATA_SPLIT = config.getfloat('TrainingDisplayParameters',
                                 'DataSplit',
                                 fallback=0.9)
    NUM_VIS = config.getint('TrainingDisplayParameters',
                            'NumberVisualizedPanels',
                            fallback=5)
    SCALE_TYPE = config.get('TrainingDisplayParameters',
                            'ScaleType',
                            fallback='m1to1')
    AUG_DATA = config.getboolean('TrainingDisplayParameters',
                                 'AugData',
                                 fallback=True)

    # Inputs
    print('....................Setting inputs...')
    # Training
    raw_img_path = os.path.join(config['DataSource']['rawimgpath'], TILES)
    traced_img_path = os.path.join(config['DataSource']['tracedimgpath'],
                                   TILES)

    print('raw_img_path', raw_img_path)
    print('traced_img_path', traced_img_path)
    raw_image_filelist = make_filelist(raw_img_path, '.png')
    traced_image_filelist = make_filelist(traced_img_path, '.png')
    # Validation
    # val_raw_img_path = '/finkbeiner/imaging/smb-robodata/mbarch/neurites_jms/TF16-PCDH10-011917/raw_images/cell_tiles/'
    # val_traced_img_path = '/finkbeiner/imaging/smb-robodata/mbarch/neurites_jms/TF16-PCDH10-011917/filaments_mask/cell_tiles'
    val_raw_image_filelist = make_filelist(raw_img_path, '.png')
    val_traced_image_filelist = make_filelist(traced_img_path, '.png')

    raw_image_filelist, traced_image_filelist, label_from_filename = make_filelists_equal(
        raw_image_filelist, traced_image_filelist, verbose=False)
    if not split_data:
        val_raw_image_filelist, val_traced_image_filelist, val_label_from_filename = make_filelists_equal(
            val_raw_image_filelist, val_traced_image_filelist, verbose=False)
        print(len(val_raw_image_filelist), len(val_traced_image_filelist),
              len(val_label_from_filename))

    # Normalization parameters
    print('....................Generating normalization data...')
    print('Normalizing raw images')
    HALF_MAX_RAW_IMG, _ = get_norm_output(raw_image_filelist,
                                          expected_dtype=np.uint16)
    print('Normalizing traced images')
    HALF_MAX_TRACED_IMG, FRACTION_ON = get_norm_output(traced_image_filelist,
                                                       expected_dtype=np.uint8,
                                                       max_val=255)
    HALF_MAX_TRACED_IMG += 1

    if verbose:
        print('Normalization summary:')
        print('FRACTION_ON', FRACTION_ON)
        print('HALF_MAX_RAW_IMG', HALF_MAX_RAW_IMG.shape,
              HALF_MAX_RAW_IMG.dtype)
        print(HALF_MAX_RAW_IMG.min(), HALF_MAX_RAW_IMG.max(), HALF_MAX_RAW_IMG)
        print('HALF_MAX_TRACED_IMG', HALF_MAX_TRACED_IMG.shape,
              HALF_MAX_TRACED_IMG.dtype)
        print(HALF_MAX_TRACED_IMG.min(), HALF_MAX_TRACED_IMG.max(),
              HALF_MAX_TRACED_IMG)

    # Splitting input and target into training/validation
    print('....................Splitting dataset for training/inference...')
    if split_data:
        tr_img_filenames_target, val_img_filenames_target = hash_split(
            zip(raw_image_filelist, traced_image_filelist,
                label_from_filename),
            p=DATA_SPLIT)
    else:
        tr_img_filenames_target = rehash_data_structure(
            zip(raw_image_filelist, traced_image_filelist,
                label_from_filename))
        val_img_filenames_target = rehash_data_structure(
            zip(val_raw_image_filelist, val_traced_image_filelist,
                val_label_from_filename))

    print(type(tr_img_filenames_target), len(tr_img_filenames_target))
    print(type(tr_img_filenames_target[0]), len(tr_img_filenames_target[0]))
    print(tr_img_filenames_target[0])
    print(type(val_img_filenames_target), len(val_img_filenames_target))
    print(type(val_img_filenames_target[0]), len(val_img_filenames_target[0]))
    print(val_img_filenames_target[0])

    training_image_filenames, training_target_filenames, training_labels = zip(
        *tr_img_filenames_target)
    validation_image_filenames, validation_target_filenames, validation_labels = zip(
        *val_img_filenames_target)
    if verbose:
        print('training_image_filenames', len(training_image_filenames))
        pprint.pprint(
            [os.path.basename(fn) for fn in training_image_filenames[0:2]])
        pprint.pprint(
            [os.path.basename(fn) for fn in training_target_filenames[0:2]])
        pprint.pprint(training_labels[0:2])
        print('validation_image_filenames', len(validation_image_filenames))
        pprint.pprint(
            [os.path.basename(fn) for fn in validation_image_filenames[0:2]])
        pprint.pprint(
            [os.path.basename(fn) for fn in validation_target_filenames[0:2]])
        pprint.pprint(validation_labels[0:2])

    # Doing tensorflow
    graph = create_graph(
        training_image_target_labels_triplet=(training_image_filenames,
                                              training_target_filenames,
                                              training_labels),
        validation_image_target_labels_triplet=(validation_image_filenames,
                                                validation_target_filenames,
                                                validation_labels))

    sv = tf.train.Supervisor(save_model_secs=30,
                             save_summaries_secs=30,
                             summary_op=None,
                             logdir=MODEL_LOCATION)

    with sv.managed_session() as sess:
        if vis_mode:
            predict_graph(sess, graph, 0, vis=True)
        elif predict_mode:
            predict_graph(sess, graph, 0, pred=True)
        else:
            for i in range(1, NUM_STEPS):
                fn = predict_graph if i % 10 == 0 or i == 0 else train_graph
                fn(sess, graph, i)
