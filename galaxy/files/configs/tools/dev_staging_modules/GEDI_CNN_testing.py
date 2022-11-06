import os
import time
import re
import galaxy.tools.dev_staging_modules.utils as utils

# suppress annoying warnings from tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import logging


class WarningFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        tf_warning = 'retry (from tensorflow.contrib.learn.python.learn.datasets.base)' in msg
        return not tf_warning


logger = logging.getLogger('tensorflow')
logger.addFilter(WarningFilter())

import tensorflow as tf
import numpy as np
import pandas as pd
from argparse import ArgumentParser
# from glob import glob
from galaxy.tools.dev_staging_modules.GEDI_other_scripts.exp_ops.tf_fun import make_dir
# from exp_ops.plotting_fun import plot_accuracies, plot_std, plot_cms, plot_pr,\
#    plot_cost
from galaxy.tools.dev_staging_modules.GEDI_other_scripts.exp_ops.preprocessing_GEDI_images import produce_patch
# from gedi_config import GEDIconfig
from galaxy.tools.dev_staging_modules.GEDI_other_scripts.models import baseline_vgg16 as vgg16
# from tqdm import tqdm
import pickle


def crop_center(img, crop_size):
    x, y = img.shape[:2]
    cx, cy = crop_size
    startx = x // 2 - (cx // 2)
    starty = y // 2 - (cy // 2)
    return img[starty:starty + cy, startx:startx + cx]


def renormalize(img, max_value, min_value):
    return (img - min_value) / (max_value - min_value)


def image_batcher(
        start,
        num_batches,
        images,
        validation_batch,
        training_max,
        training_min,
        model_image_size):
    for b in range(num_batches):
        next_image_batch = images[start:start + validation_batch]
        image_stack = []
        for f in next_image_batch:
            # 1. Load image patch
            patch = produce_patch(
                f,
                0,
                0,
                divide_panel=None,
                max_value=16117,
                min_value=0).astype(np.float32)
            # 2. Repeat to 3 channel (RGB) image
            patch = np.repeat(patch[:, :, None], 3, axis=-1)
            # 3. Renormalize based on the training set intensities
            patch = renormalize(
                patch,
                max_value=training_max,
                min_value=training_min)
            # 4. Crop the center
            patch = crop_center(patch, model_image_size[:2])
            # 5. Clip to [0, 1] just in case
            patch[patch > 1.] = 1.
            patch[patch < 0.] = 0.
            # 6. Add to list
            image_stack += [patch[None, :, :, :]]
        # Add dimensions and concatenate
        start += validation_batch
        yield np.concatenate(image_stack, axis=0), next_image_batch


def randomization_test(y, yhat, iterations=10000):
    true_score = np.mean(y == yhat)
    perm_scores = np.zeros((iterations))
    lab_len = len(y)
    for it in range(iterations):
        perm_scores[it] = np.mean(
            yhat == np.copy(y)[np.random.permutation(lab_len)])
    p_value = (np.sum(true_score < perm_scores) + 1) / float(iterations + 1)
    return p_value


# Evaluate your trained model on GEDI images
def test_vgg16(input_dict, image_dir, output_dir, models_dir):
    # get info from var_dict
    var_dict = pickle.load(open(input_dict, 'rb'))
    expt_name = var_dict['ExperimentName']
    print('Experiment Name: ' + expt_name)
    if image_dir == '':
        image_dir = os.path.join(var_dict['GalaxyOutputPath'], 'ObjectCrops')
    print('Images Folder: ' + image_dir)
    if output_dir == '':
        output_dir = var_dict['GalaxyOutputPath']
    print('Output Folder: ' + output_dir)
    assert os.path.exists(output_dir), 'Confirm that the Galaxy output folder (%s) exists.' % output_dir
    if models_dir == '':
        models_dir = '/finkbeiner/imaging/smb-robodata/Galaxy/GEDICNN_models'
    print('Models Folder: ' + models_dir)

    #    config = GEDIconfig()
    assert os.path.exists(image_dir), 'Confirm that the images folder (%s) exists,' % image_dir

    # Select and combine images
    print('Checking image list...')
    ch_token_pos = utils.get_channel_token(var_dict['RoboNumber'])
    all_files = [fname.split('_') for fname in os.listdir(image_dir) if '.tif' in fname]

    selected_files = []
    for well in var_dict['Wells']:
        for tp in var_dict['TimePoints']:
            files = [os.path.join(image_dir, '_'.join(tok_fname)) for tok_fname in all_files if
                     well == tok_fname[4] and tp == tok_fname[2] and var_dict['MorphologyChannel'] == tok_fname[
                         ch_token_pos]]
            selected_files.extend(files)
    assert len(selected_files) > 0, 'No matching images found in %s' % (image_dir)
    print('Testing %s images' % len(selected_files))

    missing_files = [fname for fname in selected_files if not os.path.exists(fname)]
    assert len(missing_files) == 0, 'Missing files: %s' % missing_files

    combined_files = np.asarray(selected_files)

    #    config = GEDIconfig()
    model_file = os.path.join(models_dir, 'model_58600.ckpt-58600')
    print('Model file:', model_file)
    # assert os.path.exists(model_file), 'No model file found (%s)' % model_file
    #    model_file_path = os.path.sep.join(model_file.split(os.path.sep)[:-1])
    #    print model_file_path
    #    meta_file_pointer = os.path.join(
    #        model_file_path,
    #        'train_maximum_value.npz')
    #    print meta_file_pointer
    meta_file = os.path.join(models_dir, 'train_maximum_value.npz')
    print('Meta file:', meta_file)
    # assert os.path.exists(meta_file), 'No training data meta file found (%s)' % meta_file
    meta_data = np.load(meta_file)

    # Prepare image normalization values
    training_max = np.max(meta_data['max_array']).astype(np.float32)
    training_min = np.min(meta_data['min_array']).astype(np.float32)

    # Find model checkpoints
    ds_dt_stamp = re.split('/', model_file)[-2]
    out_dir = os.path.join(output_dir, ds_dt_stamp)

    # Make output directories if they do not exist
    dir_list = [output_dir, out_dir]
    #    dir_list = [config.results, out_dir]
    [make_dir(d) for d in dir_list]

    # Prepare data on CPU
    model_image_size = [224, 224, 3]
    if model_image_size[:-1] < 3:
        print('*' * 60)
        print(
            'Warning: model is expecting a H/W/1 image. '
            'Do you mean to set the last dimension of '
            'model_image_size to 3?')
        print('*' * 60)

    images = tf.placeholder(
        tf.float32,
        shape=[None] + model_image_size,
        name='images')

    # Prepare model on GPU
    fine_tune_layers = [
        'conv4_1',
        'conv4_2',
        'conv4_3',
        'conv5_1',
        'conv5_2',
        'conv5_3',
        'fc6',
        'fc7',
        'fc8'
    ]
    with tf.device('/gpu:0'):
        with tf.variable_scope('cnn'):
            vgg = vgg16.model_struct(
                vgg16_npy_path=os.path.join(models_dir, 'vgg16.npy'),
                fine_tune_layers=fine_tune_layers)
            vgg.build(
                images,
                output_shape=2)

        # Setup validation op
        scores = vgg.prob
        preds = tf.argmax(vgg.prob, 1)

    # Set up saver
    saver = tf.train.Saver(tf.global_variables())

    # Loop through each checkpoint then test the entire validation set
    ckpts = [model_file]
    ckpt_yhat, ckpt_y, ckpt_scores, ckpt_file_array = [], [], [], []
    #    print '-' * 60
    #    print 'Beginning evaluation'
    #    print '-' * 60

    validation_batch = 16
    if validation_batch > len(combined_files):
        print('Trimming validation_batch size to %s (same as # of files).' % len(combined_files))
        validation_batch = len(combined_files)

    for idx, c in enumerate(ckpts):
        dec_scores, yhat, file_array = [], [], []

        # Initialize the graph
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run(
            tf.group(
                tf.global_variables_initializer(),
                tf.local_variables_initializer()))

        # Set up exemplar threading
        saver.restore(sess, c)
        start_time = time.time()
        num_batches = np.floor(
            len(combined_files) / float(
                validation_batch)).astype(int)
        for image_batch, file_batch in image_batcher(
                start=0,
                num_batches=num_batches,
                images=combined_files,
                validation_batch=validation_batch,
                training_max=training_max,
                training_min=training_min,
                model_image_size=model_image_size):
            feed_dict = {
                images: image_batch
            }
            sc, tyh = sess.run(
                [scores, preds],
                feed_dict=feed_dict)
            dec_scores = np.append(dec_scores, sc)
            yhat = np.append(yhat, tyh)
            file_array = np.append(file_array, file_batch)
        ckpt_yhat.append(yhat)
        ckpt_scores.append(dec_scores)
        ckpt_file_array.append(file_array)
        print('Job took %.3f hours' % ((time.time() - start_time) / 360))
    sess.close()

    # Save everything
    np.savez(
        os.path.join(output_dir, '%s_GEDI-CNN-validation-accuracies' % str(expt_name)),
        ckpt_yhat=ckpt_yhat,
        ckpt_scores=ckpt_scores,
        ckpt_names=ckpts,
        combined_files=ckpt_file_array)

    # Also save a csv with item/guess pairs
    try:
        dec_scores = np.asarray(dec_scores)
        yhat = np.asarray(yhat)
        df = pd.DataFrame(
            np.hstack((
                np.asarray(ckpt_file_array).reshape(-1, 1),
                yhat.reshape(-1, 1),
                dec_scores.reshape(dec_scores.shape[0] // 2, 2))),
            columns=['files', 'live_guesses', 'classifier score dead', 'classifier score live'])
        output_name = str(expt_name) + '_GEDI-CNN-results'
        if output_name is None or len(output_name) == 0:
            output_name = 'output'
        df.to_csv(os.path.join(output_dir, '%s.csv' % output_name))
        print('Saved csv to: %s' % os.path.join(
            output_dir, '%s.csv' % output_name))
    except:
        print('X' * 60)
        print('Could not save a spreadsheet of file info')
        print('X' * 60)


#    # Plot everything
#    try:
#        plot_accuracies(
#            ckpt_y, ckpt_yhat, config, ckpts,
#            os.path.join(out_dir, 'validation_accuracies.png'))
#        plot_std(
#            ckpt_y, ckpt_yhat, ckpts, os.path.join(
#                out_dir, 'validation_stds.png'))
#        plot_cms(
#            ckpt_y, ckpt_yhat, config, os.path.join(
#                out_dir, 'confusion_matrix.png'))
#        plot_pr(
#            ckpt_y, ckpt_yhat, ckpt_scores, os.path.join(
#                out_dir, 'precision_recall.png'))
#        plot_cost(
#            os.path.join(out_dir, 'training_loss.npy'), ckpts,
#            os.path.join(out_dir, 'training_costs.png'))
#    except:
#        print 'X'*60
#        print 'Could not locate the loss numpy'
#        print 'X'*60


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_dict',
                        help='Load input variable dictionary')
    parser.add_argument(
        "--image_dir",
        type=str, default='',
        dest="image_dir",
        help="Directory containing your cropped images.")
    parser.add_argument(
        "--output_dir",
        type=str, default='',
        dest="output_dir",
        help="Directory to output analysis results.")
    parser.add_argument(
        "--models_dir",
        type=str, default='',
        dest="models_dir",
        help="Directory to output analysis results.")
    args = parser.parse_args()

    # print out device info
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

    test_vgg16(**vars(args))
