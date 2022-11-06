import os
import argparse
import pickle
import numpy as np
import cv2
import utils

def burst_project(var_dict, channels, proj_method):
    '''
    loops through timepoints, wells, channels, and panels
    and applies the selected projection method
    '''
    # get number of panels
    panels = var_dict['NumberHorizontalImages'] * var_dict['NumberVerticalImages']

    # get channel token position
    ch_token_pos = utils.get_channel_token(var_dict['RoboNumber'])

    for tp in var_dict['TimePoints']:
        for well in var_dict['Wells']:
            for ch in channels:
                for panel in range(1, panels+1):

                    # get image paths for matching timepoint, well, channel, and panel
                    img_series_tokens = [x for x in image_tokens if x[3] == tp and x[5] == well and
                                         int(x[6]) == panel and ch in x[ch_token_pos+1]]

                    if len(img_series_tokens) > 1:
                        img_series = [cv2.imread(i[0], -1) for i in img_series_tokens]

                        if proj_method == 'sum':
                            img_projected = np.zeros(img_series[0].shape, dtype=img_series[0].dtype)
                            for b in img_series:
                                img_projected = cv2.add(img_projected, b)
                            if np.max(img_projected) >= 2**16:
                                print('Saturated image: %s, well %s, %s, panel %s, max intensity = %d' % (tp, well, ch, panel, np.max(img_projected)))
                            new_token = '_BSUM'

                        # get new filepath
                        last_burst = [x[0] for x in img_series_tokens if str(x[4].split('-')[1]) == str(len(img_series_tokens) - 1)][0]
                        newname = os.path.basename(last_burst.replace('.tif', (new_token + '.tif')))
                        newname_path = utils.reroute_imgpntr_to_wells(os.path.join(output_path, newname), well)

                        # save projected image
                        cv2.imwrite(newname_path, img_projected)

                    elif len(img_series_tokens) == 1:
                        # get new filepath
                        new_token = '_NOPROJ'
                        newname = os.path.basename(img_series_tokens[0][0].replace('.tif', (new_token + '.tif')))
                        newname_path = utils.reroute_imgpntr_to_wells(os.path.join(output_path, newname), well)

                        # move (non-)projected image
                        os.rename(img_series_tokens[0][0], newname_path)

if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Burst Projection.")
    parser.add_argument("input_dict",
        help="Load input variable dictionary")
    parser.add_argument("--method",
        help="projection method", default = 'sum')
    parser.add_argument("--channels",
        help="Comma-separated list of channels to process.", default = '')
    parser.add_argument("--input_image_path",
        help="Folder path to input data.", default = '')
    parser.add_argument("--output_results_path",
        help="Folder path to ouput results.", default = '')
    parser.add_argument("output_dict",
        help="Write variable dictionary.")
    args = parser.parse_args()

    # load dict
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))

    print('Projection method: %s' % args.method)

    # get paths
    input_path = utils.get_path(args.input_image_path, var_dict['GalaxyOutputPath'], 'BackgroundCorrected')
    print('Input path: %s' % input_path)

    output_path = utils.get_path(args.output_results_path, var_dict['GalaxyOutputPath'], 'BurstProjected')
    print('Output path: %s' % output_path)
    utils.create_dir(output_path)

    assert os.path.exists(input_path), 'Confirm the path for data exists (%s)' % input_path
    assert os.path.exists(output_path), 'Confirm the path for results exists (%s)' % output_path

    # get image tokens
    if var_dict['DirStructure'] == 'root_dir':
        image_paths = [os.path.join(input_path, name) for name in os.listdir(input_path) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
    elif var_dict['DirStructure'] == 'sub_dir':
        # Only traverse root and immediate subdirectories for images
        relevant_dirs = [input_path] + [os.path.join(input_path, name) for name in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, name))]
        image_paths = [os.path.join(relevant_dir, name) for relevant_dir in relevant_dirs for name in os.listdir(relevant_dir) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
    else:
        raise Exception('Unknown Directory Structure!')
    image_tokens = utils.tokenize_files(image_paths)
    assert len(image_tokens) > 0, 'No images found at %s' % input_path

    # get channels
    if str.strip(args.channels) != '':
        channels = args.channels.replace(' ', '').split(',')
        channels = [utils.get_ref_channel(x, var_dict['Channels']) for x in channels]
    else:
        channels = var_dict['Channels']
    print('Channels: %s' % channels)

    # do burst projection
    burst_project(var_dict, channels, args.method)

    # update DirStructure in dict
    var_dict['DirStructure'] = 'sub_dir'

    # Save dict to file
    with open(args.output_dict, 'wb') as ofile:
        pickle.dump(var_dict, ofile)
