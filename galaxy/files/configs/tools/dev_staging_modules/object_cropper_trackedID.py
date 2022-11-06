'''
Crops image files down to smaller size centered on each object of
a corresponding cell mask (i.e. for creating input images into
a CNN)
'''

import os, cv2, pickle, argparse, shutil, re
import pandas as pd
import utils

def crop_to_object_and_scale(img, Ystart, Yend, Xstart, Xend, scaling_factor):
    '''
    Crops image to the scaling-adjusted target dimensions, then scales the cropped image to the target dimensions.
    '''
    cropped_img = img[Ystart:Yend,Xstart:Xend]

    if scaling_factor > 1:
        cropped_img = cv2.resize(cropped_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    elif scaling_factor < 1:
        cropped_img = cv2.resize(cropped_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
    return cropped_img

def add_padding(img, crop_dim):
    '''
    Adds padding to the input image so that image crops for all objects (including those
    that are close to the image edge) will be the desired square dimensions.
    '''
    img_padded = cv2.copyMakeBorder(img, top=crop_dim, bottom=crop_dim, left=crop_dim, right=crop_dim, borderType=cv2.BORDER_CONSTANT, value=0)
    return img_padded

def main():
    ''' Main point of entry '''

    # get arguments
    parser = argparse.ArgumentParser(description='Create object crops.')
    parser.add_argument('input_dict',
        help='Load input variable dictionary')
    parser.add_argument('channel',
        help='Channel to crop on')
    parser.add_argument('crop_dim',
        help='Crop size in pixels', type=int)
    parser.add_argument('scaling_factor',
        help='Scaling factor', type=float)
    parser.add_argument('--cell_data_path', dest='cell_data_path',
        help='Path to cell_data.csv', default='')
    parser.add_argument('--images_input', dest='images_input',
        help='Path to images to crop', default='')
    parser.add_argument('--object_crops_output', dest='object_crops_output',
        help='Path to save cropped images', default='')
    parser.add_argument('output_dict',
        help='Write variable dictionary.')
    args = parser.parse_args()

    # assign arguments to variables
    var_dict = pickle.load(open(args.input_dict, 'rb'))
    channel = utils.get_ref_channel(args.channel, var_dict['Channels'])
    crop_dim = args.crop_dim
    scaling_factor = args.scaling_factor

    crop_dim_scaled = int(crop_dim / scaling_factor)

    if str.strip(args.cell_data_path) != '':
        cell_data_path = str.strip(args.cell_data_path)
    else:
        cell_data_path = os.path.join(var_dict['GalaxyOutputPath'], 'cell_data.csv')
    assert os.path.exists(cell_data_path), 'cell_data.csv not found at %s' % cell_data_path

    images_input = utils.get_path(args.images_input, var_dict['GalaxyOutputPath'], 'AlignedImages')
    assert os.path.exists(images_input), 'Input images not found at (%s)' % args.images_input

    object_crops_output = utils.get_path(args.object_crops_output, var_dict['GalaxyOutputPath'], 'ObjectCrops')
    assert os.path.exists(os.path.split(object_crops_output)[0]), 'Confirm that the output path parent folder (%s) exists.' % os.path.split(object_crops_output)[0]
    assert re.match('^[a-zA-Z0-9_-]+$', os.path.split(object_crops_output)[1]), 'Confirm that the output folder name (%s) does not contain special characters.' % os.path.split(object_crops_output)[1]

    print('Channel:', channel)
    print('Crop Dimensions:', crop_dim, 'x', crop_dim)
    print('Scaling Factor:', scaling_factor)
    print('Cell Data:', cell_data_path)
    print('Input Images:', images_input)
    print('Cropped Images Output:', object_crops_output)

    utils.create_dir(object_crops_output)

    cell_data = pd.read_csv(cell_data_path)

    # append 'T' to timepoint in cell_data so that it can be matched to the timepoint in the dict file
    cell_data['Timepoint'] = 'T' + cell_data['Timepoint'].astype(str)

    # get file list
    # get image paths
    image_paths = ''
    if var_dict['DirStructure'] == 'root_dir':
        image_paths = [os.path.join(images_input, name) for name in os.listdir(images_input) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
    elif var_dict['DirStructure'] == 'sub_dir':
        # Only traverse root and immediate subdirectories for images
        relevant_dirs = [images_input] + [os.path.join(images_input, name) for name in os.listdir(images_input) if os.path.isdir(os.path.join(images_input, name))]
        image_paths = [os.path.join(relevant_dir, name) for relevant_dir in relevant_dirs for name in os.listdir(relevant_dir) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
    else:
        raise Exception('Unknown Directory Structure!')
    image_tokens = utils.tokenize_files(image_paths)

    # loop through wells
    for well in var_dict['Wells']:
        # loop through timepoints
        for timepoint in var_dict['TimePoints']:

            # get image file path for current well, timepoint, channel
            image_paths = utils.get_filename(image_tokens, well, timepoint, channel, var_dict['RoboNumber'])
            assert len(image_paths) > 0, 'No images found for well %s at %s in %s' % (well, timepoint, images_input)
            assert len(image_paths) == 1, 'More than one image found for well %s at %s in %s' % (well, timepoint, images_input)

            # read in current image
            img = cv2.imread(image_paths[0], -1)

            # add padding to image and cell_mask so that objects on border of image are centered after cropping
            img_padded = add_padding(img, crop_dim_scaled)

            # get list of ObjectLabelsFound for current well, timepoint, and channel
            cell_IDs = cell_data[(cell_data['Sci_WellID'] == well) &
                                (cell_data['Timepoint'] == timepoint) &
                                (cell_data['MeasurementTag'] == channel)]

            # remove duplicates by taking the object with the largest area
            cell_IDs = cell_IDs.sort_values(by=['BlobArea'], ascending=False).drop_duplicates(['ObjectLabelsFound'])

            # loop through list of cell_IDs and get X and Y coodinates for object's reference channel intensity-weighted centroid
            for i, row in cell_IDs.iterrows():
                cX = int(row['BlobCentroidX_RefIntWeighted'])
                cY = int(row['BlobCentroidY_RefIntWeighted'])

                # get X/Y coordinates for the boundaries of the image to be cropped
                Ystart = int(cY + (crop_dim_scaled*0.5))
                Yend = int(cY + (crop_dim_scaled*1.5))
                Xstart = int(cX + (crop_dim_scaled*0.5))
                Xend = int(cX + (crop_dim_scaled*1.5))

			  # check to make sure crop points are within the image (for some unknown reason some cell_data coordinates are out of the image)
                if (0 <= Ystart <= img_padded.shape[0]) and (0 <= Yend <= img_padded.shape[0]) and (0 <= Xstart <= img_padded.shape[1]) and (0 <= Xend <= img_padded.shape[1]):
                    cropped_img = crop_to_object_and_scale(img_padded, Ystart, Yend, Xstart, Xend, scaling_factor)

                    orig_name = utils.extract_file_name(image_paths[0])
                    img_name = utils.reroute_imgpntr_to_wells(os.path.join(object_crops_output, orig_name + '_' + str(row['ObjectLabelsFound']) + '.tif'), well)

                    cv2.imwrite(img_name, cropped_img)

    # write out dictionary
    outfile = args.output_dict
    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    outfile = shutil.move('var_dict.p', outfile)
    utils.save_user_args_to_csv(args, object_crops_output , 'object cropper')

if __name__ == '__main__':

     main()