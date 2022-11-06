'''
Get coordinates of tiles for img_dim.
Deal only with whole tiles (check for edges)
Requires that naming scheme match:
image_filename = PID20180706_070418PINK1ParkinSurvival1_T5_60-0_A1_0_RFP-DFTrCy5_MN.tif

'''
from __future__ import absolute_import, division, print_function
import os, glob, cv2
import numpy as np
import pprint, pickle
import argparse

# IMG_DIM = 640#2048#5736# 4096# 5736
# TILE_DIM = 128

def make_filelist(path, identifier, verbose=True):
    '''
    Takes a directory and a string identifier.
    Returns a list of files.
    '''
    filelist = sorted(
        glob.glob(os.path.join(path, '*'+identifier+'*')))

    if verbose:
        print("Selection filter:")
        print(os.path.join(path, '*'+identifier+'*'))
        pprint.pprint([os.path.basename(fel) for fel in filelist])

    return filelist

def validate_image_sizes(filelist):
    """Raise value error if images are not of the same size"""
    # check that images are all same size
    print('Validating image sizes...')
    size_list = [cv2.imread(f).size for f in filelist]
    common_size = max(set(size_list), key = size_list.count)
    bad_images = [f for i, f in enumerate(filelist) if size_list[i] != common_size]
    if bad_images:
        print('Bad images:')
        print(*bad_images, sep = '\n')
        raise ValueError('Not all images are same size')
    else:
        print('Images are same size')

def make_tile_name(xycoord, img_path):
    '''Takes the coordinate the original image pointer
    and returns tile name to include tile coordinates.'''
    cell_identifier = '-'.join(['x'+str(xycoord[0]), 'y'+str(xycoord[1])])
    tile_name = '_'.join([os.path.splitext(os.path.basename(img_path))[0], cell_identifier])+'.png'
    return tile_name

def get_tile_coords(img_dim, tile_dim):
    'Takes dimensions of complete image and desired tile dimensions. Returns coordinate list.'
    list_of_coord_tuples = []
    for row_ind in range(0, img_dim, tile_dim):
        for col_ind in range(0, img_dim, tile_dim):
            list_of_coord_tuples.append((row_ind, col_ind))
    return list_of_coord_tuples

def check_for_annotation(tile_pixels):
    'Flattens and evaluates if all pixels are zero. \
     Returns True for tiles with no annotaion.'
    tile_pixels = tile_pixels.reshape((tile_pixels.size))
    unannotated_tile = sum(tile_pixels == 0) == tile_pixels.size
    # if unannotated_tile:
    #     print('This is an unannotated cell:', unannotated_tile, sum(tile_pixels == 0), tile_pixels.size)
    return unannotated_tile

def split_write_tiles(img_path, coord_list, predicted_img_path, tile_dim, annot_only=True):
    "Takes image and coordinate list and write directory. Returns written tiles."
    img = cv2.imread(img_path, -1)
    for coord_set in coord_list:
        img_tile = img[coord_set[1]:coord_set[1]+tile_dim, coord_set[0]:coord_set[0]+tile_dim]
        tile_name = make_tile_name(coord_set, img_path)
        # check for annotation, skip if lacking
        if annot_only:
            # if not check_for_annotation(img_tile) and img_tile.shape[0] == img_tile.shape[1]:
            if not check_for_annotation(img_tile) and img_tile.shape == (TILE_DIM, TILE_DIM):
                # print('Conditions:', check_for_annotation(img_tile), img_tile.shape[0] == img_tile.shape[1])
                cv2.imwrite(os.path.join(predicted_img_path, tile_name), img_tile)
        else:
            # if img_tile.shape[0] == img_tile.shape[1]:
            if img_tile.shape == (TILE_DIM, TILE_DIM):
                cv2.imwrite(os.path.join(predicted_img_path, tile_name), img_tile)
    return True

def split_write_tiles_for_all_images(image_file_list, predicted_img_path):
    for img_path in image_file_list:
        # print('img_path', img_path)
        # coord_list = get_tile_coords(IMG_DIM, TILE_DIM)
        coord_list = get_tile_coords(IMG_DIM, OVERLAP)
        split_write_tiles(img_path, coord_list, predicted_img_path, TILE_DIM, annot_only=annot)

def get_file_groups(image_filelist, token_position_list, token_annot_list):
    'Take filelist and lists of annotations and positions and returns \
    dict with keys of each combination of relevant tokens and values of \
    image lists corresponding to each.'
    output = {}
    for image_filename in image_filelist:
        token_key = get_relevant_token_keys(
            image_filename, token_position_list, token_annot_list)
        if token_key not in output:
            output[token_key] = []
        output[token_key].append(image_filename)
    return output

def get_relevant_token_keys(image_filename, token_position_list, token_annot_list):
    'Take filename and relevant token position and annotation list and generate a key\
    Time token is 2 in all naming conventions. Well token is 4 in all naming conventions.\
    @Usage\
    Example for grouping all channels for montaged images of a particular well and time point:\
    image_filename = PID20180706_070418PINK1ParkinSurvival1_T5_60-0_A1_0_RFP-DFTrCy5_MN.tif\
    token_position_list = [2,4]\
    token_annot_list = ["",""] --> name_list included for cases where token value field is \
    ambiguous and needs to be prepended with annotation (ex. montage panel and hours.'
    image_fn_tokens = os.path.basename(image_filename).split('_')
    token_keys = []
    for token_pos, token_name in zip(token_position_list, token_annot_list):
        token_keys.append(token_name+image_fn_tokens[token_pos])
    return ','.join(token_keys)

def get_tile_coords_from_token(tile_path):
    'Use coordinates token in filename to extract placement coordinates.'
    xycoord_tokens = os.path.basename(tile_path).split('_')[-1].split('.')[0].split('-')
    xycoord = (xycoord_tokens[0][1:], xycoord_tokens[1][1:])
    return xycoord

def get_cell_coords_from_token(tile_path):
    'Use cell center from token in filename to extract placement coordinates.'
    cell_coord_tokens = os.path.basename(tile_path).split('_')[-1].split('.png')[0].split('-')
    # cell_coord = (int(float(cell_coord_tokens[1][1:])), int(float(cell_coord_tokens[2][1:])))
    cell_coord = (int(float(cell_coord_tokens[0][1:])), int(float(cell_coord_tokens[1][1:])))
    return cell_coord

def make_ouput_image_path_from_tile_path(tile_path, out_path):
    made_name = '_'.join((os.path.basename(tile_path).split('_')[0:-1]))+'.tif'
    output_image_path = os.path.join(
        out_path,
        '_'.join((os.path.basename(tile_path).split('_')[0:-1]))+'.tif')
    print(output_image_path)
    return output_image_path

def join_tiles_neur(predicted_tiles_list, predicted_img_write_path, tile_dim, img_dim):
    # make empty numpy array of zeros (all black)
    # added arbitrary value such that min operation occurs between image pixel and larger value
    predicted_img = np.zeros((IMG_DIM, IMG_DIM), dtype=np.uint16)+1000
    # get output image name
    image_filename = make_ouput_image_path_from_tile_path(
        predicted_tiles_list[0], predicted_img_write_path)
    print('Length of file lists:', len(predicted_tiles_list))
    for tile_path in predicted_tiles_list:
        img_tile = cv2.imread(tile_path, -1) # read in tile
        # if img_tile.shape[0] == img_tile.shape[1]: address this at prediction time
        tile_coords = get_tile_coords_from_token(tile_path) # read tile coords
        # write predicted tile into coords
        assert tile_coords[1] != '0.tif', tile_coords[1] + tile_path
        assert tile_coords[0] != '0.tif', tile_coords[0] + tile_path
        # get current content at destination coordinates
        curr_img = predicted_img[int(tile_coords[1]):int(tile_coords[1])+tile_dim, int(tile_coords[0]):int(tile_coords[0])+tile_dim]
        # generate tile by taking minimum pixel value at each position, to exclude grid artifact
        try:
            img_tile = np.minimum(img_tile, curr_img)
            predicted_img[int(tile_coords[1]):int(tile_coords[1])+tile_dim, int(tile_coords[0]):int(tile_coords[0])+tile_dim] = img_tile
        # img_tile.shape != curr_img.shape, generate black tile
        # this happens when size of image is not divisible by tile_dim
        except ValueError:
            print('Bad tile:', tile_path)
            print('Will fit tile to curr_img.shape:', curr_img.shape)
            img_tile = cv2.resize(img_tile, (curr_img.shape[1], curr_img.shape[0]))
            img_tile = np.minimum(img_tile, curr_img)
            predicted_img[int(tile_coords[1]):int(tile_coords[1])+tile_dim, int(tile_coords[0]):int(tile_coords[0])+tile_dim] = img_tile
        # assert img_tile.shape[0] == img_tile.shape[1], tile_coords[1]+' '+str(img_tile.shape[1])+' '+tile_coords[0]+' '+str(img_tile.shape[0])
        # predicted_img[int(tile_coords[1]):int(tile_coords[1])+img_tile.shape[1], int(tile_coords[0]):int(tile_coords[0])+img_tile.shape[0]] = img_tile
    cv2.imwrite(image_filename, predicted_img)
    # return predicted img to put all predicted imgs in a list
    return predicted_img, image_filename


# Note that this should not work well even if tiles are placed correctly
# This is because the tiles are coming from soma mostly and will not trace out neurites
def join_tiles_soma(predicted_tiles_list, predicted_img_write_path, tile_dim, img_dim):
    # make empty numpy array of zeros (all black)
    predicted_img = np.zeros((IMG_DIM, IMG_DIM), dtype=np.uint16)
    # get output image name
    image_filename = make_ouput_image_path_from_tile_path(
        predicted_tiles_list[0], predicted_img_write_path)
    print('Length of file lists:', len(predicted_tiles_list))
    for tile_path in predicted_tiles_list:
        img_tile = cv2.imread(tile_path, -1) # read in tile
        if img_tile.shape[0] == img_tile.shape[1] == 128:
            tile_coords = get_cell_coords_from_token(tile_path) # read tile coords
            xstart = int(tile_coords[1])-int(tile_dim/2)
            xend = int(tile_coords[1])-int(tile_dim/2)+tile_dim
            ystart = int(tile_coords[0])+int(tile_dim/2)
            yend = int(tile_coords[0])+int(tile_dim/2)+tile_dim
            # write predicted tile into coords (shift by half tile to center)
            if xstart > 0 and ystart > 0 and xend < IMG_DIM and yend < IMG_DIM:
                predicted_img[xstart:xend, ystart:yend] = img_tile
    cv2.imwrite(image_filename, predicted_img)
    return True

def join_tiles_for_all_images(predicted_tile_path, predicted_img_path, neurite_based=True):
    image_tile_file_list = make_filelist(predicted_tile_path, '.png')
    # get list of tiles associated with image
    token_file_dict = get_file_groups(
        image_tile_file_list,
        [0,1,2,3,4,5,6],
        ['', '', '', 'H', '', 'P', ''])
    predicted_imgs = []
    predicted_filenames = []
    for token_key in token_file_dict:
        pprint.pprint([os.path.basename(fn) for fn in token_file_dict[token_key]])
        if neurite_based:
            predicted_img, image_filename = join_tiles_neur(token_file_dict[token_key], predicted_img_path, TILE_DIM, IMG_DIM)
            predicted_imgs.append(predicted_img)
            predicted_filenames.append(image_filename)
        else:
            join_tiles_soma(token_file_dict[token_key], predicted_img_path, TILE_DIM, IMG_DIM)
    if predicted_imgs and predicted_filenames:
        threshold_on_median(predicted_imgs, predicted_filenames)

def threshold_on_median(predicted_imgs, predicted_filenames):
    """Threshold predicted neurite image on median value"""
    print('Thresholding images...')
    import csv
    # get path to save thresholded images to
    predicted_img_path = os.path.dirname(predicted_filenames[0])
    exp_dir = os.path.dirname(predicted_img_path)
    threshold_img_path = os.path.join(exp_dir, 'threshold')
    # create threshold folder if it doesn't exist
    if not os.path.exists(threshold_img_path):
        print('Creating directory: ' + threshold_img_path)
        os.makedirs(threshold_img_path)
    # get threshold value by taking most common median value of all predicted images
    median_list = [np.median(img) for img in predicted_imgs]
    thresh_val = max(set(median_list), key = median_list.count)
    print('Threshold value:', thresh_val)
    # get path to save neurite_pixel_data.csv
    csv_path = os.path.join(exp_dir, 'neurite_pixel_data.csv')
    print('Writing neurite_pixel_data.csv to:', csv_path)
    # open csv file, instantiate csv writer
    with open(csv_path, 'wb') as outfile:
        header = ('filename', 'fraction_on', 'num_on', 'total_num')
        writer = csv.writer(outfile)
        writer.writerow(header)
        # iterate through predicted images, threshold, save threshold image, write neurite data to csv
        for predicted_img, img_filename in zip(predicted_imgs, predicted_filenames):
            # create image filename
            filename = os.path.basename(img_filename)
            filename = os.path.join(threshold_img_path, '_'.join(filename.rstrip('.tif').split('_') + ['THRESHOLD.tif']))
            # flag image if image median value is different than threshold value
            if np.median(predicted_img) != thresh_val:
                print('Image median difference from threshold value:', np.median(predicted_img) - thresh_val)
                print(img_filename)
            # threshold image
            threshold_img = predicted_img.copy()
            threshold_img[threshold_img > thresh_val] = 255
            threshold_img[threshold_img <= thresh_val] = 0
            # get number of on pixels, calculate fraction of on pixels
            num_on = sum(sum(threshold_img > 0))
            fraction_on = num_on / threshold_img.size
            # save threshold image, write neurite data to csv
            print(filename)
            cv2.imwrite(filename, threshold_img)
            writer.writerow((filename, fraction_on, num_on, threshold_img.size))


def get_label_from_filename(image_filename, tokenWithLabel, verbose=False):
    'Take filename path and token, return label.'
    id_tokens = os.path.basename(image_filename).split('_')
    joined_id_tokens = '-'.join([id_tokens[2], id_tokens[4], id_tokens[tokenWithLabel]])
    label = os.path.splitext(joined_id_tokens)[0]
    if verbose:
        print('Filename to generate label with:', image_filename)
        print('Token given:', tokenWithLabel)
        print('Label extracted:', label)
    # return image_filename, label
    # return label
    return os.path.basename(image_filename)

def make_filelists_equal(raw_image_filelist_in, traced_image_filelist_in, verbose=True):
    'Take filelist and lists of annotations and positions and returns \
    dict with keys of each combination of relevant tokens and values of \
    image lists corresponding to each.'

    print('Lengths of initial image files (raw, traced):',
        len(raw_image_filelist_in),
        len(traced_image_filelist_in))
    image_filelist = raw_image_filelist_in + traced_image_filelist_in
    raw_image_filelist, traced_image_filelist, labels = [], [], []
    token_file_dict = get_file_groups(
        image_filelist,
        [0,1,2,3,4,5,6,-1],
        ['', '', '', 'H', '', 'P', '',''])

    for token_key in token_file_dict:
        if len(token_file_dict[token_key])==1:
            print('Removing file:', os.path.basename(token_file_dict[token_key][0]))
            os.remove(token_file_dict[token_key][0])
        if len(token_file_dict[token_key])==2:
            for filename in token_file_dict[token_key]:
                if mask_string in os.path.basename(filename):
                    traced_image_filelist.append(filename)
                else:
                    raw_image_filelist.append(filename)
            labels.append(get_label_from_filename(filename, -1))
    print('Lengths of final image files (raw, traced):', len(raw_image_filelist), len(traced_image_filelist))
    if verbose:
        print('Raw images collected:')
        pprint.pprint(raw_image_filelist)
        print('Traced images collected:')
        pprint.pprint(traced_image_filelist)
        print('Labels:')
        pprint.pprint(labels)
    assert len(raw_image_filelist) == len(traced_image_filelist), 'Image lists are not the same lengths.'
    assert len(raw_image_filelist) == len(labels), 'Image lists and labels are not the same lengths.'
    return raw_image_filelist, traced_image_filelist, labels


def get_per_pixel_half_max(image_filenames, expected_dtype, verbose=False):
    '''Take image filenames and return a per pixel maximum image_filelist
    divided by two.'''
    half_max_image = np.zeros((TILE_DIM, TILE_DIM), dtype=expected_dtype)
    for filestring in image_filenames:
        new_img = cv2.imread(filestring, -1)
        assert new_img.dtype == expected_dtype, '%s %s' %(new_img.dtype, filestring)
        if new_img.shape != (TILE_DIM,TILE_DIM):
            print('Possibly offending file:', filestring)
            continue
        half_max_image = np.maximum(half_max_image, new_img)
        if expected_dtype == np.uint8:
            assert np.all(half_max_image < 256), filestring
    if verbose:
        print('Max image:')
        print(half_max_image[0:3,0:3])
        print('Halved max_img at generation:', ((half_max_image//2)).dtype, ((half_max_image[0:3,0:3]//2)))
    return (half_max_image//2)

def calculate_fraction_on_pixels(image_filenames, max_val=1):
    ''''Takes a list of files and returns a mean for all pixels added in annotation.
    This is to later normalize by the on pixels and reward the model for learning
    the relatively little positive annotation.'''
    running_total_pixels = 0
    running_sum_on_pixels = 0
    for filestring in image_filenames:
        new_img = cv2.imread(filestring, -1)
        running_total_pixels += new_img.size
        fraction_img_on = np.mean(new_img == max_val)
        running_sum_on_pixels += fraction_img_on * new_img.size
    return np.true_divide(running_sum_on_pixels, running_total_pixels)

def get_norm_output(image_filenames, expected_dtype, max_val=1):
    assert len(image_filenames)>0
    assert type(image_filenames)==list
    directory = os.path.dirname(image_filenames[0])
    if os.path.exists(os.path.join(directory, 'half_max_img.npy')):
        # Read in as numpy array rather than image
        half_max_img = np.load(os.path.join(directory, 'half_max_img.npy'))
        # half_max_img = cv2.imread(os.path.join(directory, 'half_max_img.png'), -1)
        print('Saved half max image properties:', half_max_img.shape)
        print('Type of half_max_img at **read in**:', half_max_img.dtype)
        print('Min:', half_max_img.min(), 'Max:', half_max_img.max())
    else:
        half_max_img = get_per_pixel_half_max(image_filenames, expected_dtype)
        # Write as numpy array rather than image
        np.save(os.path.join(directory, 'half_max_img.npy'), half_max_img)
        # cv2.imwrite(os.path.join(directory, 'half_max_img.png'), half_max_img)
        print('Wrote half max image to: ', os.path.join(directory, 'half_max_img.png'))
        print('Type of half_max_img at **creation**:', half_max_img.dtype)
        print('Min:', half_max_img.min(), 'Max:', half_max_img.max())
    if os.path.exists(os.path.join(directory, 'fraction_on_txt.txt')):
        # Read from npy
        print('Fraction on value in file:', open(os.path.join(directory, 'fraction_on_txt.txt'), 'r').read())
        fraction_on = float(open(os.path.join(directory, 'fraction_on_txt.txt'), 'r').read())
    else:
        fraction_on = calculate_fraction_on_pixels(image_filenames, max_val=max_val)
        print('*****Max value*****:', max_val, '--', 'Fraction on:', fraction_on)
        # Write as npy
        open(os.path.join(directory, 'fraction_on_txt.txt'), 'w').write(str(fraction_on))
        print('Wrote fraction on output to:', os.path.join(directory, 'fraction_on_txt.txt'))
    return half_max_img, fraction_on

def convert_to_8bit(raw_img_filelist, traced_whole_image_dir):
    """Convert all raw images to 8-bit and write to traced image directory for prediction"""
    for raw_img in raw_img_filelist:
        img_name = os.path.basename(raw_img)
        filename = '_'.join(img_name.rstrip('.tif').split('_') + ['NEURITEMASK.tif'])
        print('Converting %s to 8-bit' % img_name)
        raw = cv2.imread(raw_img, -1)
        eight_bit = raw.copy()
        eight_bit = (eight_bit - eight_bit.min()) * ((255 - 0) / (eight_bit.max() - eight_bit.min())) + 0
        eight_bit = eight_bit.astype(np.uint8)
        cv2.imwrite(os.path.join(traced_whole_image_dir, filename), eight_bit)

if __name__ == '__main__':
    # TODO: change argparser so that it accepts experiment directory with images in raw folder or well subfolders?
    # accept raw_whole_image_dir as command line argument, options to split or join
    parser = argparse.ArgumentParser(description = 'Given a path to a directory containing images, create target, predicted, \
                                        and cell_tiles directories if they do not exist. Split tiles with [-s --split], \
                                        join tiles with [-j --join].')
    parser.add_argument('raw_whole_image_dir', help = 'Path to raw whole image directory')
    parser.add_argument('-s', '--split', action = 'store_true', help = 'Split tiles. Defaults to False.')
    parser.add_argument('-j', '--join', action = 'store_true', help = 'Join tiles. Defaults to False.')
    args = parser.parse_args()

    split = args.split #True to run splitting images into tiles
    equalize = split # True (QC to remove files mismatch) ****FILES ARE REMOVED HERE SO TEST RUN AND MAKE SURE YOU HAVE A COPY.
    make_norm_files = split # True
    join = args.join # False during splitting run. True when reconstructing predicted tiles

    assert split != join, 'Must either split or join'

    mask_string = 'NEURITEMASK'#'FILAMENTSMASK' # Unique string as it appears in the traced filename. This is the string added during curation or generation of 'fake' target files
    annot = False # Always False for inference only tasks (True only for training)

    exp_dir = os.path.dirname(os.path.abspath(args.raw_whole_image_dir))

    # # Overlap test project
    raw_whole_image_dir = os.path.abspath(args.raw_whole_image_dir)
    raw_cell_tiles_dir = os.path.join(raw_whole_image_dir, 'cell_tiles')
    traced_whole_image_dir = os.path.join(exp_dir, 'target')
    traced_cell_tiles_dir = os.path.join(traced_whole_image_dir, 'cell_tiles')
    predicted_whole_image_dir = os.path.join(exp_dir, 'predicted')
    predicted_cell_tiles_dir = os.path.join(predicted_whole_image_dir, 'cell_tiles')

    # get raw images, validate image sizes
    filelist = make_filelist(raw_whole_image_dir, 'tif')
    validate_image_sizes(filelist)

    # check if raw, traced, predicted dirs exist, if not, create them
    if not os.path.exists(raw_cell_tiles_dir):
        print('Creating directory: ' + raw_cell_tiles_dir)
        os.makedirs(raw_cell_tiles_dir)
    if not os.path.exists(traced_cell_tiles_dir):
        print('Creating directory: ' + traced_cell_tiles_dir)
        os.makedirs(traced_cell_tiles_dir)
        # if no target images, convert raw images to 8-bit target for prediction
        if not [f for f in os.listdir(traced_whole_image_dir) if f.endswith('.tif')]:
            convert_to_8bit(filelist, traced_whole_image_dir)
    if not os.path.exists(predicted_cell_tiles_dir):
        print('Creating directory: ' + predicted_cell_tiles_dir)
        os.makedirs(predicted_cell_tiles_dir)

    IMG_DIM = cv2.imread(filelist[0]).shape[0]
    print('IMG_DIM', IMG_DIM)
    TILE_DIM = 128
    NEURITE_TILE_SPLIT = True
    OVERLAP = 64#42

    # Splitting images into tiles
    print('..........Generating cell image tiles..........')
    if split:
        split_write_tiles_for_all_images(make_filelist(traced_whole_image_dir, 'tif'), traced_cell_tiles_dir)
        split_write_tiles_for_all_images(filelist, raw_cell_tiles_dir)
    # Equalizing file lists and removing tiles that lack annotation
    print('..........Equalizing file lists................')
    if equalize:
        raw_image_filelist, traced_image_filelist, labels = make_filelists_equal(make_filelist(raw_cell_tiles_dir, 'png'), make_filelist(traced_cell_tiles_dir, 'png'))
        pickle.dump((raw_image_filelist, traced_image_filelist, labels), open(os.path.join(
                traced_cell_tiles_dir, 'equalized_file_lists.p'), 'wb'))
    # Generating normalization files
    print('..........Generating normalization files.......')
    if make_norm_files:
        _, _ = get_norm_output(raw_image_filelist, expected_dtype=np.uint16)
        _, _ = get_norm_output(traced_image_filelist, expected_dtype=np.uint8, max_val=255)
    # Joining images
    print('..........Joining cell image tiles.............')
    if join:
        join_tiles_for_all_images(predicted_cell_tiles_dir, predicted_whole_image_dir, neurite_based=NEURITE_TILE_SPLIT)
