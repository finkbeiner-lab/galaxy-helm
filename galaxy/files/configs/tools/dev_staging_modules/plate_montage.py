import argparse
import pickle
import utils
import cv2
import os
import numpy as np
import string
import re

def add_label(img, filename, y_pred):
    '''
    Creates image labels for samples_montage function.
    '''
    tokens = filename.split('_')
    cv2.putText(img, tokens[1], (3,10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255), 1)
    cv2.putText(img, tokens[2], (3,20), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255), 1)
    cv2.putText(img, tokens[4], (3,30), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255), 1)
    cv2.putText(img, tokens[-1].split('.')[0], (3,40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255), 1)
    cv2.putText(img, 'pred = ' + y_pred, (3,img.shape[0]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255), 1)

    return img


def prepare_img(path):
    ''' Crops, resizes, normalizes, saturates, and converts images '''
    img = cv2.imread(path, -1)
    padding = int((img.shape[0] - (img.shape[0] / crop_factor)) / 2)
    if padding > 0:
        img = img[padding:-padding, padding:-padding]
    img = cv2.resize(img, dsize=(img_size, img_size), interpolation=cv2.INTER_AREA)
    img = np.clip(img / norm_intensity * 255, 0, 255)
    img = img.astype('uint8')
    img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=(border_intensity))

    return img


if __name__ == '__main__':


    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Montage wells.")
    parser.add_argument("input_dict",
        help="Load input variable dictionary")
    parser.add_argument("img_size",
        help="Target image dimension in pixels")
    parser.add_argument("crop_factor",
        help="Crop factor")
    parser.add_argument("norm_intensity",
        help="Max intensity to normalize to")
    parser.add_argument("keep_blanks",
        help="Option to keep rows/cols with no images")
    parser.add_argument('output_dict',
        help='Write variable dictionary.')
    parser.add_argument("--images_path",
        help="Folder path to input data.", default = '')
    parser.add_argument("--output_path",
        help="Folder path to ouput results.", default = '')
    args = parser.parse_args()

    # ----Load path dict-------------------------
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))

    # ----Initialize parameters------------------
    images_path = utils.get_path(args.images_path, var_dict['GalaxyOutputPath'], 'AlignedImages')
    assert os.path.exists(images_path), 'Confirm path for images exists (%s)' % args.images_path
    print('Images input path: %s' % images_path)

    output_path = utils.get_path(args.output_path, var_dict['GalaxyOutputPath'], 'PlateMontages')
    assert os.path.exists(os.path.dirname(output_path)), 'Confirm that the output path parent folder (%s) exists.' % os.path.dirname(output_path)
    utils.create_dir(output_path)
    print('Plate Montages output path: %s' % output_path)

    img_size = int(str.strip(args.img_size))
    print('Individual image size: %ix%i' % (img_size, img_size))

    crop_factor = float(str.strip(args.crop_factor))
    assert crop_factor >= 1, 'Crop factor must be value of 1 or greater.'
    print('Crop factor: %.1f' % crop_factor)

    timepoints = var_dict['TimePoints']
    channels = var_dict['Channels']

    # non-user-modifiable parameters
    background_intensity = 0
    border_size = int(img_size/100)
    border_intensity = 0

    # get image paths
    image_paths = ''
    if var_dict['DirStructure'] == 'root_dir':
        image_paths = [os.path.join(images_path, name) for name in os.listdir(images_path) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
    elif var_dict['DirStructure'] == 'sub_dir':
        # Only traverse root and immediate subdirectories for images
        relevant_dirs = [images_path] + [os.path.join(images_path, name) for name in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, name))]
        image_paths = [os.path.join(relevant_dir, name) for relevant_dir in relevant_dirs for name in os.listdir(relevant_dir) if name.endswith('.tif') and '_FIDUCIARY_' not in name]
    else:
        raise Exception('Unknown Directory Structure!')
    img_tokens = utils.tokenize_files(image_paths)

    # filter image paths by selected wells and timepoints
    img_tokens = [x for x in img_tokens if x[5] in var_dict['Wells'] and x[3] in var_dict['TimePoints']]

    # get dataset max intensity for normalization, if needed
    norm_intensity = int(str.strip(args.norm_intensity))
    if norm_intensity == 0:
        for tokens in img_tokens:
            img = cv2.imread(tokens[0], -1)
            img_max = np.amax(img)
            if img_max > norm_intensity:
                norm_intensity = img_max
            img = None
    print('Normalization value: %i' % norm_intensity)

    img_paths = [x[0] for x in img_tokens]

    # get montage cols and rows
    wells = [x[5] for x in img_tokens]
    rows = utils.natural_sort(list(set([x[:1] for x in wells])))
    cols = sorted(list(set([x[1:] for x in wells])))

    # fill in non-existing rows/cols
    if args.keep_blanks:
        letters = list(string.ascii_uppercase)
        first_row_idx = [i for i, x in enumerate(letters) if x == rows[0]][0]
        last_row_idx = [i for i, x in enumerate(letters) if x == rows[-1]][0]
        rows = letters[first_row_idx:last_row_idx+1]

        number_start = cols[0]
        number_end = cols[-1]
        leading_zero = re.match(r'0\d+', str(number_start))
        num_range = range(int(number_start), int(number_end) + 1)
        cols = []
        for num in num_range:
            if leading_zero and num < 10:
                num = '0' + str(num)
            cols.append(str(num))

    print('Montage rows: %s' % rows)
    print('Montage columns: %s' % cols)

    # initialize montage as array of zeros
    montage_height = (len(rows)+1) * (img_size+border_size*2)
    montage_width = (len(cols)+1) * (img_size+border_size*2)
    blank_montage = np.zeros((montage_height, montage_width), dtype='uint8')
    blank_montage[:] = background_intensity  # fill blank space with gray

    label_size = int(img_size*0.01)
    label_thickness = int(img_size*0.01)
    edge_padding = int(img_size/2)

    for tp in timepoints:

        for ch in channels:
            montage = blank_montage
            r_start = edge_padding

            for row in rows:
                # create row label
                row_label = np.full((img_size, edge_padding), background_intensity, dtype='uint8')
                row_label = cv2.putText(row_label, str(row), (int(edge_padding/3), int(img_size/1.7)), cv2.FONT_HERSHEY_DUPLEX, label_size, (255), label_thickness)
                blank_montage[r_start:r_start+row_label.shape[0], 0:row_label.shape[1]] = row_label
                c_start = edge_padding

                for col in cols:
                    # create column label
                    col_label = np.full((edge_padding, img_size), background_intensity, dtype='uint8')
                    col_label = cv2.putText(col_label, str(col), (int(img_size/3), int(edge_padding/1.5)), cv2.FONT_HERSHEY_DUPLEX, label_size, (255), label_thickness)
                    blank_montage[0:col_label.shape[0], c_start:c_start+col_label.shape[1]] = col_label

                    well = row + str(col)

                    img_path = utils.get_filename(img_tokens, well, tp, ch)
                    if (len(img_path) == 0):
                        print('No image found for %s' % well)
                        # if no image for this well, insert blank image (just background color + border)
                        img = np.full((img_size, img_size), background_intensity, dtype='uint8')
                        img = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, cv2.BORDER_CONSTANT, value=(border_intensity))
                    else:
                        # if there is an image for this well (if multiple images found, will take the first one)
                        img = prepare_img(img_path[0])

                    r_end = r_start + img_size + border_size*2
                    c_end = c_start + img_size + border_size*2
                    montage[r_start:r_end, c_start:c_end] = img

                    c_start += img_size + border_size*2

                r_start += img_size + border_size*2

            tokens = utils.tokenize_files(img_paths)[0]
            montage_name = '_'.join([tokens[1],tokens[2],tp,ch,tokens[utils.get_channel_token(var_dict['RoboNumber'])],'PLATE_MONTAGE.jpg'])
            print('Montage saved: %s' % montage_name)
            cv2.imwrite(os.path.join(output_path, montage_name), montage)

    # write out dictionary
    outfile = args.output_dict
    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    timestamp = utils.update_timestring()
    utils.save_user_args_to_csv(args, output_path, 'plate_montage'+'_'+timestamp)
