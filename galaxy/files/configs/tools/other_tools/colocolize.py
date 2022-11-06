'''
Program should take path_data, write_path, channel1_name, channel1_max, channel1_min, channel2_name, channel2_max, channel2_min.
Generate a list of relevant files for each channel that should be colocolized.
For each image (well/timepoint/frame/etc), read in both channels, make a third image.  
If channel1_min<channel1<channel1_max and channel2_min<channel2<channel2_max, image3 = on-value, else image3 = 0
If both channels do not exist, skip.
Write this mask to output path.
Potentially use in the more general segmentation step.
'''
import cv2, utils, sys, os, argparse
import numpy as np
import pickle, datetime, shutil, pprint

def threshold_img(img_path, img_min, img_max):
    '''
    Reads in image from image path.
    Thresholds based on given min and max.
    '''
    img = cv2.imread(img_path, -1)
    thresh_img = np.zeros(img.shape, np.uint8)
    img_kept = np.logical_and(img > img_min, img < img_max)
    thresh_img[img_kept] = 255
    return thresh_img

def pixel_binary_coloc(c1_thresh_img, c2_thresh_img):
    '''
    Takes two images.
    Returns one binary image with coincident pixels.
    '''
    coloc_img = np.zeros(c1_thresh_img.shape, np.uint8)
    logic_matrix = np.logical_and(c1_thresh_img > 0, c2_thresh_img > 0)
    coloc_img[logic_matrix] = 255
    return coloc_img

def find_pixel_overlap(images_list, var_dict, output_path):  
    '''
    Takes a list referencing one image per channel.
    Thresholds both images based on user input of intensity min and max.
    Generates colocolization image for each pair of thresholded images.
    '''

    c1_name = var_dict["Channel1Name"]
    c2_name = var_dict["Channel1Name"]

    if len(images_list) > 2:
        print 'Images to coloc:', 
        pprint.pprint(os.path.basename([img for img in images_list]))
        assert len(images_list) == 2, 'Found more than one image for each channel.'
    
    if len(images_list) == 2:
        c1_pointer = [img_pointer for img_pointer in images_list if c1_name in img_pointer]
        c2_pointer = [img_pointer for img_pointer in images_list if c2_name in img_pointer]
        assert len(c1_pointer) == 1, 'Found '+len(c1_pointer)+' channel1 images.'
        assert len(c2_pointer) == 1, 'Found '+len(c2_pointer)+' channel2 image.'

        c1_pointer = c1_pointer[0]
        c2_pointer = c2_pointer[0]

        c1_thresh_img = threshold_img(
            c1_pointer, var_dict["Channel1Min"], var_dict["Channel1Max"])
        c2_thresh_img = threshold_img(
            c2_pointer, var_dict["Channel2Min"], var_dict["Channel2Max"])
        coloc_img = pixel_binary_coloc(c1_thresh_img, c2_thresh_img)
        print 'Number kept pixels (Img1, Img2, ColocImg):', 
        print sum(sum(c1_thresh_img)), sum(sum(c2_thresh_img)), sum(sum(coloc_img))

        coloc_name = utils.make_file_name(
            output_path, utils.extract_file_name(c1_pointer)+'_CL')
        cv2.imwrite(coloc_name, coloc_img)

def colocolization(var_dict, input_path, output_path, verbose=False):
    '''
    Main point of entry. 
    '''
   
    # Select an iterator, if needed
    iterator, iter_list = utils.set_iterator(var_dict)
    if verbose:
        print 'What will be iterated:', iterator

    for well in var_dict['Wells']:
        for timepoint in var_dict['TimePoints']:
            print 'Processing...', well, timepoint
                
            # Get relevant images
            selector = utils.make_selector(
                iterator=iterator, well=well, timepoint=timepoint)
            images_list = utils.make_filelist(input_path, selector)

            if verbose:
                print "Selecting files matching:", selector
                pprint.pprint(images_list)

            # Handles no images or not enough images
            if len(images_list) < 2:
                continue
            # Handles iterators (bursts or depths)
            elif len(images_list) > 2:
                for frame in iter_list: 
                    # Get relevant images
                    selector = utils.make_selector(
                        iterator, well, timepoint, channel, frame)
                    if verbose:
                        print "Selecting files matching with burst:", selector
                    images_list = utils.make_filelist(source_path, selector)
                    if len(images_list) < 2:
                        continue
                    else:
                        find_pixel_overlap(
                            images_list, var_dict, output_path)
            # All images will be used to calculate median image (might be many)
            else:
                find_pixel_overlap(
                    images_list, var_dict, output_path)


if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Create mask of colocolized regions.")
    parser.add_argument("input_dict", 
        help="Load input variable dictionary")
    parser.add_argument("input_path", 
        help="Folder path to input data.")
    parser.add_argument("output_path",
        help="Folder path to ouput results.")
    parser.add_argument("channel1_name",
        help="String of characters unique to channel1.")
    parser.add_argument("channel2_name",
        help="String of characters unique to channel2.")
    parser.add_argument("output_dict", 
        help="Write variable dictionary.")
    
    parser.add_argument("--channel1_min", "-min1",
        dest="channel1_min", type=int, default=10, 
        help="Minimum intensity to include for channel1.")
    parser.add_argument("--channel1_max", "-max1",
        dest="channel1_max", type=int, default=255, 
        help="Maximum intensity to include for channel1.")
    parser.add_argument("--channel2_min", "-min2",
        dest="channel2_min", type=int, default=10, 
        help="Minimum intensity to include for channel2.")
    parser.add_argument("--channel2_max", "-max2",
        dest="channel2_max", type=int, default=255, 
        help="Maximum intensity to include for channel2.")
    args = parser.parse_args()

    # ----Load path dict-------------------------
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))

    # ----Initialize parameters------------------
    path_to_images = args.input_path
    write_masks_path = args.output_path
    var_dict["Channel1Name"] = utils.get_ref_channel(
        args.channel1_name, var_dict['Channels'])
    var_dict["Channel2Name"] = utils.get_ref_channel(
        args.channel2_name, var_dict['Channels'])
    morphology_channel = var_dict['MorphologyChannel']
    var_dict["Channel1Min"] = int(args.channel1_min)
    var_dict["Channel1Max"] = int(args.channel1_max)
    var_dict["Channel2Min"] = int(args.channel2_min)
    var_dict["Channel2Max"] = int(args.channel2_max)
    outfile = args.output_dict
    resolution = -1 #actual

    # ----Confirm given folders exist--
    assert os.path.exists(path_to_images), 'Confirm the given path for data exists.'
    assert os.path.exists(write_masks_path), 'Confirm the given path for mask output exists.'

    # ----Run segmentation-----------------------
    start_time = datetime.datetime.utcnow()

    colocolization(var_dict, path_to_images, write_masks_path)

    end_time = datetime.datetime.utcnow()
    print 'Segmentation run time:', end_time-start_time
    # ----Output for user and save dict----------
    print 'Images were colocolized.'
    print 'Output was written to:'
    print write_masks_path

    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    # outfile = os.rename('var_dict.p', outfile)
    outfile = shutil.move('var_dict.p', outfile)