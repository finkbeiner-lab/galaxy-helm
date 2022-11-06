#!/bin/env python
# python script to echo a command line parameter to an output file also passed on the command line
# your name here
# your favourite OSI approved licence here
import sys
import optparse
import numpy as np
import cv2
import os
import utils

def advanced():
    """
    Trivial example
    """

    os.chdir('/Volumes/Mariya/datasets/BioP12asyn/')
    print 'Set directory:', os.getcwd()

    red_well_a3 = utils.make_filelist('.', 'T0_0_A1_'+'*'+'RFP')
    print red_well_a3
    print 'Number of files:', len(red_well_a3)
    red_well_a3_t0 = [filename for filename in red_well_a3 if 'RFP' in filename]
    print 'Number of files:', len(red_well_a3_t0)
    test_img = cv2.imread(red_well_a3_t0[0], 0)
    im_rows, im_cols = test_img.shape

    robo_num = 2
    # horizontal_image_overlap = 40
    # vertical_image_overlap = 40
    robo_num = 3
    # horizontal_image_overlap = 30
    # vertical_image_overlap = 30
    flip = True
    montage_order = [1, 2, 3, 4, 8, 7, 6, 5, 9, 10, 11, 12, 16, 15, 14, 13]
    # robo_num = 4
    # horizontal_image_overlap = 20
    # vertical_image_overlap = 20
    # flip = False
    # montage_order = [4, 3, 2, 1, 5, 6, 7, 8, 12, 11, 10, 9, 13, 14, 15, 16]

    horizontal_num_images = 4
    vertical_num_images = 4
    horizontal_image_overlap = 30
    vertical_image_overlap = 30
    total_matrix = horizontal_num_images*vertical_num_images
    total_rows = horizontal_num_images*test_img.shape[0] - horizontal_image_overlap*(horizontal_num_images-1)
    total_cols = vertical_num_images*test_img.shape[1] - vertical_image_overlap*(vertical_num_images-1)
    montaged_image = np.ones((total_rows, total_cols), dtype=np.uint8)
    print montaged_image.shape
    row_starter = 0; col_starter = 0

    for im_num in montage_order[0:total_matrix]:

        img = cv2.equalizeHist(cv2.imread(red_well_a3_t0[im_num-1], 0))
        img = cv2.flip(img, 0)
        assert img.shape == (im_rows, im_cols)

        montaged_image[row_starter:(row_starter+im_rows), col_starter:(col_starter+im_cols)] = img
        if col_starter+im_cols <= total_cols:
            row_starter = row_starter
            col_starter = col_starter+im_cols-vertical_image_overlap
        if col_starter+im_cols > total_cols:
            row_starter = row_starter+im_rows-horizontal_image_overlap
            col_starter = 0

    print 'Dimensions with '+str(horizontal_image_overlap)+' pixels overlap:', montaged_image.shape
    cv2.imwrite('../montaged_from_galaxy.tif', montaged_image)

    usage = "%s -o outfilename -s stringtowrite1 -s stringtowrite2 ..." % sys.argv[0]
    parser = optparse.OptionParser(usage = usage)
    parser.add_option("-s", "--stringtowrite",
                     action="append", type="string",dest="mystring",help="Strings to write")
    parser.add_option("-o","--outputfile",
                     action="store", type="string",dest="outputfile",help="output text file")
    (opts, args) = parser.parse_args()
    assert len(opts.mystring) > 0, "No strings to write found on command line"
    assert opts.outputfile,"No output file name found on command line"
    outf = open(opts.outputfile,'w')
    outf.write(str(img.shape))
    outf.write('\n')
    outf.write('\n'.join(opts.mystring))
    outf.write('\n')
    outf.close

if __name__ == "__main__":
        advanced()