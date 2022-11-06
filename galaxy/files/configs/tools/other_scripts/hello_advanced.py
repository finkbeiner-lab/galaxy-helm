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

    img = cv2.imread('PID20150210_BioP12asyn_T0_0_A1_10_RFP-DFTrCy5.tif', 0)
    print 'Image dimensions:', img.shape

    print 'Mean intensity:', round(img.mean(), 2)
    print 'Min intensity:', img.min()
    print 'Max intensity:', img.max()
    eimg = cv2.equalizeHist(img)
    print 'Mean intensity:', round(eimg.mean(), 2)
    print 'Min intensity:', eimg.min()
    print 'Max intensity:', eimg.max()

    height, width = eimg.shape[:2]
    res = cv2.resize(eimg, (int(.2*width), int(.2*height)), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite('../practice_tiny.tif', res)

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