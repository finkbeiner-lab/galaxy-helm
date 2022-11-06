#!/bin/env python

import sys, os, subprocess, tempfile, cv2

assert sys.version_info[:2] >= ( 2, 4 )

def ext_changer():
    """
    Allows ITK to write to '.tiff', but returns file to galaxy as '.dat'.
    """

    standing_item = sys.argv[1]
    moving_item = sys.argv[2]
    outfile = sys.argv[3]

    reg_exe = '/Users/masha/ITK/ITKb/bin/ImageRegistration1'
    # The .tiff is necessary for ITK output
    reg_outfile = outfile + '.tiff'
    cplusplus_command = [reg_exe, standing_item, moving_item, reg_outfile]
    p = subprocess.Popen(cplusplus_command)#, stderr=subprocess.PIPE)
    p.wait()

    print reg_outfile, "->", outfile
    outfile = os.rename(reg_outfile, outfile)


if __name__ == "__main__":
    ext_changer()