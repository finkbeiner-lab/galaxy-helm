import os
import cv2
import datetime
import argparse

def detect_corrupted_images(input_path, outfile):
    # Corrupted images count
    i = 0
    with open(outfile, 'wb') as ofile: 
        # input_image_names = [name for name in os.listdir(input_path) if name.endswith('.tif')]
        image_paths = [os.path.join(root, name) for root, dirs, files in os.walk(input_path) for name in files if name.endswith('.tif')]

        for i_path in image_paths:
            img = cv2.imread(i_path, -1)
            if img is None:
                i = i + 1
                ofile.write(i_path+'\n')
    return i


if __name__ == '__main__':
    # --- For Galaxy run ---
    start_time = datetime.datetime.utcnow()
    # Parser
    parser = argparse.ArgumentParser(
        description="Corrupted Images Detector")
    parser.add_argument("input_path", 
        help="Folder path to input images.")
    parser.add_argument("outfile",
        help="Output text file")
    args = parser.parse_args()

    # Initialize parameters
    input_path = args.input_path
    outfile = args.outfile

    # Confirm given input/output folders exist
    assert os.path.isdir(input_path), 'Confirm the given path for input data exists.'
     
    # Run alignment
    num_of_corrupted_images = detect_corrupted_images(input_path, outfile)

    # Print Total process time
    end_time = datetime.datetime.utcnow()
    print '%d images are corrpted. Please download the list.' % num_of_corrupted_images
    print 'Corrupted Images Detector run time: %s' %(end_time-start_time)


   