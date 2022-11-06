import platform
import cv2
import numpy
import tensorflow
import argparse


def main():
    print(platform.python_version())
    print(cv2)
    print(numpy)
    print(tensorflow)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Library tool version printer.")
    parser.add_argument("outfile",
        help="Python version string.")
    args = parser.parse_args()
    outfile = args.outfile
    main()
    print("Outfile is: %s" % outfile)
