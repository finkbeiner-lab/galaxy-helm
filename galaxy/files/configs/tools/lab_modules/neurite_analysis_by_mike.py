import subprocess
import datetime
import argparse
import os



def background_foreground_extraction(input_path, output_path):
    # use os.walk() to recursively iterate through a directory and all its subdirectories
    image_paths = [os.path.join(root, name) for root, dirs, files in os.walk(input_path) for name in files if name.endswith('.tif')]
    for image_path in image_paths:       
        input_image_filename = os.path.basename(image_path)
        output_image_path = os.path.join(output_path, os.path.basename(image_path).replace('.tif', '_NBFE.tif'))     
        external_command = ['/home/mbarch/neurite-analysis/bazel-bin/src/neurite', '--img_filename', image_path, '--out_filename', output_image_path, '--graph_filename', '/home/mbarch/neurite-analysis/data/foreground_model_v17_frozen.pbtxt']
        # Galaxy has a bug treating standard external program output as stderr(Maybe only in this neurite program, not sure, have not tested other programs yet). Thus here use PIPE to redirect and ignore the stderr error.
        # Note Do not use stdout=PIPE or stderr=PIPE with subprocess.call function as that can deadlock based on the child process output volume. Use Popen with the communicate() method when you need pipes.
        p = subprocess.Popen(external_command, stderr=subprocess.PIPE)
        # Warning Popen.wait() will deadlock when using stdout=PIPE and/or stderr=PIPE and the child process generates enough output to a pipe such that it blocks waiting for the OS pipe buffer to accept more data. Use communicate() to avoid that.
        p.communicate()






if __name__ == '__main__':
# --- For Galaxy run ---
    start_time = datetime.datetime.utcnow()
    # Parser
    parser = argparse.ArgumentParser(
        description="Neurite analysis")
    parser.add_argument("input_path", 
        help="Folder path to input data.")
    parser.add_argument("output_path",
        help="Folder path to output results.")
    parser.add_argument("outfile", 
        help="Write output information to file.")
    args = parser.parse_args()

    

    # Initialize parameters
    input_path = args.input_path
    output_path = args.output_path
     
    outfile = args.outfile
     
    # Create output folder
    try:     
        os.makedirs(output_path)
    except OSError:
        if not os.path.isdir(output_path):
            raise

    # Run neurite analysis
    background_foreground_extraction(input_path, output_path)

    # Print Total process time
    end_time = datetime.datetime.utcnow()
    print 'Neurite analysis run time:', end_time-start_time


    # Output for user
    print 'Neurite analysis has finished.'
    print 'Output was written to: \n%s' % output_path


    # Save output info to file
    with open(outfile, 'wb') as ofile: 
        ofile.write('Neurite analysis has finished. Output was written to: \n%s' % output_path)     







