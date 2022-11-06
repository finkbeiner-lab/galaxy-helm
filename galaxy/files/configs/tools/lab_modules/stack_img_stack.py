run("Stack to Images");
run("Images to Stack", "name=Stack title=[] use");



# TODO: input will be stack or split. if stack run stack if split run stack to images.
#  will also need to be able to save both

#!/usr/bin/env python
"""
Input: a tif input image; some imagej macro code
Out: output image in tif format
"""
import sys, os, subprocess, tempfile

assert sys.version_info[:2] >= ( 2, 4 )

FIJI = "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"

def __main__():
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    code = sys.argv[3]
    outfile = sys.argv[4]
    premacro = """
        f=getArgument();
        f=split(f);
        print(f[0], f[1]);
        open (f[0]);
        open (f[1]);
        """
    # code to try
    code = """
        run('Images to Stack', 'name=Stack title=[] use');
        run('StackReg', 'transformation=Translation');
        """

    postmacro = """
        saveAs('Tiff', f[2]);
        dir = getInfo('image.directory');
        tif=dir+getInfo('image.filename');
        dat=replace(f[2], '.tif', '.dat');
        done=File.rename(tif, dat);
        run('Close All');
        eval("script", "System.exit(0);");
        """

    code_full = premacro + code + postmacro

    macro_fd, macro_filename = tempfile.mkstemp()
    os.write(macro_fd, code_full)
    os.close(macro_fd)
    # TODO: handle spaces in filenames
    params = '%s %s %s' % (file1, file2, outfile)
    fiji_command = [FIJI, "--headless", "-macro", macro_filename, params]
    print "Running", fiji_command
    #subprocess.call(fiji_command, stderr=subprocess.STDOUT)
    p = subprocess.Popen(fiji_command, stderr=subprocess.PIPE)
    p.wait()

if __name__ == "__main__" : __main__()
