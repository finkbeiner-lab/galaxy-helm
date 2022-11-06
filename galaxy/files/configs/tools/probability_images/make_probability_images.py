import sys, os, subprocess, tempfile
import pickle, datetime
import utils, shutil, cv2, pprint

assert sys.version_info[:2] >= ( 2, 4 )

if 'darwin' in sys.platform:
    FIJI = "/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"
elif 'linux' in sys.platform:
    FIJI = "/usr/local/bin/Fiji.app/ImageJ-linux64"
else:
    raise RuntimeError('System path for FIJI was not set on this OS.')


def run_ij_commands(ij_code):
    '''Set up and i/o and run functions in ImageJ.'''

    macro_fd, macro_filename = tempfile.mkstemp()
    os.write(macro_fd, ij_code)
    os.close(macro_fd)
    print 'Tempfile:', macro_filename
    fiji_command = ["xvfb-run", FIJI, "-macro", macro_filename]
    print "Running", fiji_command
    # p = subprocess.call(fiji_command, stderr=subprocess.STDOUT)
    p = subprocess.Popen(fiji_command, stderr=subprocess.PIPE)
    p.wait()

def make_probimg_ijweka(images_path, write_path, model_path, files_to_segment):
    '''
    Loops through all images needing segmentation.
    Opens Weka plugin and loads specified model for each image.
    Saves probability stacks to write_path.
    '''

    # Important: IJ does not handle titles with more than 60 characters
    # img_title = getTitle() <--Does not get complete filename, 60 char limit
    # Use instead a substring of getImageInfo
    # getImageInfo holds the complete title in the first line

    # pprint.pprint(files_to_segment)

    if len(files_to_segment) == 0:
        return len(files_to_segment)
    for path in [images_path, write_path]:
    	if path[-1] != os.sep:
    		path = path+os.sep
    	else:
    		continue

    ij_code ='''
        print("Starting probability analysis...");

        list = getFileList("'''+images_path+'''");
        print(list.length);

		for (i = 0; i < list.length; i++){

		    // ----Part1: Training the model-----

		    // Open image
		    print("Opening image...");
		    print("'''+images_path+'''"+list[i]);
		    open("'''+images_path+'''"+list[i]);

            // Initiate plugin
            print("Loading weka plugin...");
            run("Trainable Weka Segmentation");
            wait(3000);

            // Set all unused filters to flase
            print("Setting parameters...");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Gaussian_blur=false");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Sobel_filter=false");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Hessian=true");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Difference_of_gaussians=true");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Gabor=true");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Memebrane_projections=true");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Laplacian=true");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Variance=false");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Mean=false");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Minimum=false");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Maximum=false");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Median=false");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Anisotropic_diffusion=false");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Bilateral=false");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Lipschitz=false");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Kuwahara=false");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Derivatives=false");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Structure=false");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Entropy=false");
            call("trainableSegmentation.Weka_Segmentation.setFeature", "Neighbors=false");

            // Set classifier names
            call("trainableSegmentation.Weka_Segmentation.changeClassName", "0", "Background");
            call("trainableSegmentation.Weka_Segmentation.changeClassName", "1", "Neurites");
            call("trainableSegmentation.Weka_Segmentation.createNewClass", "NeuronBodies");
            call("trainableSegmentation.Weka_Segmentation.createNewClass", "Debris");

            // ----Part2: Applying the model and obtaining probability images--

            // Load model (classifier)
            print("Retrieving and applying model...", "Current image:", i);
            call("trainableSegmentation.Weka_Segmentation.loadClassifier", "'''+model_pointer+'''");

            // Chose image to which to apply classifier
            print("Generating probability images...");
            //call("trainableSegmentation.Weka_Segmentation.applyClassifier", "'''+write_path+'''", list[i], "showResults=false", "storeResults=true", "probabilityMaps=false", "");
            call("trainableSegmentation.Weka_Segmentation.getProbability");

            // Save result
            print("Saving result...");
            ext = endsWith(list[i], ".tif");
            if (ext != 0) {
                s = lastIndexOf(list[i], '.');
                list[i] = substring(list[i], 0, s);
            }
            prob_name = list[i]+"_PROB"+".tif";

            //selectWindow("Classification result");
            selectWindow("Probability maps");
            //selectWindow(list_titles[0]);
            run("8-bit");
            print("'''+write_path+'''"+prob_name);
            save("'''+write_path+'''"+prob_name);

            run("Close All");

            }
        eval("script", "System.exit(0);");
        '''

    return ij_code

def learn_segmentation(images_path, write_path, model_pointer):
    '''
    Main point of entry.
    '''
    morph_channel = sys.argv[4]
    selector = utils.make_selector(channel=morph_channel)
    files_to_segment = utils.make_filelist(images_path, selector)
    pprint.pprint([os.path.basename(seg_file) for seg_file in files_to_segment])
    if len(files_to_segment) != 0:
      # Launch Weka plugin from ImageJ
        ij_code = make_probimg_ijweka(
            images_path, write_path, model_pointer, files_to_segment)
        run_ij_commands(ij_code)

    # for image_pointer in files_to_segment:
    # for well in var_dict['Wells']:
        # # Launch Weka plugin from ImageJ
        # print 'Image:', os.path.basename(image_pointer)
        # ij_code = make_probimg_ijweka(
        #     images_path, write_path, model_pointer, files_to_segment)
        # if ij_code == 0:
        #     continue
        # run_ij_commands(ij_code)

if __name__ == '__main__':

    images_path = sys.argv[1]
    write_path = sys.argv[2]
    model_pointer = sys.argv[3]

    learn_segmentation(images_path, write_path, model_pointer)


