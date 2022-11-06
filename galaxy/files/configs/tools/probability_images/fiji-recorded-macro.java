run("Image Sequence...", "open='''+images_path+''' file='''+selector+''' sort");

print("Beginning alignment...");
run("StackReg", "transformation=Translation");
print("Completed alignment!");
run("Stack to Images");

for (i=0; i<nImages; i++) {

    selectImage(i+1);
    img_info = getImageInfo();
    img_title = substring(img_info, 0, indexOf(img_info, "\\n"));

    ext = endsWith(img_title, ".tif");
    if (ext != 0) {
        s = lastIndexOf(img_title, '.');
        img_title = substring(img_title, 0, s);
    }
    aligned_name = img_title+"_ALIGNED";
    saveAs("Tiff", "'''+write_path+'''/"+aligned_name+".tif");
}
run("Close All");
eval("script", "System.exit(0);");

//take file_pointer, merged-model, write_path

write_path = "/media/mbarch/imaging-code/galaxy/tools/probability_images/"

// Part1: Training the model
// Open image
open("/media/mbarch/imaging-code/galaxy/tools/probability_images/PID20150219_BioP7asynA_T3_48_A8_1_RFP-DFTrCy5_BG_MONTAGE_ALIGNED.tif");

// Initiate plugin
run("Trainable Weka Segmentation");
wait(3000);
selectWindow("Trainable Weka Segmentation v2.2.1");

// Set all unused filters to flase
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

// Part2: Applying the model and obtaining probability images
// Load model (classifier)
call("trainableSegmentation.Weka_Segmentation.loadClassifier", "'''+write_path+'''/merged-classifier.model");

// Chose image to which to apply classifier
print("Generating probability images...");
//call("trainableSegmentation.Weka_Segmentation.applyClassifier", "/media/mbarch/imaging-code/galaxy/tools/probability_images", "PID20150219_BioP7asynA_T3_48_A8_1_RFP-DFTrCy5_BG_MONTAGE_ALIGNED.tif", "showResults=true", "storeResults=false", "probabilityMaps=true", "");
call("trainableSegmentation.Weka_Segmentation.getProbability");

// Save result
//selectWindow("Classification result");
selectWindow("Probability maps");
run("Save", "save=['''+write_path+'''Probability maps.tif]");

run("Close All");
eval("script", "System.exit(0);");

//----dkp----------------------------------------
// Interact to make labels

// Save data for multiple images
//call("trainableSegmentation.Weka_Segmentation.saveData", "/home/mariyabarch/T0-mb.arff");
// Assuming headers are the same, concat all the saved data