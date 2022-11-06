//images_path = "/media/mbarch/imaging-code/galaxy/tools/probability_images/Images/biogen/"
//write_path = "/media/mbarch/imaging-code/galaxy/tools/probability_images/ProbabilityImages/fromFIJI/"
//model_pointer = "/media/mbarch/imaging-code/galaxy/tools/probability_images/merged-classifier.model"
images_path = "/Volumes/RoboData/Mariya/NeuriteTrainingImages/Images/"
write_path = "/Volumes/RoboData/Mariya/NeuriteTrainingImages/ProbabilityImagesFIJI/"
model_pointer = "/Volumes/RoboData/Mariya/NeuriteTrainingImages/merged-classifier.model"

print("Starting probability analysis...");

list = getFileList(images_path);
print(images_path);
print(list.length);

for (i = 0; i < list.length; i++){

    // Part1: Training the model
    // Open image
    print("Opening image...");
    print(images_path+"/"+list[i]);
    open(images_path+"/"+list[i]);

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
    call("trainableSegmentation.Weka_Segmentation.loadClassifier", model_pointer);

    // Chose image to which to apply classifier
    print("Generating probability images...");
    //call("trainableSegmentation.Weka_Segmentation.applyClassifier", write_path, list[i], "showResults=true", "storeResults=false", "probabilityMaps=true", "");
    call("trainableSegmentation.Weka_Segmentation.getProbability");
    //Try this. It has to work before moving forward.
    //call("trainableSegmentation.Weka_Segmentation.applyClassifier", "/Volumes/RoboData/Mariya/NeuriteTrainingImages/Images", "PID20150215_BioP11asyn_T6_120_A8_1_RFP-DFTrCy5_BG_MONTAGE.tif", "showResults=true", "storeResults=true", "probabilityMaps=true", "");


	list_titles = getList("image.titles");
    print("Number of windows:", list_titles.length);
    for (k = 0; k < list_titles.length; k++){
    	print(list_titles[k]);
    }
    
    // Save result
    ext = endsWith(list[i], ".tif");
    if (ext != 0) {
        s = lastIndexOf(list[i], '.');
        list[i] = substring(list[i], 0, s);
    }
    prob_name = list[i]+"_PROB";
    selectWindow("Classification result");
    //selectWindow("Probability maps");
    run("8-bit");
    print(write_path+"/"+prob_name+".tif");
    save(write_path+"/"+prob_name+".tif");

    run("Close All");

    }
//eval("script", "System.exit(0);");

