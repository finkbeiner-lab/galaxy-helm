
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

// Make sure concatenate_arffs.py plugin is installed.
// Concatenate arff files (directory is set here)
run("concatenate arffs");

// Load data
//// Select folder containing merged.arff
call("trainableSegmentation.Weka_Segmentation.loadData", File.directory+"merged.arff");
print("The following file is used to train the model:", File.directory+"merged.arff");

// Train classifer
call("trainableSegmentation.Weka_Segmentation.trainClassifier");

// Save Classifer
print("Select folder to save model");
model_name = "merged-classifier"

call("trainableSegmentation.Weka_Segmentation.saveClassifier", File.directory+model_name+".model");
print("The trained model was saved to:", File.directory+model_name+".model");
run("Close All");