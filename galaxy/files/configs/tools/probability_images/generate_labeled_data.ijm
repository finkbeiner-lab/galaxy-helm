//Open image and get it's parameters
img_title = getTitle();
ext = endsWith(img_title, ".tif");
if (ext != 0) {
    s = lastIndexOf(img_title, '.');
    img_title = substring(img_title, 0, s);
}
data_name = img_title+".arff";

// Initiate plugin
run("Trainable Weka Segmentation");
wait(3000);
// selectWindow("Trainable Weka Segmentation v2.2.1");

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

// selectWindow("Trainable Weka Segmentation v2.2.1");
//run("Enhance Contrast", "saturated=0.35");

// Wait for user to label data
waitForUser("Please label the image. When done, press OK to save result.");

// Save result
call("trainableSegmentation.Weka_Segmentation.saveData", File.directory() + data_name);
print("Saved:"+File.directory+data_name);
run("Close All");