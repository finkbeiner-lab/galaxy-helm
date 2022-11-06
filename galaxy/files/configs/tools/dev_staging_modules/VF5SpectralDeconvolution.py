'''
VF5SpectralDeconvolution is a class that is designed to  be inside the galaxy anaylsis engine, 
take parameters via argparse and output N-channels of images based on the information in 
M-spectral slices. 

One method to do this is via linear least squares, this is a method that will "fit" a linear line 
to determine quantities of the fluorophore present. In practice, this method is incredibly slow 
because of the need to iteratively fit all the pixels of the stack, creating a large amount of 
computational overhead. This method also requires that all images in a spectral stack must 
open in memory for the calculation to work. 

The other method implemented in this class works by using tifffile.py and numpy to add values
to each of the channels, based on the relative quantity in each spectra. This method is not 
perfect and requires a back "scaling" and subtraction to remove parts of the image not associated 
with this channel. This scaling process is analagous to what the fit is doing in the leastSquares 
method but ignores the fact that each pixel does not have all fluorophores. 
'''

import numpy as np
import csv
import tifffile
import os
import argparse as ap

ExperimentName=""

def cleanFilenames(inputPath):
        # removes "DAPIFITC" and "CFPYFP" from the filenames
        FileList = os.listdir(inputPath)
        for i in FileList:
                if 'CFPYFP' in i:
                        os.rename(i,i.replace('CFPYFP-',''))
                if 'DAPIFITC' in i:
                        os.rename(i,i.replace('DAPIFITC-',''))
                        
                        
def createRefDataMatrix2(inputPath, refWellArray, emptyWell, outputPath):
        fileList = os.listdir(inputPath)
        #sort all files
        fileList.sort()
        refArray = np.array([])
        tempRef = np.array([])
        emptyWell = np.array(emptyWell,dtype=np.float64)
        for r in refWellArray:
                for t in range(0,10):
                        for tile in range(1,26):
                                #print 'looking for reference data at: T'+str(t)+'_ and '+r+'_'+str(tile)+'_'
                                refStackList = [i for i in fileList if '_T'+str(t)+'_' in i and '_'+r+'_'+str(tile)+'_' in i]
                                if len(refStackList)==0:
                                        continue
                                refStackList.sort()
                                tempRefArray = np.ones((len(refStackList)))
                                for i in range(len(refStackList)):
                                    img = inputPath+refStackList[i]
                                    tempImg=tifffile.TiffFile(img).asarray()
                                    tempImg = np.array(tempImg,dtype=np.float64)
                                    #obtain a thresholded value for this image
                                    tempImg /=  emptyWell[i,:,:]# correct intensity by dividing the empty well.
                                    tempImg = tempImg *( tempImg > 1.0) 
                                    tempImg = tempImg * (tempImg < (2**16) - 1) # Exclude saturated pixels
                                    if np.count_nonzero(tempImg) == 0:
                                            threshVal=0.0
                                    else:
                                            threshVal = 1.0 * np.sum(tempImg) / np.count_nonzero(tempImg)
                                    tempRefArray[i] = threshVal
                                    #print tempRefArray
                                if  len(tempRef)==0:
                                    tempRef = np.array(tempRefArray)
                                    #print tempRef
                                else:
                                    tempRef += np.array(tempRefArray,dtype=np.float64)
                                    tempRef /= 2.0
                                print tempRef
                                print ' '
                if r==refWellArray[0]:
                    refArray = np.ones((len(refWellArray),len(tempRef)),dtype=np.float64)
                tempRef /= np.max(tempRef) #Normalization
                refArray[refWellArray.index(r)] = tempRef
                print refArray
                print " "
        print np.array(refArray*int(2**15),dtype=np.uint16)
        saveFile(outputPath+'Reference.tif',np.array(refArray*int(2**15),dtype=np.uint16))
        return refArray
                        
def createStack(inputPath, Well):
        # Well is a list of the files associated with a tile in the working well
        #
        # This wil then generate a numpy array of all the images and return the stack
        FileList = Well
        # put all the wavelengths in order lowest to highest
        FileList.sort() 
        # pull the first image in the series and set array size
        tempTif = tifffile.TiffFile(inputPath+FileList[0]).asarray()
        # initialize the stack in memory 
        imgstack = np.zeros((len(FileList),tempTif.shape[0],tempTif.shape[1]))
        zIndex = 0
        for i in FileList:
                I = inputPath+i
                expName = i.split("_")[1]
                #print "Processing file: " + I
                tempTif = tifffile.TiffFile(I).asarray()
                imgstack[zIndex,:,:] = tempTif
                zIndex+=1
                tifffile.TiffFile(I).close
        return imgstack,expName

def createEmtpyWellStack(inputPath, outputPath, empty):
        emptyWellStack=np.array([])
        for t in range(0,10): #timepoints
                for i in range(1,26): # well indexPositions
                            Files = [f for f in os.listdir(inputPath) if ('_T'+str(t)+'_' in f and '_'+str(i)+'_' in f and '_'+empty[0]+'_' in f)]
                            #print inputPath
                            # Sort the files from shortest to longest wavelength
                            Files.sort()
                            plane=0
                            for f in Files:
                                F = inputPath+f
                                #print 'Processing: '+F
                                img = tifffile.TiffFile(F).asarray()
                                if emptyWellStack.size ==0:
                                    emptyWellStack = np.ones((len(Files),img.shape[0],img.shape[1]),dtype=np.uint16)
                                if i==1:
                                    emptyWellStack[plane,:,:] = img[:,:]
                                else:
                                    emptyWellStack[plane,:,:] = (emptyWellStack[plane,:,:] + img[:,:]) / 2.0
                                plane+=1
        saveFile(outputPath+'EmptyWellStack.tif', emptyWellStack)
        return emptyWellStack
                            
def transferFunction(inputPath, BlankWell):
        #Calcualte the transfer function and correction coefficient for the plate
        #
        #This is based off the paper:
        #     THIN-FILM TUNABLE FILTERS FOR HYPERSPECTRAL FLUORESCENCE MICROSCOPY
        #          Peter Favreau, Clarissa Hernandez, Ashley Stringfellow Lindsey, Diego F. Alvarez, Thomas Rich,
        #               Prashant Prabhat, Silas J. Leavesley 
        #
        # Need to acquire images from a 'bright', 'dark', and 'lightsource' at each spectral step to make the coefficient
        # Bright is an image stack filling 50% of the image register of the full spectrum
        # Dark is an image stack of no illumination (sensor noise)
        # Lamp is each CWL that will be collected in the acquisition
        
        # Collect the Blank well in to memory
        darkTiles = [i for i in os.listdir(inputPath) if (BlankWell in i and 'Dark' in i)]
        lampTiles = [i for i in os.listdir(inputPath) if (BlankWell in i and 'Lamp' in i)]
        for tile in len(blankTiles):
               darkStack += createStack(darkTiles[tile], True)
               lampStack += createStack(lampTiles[tile],True)
        darkStack = np.mean(darkStack,axis=0,dtype=np.float) # keep spatial but average through Z 
        brightstack = np.mean(lampStack,axis=0,dtype=np.float) # keep spatial but average through Z
        lampStack = np.mean(lampStack,axis=(1,2),dtype=np.float) # Keep lampStack as a Z-stack but average in XY

        TF = (brightStack - darkStack) / lampStack # Transfer Function  
        CC = 1.0 / TF # Correction Coefficient is the inverse of the transfer function

        return CC, darkStack # return back the CC and Idark as they are both needed for the spectral correction step

def processViaLinearLeastSquares(imgStack,refArray, outputPath, TP, WellPos):
        # Run the linear least squares analysis on a single pixel of data
        #
        #This function reshapes the array to arrange all pixels in a single column with spectral data in rows
        #    The function then moves through and performs a linear least squares fit to each pixel according to 
        #    the number of spectra defined in refArray
        #
        # Arguments provided are the imgStack  and the refArray
        #    Provided by parseRefArray()
        #
        # This is quite slow when needing to check all pixels of a Zyla4.2
        #
        # Uses correction coefficient to apply a spectral correction based on the transfer function
    
        # Reshape the array such that array is 2D with Z-axis in rows
        hypercube = imgStack.T.reshape(imgStack.shape[1]*imgStack.shape[2], imgStack.shape[0])
        # collect the dimensions for later use
        dimZ,dimY,dimX = imgStack.shape
        #create the refArray
        X = np.ones((refArray.shape[0],refArray.shape[1]))
        scalingFact=np.ones((X.shape[0]))
        for i in range(refArray.shape[0]):
                X[i,:]=refArray[i,:]
        X= np.array(X.T, dtype=np.float64) 
        print 'Xshape: '+str(X.shape)
        # Create a blank list of output values
        coefficientOutput = np.ones((hypercube.shape[0],X.shape[1]), dtype=np.float64)
        #print 'CO single pixel: '+str(coefficientOutput.shape)
        progress=0
        for pixel in range(hypercube.shape[0]): 
                Y =  np.array(hypercube[pixel,:], dtype=np.float64)
                scalingFact = np.max(Y*X.T,axis=1)
                #coefficientOutput[pixel,:] = np.abs(np.linalg.lstsq(X,Y)[0]) # index position 0 is the solutions, other 3 are residuals, rank and sigularities
                coefficientOutput[pixel,:] = np.abs(np.linalg.solve(X.T.dot(X), X.T.dot(Y)))
                coefficientOutput[pixel,:] /= np.max(coefficientOutput[pixel,:]) #/ 2**16 #rescale for 16-bit image
                #coefficientOutput[pixel,:] *= int(2**16-1) # saturates to 2**16-1
                #coefficientOutput[pixel,:] *= np.sum(Y); # rescale to value of original pixel
                coefficientOutput[pixel,:] *= scalingFact
                if progress%100000==0:
                        print progress
                        print coefficientOutput[pixel,:]
                progress+=1
        return np.array(coefficientOutput, dtype=np.uint16)
#        # Reshape the coefficient array to create images of each channel image and save
#        coefficientOutput = np.array(coefficientOutput)
#        chan = coefficientOutput.shape[1]
#        for i in range(chan):
#                co = coefficientOutput[:,i]
#                imgOutput = co.T.reshape(dimX,dimY)
#                saveFile(outputPath+'T'+str(TP)+'_'+WellPos+'_Epi-Ch'+repr(i)+'_0_0_0_0.tif', np.array (co.reshape(1,dimX,dimY)))
                                                                  
def saveFile(outputPath, data):
        # Save the file with outputPath as the full path and filename 
        tifffile.imsave(outputPath, data)

def getWellTiles(inputPath,Well,timePoint, index):
        FL  = os.listdir(inputPath)
        #print 'Checking for: '+Well+'_'+repr(index)+'_'
        WellTiles = [i for i in FL if ('_T'+str(timePoint)+'_' in i and Well+'_'+repr(index)+'_' in i)]
        return WellTiles
           
def Run(filePath, outputPath, channels, emptyWell, dataWells):
        # Parameter filePath is the full path to all the data.
        #
        # Parameter ouputPath is the full path to all where output should be delivered
        #
        # Parameter channels is an array of which well's contain reference data
        #      These will then be removed from the processing side of the calculation
        
        # Remove extraneous information from the filenames
        cleanFilenames(filePath)
        # Create the emptyWellStack
        print filePath
        emptyWellStack = createEmtpyWellStack(filePath, outputPath, emptyWell)
        # Create the reference data matrix
        refArray = createRefDataMatrix2(filePath,channels,emptyWellStack, outputPath)
        # Run the deconvolution iterating over the wells of interest
        for data in dataWells: # Well ID     
                for indexPosition in range(1,26): #well index positions
                        for t in range(0,10): #work through all the Time points           
                                wellTiles  = getWellTiles(filePath,data, t, indexPosition)
                                if len(wellTiles)>0:
                                        imgStack,expName = createStack(filePath, wellTiles)
                                        #divide by the average blank well for flat-field correction
                                        #imgStack = imgStack*(imgStack / emptyWellStack[:,:,:] > 1.0) # Save only the pixels above unity
                                        #imgStack /= emptyWellStack[:,:,:]
                                        wellPosition = data+'_'+repr(indexPosition)
                                        co = processViaLinearLeastSquares(imgStack, refArray, outputPath, t, wellPosition)
                                        # Reshape the coefficient array to create images of each channel image and save
                                        chan = co.shape[1]
                                        for i in range(chan):
                                                cOut = co[:,i]
                                                #todo 2048 should be pulled from the size of the image, not hardcoded
                                                dim = emptyWellStack.shape
                                                #imgOutput = cOut.T.reshape(dim[1],dim[2])
                                                saveFile(outputPath+expName+'_T'+str(t)+'_'+wellPosition+'_Epi-Ch'+repr(i)+'_0_0_0_0.tif', np.array (cOut.reshape(1,dim[1],dim[2]).T, dtype=np.uint16))
                                                                      

if __name__ == '__main__':
        args  = ap.ArgumentParser()
        args.add_argument('inputPath',help='Path to raw experiment folder')
        args.add_argument('outputPath',help='Path to the output location')
        args.add_argument('dataWells', help='wells to be deconvolved')
        args.add_argument('refWells', help='Wells to be used as reference channels (one per channel)')
        args.add_argument('emptyWell', help='Wells with no labels to act as background subtraction/intensity correction')
        
        #inputFile  = '/run/media/elliot/WorkData/SpectralDeconvolutionProjects/20170213SpectralTest-DFRCy5-YushICC/'
        #refChannels=['C2','C3','C4','C5','C7'] #Original YushICC Plate
        #dataWells = ['C6'] #Original YushICC Plate
        
        #inputFile = '/run/media/elliot/WorkData/20170507Robo5-HEK-Retransfection/raw/'
        #refChannels = ['B10','B12','B7'] #HEK Plate update
        #dataWells = ['E8','E9','E11','F8','F9','F11','G8','G9','G11'] # HEK Plate update
        
        #inputFile = '/run/media/elliot/WorkData/Elliot_SpectralDeconvolutionProjects/20170509Robo5-1-Retransfection40X06/raw/'
        #refChannels = ['B3','B6','B10','B11']#Rat Neuron Plate Robo5-1
        #dataWells = ['B5','B8','B9'] # Rat Neuron Plate Robo5-1

        
        inputArgs = args.parse_args()
        inputFile = inputArgs.inputPath
        outputFile =inputArgs.outputPath
        refChannels = inputArgs.refWells.split(",")
        dataWells = inputArgs.dataWells.split(",")
        emptyWell = inputArgs.emptyWell.split(",")
        
        if os.path.exists(outputFile) ==False:
                print outputFile +' does not exist. Making output folder.'
                os.mkdir(outputFile)
        
        # determine the wells needed for transfer function correction
        Run(inputFile, outputFile, refChannels,  emptyWell, dataWells)
