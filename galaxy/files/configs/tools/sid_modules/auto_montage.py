import os
import sys
import glob
import time
import shutil
import montage_helper as helper 

path_to_fiji = '/home/ssamsi/apps/Fiji.app/ImageJ-linux64'
logger = helper.setup_logging('main', '/home/ssamsi/logs/montage-log')

if helper.check_lock():
    logger.info('Cron cancelled. Earlier job still running')
    sys.exit(0)
else:
    helper.create_lock()
    logger.info('Created lock')

p = '/media/robodata/Robo4Images/'

all_dirs = helper.get_list_of_dirs(p)

# "T" will be tuple of direcotry paths and time points 
T = [helper.get_completed_timepoints(x) for x in all_dirs]

dest_dir = 'auto-montage'
montage_overlap = 0

for this_tuple in T:
    dirname = this_tuple[0]
    completed_timepoints = this_tuple[1]
    montage_folder = dirname + os.sep + dest_dir
    process_folder = True

    # create auto-montage directory if needed 
    if not os.path.isdir(montage_folder):
        try:
            os.makedirs(montage_folder)
            logger.info('Created directory ' + montage_folder)
        except OSError:
            if not os.path.isdir(montage_folder):
                # Unable to create folder for some reason
                # skip processing this folder
                logger.info('Unable to montage folder.Skipping ' + dirname)
                process_folder = False

    if process_folder:
        all_files,imaging_template = helper.get_filenames(dirname)
        for current_timepoint in completed_timepoints:
            logger.info(dirname + ', ' + current_timepoint)
            
            # check if the montage exists for this time point 
            montages = glob.glob(montage_folder + os.sep + '*')
            
            # figure out the wells to be processed for each time point
            files = [x for x in all_files if current_timepoint in x]
            wells, time_points, imageid, path, N, channels, excitation_wavelength = helper.generate_metadata(files)

            # for each well, do montage:
            for this_well in wells:                
                # there are multiple channels per well
                for this_channel in excitation_wavelength:
                
                    # get the files that belong to this well
                    this_well_files = [x for x in files if '_'+this_well+'_' in x and this_channel in x]
            
                    # we now have files belonging to "this_well" at "current_timepoint" with "this_channel"
                    nn = pow( len(this_well_files), 0.5)
                    
                    # only attempt to montage if we have all the files we need
                    if nn.is_integer and nn==N:
                        # get representative file name
                        rep_filename = this_well_files[0].split(os.sep)[-1]                        
                        tokens = this_well_files[0].split('_')
                        tokens[0] = tokens[0].split(os.sep)[-1] # retain just the file name, discarding the path
                        template = '_'.join(tokens[0:5])
                        template = template + '_{i}_'
                        template = template + '_'.join(tokens[6:len(tokens)])                    
                    
                        current_channel_montage_folder = montage_folder + os.sep + this_channel
                        logger.info('current montage folder: ' + current_channel_montage_folder)

                        # create if necessary
                        if not os.path.isdir(current_channel_montage_folder):
                            os.makedirs(current_channel_montage_folder)
                        temp = '_'.join(tokens[0:4]) + '_' + this_well + '_MONTAGE_' + this_channel + '.tif'
                        montage_file = current_channel_montage_folder  + os.sep + temp

                        # check if the montage file already exists 
                        if os.path.exists(montage_file) and os.path.isfile(montage_file):
                            logger.info('Montage exists. Skipping ' + montage_file)
                        else:
                            # now run the actual montaging step using ImageJ
                            # first create the macro
                            logger.info('montaging ' + rep_filename)
                            macro_text, temp_dir, macro_file = helper.create_macro(dirname, rep_filename, N, montage_overlap, current_channel_montage_folder)
                            # now run the macro
                            t0 = time.time()
                            res = helper.run_macro(path_to_fiji, temp_dir, macro_file)
                            logger.info('montage time : ' + str(time.time()-t0))
                            if not res[0]:
                                logger.info('Unable to create montage')
                            else:
                                # move montaged file to destination                                
                                shutil.move(temp_dir + os.sep + res[1], montage_file)
                                # delete the temporary files/folders
                                try:
                                    shutil.rmtree(temp_dir)
                                    logger.info('Removed ' + temp_dir)
                                except:
                                    logger.info('Exception while removing folder ' + temp_dir)
                    else:
                        # skip
                        logger.info('Did not find all image tiles ('+ str(len(this_well_files)) +' files found). Skipping ' + this_well + ' at ' + current_timepoint)


logger.info('process completed')
helper.remove_lock()
logger.info('Cron lock removed')
