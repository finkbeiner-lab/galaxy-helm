import utils, sys, os, argparse
import pickle, pprint, shutil, datetime

def zproject(project_files, input_path, projection_type, output_path, selector, verbose=False):
    '''
    Generate file strings for ImageMagick and run ImageMagick command to project. 
    '''
    name_list = os.path.basename(project_files[0]).split('_')
    proj_str = projection_type.replace('_','').upper()[0:3]
    
    tif_part = name_list[len(name_list)-1].split('.')[0]
    name_list = name_list[0:len(name_list)-1]
    name_list.append(tif_part)
    output_file_name = '_'.join(['_'.join(name_list[0:8]), '1', '_'.join(name_list[9:len(name_list)]), 'Z'+proj_str+'.tif'])
    output_file_name = os.path.join(output_path, output_file_name)
    
    if verbose:
        print os.path.join(input_path, '*'+selector)
        pprint.pprint([os.path.basename(proj_file) for proj_file in project_files])
        print 'Output file name', output_file_name

    utils.collapse_stack_magically(
        output_file_name, 
        os.path.join(input_path, '*'+selector), 
        collapse_type=projection_type)

def project_in_z(zvar_dict, input_path, output_path, projection_type, verbose=False):
    '''
    Collect all relevant files for each projection and run the projection.
    '''
    # This looping should iterate over z
    for well in zvar_dict['Wells']:
        for time in zvar_dict['TimePoints']:
            for channel in zvar_dict['UserChannels']:
                print 'Current well, time, channel:', well, time, channel  
                all_project_files = utils.get_selected_files(
                    input_path, well, time, channel)
                
                if len(all_project_files)<1:
                    continue
                
                elif (len(all_project_files)>len(zvar_dict['Depths'])) and len(zvar_dict['Panels'])>1:
                    for panel in zvar_dict['Panels']:
                        print 'Current panel:', panel
                        panel_project_files = utils.get_frame_files(
                            all_project_files, panel, 5) #panel is position 5
                        pprint.pprint([os.path.basename(prof_file) for prof_file in panel_project_files]) 
                        
                        if len(all_project_files)<1:
                            continue
                        
                        elif len(panel_project_files)==len(zvar_dict['Depths']):
                            selector = utils.make_selector_from_tokens(
                                robonum=0, well=well, time=time, channel=channel, panel=panel)
                            
                            zproject(panel_project_files, input_path, projection_type, output_path, selector)

                        elif len(panel_project_files)>len(zvar_dict['Depths']):
                            print 'Found', len(panel_project_files), 'files. Expected', len(zvar_dict['Depths']), 'files.'
                        
                        elif (len(panel_project_files)%len(zvar_dict['Depths'])) != 0:
                            print 'Did not find all files.'
                        
                        else:
                            print 'Unusual number of images found. Not projecting.'
                
                elif len(all_project_files)==len(zvar_dict['Depths']):
                    pprint.pprint([os.path.basename(prof_file) for prof_file in all_project_files])
                    selector = utils.make_selector_from_tokens(
                        robonum=0, well=well, time=time, channel=channel)
                    zproject(all_project_files, input_path, projection_type, output_path, selector)

                elif (len(all_project_files)%len(zvar_dict['Depths'])) != 0:
                    print 'Did not find all files.'
                
                else:
                    print 'Unusual number of images found. Not projecting.'
                


if __name__ == "__main__":

    '''Point of entry.'''

    # Argument parsing
    parser = argparse.ArgumentParser(description="Project z images.")
    parser.add_argument("input_path", 
        help="Folder path to input data.")
    parser.add_argument("output_path",
        help="Folder path to ouput results.")
    parser.add_argument("projection_type",
        help="Specify type of projection: max or mean.")
    parser.add_argument("channel_list",
        help="List channels to project, separated by commas")
    parser.add_argument("outfile",
        help="Name of output dictionary.")   
    args = parser.parse_args()

    # Set up I/O parameters
    #TODO: Include in args if this will plug in to workflow later
    var_dict = {} 
    input_path = args.input_path
    output_path = args.output_path
    projection_type = args.projection_type
    channel_list = args.channel_list
    outfile = args.outfile

    all_files = utils.get_all_files(input_path)
    assert len(all_files)>0, 'Input path has no files. Stopping projection module...'
    zvar_dict = {
        'Wells': utils.get_wells(all_files),
        'TimePoints': utils.get_timepoints(all_files),
        'Depths': utils.get_depths(all_files, 0),
        'Channels': utils.get_channels(all_files, 0),
        'Panels': utils.get_well_panel(all_files)
        }
    channel_list = channel_list.replace(" ","")
    user_channels = list(set(channel_list.split(',')))

    zvar_dict['UserChannels'] = []
    for user_channel in user_channels:
        user_channel = [ch for ch in zvar_dict['Channels'] if user_channel in ch][0]
        zvar_dict['UserChannels'].append(user_channel)

    print 'Wells:', zvar_dict['Wells']
    print 'Time points:', zvar_dict['TimePoints']
    print 'Depths:', zvar_dict['Depths']
    print 'Panels:', zvar_dict['Panels']
    print 'Available channels:', zvar_dict['Channels']
    print 'Selected channels', zvar_dict['UserChannels']

    # Confirm given folders exist
    assert os.path.exists(input_path), 'Confirm the given path to input data exists.'
    assert os.path.exists(output_path), 'Confirm the given path for results output exists.'

    # ----Run projections------------------------
    start_time = datetime.datetime.utcnow()

    project_in_z(zvar_dict, input_path, output_path, projection_type)

    end_time = datetime.datetime.utcnow()
    print 'Tracking run time:', end_time-start_time

    # ----Output for user and save dict----------
    print 'Stacks were unstacked.'
    print 'Output from this step is an encoded mask written to:'
    print output_path

    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    # outfile = os.rename('var_dict.p', outfile)
    outfile = shutil.move('var_dict.p', outfile)