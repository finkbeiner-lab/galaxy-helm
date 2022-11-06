import os,sys
from subprocess import Popen,PIPE
import tempfile
import logging
import logging.handlers
import glob 


def create_filesets_alt(wells, time_points, excitation_wavelengths, all_files):
    """ Create a list of all image files rganized by Well, Time Point and Excitation Wavelength
    
    Input Arguments:
    wells -- list of wells 
    time_points -- list of time points (in 'T0_0' format)
    excitation_wavelengts -- list of excitation wavelenghts extracted from file names
    all_files -- list of all files in the data set
    """
    fileset = []
    for W in wells:
        for T in time_points:
            for E in excitation_wavelengths:
                tmp = [x for x in all_files \
                       if '_'+W+'_' in x and \
                       '_'+T+'_' in x and \
                       E in x]
                if len(tmp) > 0 :
                    fileset.append((W, E, tmp))
    return fileset

def create_filesets(wells, time_points, excitation_wavelengths, all_files):
    """ Create a list of all image files rganized by Well, Time Point and Excitation Wavelength
    
    Input Arguments:
    wells -- list of wells 
    time_points -- list of time points (in 'T0_0' format)
    excitation_wavelengts -- list of excitation wavelenghts extracted from file names
    all_files -- list of all files in the data set
    """
    fileset = []
    for W in wells:
        for T in time_points:
            for E in excitation_wavelengths:
                tmp = [x for x in all_files \
                       if '_'+W+'_' in x and \
                       '_'+T+'_' in x and \
                       '_'+E+'_' in x]
                if len(tmp) > 0 :
                    fileset.append((W, E, tmp))
    return fileset

def create_lock():
    try:
        os.mkdir('/home/ssamsi/lock/auto-montage')
    except:
        # hosed. unable to create lock file
        print 'failed to create lock file'
    return

def remove_lock():
    try:
        os.rmdir('/home/ssamsi/lock/auto-montage')
    except:
        print 'failed to remove lock'
    return

def check_lock():    
    return os.path.isdir('/home/ssamsi/lock/auto-montage')

def setup_logging(name, LOGFILE):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.handlers.TimedRotatingFileHandler(LOGFILE, when='midnight')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def gen_path(N):
    if N is None:
        return None
    N = int(N)
    path = range(1,N*N+1) 
    flipid = range(1, N*N+1, N)[0::2]
    for ii in flipid:
        path[ii-1:ii+N-1] = reversed(path[ii-1:ii+N-1])
    return(path)

def get_list_of_dirs(pathname):    
    all_dirs = []
    print 'this is returning wrong results'
    for name in glob.glob(pathname+os.sep+'*'):
        all_dirs.append(name)
    return all_dirs

# get a list of completed time points for the input directory
def get_completed_timepoints(pathname):
    T = []
    for name in glob.glob(pathname+os.sep+'T?_done'):
        name = name.split(os.sep)
        T.append(name[-1].split('_')[0])
    return (pathname, T)

def generate_metadata(filenames):
    # assumes that the filenames have been filtered so that only valid filenames are present here
    # also assumes that filenames contain full paths to the images
    exp_name = filenames[0].split(os.sep)[-2] # experiment name is the same as folder name. extract this
    
    #remove path information from the filenames
    filenames = [x.split(os.sep)[-1] for x in filenames]
    fields = [x.split('_') for x in filenames] # this can be messed up if experiment name has underscores
    # move the starting point based on the number of underscores in the experiment name 
    # only keep the entries that are sufficiently long -- this ensures that any extraneous files that were missed 
    # will be skipped
    #a = list(set([len(x) for x in fields]))
    # test data : /Volumes/RoboData/Robo4Images/AliciaRobo4TestPlate5_withConfocal_7_PFSTest_6

    idx = len(exp_name.split('_'))
    # filter out short file names 
    fields = [x for x in fields if len(x)>=(idx+5)]

    time = list( set( [x[idx+1] for x in fields] ) )
    hours = list( set( [x[idx+2] for x in fields] ) )
    wells = list( set( [x[idx+3] for x in fields] ) )
    time_points = list( set( [x[idx+1]+'_'+x[idx+2] for x in fields] ) ) 
    imageid = list( set( [int(x[idx+4]) for x in fields] ) )

    # temporary hack to take care of changes in filename metadata format on robo4
    if 'ELWD' in fields[0][idx+5]:
        excitation_wavelength = list( set( [x[idx+6] for x in fields] ) )        
    else:
        excitation_wavelength = list( set( [x[idx+5] for x in fields] ) )

    N = pow( len(imageid), 0.5)    
    if not N==round(N):
        N = None
    else:
        N = int(N)

    path = gen_path(N)
    return(wells, time_points, imageid, path, N, excitation_wavelength, exp_name)

def create_macro(pathname, rep_image, grid_size, overlap, montagedir):
    ''' 
        grid_size : int
    '''
    temp_dir = tempfile.mkdtemp()
    temp_file = tempfile.NamedTemporaryFile('wt', dir=temp_dir, delete=False)    

    tokens = rep_image.split('_')
    rep_filename = '_'.join(tokens[0:5])+'_{i}_'+'_'.join(tokens[6:len(tokens)])
    montage_file = '_'.join(tokens[0:5])+'_MONTAGED_'+'_'.join(tokens[6:len(tokens)])
    montage_file = montagedir + os.sep + montage_file

    macro_text = 'run("Grid/Collection stitching", "type=[Grid: snake by rows] order=[Left & Down] grid_size_x=' 
    macro_text +=  str(grid_size) + ' grid_size_y=' + str(grid_size) + ' tile_overlap=' + str(overlap) 
    macro_text += ' first_file_index_i=1 directory=' + pathname + ' file_names=' + rep_filename 
    macro_text += ' output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] '
    macro_text += 'regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 '
    macro_text += 'computation_parameters=[Save memory (but be slower)] image_output=[Write to disk] '
    macro_text += 'output_directory=' + temp_dir + '"); run("Quit");'
    
    fp = open(temp_file.name, 'wt')
    fp.writelines(macro_text)
    fp.writelines('\n')
    fp.close()
    
    return (macro_text, temp_dir, temp_file.name)

def create_ij_macro(pathname, rep_image, grid_size, overlap, montagedir):
    ''' This function returns the text of the macro to be run. The macro
    does not contain the Quit command
    '''
    temp_dir = tempfile.mkdtemp()
    
    tokens = rep_image.split('_')
    rep_filename = '_'.join(tokens[0:5])+'_{i}_'+'_'.join(tokens[6:len(tokens)])
    montage_file = '_'.join(tokens[0:5])+'_MONTAGED_'+'_'.join(tokens[6:len(tokens)])
    montage_file = montagedir + os.sep + montage_file

    macro_text1 = 'Grid/Collection stitching'
    macro_text = 'type=[Grid: snake by rows] order=[Left & Down] grid_size_x=' 
    macro_text +=  str(grid_size) + ' grid_size_y=' + str(grid_size) + ' tile_overlap=' + str(overlap) 
    macro_text += ' first_file_index_i=1 directory=' + pathname + ' file_names=' + rep_filename 
    macro_text += ' output_textfile_name=TileConfiguration.txt fusion_method=[Linear Blending] '
    macro_text += 'regression_threshold=0.30 max/avg_displacement_threshold=2.50 absolute_displacement_threshold=3.50 '
    macro_text += 'computation_parameters=[Save memory (but be slower)] image_output=[Write to disk] '
    macro_text += 'output_directory=' + temp_dir 
    return ((macro_text1, macro_text), temp_dir)

def run_macro(path_to_fiji, path_to_macro, macro_file):
    this_dir = os.getcwd()
    os.chdir(path_to_macro) # change to tmp dir
    list_before = os.listdir(path_to_macro)
    #print list_before
    # Using the --headless switch results in wrong answers due to a bug in Fiji
    #p = Popen([path_to_fiji, '--headless', '-macro', macro_file], stdin=PIPE, stdout=PIPE, stderr=PIPE) 
    p = Popen([path_to_fiji, '-macro', macro_file], stdin=PIPE, stdout=PIPE, stderr=PIPE) 
    p.communicate()
    list_after = os.listdir(path_to_macro)
    #print list_after
    new_file = [x for x in list_after if x not in list_before]
    os.chdir(this_dir) # go back to previous dir

    # Check if macro ran successfully. Then, move the file just created to the actual montage folder
    if len(new_file) == 1:
        # success assumed here. rename and move montage to actual location
        #print 'montage created'
        #print 'montage file : ', new_file[0]
        return (True, new_file[0])
    else:
        return (False, None)

def get_array_size(filename):
    ''' Assumption : We have access to the imaging template and image array is the same 
    for all wells
    '''
    f = open(filename, 'rt')
    lines = f.readlines()
    f.close()    
    found = False
    array_size = -1
    i = 0
    while not found:
        s = lines[i].split(',')
        i = i + 1
        if s[0]=='Well':            
            index = s.index('arraySize')
            found = True    
    if found:
        # just need to read one more line to get the array count
        l1 = lines[i]
        array_size = int(l1.split(',')[index])
    return array_size

def get_filenames(pathname):
    ''' Returns the full path to all tif images in specified directory and the imaging template
    associated with this experiment
    '''
    files = os.listdir(pathname)
    # get the imaging template
    imaging_template = [pathname+os.sep+x for x in files if x.endswith('.csv')]
    if len(imaging_template)>1:
        imaging_template = imaging_template[0]
    # filter out non-tif images and images with 'montage' in the filename
    files = [x for x in files if (x.endswith('.tif') or x.endswith('.tiff')) and \
             x.find('MONTAGE')<0 and \
             x.find('FIDUCIARY_STACK')<0 and \
             x.startswith('PID')]
    files = [pathname+os.sep+x for x in files]
    return files,imaging_template

