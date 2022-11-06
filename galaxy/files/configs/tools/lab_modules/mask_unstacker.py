import utils, sys, os, argparse
import pickle, pprint, shutil, datetime

def take_stack_make_singles(var_dict, input_path, output_path, unstack_step, verbose=False):
    # This looping should iterate over z
    for well in var_dict['Wells']:
        for channel in var_dict['Channels']:
            if unstack_step == 'mask':
                selector = '*_'+well+'_*'+'MONTAGE_MASK.tif'
            if unstack_step == 'aligned':
                selector = '*_'+well+'_*'+channel+'.tif'
            stack_name = utils.make_filelist(input_path, selector)
            if len(stack_name)<1:
                continue
            assert len(stack_name) == 1, 'The file is not singular.'
            stack_name = stack_name[0]
            name_list = os.path.basename(stack_name).split('_')
            if unstack_step == 'mask':
                output_file_name = '_'.join(['_'.join(name_list[0:2]), 'T%d', '0', '_'.join(name_list[4:6]), name_list[6].split('.')[0], 'ENCODED.tif'])
            if unstack_step == 'aligned':
                output_file_name = '_'.join(['_'.join(name_list[0:2]), 'T%d', '0', '_'.join(name_list[4:7])])
            output_file_name = os.path.join(output_path, output_file_name)
            if verbose:
                print "Stack treated is:", stack_name
                print "File will be written as:", output_file_name
            utils.split_stack_magically(
                os.path.join(input_path, selector), output_file_name)


if __name__ == "__main__":

    '''Point of entry.'''

    # Argument parsing
    parser = argparse.ArgumentParser(description="Unstack images.")
    parser.add_argument("input_path", 
        help="Folder path to encoded masks.")
    parser.add_argument("output_path",
        help="Folder path to ouput results.")
    parser.add_argument("step_to_unstack",
        help="Chose 'aligned' or 'mask' images.")
    parser.add_argument("outfile",
        help="Name of output dictionary.")   
    args = parser.parse_args()

    # Set up I/O parameters
    input_path = args.input_path
    unstack_step = args.step_to_unstack
    output_path = args.output_path
    outfile = args.outfile

    all_files = utils.get_all_files(input_path)
    var_dict = {
        'Wells': utils.get_wells(all_files),
        'Channels': utils.get_channels(all_files, 3)
        }

    # Confirm given folders exist
    if not os.path.exists(input_path):
        print 'Confirm the given path for data exists.'
    assert os.path.exists(input_path), 'Confirm the given path for data exists.'
    if not os.path.exists(output_path):
        print 'Confirm the given path for results exists.'
    assert os.path.exists(output_path), 'Confirm the given path for results exists.'

    # ----Run unstacking---------------------------
    start_time = datetime.datetime.utcnow()

    take_stack_make_singles(var_dict, input_path, output_path, unstack_step)

    end_time = datetime.datetime.utcnow()
    print 'Tracking run time:', end_time-start_time

    # ----Output for user and save dict----------
    print 'Stacks were unstacked.'
    print 'Output from this step: individual encoded masks written to:'
    print output_path

    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    # outfile = os.rename('var_dict.p', outfile)
    outfile = shutil.move('var_dict.p', outfile)
