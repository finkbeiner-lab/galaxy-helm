'''
Changes values in var_dict.
'''
import utils, sys, shutil, argparse, pickle

if __name__ == '__main__':

   #----Argument parsing--------
    parser = argparse.ArgumentParser(description="Process iPSC data.")

    parser.add_argument("var_dict",
        help="Dictionary of variables from previous step.")

    parser.add_argument("outfile",
        help="Name of output dictionary.")

    args = parser.parse_args()

    # ----Load parameters------------------------
    infile = args.var_dict
    var_dict = pickle.load(open(infile, 'rb'))

    print 'Initial:'
    # print var_dict['MorphologyChannel']
    # print var_dict['InputPath']
    # print var_dict['outfileutputPath']


    # channels = var_dict['Channels']
    # var_dict['MorphologyChannel'] = utils.get_ref_channel('Green', channels)
    var_dict['InputPath'] = '/media/robodata/Robo3Images/MariyaLiveDeadPlate4ICC/Processed/CellMasks/'
    var_dict['OutputPath'] = '/media/robodata/Robo3Images/MariyaLiveDeadPlate4ICC/Processed/CellMasks/'

    print 'Modified:'
    # print var_dict['MorphologyChannel']
    print var_dict['InputPath']
    print var_dict['OutputPath']

    outfile = args.outfile
    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    outfile = shutil.move('var_dict.p', outfile)
