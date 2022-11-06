import os
import argparse
import pickle 
import datetime
import json


def main():
    start_time = datetime.datetime.utcnow()
    # Argument parser
    parser = argparse.ArgumentParser(
        description="Save metadata for Galaxy history.")
    parser.add_argument("input_dict",
        help="Load input variable dictionary")
    parser.add_argument("out_text",
        help="Write log")
    args = parser.parse_args()

    # Load dict
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))
    out_text = args.out_text

    # Get json output path
    output_path = var_dict['GalaxyOutputPath']
    experiment_name = var_dict['ExperimentName']
    json_name = experiment_name + 'GalaxyMeta.json'
    json_output_path = os.path.join(output_path, json_name)

    # The value of CalculatedShift is a dict of tuple experiment-well of dict timepoint.
    # To avoid json.dump() "TypeError: keys must be a string"
    # This will try to convert any key that is not a string into a string. 
    # Any key that could not be converted into a string will be deleted.
    for key in var_dict.keys():
        if type(var_dict[key]) is dict:
            for keyin in var_dict[key].keys():
                if type(keyin) is not str:
                    var_dict[key][str(keyin)] = var_dict[key][keyin]
                del var_dict[key][keyin]    

    # Convert dict to json and save to storage
    with open(json_output_path, 'w') as outfile:
        json.dump(var_dict, outfile)

    end_time = datetime.datetime.utcnow()
    print 'Total run time: %s' %(end_time-start_time)
    print 'The Galaxy history metadata will be saved to: \n%s' % json_output_path

if __name__ == '__main__':
    main()

    