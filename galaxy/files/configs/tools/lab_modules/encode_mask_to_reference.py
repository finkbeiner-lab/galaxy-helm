'''
Takes two sets of masks (reference mask will be encoded).
Evaluates overlap between each object in query and reference mask.
Generates encoding in query mask based on encoding in reference mask.

Whole pipeline steps:
Track reference masks (then overlay and extract as before)
Run encoding_to_reference_mask.py
Track from encoding for the reference mask (track cells extra)
Then extract and overlay as usual.

Known issue: relies on tracking

vs.

Whole pipeline steps:
Run encoding_to_reference_mask.py with pointers to both untracked masks
Track from encoding for the reference mask and for the query mask (track cells extra)
Then extract and overlay as usual for both steps
'''

import re, os, glob, sys
import numpy as np
import cv2, pickle, pprint
import subprocess, datetime
import string, shutil
import utils, argparse

def main(path_to_ref_masks, path_to_query_masks, verbose=False):
    '''
    Point of entry.
    '''
    print 'Given reference masks location:', path_to_query_masks
    print 'Given query masks location:', path_to_ref_masks
    # Collect two lists of masks
    # reference_mask_list = utils.make_filelist(path_to_ref_masks, '_CELLMASK.tif', verbose=False)
    # query_mask_list = utils.make_filelist(path_to_query_masks, '_CELLMASK.tif', verbose=False)
    reference_mask_list = utils.make_filelist_wells(path_to_ref_masks, '_CELLMASK.tif', verbose=False)
    query_mask_list = utils.make_filelist_wells(path_to_query_masks, '_CELLMASK.tif', verbose=False)

    global_cnts = 0
    # Pair masks by time point and well
    mask_pairs_dict = find_pairs(reference_mask_list, query_mask_list)
    for key, pair in mask_pairs_dict.items():
        ref_mask_pointer, query_mask_pointer = (pair[0], pair[1])
        if verbose:
            print os.path.basename(ref_mask_pointer)
            print os.path.basename(query_mask_pointer)

        query_mask = cv2.imread(query_mask_pointer, 0)
        reference_mask = cv2.imread(ref_mask_pointer, 0)
        img_dims = reference_mask.shape
        contours_ref, contours_query = get_ref_quer_cnts(
            reference_mask, query_mask)
        # print "Number of reference vs query contours:", len(contours_ref), len(contours_query)
        encoded_query_mask, encoded_ref_mask, global_cnts = encode_intersections(
            img_dims, contours_ref, contours_query, global_cnts)
        # Handle naming and write
        ref_mask_enc_name = create_new_name(
            ref_mask_pointer, path_to_ref_masks, suffix='_ENCODED')
        query_mask_enc_name = create_new_name(
            query_mask_pointer, path_to_query_masks, suffix='_ENCODED')
        cv2.imwrite(ref_mask_enc_name, encoded_ref_mask)
        cv2.imwrite(query_mask_enc_name, encoded_query_mask)
    print 'Total number of intersections found:', global_cnts

def create_new_name(img_pointer, write_path, suffix=''):
    '''Creates filename from original.'''
    orig_name = utils.extract_file_name(img_pointer)
    new_file_name = utils.make_file_name(write_path, orig_name+suffix)
    well_name = os.path.basename(img_pointer).split('_')[4]
    new_file_name = utils.reroute_imgpntr_to_wells(new_file_name, well_name)
    return new_file_name

# TODO: Deal with reading encoding
# for i in np.unique(img)[np.unique(img)!=0]:
#     cnt_values = img[np.nonzero(img==i)]
#     mask = np.zeros(img.shape[:2], np.uint16)
#     mask[np.nonzero(img==i)]=2**16
#     contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     if len(contours)>1:
#         print "Found", len(contours), "objects for encoding", i
#         contours = sorted(contours, key = cv2.contourArea, reverse = True)
#         for cnt in contours:
#             center, radius = cv2.minEnclosingCircle(cnt)
#             print center
#     cnt_ind = cnt_values.max()

# Getting the pairs of masks
def find_pairs(reference_mask_list, query_mask_list, verbose=False):
    '''
    Takes two lists of images and returns a dictionary of matching pairs.
    '''
    mask_pairs = {}
    img_tokens_dict = {'well':4,'channel':6,'pid':0,'time':2}

    for ref_mask_pointer in reference_mask_list:
        ref_img_tokens = os.path.basename(ref_mask_pointer).split('_')
        for query_mask_pointer in query_mask_list:
            query_img_tokens = os.path.basename(query_mask_pointer).split('_')

            # Evaluate sameness and keep pairs
            wells_same = query_img_tokens[img_tokens_dict['well']] == ref_img_tokens[img_tokens_dict['well']]
            times_same = query_img_tokens[img_tokens_dict['time']] == ref_img_tokens[img_tokens_dict['time']]
            pid_same = query_img_tokens[img_tokens_dict['pid']] == ref_img_tokens[img_tokens_dict['pid']]
            # ch_same = query_img_tokens[img_tokens_dict['channel']] == ref_img_tokens[img_tokens_dict['channel']]
            if wells_same and times_same and pid_same:
                mask_pairs[str(
                    ref_img_tokens[img_tokens_dict['pid']])+str(
                    ref_img_tokens[img_tokens_dict['time']])+str(
                    ref_img_tokens[img_tokens_dict['well']])] = [ref_mask_pointer, query_mask_pointer]
    if verbose:
        print len(mask_pairs)
        print 'Mask pairs'
        pprint.pprint(mask_pairs)

    return mask_pairs

# Getting the contours
def get_ref_quer_cnts(reference_mask, query_mask, verbose=False):
    '''
    Take paths for two binary masks and return their contours.
    '''
    if verbose:
        print 'Reference mask', reference_mask.shape
        print 'Query mask', query_mask.shape

    # Collect contours from query mask
    contours_query, hierarchy = cv2.findContours(
        query_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # Collect contours from reference mask
    contours_ref, hierarchy = cv2.findContours(
        reference_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if verbose:
        print 'Found reference contours:', len(contours_ref),
        print 'Found query contours:', len(contours_query)

    return contours_ref, contours_query

# Comparing the masks
def encode_intersections(img_dims, contours_ref, contours_query, global_cnts, verbose=True):
    '''
    Take contours for a reference mask and a query mask.
    Returns a query mask encoded by reference mask's id number.
    or
    Returns a query and a reference mask with matching encodings.
    '''
    # Initiate image and dictionary to collect meta data
    encoded_query_mask = np.zeros(img_dims, np.uint16)
    encoded_ref_mask = np.zeros(img_dims, np.uint16)

    count_ints = 0
    # Test each junction contour against every point in neuron contour
    for ref_cnt_id, cnt_ref in enumerate(contours_ref):
        cv2.drawContours(encoded_ref_mask, [cnt_ref], 0, ref_cnt_id+1, -1)
        for q_cnt_id, cnt_q in enumerate(contours_query):

            # Evaluate overlap
            (x,y), radius = cv2.minEnclosingCircle(cnt_q)
            if cv2.pointPolygonTest(cnt_ref, (x,y), False) == 1:
                if verbose:
                    print q_cnt_id, 'is INSIDE cnt_ref:', ref_cnt_id+1
                cv2.drawContours(encoded_query_mask, [cnt_q], 0, ref_cnt_id+1, -1)
                count_ints += 1
            else:
                # if verbose:
                    # print '--', q_cnt_id, 'is OUTSIDE ref_cnt', ref_cnt_id
                continue

    global_cnts = global_cnts+count_ints
    if verbose:
        print 'Number of reference contours:', len(contours_ref)
        print 'Number of query contours:', len(contours_query)
        print 'Number of intersections:', count_ints
    return encoded_query_mask, encoded_ref_mask, global_cnts

if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Encode query mask to reference mask.")
    parser.add_argument("input_dict",
        help="Load input variable dictionary")
    parser.add_argument("path_to_ref_masks",
        help="Folder path to reference masks.")
    parser.add_argument("path_to_query_masks",
        help="Folder path to query masks.")
    parser.add_argument("output_dict",
        help="Write variable dictionary.")
    args = parser.parse_args()

    # ----Load path dict-------------------------
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))

    # ----Initialize parameters------------------
    path_to_ref_masks = args.path_to_ref_masks
    path_to_query_masks = args.path_to_query_masks

    outfile = args.output_dict

    # ----Confirm given folders exist--
    assert os.path.exists(path_to_ref_masks), 'Confirm the given path for reference masks exists.'
    assert os.path.exists(path_to_query_masks), 'Confirm the given path for query masks exists.'

    # ----Run segmentation-----------------------
    start_time = datetime.datetime.utcnow()

    main(path_to_ref_masks, path_to_query_masks)

    end_time = datetime.datetime.utcnow()
    print 'Reference pairing run time:', end_time-start_time
    # ----Output for user and save dict----------
    print 'Selected masks were encoded to reference masks.'
    print 'Encoded output was written to the reference and query mask paths, respectively:'
    print path_to_query_masks
    print path_to_ref_masks
    print 'Use encoded masks with Track Cells Extra module to "Track from encoded mask"'

    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    outfile = shutil.move('var_dict.p', outfile)
    utils.save_user_args_to_csv(args, path_to_query_masks, 'encode_masks_to_reference')


