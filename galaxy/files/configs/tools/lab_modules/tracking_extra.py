import cv2, utils, sys, os, argparse
import numpy as np
import scipy.stats as stat
import pickle, collections
import pprint, datetime, shutil
from segmentation import filter_contours, find_cells
from scipy import ndimage as ndi
from skimage.filters import gabor_kernel, scharr, laplace
from skimage.feature import hessian_matrix
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy import signal

# Initiate STAR detector
star = cv2.FeatureDetector_create("STAR")
# Initiate BRIEF extractor
brief = cv2.DescriptorExtractor_create("BRIEF")

class Cell(object):
    '''
    A class that makes cells from contours.
    '''
    def __init__(self, cnt, ch_images=None):
        self.cnt = cnt
        self.all_ch_int_stats = None

        if ch_images:
            self.collect_all_ch_intensities(ch_images)

    def __repr__(self):
        return "Cell instance (%s center, %s area)" % (str(self.get_circle()[0]), str(cv2.contourArea(self.cnt)))

    # ----Contours-------------------------------
    def get_circle(self):
        '''Returns centroid of contour.'''
        center, radius = cv2.minEnclosingCircle(self.cnt)
        center, (MA, ma), angle = cv2.fitEllipse(self.cnt)

        return center, radius

    def intersects_contour(self, centroid, verbose=False):
        '''
        Determines if a given centroid overlaps with cell's contour.
        '''
        if verbose:
            print "\t\tContour", self.cnt, "Centroid", centroid

        overlap_value = cv2.pointPolygonTest(
            self.cnt, centroid, False)
        return overlap_value

    def evaluate_distance(self, shift):
        '''
        Make certain x's and y's are being compared with x's and y's.
        '''
        center, _ = self.get_circle()

        distance = (
            (center[0] - shift[0])**2 +
            (center[1] - shift[1])**2)**0.5

        return distance

    def evaluate_overlap(self, circle2):
        '''
        Calculates distance between centroids.
        Evaluates if distance is within 80'%' of both radii.

        @Usage
        circle2 is passed in as (center, radius) tuple
        center is an (x,y) tuple
        '''
        center1, radius1 = self.get_circle()
        center2, radius2 = circle2

        distance = (
            (center1[0] - center2[0])**2 +
            (center1[1] - center2[1])**2)**0.5

        return distance < (radius1 + radius2)*0.8

    def calculate_cnt_parameters(self):
        '''Extracts all cell-relevant parameters.'''
        cell_params = {}
        area_cnt = cv2.contourArea(self.cnt)
        cell_params['BlobArea'] = area_cnt

        xcnt, ycnt, wcnt, hcnt = cv2.boundingRect(self.cnt)
        rect_area = wcnt*hcnt
        cell_params['Extent'] = float(area_cnt)/rect_area
        cell_params['AspectRatio'] = float(wcnt)/hcnt

        perimeter = cv2.arcLength(self.cnt, True)
        cell_params['BlobPerimeter'] = perimeter

        center, radius = cv2.minEnclosingCircle(self.cnt)
        cell_params['Radius'] = radius

        (x, y), (MA, ma), angle = cv2.fitEllipse(self.cnt)
        cell_params['BlobCentroidX'] = x
        cell_params['BlobCentroidY'] = y
        ecc = np.sqrt(1-((MA)**2/(ma)**2))
        cell_params['BlobCircularity'] = ecc
        cell_params['Angle'] = angle
        cell_params['MajorAxis'] = MA
        cell_params['MinorAxis'] = ma

        hull = cv2.convexHull(self.cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area_cnt)/hull_area
        cell_params['Spread'] = solidity

        convexity = cv2.isContourConvex(self.cnt)
        cell_params['Convexity'] = convexity

        return cell_params

    def get_cell_patch(self, img, dim=False):
        '''
        Returns the smallest (or dim x dim) rectangular region around contour from given image.
        '''
        rows, cols = img.shape
        x,y,w,h = cv2.boundingRect(self.cnt)
        if not dim:
            cnt_rect = img[y:y+h,x:x+w]
        else:
            if y+dim > rows:
                dim = rows - y
            elif x+dim > cols:
                dim = cols - x
            print 'Resetting dimensions to dim...', dim
            cnt_rect = img[y:y+dim,x:x+dim]

        return cnt_rect

    def get_cell_square(self, img, dim=128, show=False):
        '''
        Returns either a dim x dim square centered at cell centroid
        or a patch of zeros dim x dim if a cell intersects an edge.
        '''
        rows, cols = img.shape
        (x,y), _ = self.get_circle()
        x = int(x)
        y = int(y)
        if y+(dim/2)>rows or y-(dim/2)<0 or x+(dim/2)>cols or x-(dim/2)<0:
            print 'Exluding cell:', self
            cnt_patch = 120 + np.zeros((dim, dim), np.uint16)
        else:
            cnt_patch = img[y-(dim/2):y+(dim/2),x-(dim/2):x+(dim/2)]
        if show:
            plt.subplot(121),plt.imshow(cnt_patch,'gray'),plt.title('cnt_patch')
            plt.subplot(122),plt.imshow(cnt_patch.flatten().reshape((
                dim/2),(dim/2)),'gray'),plt.title('re_shaped')
            plt.show()
        return cnt_patch.flatten()

    # ----Intensities----------------------------
    def find_cnt_int_dist(self, img):
        '''
        Finds pixels associated with contour.
        Returns intensity parameters.
        This is one of the required parameters to instnatiate a Cell_obj.
        '''

        # These are the edge intensities
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [self.cnt], 0, 256, 1)
        cnt_ints_edge = img[np.nonzero(mask)]

        # These are the internal intensities
        mask = np.zeros(img.shape[:2], np.uint8)
        cv2.drawContours(mask, [self.cnt], 0, 256, -1)
        cnt_ints = img[np.nonzero(mask)]

        cell_int_stats = {}

        # SIFT-like descriptors
        # cell_int_stats['Descriptors'] = self.find_cnt_star_brief_descriptors(img, mask)

        # Intensity params for cell
        cell_int_stats['PixelIntensityTotal'] = cnt_ints.sum()
        cell_int_stats['PixelIntensityMinimum'] = cnt_ints.min()
        cell_int_stats['PixelIntensityMaximum'] = cnt_ints.max()
        cell_int_stats['PixelIntensityMean'] = cnt_ints.mean()
        cell_int_stats['PixelIntensityStdDev'] = cnt_ints.std()
        cell_int_stats['PixelIntensityVariance'] = cnt_ints.var()
        cell_int_stats['PixelIntensityMeanSD'] = cnt_ints.mean()/cnt_ints.std()
        cell_int_stats['PixelIntensityRangeSD'] = (cnt_ints.max()-cnt_ints.min())/cnt_ints.std()

        (q1, q5, q10, q25, q50, q75, q90, q95, q99) = np.percentile(
            cnt_ints, [1, 5, 10, 25, 50, 75, 90, 95, 99])
        cell_int_stats['PixelIntensity1Percentile'] = q1
        cell_int_stats['PixelIntensity5Percentile'] = q5
        cell_int_stats['PixelIntensity10Percentile'] = q10
        cell_int_stats['PixelIntensity25Percentile'] = q25
        cell_int_stats['PixelIntensity50Percentile'] = q50
        cell_int_stats['PixelIntensity75Percentile'] = q75
        cell_int_stats['PixelIntensity90Percentile'] = q90
        cell_int_stats['PixelIntensity95Percentile'] = q95
        cell_int_stats['PixelIntensity99Percentile'] = q99
        cell_int_stats['PixelIntensityInterquartileRange'] = q75-q25

        cell_int_stats['PixelIntensitySkewness'] = stat.skew(cnt_ints)
        cell_int_stats['PixelIntensityKurtosis'] = stat.kurtosis(cnt_ints)

        # Intensities for edges
        cell_int_stats['EdgeIntensityTotal'] = cnt_ints_edge.sum()
        cell_int_stats['EdgeIntensityMaximum'] = cnt_ints_edge.max()
        cell_int_stats['EdgeIntensityMinimum'] = cnt_ints_edge.min()
        cell_int_stats['EdgeIntensityMean'] = cnt_ints_edge.mean()
        cell_int_stats['EdgeIntensityVariance'] = cnt_ints_edge.var()
        cell_int_stats['EdgeIntensityStdDev'] = cnt_ints_edge.std()
        cell_int_stats['EdgeIntensityMeanSD'] = cnt_ints_edge.mean()/cnt_ints_edge.std()
        cell_int_stats['EdgeIntensityRangeSD'] = (cnt_ints_edge.max()-cnt_ints_edge.min())/cnt_ints_edge.std()
        cell_int_stats['EdgeIntensitySkewness'] = stat.skew(cnt_ints_edge)
        cell_int_stats['EdgeIntensityKurtosis'] = stat.kurtosis(cnt_ints_edge)

        # These are the line intensities
        (x, y), (MA, ma), angle = cv2.fitEllipse(self.cnt)
        # print 'x', x, 'y', y
        cell_center_int = img[int(y), int(x)]
        cell_int_stats['CellCenterIntensity'] = cell_center_int
        # Slope along major axis/minor axis
        mask = np.zeros(img.shape[:2], np.uint8)
        [vx, vy, x, y] = cv2.fitLine(self.cnt, cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
        cv2.line(mask,(int(x),int(y)),(int(x+(2*MA/3)*vx), int(y+(2*MA/3)*vy)), 250, 1)
        line_ints = img[np.nonzero(mask)]
        cv2.line(mask,(int(x),int(y)),(int(x-(2*MA/3)*vx), int(y-(2*MA/3)*vy)), 250, 1)
        full_MA_ints = img[np.nonzero(mask)]

        # Spatial profile
        cell_int_stats['LineIntensityTotal'] = line_ints.sum()
        cell_int_stats['LineIntensityMaximum'] = line_ints.max()
        cell_int_stats['LineIntensityMinimum'] = line_ints.min()
        cell_int_stats['LineIntensityMean'] = line_ints.mean()
        cell_int_stats['LineIntensityVariance'] = line_ints.var()
        cell_int_stats['LineIntensityStdDev'] = line_ints.std()
        cell_int_stats['LineIntensityMeanSD'] = line_ints.mean()/line_ints.std()
        cell_int_stats['LineIntensityRangeSD'] = (line_ints.max()-line_ints.min())/line_ints.std()
        cell_int_stats['LineIntensitySkewness'] = stat.skew(line_ints)
        cell_int_stats['LineIntensityKurtosis'] = stat.kurtosis(line_ints)

        # Fit a line to the intensities (note "relevance" for non-reference channels)
        xline = np.arange(0, len(line_ints))
        slope, intercept, r_value, p_value, std_err = stat.linregress(xline,line_ints)
        cell_int_stats['LineIntensitySlope'] = slope
        cell_int_stats['LineIntensityRsquared'] = r_value
        # Fit a gaussian to the the line intensities
        # mu, sigma = fitgauss(xline, full_MA_ints)
        # cell_int_stats['LineIntensitySpread'] = sigma

        # Intensity @percent distance from center actual
        cell_int_stats['LineIntensity25PercentRaw'] = line_ints[int(len(line_ints)*0.25)]#/cell_center_int #line_ints[0]
        cell_int_stats['LineIntensity50PercentRaw'] = line_ints[int(len(line_ints)*0.50)]#/cell_center_int
        cell_int_stats['LineIntensity75PercentRaw'] = line_ints[int(len(line_ints)*0.75)]#/cell_center_int
        cell_int_stats['LineIntensity99PercentRaw'] = line_ints[int(len(line_ints)*0.99)]#/cell_center_int

        # Intensity @percent distance from center cumulative
        cell_int_stats['LineIntensity25PercentCum'] = line_ints[0:int(len(line_ints)*0.25)].sum()
        cell_int_stats['LineIntensity50PercentCum'] = line_ints[0:int(len(line_ints)*0.50)].sum()
        cell_int_stats['LineIntensity75PercentCum'] = line_ints[0:int(len(line_ints)*0.50)].sum()
        cell_int_stats['LineIntensity99PercentCum'] = line_ints[0:int(len(line_ints)*0.50)].sum()

        # These are the coordinates of their extrema values
        # cnt_ints_edge_max = np.where(img[np.nonzero(mask)] == cnt_ints_edge.max())
        # cnt_ints_edge_max = np.ravel(cnt_ints_edge_max)[0]
        # cnt_ints_edge_min = np.where(img[np.nonzero(mask)] == cnt_ints_edge.min())
        # cnt_ints_edge_min = np.ravel(cnt_ints_edge_min)[0]

        # Frequencies and kernals
        # http://scikit-image.org/docs/dev/auto_examples/plot_blob.html
        # http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_gabor.html?highlight=gabor
        # Generate kernels: 0-10 Gabor, 11 Hesian, 12 Laplacian
        # Hessian, Gabor, Difference of gaussians (blob_dog), laplacian (blob_log)

        # This is the smallest rectangular region around contour
        cnt_rect = self.get_cell_patch(img)

        # Hu's transformation invariant moments
        hu_moments = cv2.HuMoments(cv2.moments(cnt_rect)).flatten()
        cell_int_stats['HuOne'] = hu_moments[0]
        cell_int_stats['HuTwo'] = hu_moments[1]
        cell_int_stats['HuThree'] = hu_moments[2]
        cell_int_stats['HuFour'] = hu_moments[3]
        cell_int_stats['HuFive'] = hu_moments[4]
        cell_int_stats['HuSix'] = hu_moments[5]
        cell_int_stats['HuSeven'] = hu_moments[6]

        # Gabor - textures
        filtered = ndi.convolve(cnt_rect, np.real(gabor_kernel(0.6)), mode='wrap')
        cell_int_stats['GaborMean'] = filtered.mean()
        cell_int_stats['GaborVariance'] = filtered.var()

        # Scharr - edges
        filtered = scharr(cnt_rect)
        cell_int_stats['ScharrMean'] = filtered.mean()
        cell_int_stats['ScharrVariance'] = filtered.var()

        # Hessian - vesselness
        Hxx, Hxy, Hyy = hessian_matrix(cnt_rect, sigma=0.1)
        cell_int_stats['HesianXXMean'] = Hxx.mean()
        cell_int_stats['HesianXXVariance'] = Hxx.var()
        cell_int_stats['HesianXYMean'] = Hxy.mean()
        cell_int_stats['HesianXYVariance'] = Hxy.var()
        cell_int_stats['HesianYYMean'] = Hyy.mean()
        cell_int_stats['HesianYYVariance'] = Hyy.var()

        # Blobs - returns (y,x, sigma) for each blob.
        # DOH should be better than DOG if objects are above 3px
        # print "DOH"
        # det_of_hess = blob_doh(cnt_rect)
        # print det_of_hess
        cell_int_stats['DohMean'] = 0#det_of_hess[:, 2].mean() * np.sqrt(2)
        cell_int_stats['DohVariance'] = 0#det_of_hess[:, 2].var() * np.sqrt(2)

        # Laplacian
        laplacian = np.array([[0, 1, 0], [1,-4, 1], [0, 1, 0]])
        filtered = ndi.convolve(cnt_rect, laplacian, mode='wrap')
        # # or
        # filtered = laplace(cnt_rect, mask=mask)
        cell_int_stats['LaplacianMean'] = filtered.mean()
        cell_int_stats['LaplacianVariance'] = filtered.var()

        # Entropy
        from skimage.morphology import disk
        filtered = entropy(cnt_rect, disk(5))
        cell_int_stats['EntropyMean'] = filtered.mean()
        cell_int_stats['EntropyVariance'] = filtered.var()

        # Cell patch pixels
        cell_int_stats["CellPatchPixels"] = self.get_cell_square(img, dim=128)

        return cell_int_stats

    def find_cnt_star_brief_descriptors(self, img, mask):
        '''
        Per channel (img), finds descriptors for key points in contour.
        Returns linearized descriptor sequence for all key points found.
        '''
        global star
        global brief
        # mask = np.zeros(img.shape[:2], np.uint8)
        # cv2.drawContours(mask, [self.cnt], 0, 256, -1)
        img = img.astype(np.uint8)
        kp = star.detect(img, mask)
        kp, des = brief.compute(img, kp)
        if len(kp) > 0:
            descriptors = des.ravel().tolist()
        else:
            descriptors = [0]

        return descriptors

    def find_correlations(img1, img2, cell_int_stats):
        '''
        Calculate correlation between any two cells (probably ifft).
        Return a vector of correlations (indices of correlation maximums as a feature).
        '''
        corr = signal.correlate2d(img1, img2)
        y, x = np.unravel_index(np.argmax(corr), corr.shape)
        cell_int_stats['CorrMaxX'] = x
        cell_int_stats['CorrMaxY'] = y
        return cell_int_stats

    def collect_all_ch_intensities(self, ch_images, verbose=False):
        '''
        Takes list of already opened images (per timepoint, well, and frame).
        Calculates intensity statistics based on morphology contour.
        '''

        if verbose:
            print 'Get collect_all_ch_intensities:'
            pprint.pprint(ch_images)

        self.all_ch_int_stats = {}
        for color in ch_images.keys():
            # if var_dict['MorphologyChannel'] not in ch_images.keys():
            #     continue
            self.all_ch_int_stats[color] = {}

            # Read image-holding dictionary back in and collect all intensities/image
            for frame, ch_image in ch_images[color].items():
                self.all_ch_int_stats[color][frame] = self.find_cnt_int_dist(ch_image)

        if verbose:
            pprint.pprint(self.all_ch_int_stats)


def sort_cell_info_by_index(time_dictionary, time_list):
    '''
    Takes arrays for each timepoint (key) and sorts on index of tuple.
    '''
    for timepoint in time_list:
        cell_inds = [int(cell[0]) if cell[0] not in ['n','e'] else cell[0] for cell in time_dictionary[timepoint]]
        cell_objs = [cell[1] for cell in time_dictionary[timepoint]]
        inds_and_objs = zip(cell_inds, cell_objs)

        sorted_cell_objs = sorted(inds_and_objs, key=lambda pair: pair[0])
        time_dictionary[timepoint] = sorted_cell_objs

    return time_dictionary

# ----The main tracking function-----------------
def populate_cell_ind_overlap(time_dictionary, time_list, verbose=False):
    '''
    Updates cell_ind from 'n' to value.
    Value is determined from match with previous time point.
    Each cell record is ['n', CellObj]
    '''

    if verbose:
        print '--time_dictionary before--'
        pprint.pprint(time_dictionary.items())

    assert len(time_dictionary.keys()) > 0, 'No time point data given.'
    first_entry_time = time_list[0]
    # Ordering first entry
    for ind, cell_record in enumerate(time_dictionary[first_entry_time], 1):
        cell_record[0] = ind
    # Numbering the rest
    print 'Initial number of cells:', len(time_dictionary[first_entry_time])
    num_cell = len(time_dictionary[first_entry_time])+1

    for time_ind in range(1, len(time_list)):
        t_curr = time_list[time_ind]
        t_prev = time_list[time_ind-1]

        # Definition: cell_record = (cell_ind, cell_obj)
        for cell_record_c in time_dictionary[t_curr]:
            cell_curr = cell_record_c[1]
            circle_curr = cell_curr.get_circle()

            found = False
            # Sweep previous cells and look for intersection.
            for cell_record_p in time_dictionary[t_prev]:
                cell_prev = cell_record_p[1]
                overlap = cell_prev.evaluate_overlap(circle_curr)

                if overlap == True:
                    found = True
                    cell_record_c[0] = cell_record_p[0]
                    break

            if not found:
                cell_record_c[0] = num_cell
                num_cell += 1

    print 'Final number of cells:', num_cell
    if verbose:
        print '--time_dictionary after--'
        pprint.pprint(time_dictionary.items())

    # Make sure all 'n' were registered.
    for time, cell_records in time_dictionary.items():
        for cell_record in cell_records:
            index = cell_record[0]
            assert index != 'n', cell_records

    return time_dictionary

def populate_cell_ind_closest(time_dictionary, time_list, max_dist=100, verbose=False):
    '''
    Updates cell_ind from 'n' to value.
    Value is determined from match with previous time point based on proximity.
    Each cell record is ['n', CellObj]
    '''
    if verbose:
        print '--time_dictionary before--'
        pprint.pprint(time_dictionary.items())

    assert len(time_dictionary.keys()) > 0, 'No time point data given.'
    first_entry_time = time_list[0]
    # Ordering first entry
    for ind, cell_record in enumerate(time_dictionary[first_entry_time],1):
        cell_record[0] = ind

    # Numbering the rest
    print 'Initial number of cells:', len(time_dictionary[first_entry_time])
    num_cell = len(time_dictionary[first_entry_time])+1
    for time_ind in range(1, len(time_list)):
        t_curr = time_list[time_ind]
        t_prev = time_list[time_ind-1]

        # Definition: cell_record = (cell_ind, cell_obj)
        for cell_record_c in time_dictionary[t_curr]:
            cell_curr = cell_record_c[1]
            circle_curr = cell_curr.get_circle()

            # Rounded number of pixels on image side
            dist_found = 4000
            # Sweep previous cells and look for intersection.
            for cell_record_p in time_dictionary[t_prev]:
                cell_prev = cell_record_p[1]
                dist_delta = cell_prev.evaluate_dist(circle_curr)
                overlap = dist_delta < dist_found

                if overlap == True:
                    # Update the distance to the cell-cell distance
                    dist_found = dist_delta
                    cell_record_c[0] = cell_record_p[0]

            if dist_found > max_dist:
                cell_record_c[0] = num_cell
                num_cell += 1

    print 'Final number of cells:', num_cell
    if verbose:
        print '--time_dictionary after--'
        pprint.pprint(time_dictionary.items())

    # Make sure all 'n' were registered.
    for time, cell_records in time_dictionary.items():
        for cell_record in cell_records:
            index = cell_record[0]
            assert index != 'n', cell_records

    return time_dictionary

def populate_cell_ind_csv(well, pointer_to_csv, time_dictionary, verbose=True):
    '''
    Updates cell_ind from 'n' to value.
    Value is determined from match with csv x,y coordinates for the same time point.
    Each cell record is ['n', CellObj]
    '''
    import pandas as pd
    untracked_csv = pd.read_csv(pointer_to_csv)
    print 'Imported csv with (rows,columns)', untracked_csv.shape

    for time_id in time_dictionary.keys():
        for cr_num, cell_record in enumerate(time_dictionary[time_id]):
            print cell_record[1]
            cnt_querry = cell_record[1].cnt
            (x, y), (ma, MA), angle = cv2.fitEllipse(cnt_querry)
            area_cnt = cv2.contourArea(cnt_querry)
            # Get some parameters that are not x and y
            # ecc = np.sqrt(1-((MA)**2/(ma)**2))
            print "Querying csv for:", x,y, time_id, area_cnt
            # Original join options for Galaxy csv to cv2 calculation comparison
            sub_untracked = untracked_csv[(
                untracked_csv.Timepoint==int(time_id[1:len(time_id)])) & (
                untracked_csv.BlobArea.round(2)==round(
                    cv2.contourArea(cell_record[1].cnt), 2)) & (
                untracked_csv.BlobCentroidX.round()==round(x)) & (
                untracked_csv.BlobCentroidY.round()==round(y))]
            # Other join options
                # untracked_csv.Timepoint==int(time_id[1:len(time_id)])) & (
                # untracked_csv.BlobArea.round(2)==round(
                #     cv2.contourArea(cell_record[1].cnt), 2)) & (
                # untracked_csv.Angle.round()==round(angle)) & (
                # untracked_csv.BlobCircularity.round()==round(ecc))]

            # Updated join options for PP csv to cv2 calculation comparison
            # sub_untracked = untracked_csv[(
            #     untracked_csv.Timepoint==int(time_id[1:len(time_id)])) & (
            #     untracked_csv.Sci_WellID==well) & (
            #     # untracked_csv.BlobArea.round(2)<(round(cv2.contourArea(cell_record[1].cnt), 2)+500)) & (
            #     # untracked_csv.BlobArea.round(2)>(round(cv2.contourArea(cell_record[1].cnt), 2)-500)) & (
            #     untracked_csv.BlobCentroidX.round()<(round(x)+5)) & (
            #     untracked_csv.BlobCentroidX.round()>(round(x)-5)) & (
            #     untracked_csv.BlobCentroidY.round()<(round(y)+5)) & (
            #     untracked_csv.BlobCentroidY.round()>(round(y)-5))]

            print sub_untracked.shape[0]
            if sub_untracked.shape[0]==1:
                if verbose:
                    print 'Before setting:', time_dictionary[time_id][cr_num]
                time_dictionary[time_id][cr_num] = list(
                    time_dictionary[time_id][cr_num])
                time_dictionary[time_id][cr_num][0] = str(
                    sub_untracked.iloc[0]["ObjectLabelsFound"])
                time_dictionary[time_id][cr_num] = tuple(
                    time_dictionary[time_id][cr_num])
                if verbose:
                    print 'After setting:', time_dictionary[time_id][cr_num]
                    # print sub_untracked[["ObjectLabelsFound", "Timepoint", "BlobCentroidX", "BlobCentroidY", "Sci_WellID", "BlobArea"]]

            else:
                if verbose:
                    print "Number of records matching dictionary item in csv is:", sub_untracked.shape[0]
                    print sub_untracked[["ObjectLabelsFound", "Timepoint", "BlobCentroidX", "BlobCentroidY", "Sci_WellID", "BlobArea"]]

    # Make sure all 'n' were registered.
    for time, cell_records in time_dictionary.items():
        for cell_record in cell_records:
            index = cell_record[0]
            # assert index != 'n', cell_records
            if verbose:
                if index == 'n':
                    print cell_record

    if verbose:
        print "Time dict from csv:"
        pprint.pprint(time_dictionary)

    return time_dictionary

# ----Handling dictionary structure--------------
def time_all_cell_dict(well_filelist, time_list, resolution, tracking_type, var_dict):
    '''
    Initialize time dctionary to keep track of cells either as blank entries or from encoded masks.
    '''
    time_dictionary = collections.OrderedDict()

    for img_pointer, time_id in zip(well_filelist, time_list):
        # print "Current image handled:", img_pointer
        if tracking_type=="TrackFromMasks":
            img = cv2.imread(img_pointer, -1)
            time_dictionary = time_id_cell_dict_encoded(
                img, time_id, time_dictionary)
        elif any(tracking_type==track_opt for track_opt in
                 ["TrackSpatioTemporal", "SkipTracking", "TrackFromCSV"]):
            img = cv2.imread(img_pointer, 0)
            # print "What image used for cnt is like:", img.min(), img.max()
            time_dictionary = time_id_cell_dict(
                img, time_id, time_dictionary, var_dict)
        else:
            print "Unknown tracking request..."
            break

    return time_dictionary

def time_id_cell_dict(img, time_id, time_dictionary, var_dict):
    '''
    Finds contours and filters for cells.
    Then adds cells to time_dictionary.
    '''

    kept_contours = find_cells(img, img_is_mask=True)
    kept_contours = filter_contours(kept_contours,
        small=var_dict["MinCellSize"], large=var_dict["MaxCellSize"])

    time_dictionary[time_id] = []
    for cnt in kept_contours:
        time_dictionary[time_id].append(
            ['n', Cell(cnt)])

    return time_dictionary

def time_id_cell_dict_encoded(img, time_id, time_dictionary):
    '''
    Finds contours and filters for cells.
    Then adds cells to time_dictionary.
    '''
    time_dictionary[time_id] = []
    print "Number of masked cells", len(np.unique(img)[np.unique(img)!=0])

    for i in np.unique(img)[np.unique(img)!=0]:
        cnt_values = img[np.nonzero(img==i)]
        mask = np.zeros(img.shape[:2], np.uint8)
        mask[np.nonzero(img==i)]=255
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours)>1:
            print "Found", len(contours), "objects for encoding", i
            contours = sorted(contours, key = cv2.contourArea, reverse = True)
            for cnt in contours:
                center, radius = cv2.minEnclosingCircle(cnt)
                print center
                time_dictionary[time_id].append(
                    [cnt_values.max(), Cell(cnt)])
        else:
            time_dictionary[time_id].append(
                [cnt_values.max(), Cell(contours[0])])

    # Make sure all indices were registered.
    for time, cell_records in time_dictionary.items():
        for cell_record in cell_records:
            index = cell_record[0]
            assert index < 2**16 and index >= 0, cell_records

    # print "Time dict from encoded masks:"
    # pprint.pprint(time_dictionary)
    return time_dictionary

# ----Encoding dictionary structure--------------
def make_encoded_mask(sorted_time_dict, well_filelist, time_list, write_path, resolution, verbose=False):
    '''
    Draw kept cells onto image with intensity value corresponding to cell number.
    '''
    # Should be 16-bit encode > 256 objects
    d_type = np.uint16

    if verbose:
        print 'Files to use:'
        pprint.pprint(well_filelist)
        print 'Time list'
        print time_list

    for img_pointer, time_id in zip(well_filelist, time_list):

        if 'MASK.tif' not in img_pointer and 'MASKS.tif' not in img_pointer:
            continue

        mask_shape = cv2.imread(img_pointer, resolution).shape[0:2]
        mask = np.zeros(mask_shape, dtype=d_type)
        orig_name = utils.extract_file_name(img_pointer)
        img_name = utils.make_file_name(write_path, orig_name+'_ENCODED')
        well_name = os.path.basename(img_pointer).split('_')[4]
        img_name = utils.reroute_imgpntr_to_wells(img_name, well_name)

        # Loop through Cells in dictionary and encode into mask
        for cnt_ind, cell_obj in sorted_time_dict[time_id]:
            if cnt_ind in ['n','e']:
                continue
            cv2.drawContours(mask, [cell_obj.cnt], 0, int(cnt_ind), -1)
            cv2.drawContours(mask, [cell_obj.cnt], 0, int(cnt_ind), 5)

        cv2.imwrite(img_name, mask)

def tracking(var_dict, path_to_masks, write_path, tracking_type, pointer_to_csv=None):
    '''
    Main point of entry.
    '''
    resolution = 0

    morph_channel = var_dict["MorphologyChannel"]
    try:
        var_dict['TrackedCells']
    except KeyError:
        var_dict['TrackedCells'] = {}

    for well in var_dict['Wells']:
        if tracking_type=="TrackFromMasks":
            selector = utils.make_selector(well=well)
            # well_filelist = utils.make_filelist(path_to_masks, '*'+selector)
            well_filelist =  utils.make_filelist_wells(path_to_masks, '*'+selector)
            well_filelist = [imp for imp in well_filelist if 'ENCODED' in imp]
        else:
            selector = utils.make_selector(well=well)#, channel=var_dict["MorphologyChannel"])
            # print 'Current selector', selector
            # well_filelist = utils.make_filelist(path_to_masks, '*'+selector)
            well_filelist =  utils.make_filelist_wells(path_to_masks, '*'+selector)
            well_filelist = [imp for imp in well_filelist if 'ENCODED' not in imp]

        if len(well_filelist) == 0:
            # print 'No files associated with morphology channel.'
            print 'Confirm that CellMasks folder contains files.'
            continue

        # time_list = utils.get_timepoints(well_filelist)
        time_list = var_dict['TimePoints']
        print 'Time points that have a morphology image:'
        print 'Well', well, time_list
        print 'Testing file order:'
        pprint.pprint([os.path.basename(fn) for fn in well_filelist])

        try:
            time_dictionary = var_dict['TrackedCells'][well]
            print 'Using the supplied initial cell dictionary.'
        except KeyError:
            time_dictionary = time_all_cell_dict(
                well_filelist, time_list, resolution, tracking_type, var_dict)

        if tracking_type=='TrackFromCSV':
            time_dictionary = populate_cell_ind_csv(
                well, pointer_to_csv, time_dictionary)

        if tracking_type=="TrackSpatioTemporal": #should rename to 'overlap'
            time_dictionary = populate_cell_ind_overlap(time_dictionary, time_list)
            print "Numbers should be assigned now..."

        if tracking_type=="TrackProximity":
            time_dictionary = populate_cell_ind_closest(time_dictionary, time_list)
            print "Numbers should be assigned now..."

        sorted_time_dict = sort_cell_info_by_index(time_dictionary, time_list)
        var_dict['TrackedCells'][well] = sorted_time_dict
        # print 'sorted_time_dict'
        # pprint.pprint(sorted_time_dict.items())

        # Bring back encoded mask after pipeline pilot work completed:
        if tracking_type in ["TrackSpatioTemporal", "TrackFromCSV", "TrackProximity"]:

            make_encoded_mask(
                sorted_time_dict, well_filelist, time_list, write_path, resolution)

    # For select_analysis_module input, set var_dict['OutputPath']
    var_dict["OutputPath"] = write_path


if __name__ == '__main__':

    # ----Parser-----------------------
    parser = argparse.ArgumentParser(
        description="Track cells from cell masks.")
    parser.add_argument("input_dict",
        help="Load input variable dictionary")
    parser.add_argument("input_path",
        help="Folder path to cell masks.")
    parser.add_argument("tracking_type",
        help="Is tracking information avaialble? Possible options are: TrackSpatioTemporal, TrackProximity, TrackFromMasks, TrackFromCSV, SkipTracking")
    parser.add_argument("output_dict",
        help="Write variable dictionary.")
    parser.add_argument("--pointer_to_csv",
        dest="pointer_to_csv", default=None,
        help="Pointer to .csv with cluter-based tracking.")
    parser.add_argument("--min_cell",
        dest="min_cell", type=int,
        help="Minimum feature size considered as cell.")
    parser.add_argument("--max_cell",
        dest="max_cell", type=int,
        help="Maximum feature size considered as cell.")
    parser.add_argument("--max_dist",
        dest="max_dist", type=int,
        help="Maximum distance a cell can travel between time points in pixels.")

    args = parser.parse_args()

    # ----Load path dict-------------------------
    infile = args.input_dict
    var_dict = pickle.load(open(infile, 'rb'))
    var_dict["MaxDistance"] = int(args.max_dist)

    try:
        min_cell = var_dict["MinCellSize"]
        max_cell = var_dict["MaxCellSize"]
    except KeyError:
        print "Using updated min/max size object parameters."
        try:
            var_dict["MinCellSize"] = int(args.min_cell)
            var_dict["MaxCellSize"] = int(args.max_cell)
            assert var_dict["MinCellSize"] < var_dict["MaxCellSize"], 'Minimum size should be smaller than maximum size.'
        except TypeError:
            assert type(args.min_cell)==int, 'Please provide numerical minimum value.'
            assert type(args.max_cell)==int, 'Please provide numerical maximum value.'


    print "Minimum object size set to:", var_dict["MinCellSize"]
    print "Maximum object size set to:", var_dict["MaxCellSize"]

    # ----Initialize parameters------------------
    path_to_masks = args.input_path
    write_path = path_to_masks
    tracking_type = args.tracking_type # possible values: TrackSpatioTemporal, TrackFromMasks, TrackFromCSV, SkipTracking
    outfile = args.output_dict
    if tracking_type=="TrackFromCSV":
        pointer_to_csv = args.pointer_to_csv
        assert os.path.exists(pointer_to_csv), 'Confirm the csv exists.'

    # ----Confirm given folders exist--
    assert os.path.exists(path_to_masks), 'Confirm the given path for data exists.'

    # ----Run tracking---------------------------
    start_time = datetime.datetime.utcnow()

    if tracking_type == "TrackFromCSV":
        tracking(
            var_dict, path_to_masks, write_path,
            tracking_type, pointer_to_csv=pointer_to_csv)
    else:
        tracking(
            var_dict, path_to_masks, write_path, tracking_type)

    end_time = datetime.datetime.utcnow()
    print 'Tracking run time:', end_time-start_time

    # ----Output for user and save dict----------
    print 'Cells were tracked for each time point.'
    print 'Output from this step is an encoded mask written to:'
    print write_path

    pickle.dump(var_dict, open('var_dict.p', 'wb'))
    # outfile = os.rename('var_dict.p', outfile)
    outfile = shutil.move('var_dict.p', outfile)
    utils.save_user_args_to_csv(args, write_path, 'tracking_extra')
