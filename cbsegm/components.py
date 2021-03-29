import numpy as np
import cv2
import scipy.ndimage
import json

def connected_components(img, connectivity=4, remove_zero=True):
    img = (img>0).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((img).astype(np.uint8) * 255, connectivity,
                                                                            cv2.CV_32S)
    print("num_labels:",num_labels)
    bboxes = stats[int(remove_zero):, :4].copy()
    bboxes[:, 2:] += bboxes[:, :2]
    # labels, bboxes, centroids,
    return labels, bboxes, centroids[int(remove_zero):,:], stats[int(remove_zero):,4]

def rlsa(img, gap, horizontal=True):
    if gap==0:
        return img
    if horizontal:
        structure = np.zeros([3, 3])
        structure[1, :] = 1
    else:
        structure = np.zeros([3, 3])
        structure[:, 1] = 1
    img = img > 0
    structure = structure>0
    #print(f"img:{img.mean()}")
    dilated = scipy.ndimage.morphology.binary_dilation(img, structure=structure, iterations=gap//2);
    #print(f"dilated:{dilated.mean()} gap:{gap//2}")
    smeared = scipy.ndimage.morphology.binary_erosion(dilated, structure=structure, iterations=gap//2)
    #print(f"smeared:{smeared.mean()}")
    return smeared


def get_intencity_sum(bin_img, bboxes):
    padded_img=np.zeros([bin_img.shape[0]+2,bin_img.shape[1]+2])
    padded_img[:-2,:-2]=bin_img
    int_img=padded_img.cumsum(axis=0).cumsum(axis=1)
    left = bboxes[:, 0]
    right = bboxes[:, 2]
    top = bboxes[:, 1]
    bottom = bboxes[:, 3]
    top_left = int_img[top, left]
    #print("int_img:",int_img.shape,"  bottom:", bottom.max()+1, "  left:",left.max())
    bottom_left = int_img[bottom+1, left]
    top_right = int_img[top, right+1]
    bottom_right = int_img[bottom+1, right+1]
    return (top_left+bottom_right)-(bottom_left+top_right)


def extract_bbox_features(bboxes,bin_img):
    page_height, page_width = bin_img.shape
    width = 1 + bboxes[:,2]-bboxes[:,0]
    height = 1 + bboxes[:, 3] - bboxes[:, 1]
    norm_width = width/page_width
    norm_height = height/page_height
    center_x = bboxes[:, [0, 2]].mean(axis=1)
    center_y = bboxes[:, [1, 3]].mean(axis=1)
    norm_center_x = center_x / page_width
    norm_center_y = center_y / page_height
    areas = width*height
    norm_areas = areas**.5/(page_width*page_height)**.5
    wideness = width/height
    inv_wideness = height/width
    foreground_count = get_intencity_sum(bin_img, bboxes)

    foreground_density = foreground_count/areas
    #feats1d = [width, height,center_x,center_y,norm_center_x,norm_center_y,areas,norm_areas,wideness,inv_wideness,foreground_count,foreground_density]
    #feats1d = [width, height, areas**.5, norm_areas, wideness,inv_wideness, foreground_density]
    feats1d = [norm_width, norm_height, wideness, inv_wideness, norm_areas, foreground_density]
    feats1d = [f.reshape([-1, 1]) for f in feats1d]
    #result = np.concatenate(feats1d+[bboxes], axis=1)
    result = np.concatenate(feats1d , axis=1)
    #print(result)
    return result


class BoxLikelihoodEstimator(object):
    def __init__(self, epsilon=.0000001):
        self.means=np.zeros([1,16])
        self.deviations = np.ones([1, 16])
        self.epsilon=epsilon

    def train(self, features):
        self.means = features.mean(axis=0).reshape([1,-1])
        self.deviations = (features-self.means).std(axis=0).reshape([1,-1])
        self.deviations[self.deviations<self.epsilon]=self.epsilon

    def save(self, fname):
        obj = {"means":self.means.tolist(),"deviations":self.deviations.tolist()}
        with open(fname,"w") as fd:
            json.dump(obj, fd)

    @classmethod
    def load(cls, fname):
        res=cls()
        with open(fname) as fd:
            obj = json.load(fd)
        res.means = np.array(obj["means"])
        res.deviations = np.array(obj["deviations"])
        return res

    def __call__(self, features: np.array)->np.array:
        normalised_features = (features-self.means)/self.deviations
        feature_probs = scipy.stats.norm().pdf(normalised_features)
        logits = np.log(feature_probs+.001)
        return np.exp(logits.mean(axis=1))-.001 # returning the geometric mean of all feature probabillities