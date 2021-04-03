import numpy as np
import cv2
import scipy.ndimage
import json
import torch
from .utils import load_annotator_json_words
from .nms import get_iou, get_inter_union
from PIL import Image

def connected_components(img, connectivity=4, remove_zero=True):
    img = (img>0).astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats((img).astype(np.uint8) * 255, connectivity,
                                                                            cv2.CV_32S)
    bboxes = stats[int(remove_zero):, :4].copy()
    bboxes[:, 2:] += bboxes[:, :2]
    # labels, bboxes, centroids,
    return labels, bboxes, centroids[int(remove_zero):,:], stats[int(remove_zero):,4]

def rlsa(img, gap, horizontal=True):
    if gap==0:
        return img
    if isinstance(gap, int):
        if horizontal:
            structure = np.zeros([3, 3])
            structure[1, :] = 1
        else:
            structure = np.zeros([3, 3])
            structure[:, 1] = 1
        img = img > 0
        structure = structure>0
        dilated = scipy.ndimage.morphology.binary_dilation(img, structure=structure, iterations=gap//2);
        smeared = scipy.ndimage.morphology.binary_erosion(dilated, structure=structure, iterations=gap//2)
    else:
        h_structure = np.zeros([3, 3])
        h_structure[1, :] = 1
        v_structure = np.zeros([3, 3])
        v_structure[:, 1] = 1
        v_dilated = scipy.ndimage.morphology.binary_dilation(img, structure=h_structure, iterations=gap[0] // 2);
        h_smeared = scipy.ndimage.morphology.binary_erosion(v_dilated, structure=h_structure, iterations=gap[0] // 2)
        v_dilated = scipy.ndimage.morphology.binary_dilation(img, structure=v_structure, iterations=gap[1] // 2);
        v_smeared = scipy.ndimage.morphology.binary_erosion(v_dilated, structure=v_structure, iterations=gap[1] // 2)
        smeared = np.maximum(h_smeared, v_smeared)
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
    feats1d = [width, height,center_x,center_y,norm_center_x,norm_center_y,areas,norm_areas,wideness,inv_wideness,foreground_count,foreground_density]
    #feats1d = [width, height, areas**.5, norm_areas, wideness,inv_wideness, foreground_density]
    #feats1d = [norm_width, norm_height, wideness, inv_wideness, norm_areas, foreground_density]
    feats1d = [f.reshape([-1, 1]) for f in feats1d]
    result = np.concatenate(feats1d+[bboxes], axis=1)
    #result = np.concatenate(feats1d , axis=1)
    return result


_bbox_feature_size = extract_bbox_features(np.array([[0,0,1,1]]), np.array([[1]])>0).shape[1]


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

class BoxIOUPredictor(torch.nn.Module):
    def __init__(self, feature_sizes=_bbox_feature_size):
        super().__init__()
        self.feature_sizes=feature_sizes
        self.fc1 = torch.nn.Linear(self.feature_sizes, 200)
        self.fc2 = torch.nn.Linear(200, 30)
        self.fc3 = torch.nn.Linear(30, 2)
        self.non_lineariry = torch.nn.ReLU()
        self.non_lineariry = torch.nn.Tanh()
        self.dropout = torch.nn.Dropout()


    def forward(self, x):
        x = self.non_lineariry(self.fc1(x))
        x = self.dropout(x)
        x = self.non_lineariry(self.fc2(x))
        x = self.dropout(x)
        return torch.log_softmax(self.fc3(x), dim=1)

    def train_supervised(self, dataset):
        pass

    @staticmethod
    def make_supervised_dataset(gt_bbox_paths, proposed_bbox_paths, bin_img_paths, iou_threshold=-1):
        id_to_gt = {p.split("/")[-1].split(".")[0]: p for p in gt_bbox_paths}
        id_to_proposals = {p.split("/")[-1].split(".")[0]: p for p in proposed_bbox_paths}
        id_to_binimg = {p.split("/")[-1].split(".")[0]: p for p in bin_img_paths}
        assert set(id_to_gt.keys()) == set(id_to_proposals.keys()) == set(id_to_binimg.keys())
        inputs = []
        targets = []
        recalls = []
        for page_id in id_to_gt.keys():
            gt_bboxes, _ = load_annotator_json_words(id_to_gt[page_id])
            proposal_bboxes, _ = load_annotator_json_words(id_to_proposals[page_id])
            prob_img = np.array(Image.open(id_to_binimg[page_id]))
            inputs.append(extract_bbox_features(proposal_bboxes, prob_img))
            i,u = get_inter_union(gt_bboxes, proposal_bboxes)
            iou = get_iou(gt_bboxes, proposal_bboxes)
            best_iou = iou.max(axis=0)
            recall = iou.max(axis=1)
            recalls.append(recall)
            print(best_iou.shape,recall.shape)
            print(f"Recall@50: {(recall>.5).mean().item():6}    Recall@75:{(recall>.75).mean().item():6}    Recall@90:{(recall>.9).mean().item():6}    Recall@100:{(recall>=1).mean().item():6}    Recall>100:{(recall>1).mean().item():6}")
            print(i.max(),u.max(), (i/u).max())
            print()
            if iou_threshold > 0:
                targets.append(best_iou>iou_threshold)
            else:
                targets.append(best_iou)
        inputs = torch.Tensor(np.concatenate(inputs, axis=0))
        targets = torch.Tensor(np.concatenate(targets, axis=0))
        recalls = np.concatenate(recalls, axis=0)
        print(
            f"GTBOXES:{recalls.shape[0]} PROPOSALS:{targets.shape[0]} Recall@50: {(recalls > .5).mean().item():6}    Recall@75:{(recalls > .75).mean().item():6}    Recall@90:{(recalls > .9).mean().item():6}")
        assert inputs.shape[0] == targets.shape[0]
        return [(inputs[n, :], targets[n]) for n in range(len(targets))]

    def train_supervised(self, batch_size=128, ):
        pass



