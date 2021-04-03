from .components import BoxLikelihoodEstimator, connected_components, extract_bbox_features, rlsa
import numpy as np
from .nms import nms

def word_proposals(prob_img, box_likelihood_estimator:BoxLikelihoodEstimator, thresholds=[1, 128, 255], rlsa_gaps=[1, 4, 8], iou_threshold=.5):
    features = []
    modality_id = 0
    for threshold in thresholds:
        for rlsa_gap in rlsa_gaps:
            bin_img = (prob_img < threshold).astype(np.uint8)
            smeared_img = rlsa(bin_img, rlsa_gap)
            label, bbox, centroid, component_size = connected_components(smeared_img)
            #print(f"rlsa:{rlsa_gap}, threshold:{threshold}, nb_components:{bbox.shape[0]}, bin_img:{bin_img.mean()}, smeared_img:{smeared_img.mean()}, ")
            features.append(extract_bbox_features(bbox,smeared_img))
            modality_id += 1
    features = np.concatenate(features, axis=0)
    bboxes = features[:, -4:].astype(np.long)
    likelihoods = box_likelihood_estimator(features)

    keep = nms(bboxes, likelihoods, iou_threshold)
    #print("Proposals:", likelihoods.shape[0], "   Keep", keep.shape[0])
    return bboxes[keep, :]