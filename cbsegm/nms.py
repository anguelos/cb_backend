import torch
import numpy as np
import torchvision


def _torch_get_inter_union(boxes_ltrb: torch.Tensor):
    left_col = boxes_ltrb[:, 0:1]
    right_col = boxes_ltrb[:, 2:3]
    top_col = boxes_ltrb[:, 1:2]
    bottom_col = boxes_ltrb[:, 3:4]

    left_row = left_col.view([1, -1])
    right_row = right_col.view([1, -1])
    top_row = top_col.view([1, -1])
    bottom_row = bottom_col.view([1, -1])

    width_row = right_row - left_row
    height_row = right_row - left_row
    surface_row = width_row * height_row
    surface_col = surface_row.view([1, -1])

    h_intersect = torch.relu(torch.min(right_row, right_col) - torch.max(left_row, left_col))
    v_intersect = torch.relu(torch.min(bottom_row, bottom_col) - torch.max(top_row, top_col))
    intersection = h_intersect * v_intersect

    union = surface_col + surface_row - intersection
    return intersection, union


def _torch_get_iou(boxes_ltrb, epsilon=.000001):
    intersection, union = _torch_get_inter_union(boxes_ltrb)
    iou = intersection.to(torch.float)/(union.to(torch.float) + epsilon)
    return iou


def get_inter_union(boxes_ltrb_cols: np.array, boxes_ltrb_rows: np.array):
    left_col = boxes_ltrb_cols[:, 0:1]
    right_col = boxes_ltrb_cols[:, 2:3]
    top_col = boxes_ltrb_cols[:, 1:2]
    bottom_col = boxes_ltrb_cols[:, 3:4]

    left_row = boxes_ltrb_rows[:, 0:1].T
    right_row = boxes_ltrb_rows[:, 2:3].T
    top_row = boxes_ltrb_rows[:, 1:2].T
    bottom_row = boxes_ltrb_rows[:, 3:4].T

    width_col = 1 + right_col - left_col
    height_col = 1 + bottom_col - top_col
    surface_col = width_col * height_col

    width_row = 1 + right_row - left_row
    height_row = 1 + bottom_row - top_row
    surface_row = width_row * height_row

    h_intersect = 1 + np.minimum(right_row, right_col) - np.maximum(left_row, left_col)
    h_intersect = (h_intersect > 0) * h_intersect

    v_intersect = 1 + np.minimum(bottom_row, bottom_col) - np.maximum(top_row, top_col)
    v_intersect = (v_intersect > 0) * v_intersect

    intersection = h_intersect * v_intersect

    union = (surface_col + surface_row) - intersection
    return intersection, union


def get_iou(row_boxes_ltrb: np.array, col_boxes_ltrb=None, epsilon=.000001):
    if col_boxes_ltrb is None:
        col_boxes_ltrb = row_boxes_ltrb
    intersection, union = get_inter_union(row_boxes_ltrb, col_boxes_ltrb)
    iou = intersection/(union + epsilon)
    return iou


def precision_recall(iou_mat, threshold, epsilon=.00000001):
    correct = iou_mat > threshold
    precision = correct.max(axis=0).mean()
    recall = correct.max(axis=1).mean()
    fscore = (2*precision*recall)/(precision+recall+epsilon)
    return precision, recall, fscore


def nms(bboxes_ltrb, scores, thresh):
    bboxes_ltrb=torch.tensor(bboxes_ltrb).float()
    scores=torch.tensor(scores).float()
    keep = torchvision.ops.nms(bboxes_ltrb, scores, thresh)
    return keep.numpy()

