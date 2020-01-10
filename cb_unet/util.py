import cv2
import numpy as np
import torch


def get_otsu_threshold(img):
    if type(img) is torch.Tensor:
        threshold, _ = cv2.threshold((img * 255).to("cpu").numpy().astype("uint8"), 0, 1.0,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif type(img) is np.array and img.dtype==np.uint8:
        threshold, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif type(img) is np.array:
        threshold, _ = cv2.threshold((img*255).astype("uint8"), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold/=255.0
    else:
        raise ValueError("Img should be a 2D numpy or pytorch")
    return threshold

def evaluate_binarization_improvement(input_gray, prediction_gray, gt_img,epsilon=.000001):
    gt_img = gt_img > 0
    print("Thr 1")
    input_bin = input_gray>get_otsu_threshold(input_gray)
    print("Thr 2")
    prediction_bin = prediction_gray > get_otsu_threshold(prediction_gray)
    print("input_precision")
    input_precision= (input_bin & gt_img).sum() / float(gt_img.sum()+epsilon)
    print("input_recall")
    input_recall = (input_bin & gt_img).sum() / float(input_bin.sum()+epsilon)
    print("input_fscore")
    input_fscore=2*input_precision*input_recall/(input_precision+input_recall+epsilon)
    print("input_precision")
    prediction_precision= (prediction_bin * gt_img).sum() / float(gt_img.sum().float()+epsilon)
    prediction_recall = (prediction_bin * gt_img).sum() / float(prediction_bin.sum()+epsilon)
    prediction_fscore=2*prediction_precision*prediction_recall/(prediction_precision+prediction_recall+epsilon)
    return prediction_fscore,input_fscore

def validate(model,validation_loader):
    device=model.get_device()
    for input_imgs,target in validation_loader:
        dev_input_imgs=input_imgs.to(device)
        target
        dev_out=model(dev_input_imgs)
        prediction_fscore, input_fscore = evaluate_binarization_improvement(
            dev_input_imgs[0,0,:,:], dev_out[0,0,:,:], target[0,0,:,:])

