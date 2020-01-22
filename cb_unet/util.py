import cv2
import numpy as np
import torch

def render_confusion(prediction,gt,tp_col=[0,0,0],tn_col=[255,255,255],fp_col=[255,0,0],fn_col=[0,0,255]):
    prediction=(prediction.cpu().numpy())
    gt = (gt.cpu().numpy())
    res=np.zeros(prediction.shape+(3,))
    tp = gt & prediction
    tn = (~gt) & (~prediction)
    fp = (~gt) & prediction
    fn = (gt) & (~prediction)
    res[tp, :] = tp_col
    res[tn, :] = tn_col
    res[fp, :] = fp_col
    res[fn, :] = fn_col
    precision = (1+tp.sum())/float(1+tp.sum()+fp.sum())
    recall = (1+tp.sum()) / float(1+tp.sum() + fn.sum())
    Fscore=(2*precision*recall)/(precision+recall)
    return res,precision,recall,Fscore

def render_optimal_confusion(prediction_cont,gt,tp_col=[0,0,0],tn_col=[255,255,255],fp_col=[255,0,0],fn_col=[0,0,255]):
    prediction_cont=prediction_cont.cpu()
    gt=gt.cpu().numpy()
    results=[]
    for thr in [n/100.0 for n in range(0,100,1)]:
        prediction=(prediction_cont<thr).numpy()
        res=np.zeros(prediction.shape+(3,))
        tp = gt & prediction
        tn = (~gt) & (~prediction)
        fp = (~gt) & prediction
        fn = (gt) & (~prediction)
        res[tp, :] = tp_col
        res[tn, :] = tn_col
        res[fp, :] = fp_col
        res[fn, :] = fn_col
        precision = (1+tp.sum())/float(1+tp.sum()+fp.sum())
        recall = (1+tp.sum()) / float(1+tp.sum() + fn.sum())
        Fscore=(2*precision*recall)/(precision+recall)
        results.append((Fscore,(res,precision,recall,Fscore)))
    results=sorted(results)
    return results[-1][1]


def get_otsu_threshold(img):
    if type(img) is torch.Tensor:
        threshold, _ = cv2.threshold((img * 255).detach().to("cpu").numpy().astype("uint8"), 0, 1.0,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold /= 255.0
    elif type(img) is np.array and img.dtype==np.uint8:
        threshold, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif type(img) is np.array:
        threshold, _ = cv2.threshold((img*255).astype("uint8"), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        threshold/=255.0
    else:
        raise ValueError("Img should be a 2D numpy or pytorch")
    return threshold

def evaluate_binarization_improvement(input_gray, C,epsilon=.000001):
    return 0,0
    gt_img = gt_img > 0
    #print("Thr 1")
    input_bin = input_gray < get_otsu_threshold(input_gray)
    #print("Thr 2")
    prediction_bin = prediction_gray < get_otsu_threshold(prediction_gray)
    #print("input_precision")
    input_precision= (input_bin & gt_img).sum().item() / float(gt_img.sum()+epsilon)
    #print("input_recall")
    input_recall = (input_bin & gt_img).sum().item() / float(input_bin.sum()+epsilon)
    #print("input_fscore")
    input_fscore=2*input_precision*input_recall/(input_precision+input_recall+epsilon)
    #print("prediction_precision")
    prediction_precision= (prediction_bin * gt_img).float().sum().item() / (float(gt_img.sum().float())+epsilon)
    #print("prediction_recall",(float(prediction_bin.sum())+epsilon))
    prediction_recall = (prediction_bin * gt_img).float().sum().item() / (float(prediction_bin.sum())+epsilon)
    #print("prediction_fscore",(prediction_precision+prediction_recall+epsilon))
    prediction_fscore=2*prediction_precision*prediction_recall/(prediction_precision+prediction_recall+epsilon)
    #print("return")
    print(prediction_fscore, input_fscore)
    return prediction_fscore, input_fscore


def draw_images(prediction_gray, gt_img):
    pass

def validate(model,validation_loader):
    device=model.get_device()
    for input_imgs,target in validation_loader:
        dev_input_imgs=input_imgs.to(device)
        target
        dev_out=model(dev_input_imgs)
        prediction_fscore, input_fscore = evaluate_binarization_improvement(
            dev_input_imgs[0,0,:,:], dev_out[0,0,:,:], target[0,0,:,:])

