import cv2
import numpy as np
from collections import namedtuple
import torch
import tqdm

from matplotlib import pyplot as plt

def connected_component_labeling(bin_img):
    integrable = torch.zeros([bin_img.size(0),bin_img.size(1)+1], dtype=torch.int)
    integrable[:,1:-1] = ((bin_img[:, 1:]!=bin_img[:, :-1])) * 1
    integrable[:,0] = bin_img[:,0] * 1
    integrable[:, -1] = bin_img[:, -1] * 1

    #integrable[:, 1:-1] += ((bin_img[:, :-1] - bin_img[:, 1:]) == 1).int() * -1
    #integrable[:, -1] = 2#bin_img[:, -1] * -1

    v_connected = (integrable.view(-1).contiguous().cumsum(dim=0).view(integrable.size()))#
    v_connected = v_connected[:, :-1]
    v_connected = v_connected * bin_img.long()

    h_touching_right = torch.zeros_like(bin_img)
    h_touching_right[:, 1:] = (bin_img[:, 1:] * bin_img[:, :-1]).byte()

    h_touching_left = torch.zeros_like(bin_img)
    h_touching_left[:, :-1] = (bin_img[:, :-1] * bin_img[:, 1:]).byte()

    to_connect = torch.cat([v_connected[h_touching_left].view(-1, 1), v_connected[h_touching_right].view(-1, 1)], dim=1)
    keep_idx = torch.ones(to_connect.size(0), dtype=torch.uint8)
    keep_idx[1:]=1-(to_connect[1:,0]==to_connect[:-1,0])*(to_connect[1:,1]==to_connect[:-1,1])
    to_connect=to_connect[keep_idx,:]
    
    labels_map = torch.arange(v_connected.max()+1,dtype=torch.int32)


def get_component_ds(dibco_ds,unet,device):
    unet = unet.train().to(device)
    result_ds=[]
    with torch.no_grad():
        for n in tqdm.tqdm(range(len(dibco_ds))):
            input,gt, _ = dibco_ds[n]
            gt=gt.byte().numpy()[1,:,:]
            prediction = unet(input.unsqueeze(dim=0).to(device)).cpu()
            prediction = torch.nn.functional.softmax(prediction, dim=1)[0,1,:,:].cpu().numpy()

            bin_img = (prediction>.5)
            #from matplotlib import pyplot as plt;plt.imshow(prediction);plt.colorbar();plt.show()
            #from matplotlib import pyplot as plt;plt.imshow(bin_img);plt.colorbar();plt.show()
            img_height, img_width = gt.shape
            img_height, img_width = float(img_height), float(img_width)
            img_surface = img_height * img_width
            img_density = bin_img.sum() / img_surface

            components=get_components(bin_img)
            #plot_components(components);plt.show()
            fscores,precisions,recalls = get_component_fscore(components,gt)

            total_fscores, total_precisions, total_recalls = fscores[0], precisions[0], recalls[0]
            fscores, precisions, recalls = fscores[1:], precisions[1:], recalls[1:]
            delta_fscores, delta_precisions, delta_recalls = fscores-total_fscores, precisions-total_precisions, recalls-total_recalls

            nb_pixels = components.nb_pixels.astype("float")[1:]
            left=components.left.astype("float")[1:]
            top = components.top.astype("float")[1:]
            right=components.right.astype("float")[1:]
            bottom = components.bottom.astype("float")[1:]
            centroid_x = components.x_centroid.astype("float")[1:]
            centroid_y = components.y_centroid.astype("float")[1:]
            probabilities = components.probabilities.astype("float")[1:]

            widths = 1+right - left
            heights = 1 + bottom - top
            surfaces = widths * heights
            densities = nb_pixels/surfaces

            relative_centroid_x = (left + right) / 2 -  centroid_x
            relative_centroid_y = (top + bottom) / 2 - centroid_y

            relative_left = left/img_width
            relative_right = right / img_width
            relative_top = left/img_height
            relative_bottom = right / img_height
            relative_widths = widths/img_width
            relative_heights = heights / img_height
            relative_surfaces = surfaces / img_surface
            relative_densities = densities / img_density
            features = [nb_pixels,left,top,right,bottom,centroid_x,centroid_y,probabilities,widths,heights,
                        surfaces,densities, relative_centroid_x,relative_centroid_y,relative_left,relative_top,
                        relative_right,relative_bottom,relative_widths,relative_heights,relative_surfaces,
                        relative_densities]
            features=np.array(features)
            print("Features Size:",features.shape)
            for n in range(features.shape[0]):
                sample_features = torch.Tensor(features[n])
                sample_outputs = torch.Tensor([delta_fscores[n]>0,fscores[n],precisions[n],recalls[n],delta_fscores[n],delta_precisions[n],delta_recalls[n]])
            result_ds.append((sample_features, sample_outputs))

    return result_ds







def get_components(bin_img, device='cuda', fg_prob=None):
    if fg_prob is None:
        fg_prob = torch.ones(bin_img.shape)
    fg_logits=torch.log(fg_prob)
    nb_labels, labels = cv2.connectedComponents(bin_img.astype("uint8") * 255)
    nb_labels = labels.max()+1 # occasionally nb_lables is wrong
    #from matplotlib import pyplot as plt;plt.imshow(labels);plt.colorbar();plt.show()
    labels = torch.Tensor(labels).int()
    height, width = bin_img.shape

    x_field = torch.ones(bin_img.shape, dtype=torch.int32) * torch.arange(bin_img.shape[1], dtype=torch.int32).view(1, -1)
    y_field = torch.ones(bin_img.shape, dtype=torch.int32) * torch.arange(bin_img.shape[0], dtype=torch.int32).view(-1, 1)
    x_field = labels * width + x_field
    y_field = labels * height + y_field

    sorted_xfield = torch.sort(x_field.view(-1).to(device))[0]
    sorted_yfield = torch.sort(y_field.view(-1).to(device))[0]

    comp_id = (sorted_xfield / width).view(-1)#.int()
    X = sorted_xfield % width
    Y = sorted_yfield % height
    #comp_id = (sorted_yfield / width).astype(np.int32) as good
    change_idx=torch.zeros(nb_labels+1,dtype=torch.int32).view(-1)
    change_idx[1:-1]=(torch.nonzero(comp_id[1:] - comp_id[:-1]).view(-1))+1
    change_idx[-1]=comp_id.size(0)#-1
    change_idx=change_idx.view(-1)
    component_start=[change_idx[n].item() for n in range(nb_labels)]
    component_end = [change_idx[n].item() for n in range(1,nb_labels+1)]
    component_X = np.array([X[component_start[n]:component_end[n]].cpu().numpy() for n in range(nb_labels)])
    component_Y = np.array([Y[component_start[n]:component_end[n]].cpu().numpy() for n in range(nb_labels)])


    pixelcount = torch.Tensor(component_end)-torch.Tensor(component_start)

    # sorting by component size
    _, sorted_idx = torch.sort(pixelcount)
    sorted_idx = sorted_idx.flip(0)
    component_X = component_X[sorted_idx]
    component_Y = component_Y[sorted_idx]
    pixelcount = pixelcount[sorted_idx]

    # creating bbox features
    left= np.empty(pixelcount.size(),dtype=np.int32)
    right = np.empty(pixelcount.size(),dtype=np.int32)
    top = np.empty(pixelcount.size(),dtype=np.int32)
    bottom = np.empty(pixelcount.size(),dtype=np.int32)
    centroid_x = np.empty(pixelcount.size(),dtype=np.float32)
    centroid_y = np.empty(pixelcount.size(), dtype=np.float32)
    component_prob = np.empty(pixelcount.size(), dtype=np.float32)
    for n in range(nb_labels):
        left[n] = component_X[n].min()
        right[n] = component_X[n].max()
        centroid_x[n] = component_X[n].mean()

        top[n] = component_Y[n].min()
        bottom[n] = component_Y[n].max()
        centroid_y[n] = component_Y[n].mean()

        component_prob[n]=torch.exp(fg_logits[component_Y[n], component_X[n]].mean())

    Components = namedtuple('Components', ['bin_img', 'label_img', 'nb_components', 'nb_pixels', 'x_coords', 'y_coords','left','right','top','bottom','x_centroid','y_centroid','probabilities'])
    return Components(bin_img=bin_img, label_img=labels, nb_components=nb_labels, nb_pixels=pixelcount.numpy(), x_coords=component_X, y_coords=component_Y, left=left, right=right, top=top, bottom=bottom, x_centroid=centroid_x, y_centroid=centroid_y,probabilities=component_prob)


def get_component_fscore(components, gt):
    """Measures the effect each connected component has to a binarization FScore.

    :param components: A named tuple with components.
    :param gt: A pytorch ByteTensor containing a binary image with the binarization groundtruth.
    :return: three vectors containing the fscores, precisions, and recalls if any component was removed.
        The zeros component contains the total fscore, precision, and recall.
    """

    tp = (gt * components.bin_img).sum()
    selected = components.bin_img.sum()
    relevant = gt.sum()

    total_precision = tp/float(selected)
    total_recall = tp / float(relevant)
    total_fscore = (2 * total_precision * total_recall) / (total_recall + total_precision)
    precisions = np.zeros_like(components.probabilities)
    recalls = np.zeros_like(components.probabilities)
    fscores = np.zeros_like(components.probabilities)
    precisions[0] = total_precision
    recalls[0] = total_recall
    fscores[0] = total_fscore
    for n in range(1,components.nb_components):
        pixels = gt[components.y_coords[n],components.x_coords[n]]
        correct = pixels.sum()
        precision = (tp - correct) / (relevant - correct)
        recall = (tp - correct) / (selected - pixels.size)
        fscore = (2 * precision * recall) / (recall + recall)
        precisions[n] = precision
        recalls[n] = recall
        fscores[n] = fscore
    return fscores, precisions, recalls


def filter_components(components):
    width = 1 + components.right-components.left
    height = 1 + components.bottom - components.top
    surface = width*height
    longality = width/height
    bbox_centre_x = (components.right + components.left) / 2
    bbox_centre_y = (components.top - components.bottom) / 2
    offcenter_x = .5*(components.centroid_x - bbox_centre_x)/width
    offcenter_y = .5 * (components.centroid_y - bbox_centre_y) / height


def erase_components(components, erase_labels):
    for erase_label in erase_labels:
        components.label_img[components.y_coords[erase_label], components.x_coords[erase_label]] = 0
    keep = np.ones(components.nb_components, np.bool)
    keep[erase_labels] =0
    new_labels = np.zeros(components.nb_components)  # np.arange(components.nb_components)

    components.nb_components = keep.sum()
    new_labels[keep]=np.arange(components.nb_components)
    components.label_img[:, :]=new_labels[components.label_img[:, :]]
    components.bin_img[:, :] = components.label_img > 0

    components.x_coords = components.x_coords[keep]
    components.y_coords = components.y_coords[keep]

    components.left = components.left[keep]
    components.right = components.right[keep]
    components.top = components.top[keep]
    components.bottom = components.bottom[keep]

    components.x_centroid = components.x_centroid[keep]
    components.y_centroid = components.y_centroid[keep]
    components.nb_pixels = components.nb_pixels[keep]
    components.probability = components.nb_pixels[keep]


def print_components(components, id_list=None):
    if id_list is None:
        id_list=list(range(components.nb_components))

    for n in id_list:
        descr="ID:{}\nSize:{}\nBBox:[{},{},{},{}]\nCentroid:[{:02f},{:02f}]\nProbabillity: {:05f}\n".format(n,components.nb_pixels[n],
                                                                                      components.left[n],components.top[n],components.right[n],components.bottom[n],
                                                                                      components.x_centroid[n],components.y_centroid[n],components.probabillities[n])
        print(descr)

def plot_components(components,id_list=None):
    plt.figure()
    plt.imshow(components.label_img)

    if id_list is None:
        id_list=list(range(components.nb_components))

    for n in id_list:
        X = components.left[n],components.left[n],components.right[n],components.right[n],components.left[n]
        Y = components.bottom[n], components.top[n], components.top[n], components.bottom[n], components.bottom[n]
        plt.plot(X,Y)
    plt.plot(components.x_centroid ,components.y_centroid,"*")
