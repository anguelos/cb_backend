import numpy as np
import torch
try:
    import fastai
    import fastai.vision
except ImportError:
    print("could not load fastai: dynamic unets not available")
from .unet import UNet
from .unet2 import R2AttU_Net, AttU_Net, R2U_Net, U_Net
import iunets
import time
import tqdm
import torchvision
from matplotlib import pyplot as plt
from PIL import Image, ImageOps

import torch
import torch.nn.functional as F
import types
import PIL
from .dibco import dibco_transform_color_input, dibco_transform_gt, dibco_transform_gray_input
from .otsu import OtsuPtPIL


class Discriminator(torch.nn.Module):
    def __init__(self, ngpu, ndf = 64, nc =1):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = torch.nn.Sequential(
            # input is (nc) x 64 x 64
            torch.nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            torch.nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 2),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            torch.nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 4),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            torch.nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ndf * 8),
            torch.nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            torch.nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            torch.nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

class SingleImageDataset(object):
    def __init__(self, image_filename_list, transform=dibco_transform_gray_input, gt_transform=dibco_transform_gt,  cache_images=False, add_mask=0):
        self.image_filenames = image_filename_list
        self.cache_images = cache_images
        self.transform = transform
        self.gt_transform = gt_transform
        self.add_mask = add_mask

        if self.cache_images:
            self.cache=[]
            for filename in self.image_filenames:
                self.cache.append(PIL.Image.open(filename))
        else:
            self.last_img_idx = -1

    def __getitem__(self, item):
        if self.cache_images:
            res = self.cache[item]
        elif self.last_img_idx == item:
            res = self.last_img
        else:
            img = PIL.Image.open(self.image_filenames[item])
            self.last_img = img
            res = self.last_img
        res = self.transform(res), self.gt_transform(res)
        if self.add_mask:
            res=res+(torch.ones_like(res[0][:1,:,:]),)
        return res

    def __len__(self):
        return len(self.image_filenames)


class TiledDataset(object):
    def _create_idx(self):
        image_in_sample = self.is_image_in_sample.find(True)
        patch_idx = 0
        self.patch_idx = []
        self.sample_idx = []
        self.left = []
        self.top = []
        self.sizes=[]
        self.inverse_starts = []
        self.inverse_ends=[]
        total_counter=0
        for img_idx,sample in enumerate(self.dataset):
            self.inverse_starts.append(total_counter)
            img = sample[image_in_sample]
            sz = img.size
            self.sizes.append(sz)
            self.patch_idx = 0
            for horiz in range(0, sz[0], self.tile_size[0]):
                for vert in range(0, sz[1], self.tile_size[1]):
                    self.left.append(horiz)
                    self.top.append(vert)
                    self.patch_idx.append(patch_idx)
                    self.sample_idx.append(img_idx)
                    patch_idx += 1
                    total_counter += 1
            self.inverse_ends.append(total_counter)

    def __init__(self, dataset, is_image_in_sample=None, tile_size=256, ltrb_pad=(64,64,64,64), input_transform=None, output_transform=None):
        if is_image_in_sample is None:
            is_image_in_sample = [isinstance(datum, PIL.Image) for datum in dataset[0]]
        self.dataset = dataset
        self.is_image_in_sample = is_image_in_sample
        self.tile_size = tile_size
        self.ltrb_pad = ltrb_pad
        self._create_idx()

    def __getitem__(self, item):
        sample = self.ds[self.sample_idx[item]]
        res_sample = []
        for n, datum in enumerate(sample):
            if self.is_image_in_sample[n]:
                patch = datum.crop((self.left[item]-self.ltrb_pad[0], self.top[item]-self.ltrb_pad[1], self.left[item] + self.tile_size[0]+self.ltrb_pad[1], self.top[item] + self.tile_size[1]+self.ltrb_pad[1]))
                res_sample.append(patch)
            else:
                res_sample.append(datum)
        return res_sample

    def __len__(self):
        return len(self.sample_idx)

    def sample_as_list(self, n_sample):
        res=[]
        for n in range(self.inverse_starts[n_sample], self.inverse_ends[n_sample]):
            res.append(self[n])
        return res

    def stich_image_tensors(self, image_list, sz):
        image_stack = image_list.copy()
        out_image = []
        for horiz in range(0, sz[0], self.tile_size[0]):
            column = []
            for vert in range(0, sz[1], self.tile_size[1]):
                patch = image_stack.pop(0)
                left,top,right,bottom = self.ltrb_pad[0], self.ltrb_pad[1],patch.size(2)-self.ltrb_pad[2], patch.size(3)-self.ltrb_pad[3]
                patch = patch[:,top:bottom,left:right]
                column.append(patch)
            column=torch.cat(column,dim=2)
            out_image.append(column)
        out_image = torch.cat(out_image, dim=3)
        return out_image[:, :sz[1], :sz[0]]


    def apply_network(self, network, sample_pos=0, datum_idx=0):
        images = [sample[datum_idx] for sample in self.sample_as_list(sample_pos)]




def validate_epoch(p, device, loader, net, save_images=True, save_input_images=False, file_prefix="val"):
    with torch.set_grad_enabled(False):
        net = net.to(device)
        net.eval()
        fscores = []
        precisions = []
        recalls = []
        losses = []
        t=time.time()
        for n, data in tqdm.tqdm(enumerate(loader)):
            if len(data) == 3:
                (input_img, gt, mask) = data
            elif len(data) == 2:
                input_img, gt = data
                mask = torch.ones_like(gt[:,:1,:,:])
            input_img, gt, mask = input_img.to(device), gt.to(device), mask.to(device)

            prediction = net(input_img)
            prediction = torch.nn.functional.softmax(prediction, dim=1)
            loss = loss.sum()
            confusion, precision, recall, fscore = render_confusion(prediction[0,0,:,:]<prediction[0,1,:,:], gt[0, 0, :, :] < .5, mask[0,0,:,:] > .5)

            fscores.append(fscore)
            precisions.append(precision)
            recalls.append(recall)
            losses.append(loss.item()/gt.view(-1).size()[0])
            if save_images:
                conf_img=Image.fromarray((confusion[:,:,[2,1,0]]).astype("uint8")).convert('RGB')
                conf_img.save(f"/tmp/{file_prefix}_{n}_conf.png")
                prediction = torch.softmax(prediction[0, :, :, :], dim=0)
                Image.fromarray((prediction[0,:,:].detach().cpu().numpy() * 255)).convert('RGB').save(f"/tmp/{file_prefix}_{n}_output.png")
                Image.fromarray((mask[0, 0, :, :].detach().cpu().numpy() * 255)).convert('RGB').save(f"/tmp/{file_prefix}_{n}_mask.png")
                Image.fromarray((gt[0, 1, :, :].detach().cpu().numpy() * 255)).convert('RGB').save(f"/tmp/{file_prefix}_{n}_gt.png")
            if save_input_images:
                torchvision.transforms.ToPILImage()(input_img[0,:,:,:].detach().cpu()).save(f"/tmp/{file_prefix}_{n}_input.png")
    return sum(fscores) / len(fscores),sum(precisions) / len(precisions), sum(recalls) / len(recalls),sum(losses) / len(losses)









def patch_forward(net, img, patch_width, patch_height):
    padded_img = torch.zeros([img.size(0), img.size(1), (1+img.size(2)//patch_height)*patch_height,(1+img.size(3)//patch_width)*patch_width],device=img.device)
    print("Padded:", padded_img.size())
    padded_img[:, :, :img.size(2), :img.size(3)]=img
    res = torch.zeros([img.size(0), 2, padded_img.size(2), padded_img.size(3)], device=img.device)
    for left in range(0, padded_img.size(3), patch_width):
        for top in range(0, padded_img.size(2), patch_height):
            patch = padded_img[:, :, top:top+patch_height, left:left+patch_width]
            print("patch:",patch.size())
            out_patch = net(patch)
            res[:, :, top:top+patch_height, left:left+patch_width] = out_patch
    return res[:, :, :res.size(2), :res.size(3)]


def create_net(arch,n_channels,n_classes, bn_momentum,rnd_pad, pretrained=True):
    if arch == "dunet34":
        print("Creating Resnet")
        body = fastai.vision.learner.create_body(fastai.vision.models.resnet34, pretrained=pretrained, cut=-2)
        print("Creating Unet")
        net = fastai.vision.models.unet.DynamicUnet(body, n_classes)
        print("Done")
        if bn_momentum <0:
            bn_momentum=None
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.momentum = bn_momentum
    elif arch == "runet":
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.in2u=torch.nn.Conv2d(n_channels, 64, 1)
                self.unet=iunets.iUNet(in_channels=64, architecture=[3,3,4,4,5],dim=2)
                self.u2out=torch.nn.Conv2d(64, n_classes, 1)

            def forward(self, x):
                x=self.in2u(x)
                x= self.unet(x)
                x= self.u2out(x)
                return x
        net=Network()
    elif arch == "srunet":
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.in2u=torch.nn.Conv2d(n_channels, 32, 1)
                self.unet=iunets.iUNet(in_channels=32, architecture=[3,3,3,3],dim=2)
                self.u2out=torch.nn.Conv2d(32, n_classes, 1)

            def forward(self, x):
                x=self.in2u(x)
                x= self.unet(x)
                x= self.u2out(x)
                return x
        net=Network()
    elif arch == "wrunet":
        class Network(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.in2u=torch.nn.Conv2d(n_channels, 128, 1)
                self.unet=iunets.iUNet(in_channels=128, architecture=[3,3,4,5],dim=2)
                self.u2out=torch.nn.Conv2d(128, n_classes, 1)

            def forward(self, x):
                x=self.in2u(x)
                x= self.unet(x)
                x= self.u2out(x)
                return x
        net=Network()
    elif arch == "dunet50":
        print("Creating Resnet")
        body = fastai.vision.learner.create_body(fastai.vision.models.resnet50, pretrained=pretrained, cut=-2)
        print("Creating Unet")
        net = fastai.vision.models.unet.DynamicUnet(body, n_classes)
        print("Done")
        if bn_momentum <0:
            bn_momentum=None
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.momentum = bn_momentum
    elif arch=="dunet18":
        print("Creating Resnet")
        body = fastai.vision.learner.create_body(fastai.vision.models.resnet18, pretrained=pretrained, cut=-2)
        print("Creating Unet")
        net = fastai.vision.models.unet.DynamicUnet(body, n_classes)
        print("Done")
        if bn_momentum <0:
            bn_momentum=None
        for m in net.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.momentum = bn_momentum
    elif arch== "unet":
        net = UNet(n_channels=n_channels, n_classes=n_classes)
    elif arch == "R2AttUNet":
        net = R2AttU_Net(img_ch=n_channels, output_ch=n_classes)
    elif arch == "AttUNet":
        net = AttU_Net(img_ch=n_channels, output_ch=n_classes)
    elif arch == "R2UNet":
        net = R2U_Net(img_ch=n_channels, output_ch=n_classes)
    elif arch == "UNet":
        net = U_Net(img_ch=n_channels, output_ch=n_classes)
    elif arch == "otsu":
        net = OtsuPtPIL(rgb_to_gray=True,n_outputs=2)
    else:
        raise ValueError("arch must be either dunet34, dunet50, dunet18, or unet")
    return net


def save(param_hist, per_epoch_train_errors,per_epoch_validation_errors,epoch,net):
    p=param_hist[sorted(param_hist.keys())[-1]]
    save_dict = {"net_state":net.state_dict()} # version2
    save_dict["param_hist"]=param_hist
    save_dict["per_epoch_train_errors"]=per_epoch_train_errors
    save_dict["per_epoch_validation_errors"] = per_epoch_validation_errors
    save_dict["epoch"]=epoch
    torch.save(save_dict, p["resume_fname"])
    if p["archive_nets"]:
        folder="/".join(p["resume_fname"].split("/")[:-1])
        if folder == "":
            folder = "."
        torch.save(save_dict, f"{folder}/{p['arch']}_{epoch:05}.pt")

def resume(net,resume_fname, device):
    try:
        save_dict=torch.load(resume_fname,map_location=device)
        if "param_hist" in save_dict.keys():
            param_hist=save_dict["param_hist"]
            del save_dict["param_hist"]
        else:
            param_hist={}
        per_epoch_train_errors=save_dict["per_epoch_train_errors"]
        del save_dict["per_epoch_train_errors"]
        per_epoch_validation_errors=save_dict["per_epoch_validation_errors"]
        del save_dict["per_epoch_validation_errors"]
        start_epoch=save_dict["epoch"]
        del save_dict["epoch"]
        if "net_state" in save_dict.keys():
            net.load_state_dict(save_dict["net_state"])
        else:
            net.load_state_dict(save_dict)
        print("Resumed from ",resume_fname)
        return param_hist, per_epoch_train_errors, per_epoch_validation_errors,start_epoch,net
    except FileNotFoundError as e:
        print("Failed to resume from ", resume_fname)
        return {}, {}, {}, 0, net


def render_confusion(prediction,gt,valid_mask=None,tp_col=[0, 0, 0],tn_col=[255,255,255],fp_col=[255,0,0],fn_col=[0,0,255], undetermined_col=[128,128,128]):
    prediction=(prediction.cpu().numpy()>.5)
    gt = (gt.cpu().numpy()>.5)
    res=np.zeros(prediction.shape+(3,))
    if valid_mask is not None:
        valid_mask = valid_mask.cpu().numpy()
        tp = gt & prediction & valid_mask
        tn = (~gt) & (~prediction) & valid_mask
        fp = (~gt) & prediction & valid_mask
        fn = (gt) & (~prediction) & valid_mask
    else:
        tp = gt & prediction
        tn = (~gt) & (~prediction)
        fp = (~gt) & prediction
        fn = (gt) & (~prediction)
        valid_mask = np.ones_like(gt)
    res[~valid_mask, :] = undetermined_col
    res[tp, :] = tp_col
    res[tn, :] = tn_col
    res[fp, :] = fp_col
    res[fn, :] = fn_col
    precision = (1+tp.sum())/float(1+tp.sum()+fp.sum())
    recall = (1+tp.sum()) / float(1+tp.sum() + fn.sum())
    Fscore=(2*precision*recall)/(precision+recall)
    return res, precision, recall, Fscore
