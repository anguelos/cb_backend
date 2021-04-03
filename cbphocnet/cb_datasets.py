from PIL import Image
import numpy as np
import torch
import json
import torchvision
import unidecode
import string

from .phoc import build_phoc_descriptor


class CBDataset(object):
    @staticmethod
    def compile(ids , id2gt, id2img, fixed_size, pyramids, unigrams, input_channels):
        #build_phoc_descriptor
        data = []
        for id in ids:
            gt = json.load(open(id2gt[id]))
            #page = torch.tensor(np.array(Image.open(id2img[id]),dtype=np.float) / 255.)
            #page = page.transpose(0, 2).transpose(1, 2)
            page = Image.open(id2img[id])
            word_imgs = []
            word_captions = []
            for n in range(len(gt["captions"])):
                caption = gt["captions"][n]
                if caption.startswith("W@"):
                    caption = unidecode.unidecode(caption[2:])
                    l, t, r, b = gt["rectangles_ltrb"][n]
                    word = page.crop((l, t, r, b))
                    if fixed_size is not None:
                        word = word.resize(fixed_size)
                    word = torch.tensor(np.array(word, dtype=np.float) / 255.)
                    word = word.transpose(0, 2).transpose(1, 2)
                    word_imgs.append(word)
                    word_captions.append(caption)
            phocs = torch.tensor(build_phoc_descriptor(word_captions, phoc_unigrams=unigrams, unigram_levels=pyramids))
            data = data + [(word_imgs[n],phocs[n,:]) for n in range(len(word_imgs))]
        return data

    def __init__(self,input_channels, img_paths, gt_paths, train=True, phoc_pyramids=[1,2,3,4,5,7,11], fixed_size=None,
                 unigrams=string.digits+string.ascii_lowercase):
        id2gt = {f.split("/")[-1].split(".")[0]:f for f in gt_paths}
        id2img = {f.split("/")[-1].split(".")[0]: f for f in img_paths}
        ids = [pageid for n, pageid in enumerate(sorted(id2gt.keys())) if n % 5 != train]
        self.data = CBDataset.compile(ids, id2gt, id2img,fixed_size=fixed_size,pyramids=phoc_pyramids,
                                      unigrams=unigrams, input_channels=input_channels)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
