import PIL.Image
from PIL import Image
import numpy as np
import torch
import json
import unidecode
import string
import glob
from random import Random
import collections
import os
import tqdm

from .phoc import build_phoc_descriptor


class CBDatasetFiles(object):
    def __init__(self, image_files, gt_files, fixed_size=(0, 0)):
        pass

class CBDataset(object):
    "wget http://rr.visioner.ca/assets/cbws/fake_db.tar.bz2"
    @staticmethod
    def compile(id2gt, id2img, fixed_size, pyramids, unigrams, input_channels, keep_singletons, pad_mode):
        data = []
        for id in tqdm.tqdm(id2gt.keys(), desc=f"Compiling Ds"):
            gt = json.load(open(id2gt[id]))
            page = Image.open(id2img[id])
            if input_channels == 1:
                page=page.convert('L')
            elif input_channels == 3:
                page = page.convert('RGB')
            else:
                raise ValueError
            word_imgs = []
            word_captions = []
            for n in range(len(gt["captions"])):
                caption = gt["captions"][n]
                if caption.startswith("W@"):
                    caption = unidecode.unidecode(caption[2:].lower())
                    l, t, r, b = gt["rectangles_ltrb"][n]
                    word = page.crop((l, t, r, b))
                    if fixed_size is not None and pad_mode=="scale":
                        word = word.resize(fixed_size)
                    elif fixed_size is not None and pad_mode=="padcropscale":
                        in_w, in_h = word.size
                        out_w, out_h = fixed_size
                        if in_w/out_w > 1 or in_h/out_w > 1:
                            ratio = min(out_h/in_h, out_w/out_h)
                            word = word.resize((int(word.size[0]*ratio), int(word.size[1]*ratio)))
                        in_w, in_h = word.size
                        assert in_w <= out_w and in_h <= out_h
                        left, top = (out_w-in_w)//2, (out_h - in_h) // 2
                        out_word = PIL.Image.new(mode=word.mode, size=fixed_size)
                        out_word.paste(word, (left, top))
                        word = out_word
                    else:
                        raise NotImplemented
                    word = torch.from_numpy(np.array(word, dtype=np.float) / 255.).float()
                    if len(word.size()) == 2:
                        word = word.unsqueeze(dim=2)
                    word = word.transpose(0, 2).transpose(1, 2)
                    word_imgs.append(word)
                    word_captions.append(caption)
            phocs = build_phoc_descriptor(word_captions, phoc_unigrams=unigrams, unigram_levels=pyramids)
            phocs = torch.from_numpy(phocs).float()
            data = data + [(word_imgs[n], phocs[n,:], word_captions[n]) for n in range(len(word_imgs))]
        if keep_singletons==True:
            return [d[:2] for d in data]
        else:
            occurrences = collections.Counter([d[2] for d in data])
            res = [d[:2] for d in data if occurrences[d[2]]>1]
            #print(f"Words:{len(data)}, Unique:{len(occurrences)}, Non singleton:{len([k for k,v in occurrences.items() if v>1])}  keeping: {len(res)}")
            return res

    def __init__(self, img_paths=[], gt_paths=[], img_glob="", gt_glob="", cache_fname="/tmp/cb_ds_P{}_T{}_C{}_{}x{}.pt", net=None, train=True, input_channels=3, phoc_pyramids=[1,2,3,4,5,7,11], fixed_size=None,
                 unigrams=string.digits+string.ascii_lowercase, max_items=-1, keep_singletons=True, pad_mode="padcropscale"):
        assert pad_mode in ["padcropscale", "scale"]
        if net is not None:
            phoc_pyramids = net.params["unigram_pyramids"]
            unigrams = net.params["unigrams"]
            input_channels = net.params["input_channels"]
            fixed_size = net.params["fixed_size"]
        if cache_fname:
            p = 'h'.join([str(h) for h in phoc_pyramids])
            if fixed_size is None:
                width, height = [0,0]
            else:
                width, height = fixed_size
            cache_fname = cache_fname.format(p, int(train),input_channels, width,height)
        if os.path.exists(cache_fname):
            self.data = torch.load(cache_fname)
        else:
            if img_paths == gt_paths == []:
                img_paths = glob.glob(img_glob)
                gt_paths = glob.glob(gt_glob)
            # if test_glob:
            #     if train: # test_glob and test
            #         gt_paths = sorted(set(gt_paths) - set(glob.glob(test_glob)))
            #         img_paths = sorted(set(img_paths) - set(glob.glob(test_glob)))
            #     else: # test_glob and train
            #         gt_paths = sorted(set(gt_paths).intersection(set(glob.glob(test_glob))))
            #         img_paths = sorted(set(img_paths).intersection(set(glob.glob(test_glob))))
            #     id2gt = {f.split("/")[-1].split(".")[0]: f for f in gt_paths}
            #     id2img = {f.split("/")[-1].split(".")[0]: f for f in img_paths}
            # else: # no glob defining test set
            #     keep_page_ids = sorted([f.split("/")[-1].split(".")[0] for f in gt_paths])
            #     Random(0).shuffle(keep_page_ids)
            #     if train:
            #         keep_page_ids=keep_page_ids[:len(keep_page_ids)//5]
            #     else:
            #         keep_page_ids = keep_page_ids[len(keep_page_ids) // 5:]
            id2gt = {f.split("/")[-1].split(".")[0]: f for f in gt_paths}
            id2img = {f.split("/")[-1].split(".")[0]: f for f in img_paths}
            #     id2gt = {k:id2gt[k] for k in keep_page_ids}
            #     id2img = {k:id2img[k] for k in keep_page_ids}

            self.data = CBDataset.compile(id2gt=id2gt, id2img=id2img, fixed_size=fixed_size,pyramids=phoc_pyramids,
                                          unigrams=unigrams, input_channels=input_channels,keep_singletons=keep_singletons, pad_mode=pad_mode)
            Random(1).shuffle(self.data)
            if max_items > 0:
                self.data = self.data[:max_items]
            if cache_fname:
                torch.save(obj=self.data, f=cache_fname)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
