import torch
import torch.nn as nn
import torch.nn.functional as F
from .gpp import GPP
from .phoc import build_phoc_descriptor
import unidecode
from PIL import Image
import numpy as np
import pathlib
import hashlib
from .cb_datasets import resize_word

def resume_embedder(fname, allow_fail=True, net=None):
    if net is not None and allow_fail and not pathlib.Path(fname).is_file():
        return net, {}
    store_data = torch.load(fname, map_location="cpu")
    if net is None:
        if "resize_mode" not in store_data["contructor_params"].keys():
            store_data["contructor_params"]["resize_mode"] = "padcropscale"
        net = eval(store_data["class_name"])(**store_data["contructor_params"])
    else:
        assert type(net).__name__ == store_data["class_name"]
        assert net.params == store_data["contructor_params"]
    net.load_state_dict(store_data["state_dict"])
    del store_data["state_dict"]
    del store_data["contructor_params"]
    del store_data["class_name"]
    return net, store_data


class Embedder(nn.Module):
    def __init__(self, unigrams, unigram_pyramids, fixed_size=None, input_channels=1, gpp_type='spp', pooling_levels=3, pool_type='max_pool', resize_mode="padcropscale"):
        super().__init__()
        if gpp_type not in ['spp', 'tpp', 'gpp']:
            raise ValueError('Unknown pooling_type. Must be either \'gpp\', \'spp\' or \'tpp\'')
        self.resize_mode = resize_mode
        self.unigrams = unigrams
        self.unigram_pyramids = unigram_pyramids
        self.fixed_size = fixed_size
        self.params = {"unigrams": unigrams, "unigram_pyramids": unigram_pyramids, "fixed_size": fixed_size,
                       "input_channels": input_channels, "gpp_type": gpp_type, "pooling_levels": pooling_levels,
                       "pool_type": pool_type, "resize_mode": resize_mode}

    def retrieval_distance_metric(self):
        raise NotImplemented()

    def arch_hash(self):
        raise NotImplemented()
        #return hashlib.md5(("Embedder"+repr(sorted(self.params.items()))).encode("utf-8")).hexdigest()

    def forward(self, x):
        raise NotImplemented()

    def save(self, fname, **kwargs):
        store_data = {k: v for k, v in kwargs.items()}
        store_data["class_name"] = type(self).__name__
        store_data["contructor_params"] = self.params
        store_data["state_dict"] = self.state_dict()
        torch.save(store_data, fname)

    @classmethod
    def resume(cls, fname, allow_fail=True, net=None):
        if net is not None and allow_fail and not pathlib.Path(fname).is_file():
            return net, {}
        store_data = torch.load(fname, map_location="cpu")
        if net is None:
            net = cls(**store_data["contructor_params"])
        else:
            assert net.params == store_data["contructor_params"]
        net.load_state_dict(store_data["state_dict"])
        del store_data["state_dict"]
        del store_data["contructor_params"]
        del store_data["class_name"]
        return net, store_data


    def embed_strings(self, words):
        words = [unidecode.unidecode(w) for w in words]
        return build_phoc_descriptor(words, self.unigrams, self.unigram_pyramids)

    def embed_image(self, word_image: Image, device):
        word_image = resize_word(word_image, fixed_size = self.fixed_size, pad_mode=self.resize_mode)
        if self.params["input_channels"] == 1:
            word_img = torch.from_numpy(np.array(word_image.convert("LA"))[:, :, 0]).float().to(device)
            word_img = word_img.unsqueeze(dim=2)
        else:
            word_img = torch.from_numpy(np.array(word_image.convert("RGB"))).float().to(device)
        word_img = word_img.transpose(0, 2).transpose(1, 2) / 255.
        dl = torch.utils.data.DataLoader([[word_img, 0]]) # we use a data loader because under some conditions raw tensors can cause a memory leak.
        embeddings = []
        for data, _ in dl:
            embeddings.append(torch.sigmoid(self(data)).detach().cpu().numpy())
        return embeddings[0]

    # def embed_rectangles(self, img: Image, ltrb: np.array, device: str, batch_size: int):
    #     with torch.no_grad():
    #         if self.params["input_channels"] == 1:
    #             page = torch.from_numpy(np.array(img.convert("LA"))[:, :, 0]).float().to(device)
    #             page = page.unsqueeze(dim=2)
    #         else:
    #             page = torch.from_numpy(np.array(img.convert("RGB"))).float().to(device)
    #         page = page.transpose(0, 2).transpose(1, 2)
    #         dataset = []
    #         boxes = ltrb.tolist()
    #         for left, top, right, bottom in boxes:
    #             right_end = min(right + 1, page.size(2))
    #             bottom_end = min(bottom + 1, page.size(1))
    #             dataset.append((page[:, top:bottom_end, left:right_end], torch.tensor([left, top, right, bottom])))
    #         self.to(device)
    #         self.train(False)
    #         dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    #         embedings = []
    #         rectangles = []
    #         for data, boxes in dataloader:
    #             embedings.append(torch.sigmoid(self(data)).detach().cpu().numpy())
    #             rectangles.append(boxes.numpy())
    #         embedings = np.concatenate(embedings, axis=0)
    #         rectangles = np.concatenate(rectangles, axis=0)
    #         return rectangles, embedings

    def embed_rectangles(self, img: Image, ltrb: np.array, device: str, batch_size: int):
        with torch.no_grad():
            if self.params["input_channels"] == 1:
                page = img.convert("LA")
            else:
                page = img.convert("RGB")
            dataset = []
            boxes = ltrb.tolist()
            for left, top, right, bottom in boxes:
                word_img = page.crop((left, top, right, bottom))
                word_img = resize_word(word_img, fixed_size=self.fixed_size, pad_mode=self.resize_mode)
                if word_img.mode == "LA":
                    word_tensor = torch.from_numpy(np.array(word_img)).unsqueeze(dim=2).transpose(0, 2).transpose(1, 2).to(device)
                else:  # mode == "RGB"
                    word_tensor = torch.from_numpy(np.array(word_img)).transpose(0, 2).transpose(1, 2).to(device) /255.
                dataset.append((word_tensor, torch.tensor([left, top, right, bottom])))
            self.to(device)
            self.train(False)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
            embedings = []
            rectangles = []
            for data, boxes in dataloader:
                embedings.append(torch.sigmoid(self(data)).detach().cpu().numpy())
                rectangles.append(boxes.numpy())
            embedings = np.concatenate(embedings, axis=0)
            rectangles = np.concatenate(rectangles, axis=0)
            return rectangles, embedings


class PHOCNet(Embedder):
    def __init__(self, unigrams, unigram_pyramids, fixed_size=None, input_channels=1, gpp_type='spp', pooling_levels=3, pool_type='max_pool', resize_mode="padcropscale"):
        super().__init__(unigrams=unigrams, unigram_pyramids=unigram_pyramids, fixed_size=fixed_size, input_channels=input_channels,gpp_type=gpp_type,pooling_levels=pooling_levels, pool_type='max_pool')
        # some sanity checks
        # set up Conv Layers
        n_out = len(unigrams)*sum(unigram_pyramids)

        self.conv1_1 = nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv3_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        # create the spatial pooling layer
        self.pooling_layer_fn = GPP(gpp_type=gpp_type, levels=pooling_levels, pool_type=pool_type, input_channels=512)
        pooling_output_size = self.pooling_layer_fn.pooling_output_size
        self.fc5 = nn.Linear(pooling_output_size, 4096)
        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, n_out)
        self.params = {"unigrams": unigrams, "unigram_pyramids": unigram_pyramids, "fixed_size": fixed_size,
                       "input_channels": input_channels, "gpp_type": gpp_type, "pooling_levels": pooling_levels,
                       "pool_type": pool_type}

    def retrieval_distance_metric(self):
        return "cosine"

    def arch_hash(self):
        return hashlib.md5(("PHOCNet"+repr(sorted(self.params.items()))).encode("utf-8")).hexdigest()

    def forward(self, x):
        if x.size(2) < 8:
            x = F.pad(x, (0, 0, 4, 4))
        if x.size(3) < 8:
            x = F.pad(x, (4, 4, 0, 0))
        y = F.relu(self.conv1_1(x))
        y = F.relu(self.conv1_2(y))
        y = F.max_pool2d(y, kernel_size=2, stride=2, padding=0)
        y = F.relu(self.conv2_1(y))
        y = F.relu(self.conv2_2(y))
        y = F.max_pool2d(y, kernel_size=2, stride=2, padding=0)
        y = F.relu(self.conv3_1(y))
        y = F.relu(self.conv3_2(y))
        y = F.relu(self.conv3_3(y))
        y = F.relu(self.conv3_4(y))
        y = F.relu(self.conv3_5(y))
        y = F.relu(self.conv3_6(y))
        y = F.relu(self.conv4_1(y))
        y = F.relu(self.conv4_2(y))
        y = F.relu(self.conv4_3(y))
        y = self.pooling_layer_fn.forward(y)
        y = F.relu(self.fc5(y))
        y = F.dropout(y, p=0.5, training=self.training)
        y = F.relu(self.fc6(y))
        y = F.dropout(y, p=0.5, training=self.training)
        y = self.fc7(y)
        return y

    def init_weights(self):
        self.apply(PHOCNet._init_weights_he)

    @staticmethod
    def _init_weights_he(m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, (2. / n) ** (1 / 2.0))
        if isinstance(m, nn.Linear):
            n = m.out_features
            m.weight.data.normal_(0, (2. / n) ** (1 / 2.0))
            #nn.init.kaiming_normal(m.weight.data)
            nn.init.constant(m.bias.data, 0)


class ResnetBottleneck(torch.nn.Module):
    def __init__(self, channels_in, channels_out, bottleneck_sz=0, add_batch_norm=False):
        super().__init__()
        if bottleneck_sz == 0:
            bottleneck_sz = channels_out // 4
        if add_batch_norm:
            self.bottleneck = torch.nn.Sequential(
                torch.nn.Conv2d(channels_in, bottleneck_sz, 1),
                torch.nn.BatchNorm2d(bottleneck_sz),
                torch.nn.ReLU(),
                torch.nn.Conv2d(bottleneck_sz, bottleneck_sz, 3, padding=1),
                torch.nn.BatchNorm2d(bottleneck_sz),
                torch.nn.ReLU(),
                torch.nn.Conv2d(bottleneck_sz, channels_in, 1),
                torch.nn.BatchNorm2d(channels_out))
        else:
            self.bottleneck = torch.nn.Sequential(
                torch.nn.Conv2d(channels_in, bottleneck_sz, 1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(bottleneck_sz, bottleneck_sz, 3, padding=1),
                torch.nn.ReLU(),
                torch.nn.Conv2d(bottleneck_sz, channels_in, 1))
        if channels_out != channels_in:
            self.adapt = torch.nn.Conv2d(in_channels=channels_in, out_channels=channels_out, kernel_size=1)
        else:
            self.adapt = None

    def forward(self, x):
        x = F.relu(x + self.bottleneck(x))
        if self.adapt is None:
            return x
        else:
            return self.adapt(x)


class PHOCResNet(Embedder):
    def retrieval_distance_metric(self):
        return "cosine"

    def arch_hash(self):
        return hashlib.md5(("PHOCNet"+repr(sorted(self.params.items()))).encode("utf-8")).hexdigest()

    def __init__(self, unigrams, unigram_pyramids, fixed_size=None, input_channels=1, gpp_type='spp', pooling_levels=3, pool_type='max_pool', resize_mode="padcropscale"):
        super().__init__(unigrams=unigrams, unigram_pyramids=unigram_pyramids, fixed_size=fixed_size, input_channels=input_channels,gpp_type=gpp_type,pooling_levels=pooling_levels, pool_type='max_pool')
        n_out = len(unigrams)*sum(unigram_pyramids)

        self.intro_layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=input_channels, out_channels=256, kernel_size=7, stride=1, padding=3),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(3)
        )
        self.layers_256 = torch.nn.Sequential(
            ResnetBottleneck(256, 256),
            ResnetBottleneck(256, 256),
            ResnetBottleneck(256, 512),
        )
        self.layers_512 = torch.nn.Sequential(
            ResnetBottleneck(512, 512),
            ResnetBottleneck(512, 512),
            ResnetBottleneck(512, 512),
            ResnetBottleneck(512, 1024),
        )
        self.layers_1024 = torch.nn.Sequential(
            ResnetBottleneck(1024, 1024),
            ResnetBottleneck(1024, 1024),
            ResnetBottleneck(1024, 1024),
            ResnetBottleneck(1024, 1024),
            ResnetBottleneck(1024, 1024),
            ResnetBottleneck(1024, 1024),
        )
        self.pooling_layer_fn = GPP(gpp_type=gpp_type, levels=pooling_levels, pool_type=pool_type, input_channels=1024)
        pooling_output_size = self.pooling_layer_fn.pooling_output_size
        self.mlp = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(pooling_output_size, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=.5),
            torch.nn.Linear(4096, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=.5),
            torch.nn.Linear(4096, n_out))
        self.params = {"unigrams": unigrams, "unigram_pyramids": unigram_pyramids, "fixed_size": fixed_size,
                       "input_channels": input_channels, "gpp_type": gpp_type, "pooling_levels": pooling_levels,
                       "pool_type": pool_type}

    def forward(self, x):
        if x.size(2) < 8:
            x = F.pad(x, (0, 0, 4, 4))
        if x.size(3) < 8:
            x = F.pad(x, (4, 4, 0, 0))
        x = self.intro_layers(x)
        x = self.layers_256(x)
        x = self.layers_512(x)
        x = self.layers_1024(x)
        x = self.pooling_layer_fn.forward(x)
        x = self.mlp(x)
        return x
