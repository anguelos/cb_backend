import skimage
import skimage.filters
import torch
import kornia as K
from matplotlib import pyplot as plt

class OtsuPtPIL(torch.nn.Module):
    def __init__(self, rgb_to_gray=True, clamp=True,n_outputs=-1):
        super().__init__()
        self.rgb_to_gray = rgb_to_gray
        self.clamp = clamp
        self.n_outputs = n_outputs

    def forward(self, x):
        if x.size(1) == 3 and self.rgb_to_gray:
            img = K.color.rgb_to_grayscale(x)
        else:
            assert x.size(1) == 1
            img = x
        if self.n_outputs == 2 and img.size(1) == 1:
            img = torch.cat([img, 1 - img], dim=1)
        res = torch.empty_like(img)
        for batch_n in range(img.size(0)):
            for channel_n in range(img.size(1)):
                gray_img = img[batch_n, channel_n, :, :]
                if self.clamp:
                    gray_img = torch.clamp(gray_img, 0, 1)
                thr = skimage.filters.threshold_otsu((gray_img * 255).byte().detach().cpu().numpy())/255.
                plt.imshow(gray_img.cpu().numpy() - thr)
                res[batch_n, channel_n, :, :] = torch.clamp((gray_img > thr).float(),0,1.)
        return res