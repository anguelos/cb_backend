import tormentor
from diamond_square import functional_diamond_square
import torch
from tormentor import AugmentationState


class Bleedthrough(tormentor.StaticImageAugmentation):
    center_x = tormentor.Uniform([-1, 1])
    center_y = tormentor.Uniform([-1, 1])
    h_flip = tormentor.Bernoulli(.5)
    v_flip = tormentor.Bernoulli(.5)
    blend_roughness = tormentor.Uniform(value_range=(.3, .6))
    max_scale = tormentor.Uniform(value_range=(.01, .5))

    def generate_batch_state(self, batch_tensor: torch.Tensor) -> AugmentationState:
        batch_sz, channels, height, width = batch_tensor.size()
        blend_roughness = type(self).blend_roughness(batch_sz, device=batch_tensor.device)
        plasma_sz = (batch_sz, 1, height, width)
        max_scale = type(self).max_scale([batch_sz, 1, 1, 1], device=batch_tensor.device)
        scale = functional_diamond_square(plasma_sz, roughness=blend_roughness, device=batch_tensor.device)
        scale = scale * max_scale
        center_x = type(self).center_x(batch_sz)
        center_y = type(self).center_y(batch_sz)
        h_flip = type(self).h_flip(batch_sz)
        v_flip = type(self).v_flip(batch_sz)
        return scale, center_x, center_y, h_flip, v_flip

    @classmethod
    def functional_image(cls, batch_images, scale, center_x, center_y, h_flip, v_flip):
        batch_size, channels, height, width = batch_images.size()
        for n in range(batch_images.size(0)):
            flip_dims = []
            if h_flip[n]:
                flip_dims.append(2)
            if v_flip[n]:
                flip_dims.append(1)
            #interference = batch_images[n, :, :, :].flip(flip_dims)
            interference = batch_images[n, :, :, :]
            if center_x[n] < 0:
                interference_left = int(-center_x[n] * (width // 2))
                interference_right = width - 1
            else:
                interference_left = 0
                interference_right = int(width - center_x[n] * (width // 2))
            if center_y[n] < 0:
                interference_top = int(-center_y[n] * (height // 2))
                interference_bottom = height - 1
            else:
                interference_top = 0
                interference_bottom = int(height - center_y[n] * (height // 2))
            img_left = width - (1+ interference_right)
            img_right = width - (1+ interference_left)
            img_top = height - (1+ interference_bottom)
            img_bottom = height - (1+ interference_top)
            cropped_scale = scale[n, :, interference_top:interference_bottom, interference_left:interference_right]
            inv_cropped_scale = 1 - cropped_scale
            scaled_interference_patch = cropped_scale * interference[:, interference_top:interference_bottom,
                            interference_left:interference_right]
            scaled_image_patch = inv_cropped_scale * batch_images[n, :,img_top:img_bottom,img_left:img_right]
            batch_images[n,:,img_top:img_bottom,img_left:img_right] = scaled_interference_patch + scaled_image_patch
        return batch_images



class ErodeInpainting(tormentor.StaticImageAugmentation):
    r"""Augmentation Shred.


    Distributions:
        ``roughness``: Quantification of the local inconsistency of the distortion effect.
        ``erase_percentile``: Quantification of the surface that will be erased.
        ``inside``: If True

    .. image:: _static/example_images/Shred.png
    """
    roughness = tormentor.Uniform(value_range=(.1, .5))
    inside = tormentor.Bernoulli(prob=.5)
    erase_percentile = tormentor.Uniform(value_range=(.0, .2))
    bg_roughness = tormentor.Uniform(value_range=(.1, .5))
    bg_range = tormentor.Uniform(value_range=(.01, .6))
    bg_invert = tormentor.Bernoulli(.5)

    def generate_batch_state(self, image_batch: torch.Tensor) -> AugmentationState:
        batch_sz, _, width, height = image_batch.size()
        roughness = type(self).roughness(batch_sz, device=image_batch.device)
        plasma_sz = (batch_sz, 1, width, height)
        plasma = functional_diamond_square(plasma_sz, roughness=roughness, device=image_batch.device)
        erase_percentile = type(self).erase_percentile(batch_sz, device=image_batch.device)
        inside = type(self).inside(batch_sz, device=image_batch.device).float()
        bg_roughness = type(self).bg_roughness(batch_sz, device=image_batch.device)
        bg_range, _ = type(self).bg_range((batch_sz, 2), device=image_batch.device).sort(dim=1)
        bg_invert = type(self).bg_invert((batch_sz,1,1,1), device=image_batch.device)
        bg_plasma = functional_diamond_square(plasma_sz, roughness=bg_roughness, device=image_batch.device) * (bg_range[:, 1:]-bg_range[:, :1]) + bg_range[:, :1]
        bg_plasma = bg_plasma * (1-bg_invert) + (1-bg_plasma) * bg_invert
        return plasma, inside, erase_percentile, bg_plasma

    @classmethod
    def functional_image(cls, image_batch: torch.Tensor, plasma: torch.FloatTensor, inside: torch.FloatTensor,
                                  erase_percentile: torch.FloatTensor, bg_plasma: torch.FloatTensor) -> torch.Tensor:
        inside = inside.view(-1, 1, 1, 1)
        erase_percentile = erase_percentile.view(-1, 1, 1, 1)
        plasma = inside * plasma + (1 - inside) * (1 - plasma)
        plasma_pixels = plasma.view(plasma.size(0), -1)
        thresholds = []
        for n in range(plasma_pixels.size(0)):
            thresholds.append(torch.kthvalue(plasma_pixels[n], int(plasma_pixels.size(1) * erase_percentile[n]))[0])
        thresholds = torch.Tensor(thresholds).view(-1, 1, 1, 1).to(plasma.device)
        erase = (plasma < thresholds).float()
        return image_batch * (1 - erase) + bg_plasma * erase


class ErodeSimple(ErodeInpainting):
    r"""Augmentation Shred.


    Distributions:
        ``roughness``: Quantification of the local inconsistency of the distortion effect.
        ``erase_percentile``: Quantification of the surface that will be erased.
        ``inside``: If True

    .. image:: _static/example_images/Shred.png
    """
    def forward_mask(self, X: torch.Tensor) -> torch.Tensor:
        images = X.float()
        res = self.forward_img(images)
        if X.dtype == torch.bool:
            return res.bool
        else:
            return res


class ScaleIntencity(tormentor.ColorAugmentation):
    r"""Lowers   the brightness of the image over a random mask.

    .. image:: _static/example_images/PlasmaShadow.png
   """
    min_inv_max = tormentor.Uniform(value_range=(.0, .3))

    def generate_batch_state(self, batch_tensor: torch.FloatTensor) -> torch.FloatTensor:
        batch_sz = batch_tensor.size(0)
        min_inv_max = type(self).min_inv_max((batch_sz, 2), device=batch_tensor.device)
        min = min_inv_max[:, 0]
        max = 1 - min_inv_max[:, 1]
        return min, max

    @classmethod
    def functional_image(cls, batch: torch.FloatTensor, min: torch.FloatTensor, max: torch.FloatTensor) -> torch.FloatTensor:
        min = min.view(-1,1,1,1)
        max = max.view(-1,1,1,1)
        scale = max - min
        return torch.clamp(batch * scale + min, 0, 1)

#tormentor.RandomIdentity ^ tormentor.RandomWrap ^ BleedThrough ^ Erode ^ tormentor.RandomIdentity

perspective = tormentor.RandomPerspective.custom(x_offset=tormentor.Uniform(value_range=(0.9, 1.1)), y_offset=tormentor.Uniform(value_range=(0.9, 1.11)))
wrap = tormentor.RandomWrap.custom(roughness=tormentor.Uniform(value_range=(0.232, 0.352)), intensity=tormentor.Uniform(value_range=(0.4, 0.48)))

#augmentation_pipeline = tormentor.AugmentationFactory(tormentor.AugmentationChoice.create([tormentor.RandomIdentity , tormentor.RandomWrap , BleedThrough , Erode , tormentor.RandomIdentity]))
pre_spatials0 = tormentor.AugmentationFactory(ScaleIntencity)
pre_spatials1 = tormentor.AugmentationFactory(tormentor.AugmentationChoice.create([tormentor.RandomIdentity , Bleedthrough]))
pre_spatials2 = tormentor.AugmentationFactory(tormentor.AugmentationChoice.create([tormentor.RandomIdentity, ErodeSimple]))
pre_spatials3 = tormentor.AugmentationFactory(tormentor.AugmentationChoice.create([tormentor.RandomIdentity, tormentor.RandomPlasmaBrightness, tormentor.PlasmaShadow]))
spatials = tormentor.AugmentationFactory(tormentor.AugmentationChoice.create([tormentor.RandomIdentity , wrap ,perspective]))
#augmentation_pipeline = tormentor.AugmentationFactory(GaussianAdditiveNoise)
augmentation_pipeline = pre_spatials1|pre_spatials2|spatials|pre_spatials3
augmentation_pipeline_flat =  tormentor.AugmentationFactory(tormentor.AugmentationChoice.create([tormentor.RandomIdentity ]*3+[ Bleedthrough, ErodeSimple, wrap, perspective]))
