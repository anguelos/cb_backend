import tormentor
import diamond_square
import torch
from tormentor import SpatialAugmentationState


class BleedThrough(tormentor.ColorAugmentation):
    center_x = tormentor.Uniform([-1, 1])
    center_y = tormentor.Uniform([-1, 1])
    blend_roughness = tormentor.Uniform(value_range=(.3, .6))
    max_scale = tormentor.Uniform(value_range=(.01, .3))

    def generate_batch_state(self, batch_tensor: torch.Tensor) -> SpatialAugmentationState:



    @classmethod
    def functional_image(cls, image_tensor, *state):
        pass
