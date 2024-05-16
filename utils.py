import torch
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid
import numpy as np
import gc


def unload_from_gpu(var):
    var = var.to(torch.device("cpu"))
    gc.collect()
    torch.cuda.empty_cache()


def denorm(x, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
    return TF.normalize(
        x, mean=-(np.array(mean) / np.array(std)), std=(1 / np.array(std)),
    )


def image_to_grid(image, n_cols):
    tensor = image.clone().detach().cpu()
    tensor = denorm(tensor)
    grid = make_grid(tensor, nrow=n_cols, padding=1, pad_value=1)
    grid.clamp_(0, 1)
    grid = TF.to_pil_image(grid)
    return grid
