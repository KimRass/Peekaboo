import torch
import torchvision.transforms.functional as TF
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
