import torch
import gc


def unload_from_gpu(var):
    var = var.to(torch.device("cpu"))
    gc.collect()
    torch.cuda.empty_cache()
