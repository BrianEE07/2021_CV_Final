import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from flownet import estimate

def genFlow(frameA_path, frameB_path):

    tsframeA = torch.FloatTensor(np.ascontiguousarray(np.array(Image.open(frameA_path))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
    tsframeB = torch.FloatTensor(np.ascontiguousarray(np.array(Image.open(frameB_path))[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))

    #####################################################################################################
    # Predict Flow
    #####################################################################################################
    flow01 = estimate(tsframeA, tsframeB)
    flow10 = estimate(tsframeB, tsframeA)

    optical_flow01 = flow01.squeeze(0).cpu().numpy().transpose(1, 2, 0) # (h, w, 2)
    optical_flow10 = flow10.squeeze(0).cpu().numpy().transpose(1, 2, 0) # (h, w, 2)

    return optical_flow01, optical_flow10