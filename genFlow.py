import numpy as np
import torch
import torch.nn as nn
import math
from model import pwcnet

def genFlow(frameA, frameB, model):

    frameA = np.moveaxis(frameA, -1, 0) / 255.0 # (3, h, w)
    frameB = np.moveaxis(frameB, -1, 0) / 255.0 # (3, h, w)
    assert(frameA.shape[1] == frameB.shape[1])
    assert(frameA.shape[2] == frameB.shape[2])

    intWidth = frameA.shape[2]
    intHeight = frameA.shape[1]

    #####################################################################################################
    # Predict Flow
    #####################################################################################################

    # Move to device and unsqueeze in the batch dimension (to have batch size 1)
    frameA_cuda = torch.from_numpy(frameA).cuda().unsqueeze(0).float()
    frameB_cuda = torch.from_numpy(frameB).cuda().unsqueeze(0).float()
    # image need to ber divisible by 64
    intWidth64= int(math.ceil(intWidth / 64.0) * 64.0)
    intHeight64 = int(math.ceil(intHeight / 64.0) * 64.0)
    frameA64_cuda = torch.nn.functional.interpolate(input=frameA_cuda, size=(intHeight64, intWidth64), mode='bilinear', align_corners=False)
    frameB64_cuda = torch.nn.functional.interpolate(input=frameB_cuda, size=(intHeight64, intWidth64), mode='bilinear', align_corners=False)
    with torch.no_grad():
        flow = model.forward(frameA64_cuda, frameB64_cuda) # both (1, 3, 64n, 64m)

    flow = 20.0 * torch.nn.functional.interpolate(input=flow, size=(intHeight, intWidth), mode='bilinear', align_corners=False)
    flow[:, 0, :, :] *= float(intWidth) / float(intWidth64)
    flow[:, 1, :, :] *= float(intHeight) / float(intHeight64)

    #####################################################################################################
    # Predict Flow
    #####################################################################################################

    # Flow is stored row-wise in order [channels, height, width].
    optical_flow = flow.squeeze(0).cpu().numpy().transpose(1, 2, 0) # (h, w, 2)
    assert len(optical_flow.shape) == 3

    return optical_flow