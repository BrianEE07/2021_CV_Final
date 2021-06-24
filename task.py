import sys, os, json
import numpy as np
import torch
import torch.nn as nn
import flow_vis
from skimage import io
from genFlow import genFlow
from warping import Fwarp, Bwarpt0, Bwarpt1, occlusion
from splatting import backwarp
import softsplat

def task_center(frameA_path, frameB_path, output_dir, id):
    frameA = io.imread(frameA_path) # (h, w, 3) RGB
    frameB = io.imread(frameB_path) # (h, w, 3) RGB

    flow01, flow10 = genFlow(frameA_path, frameB_path)

    midframe = interp_frame(frameA, frameB, flow01, flow10, 0.5)

    if not os.path.isdir(os.path.join(output_dir, id)):
        os.makedirs(os.path.join(output_dir, id))
    io.imsave(os.path.join(output_dir, id, "frame10i11.jpg"), midframe)

    # visualize flow (for debug)
    # flow_color = flow_vis.flow_to_color(flow01, convert_to_bgr=False)
    # io.imsave(os.path.join(output_dir, id, "flow01.png"), flow_color)
    return midframe

def task_30to240(frameA_path, frameB_path, output_dir, id):
    frameA = io.imread(frameA_path) # (h, w, 3) RGB
    frameB = io.imread(frameB_path) # (h, w, 3) RGB

    flow01, flow10 = genFlow(frameA_path, frameB_path)

    frames = []
    for i in range(7):
        frames.append(interp_frame(frameA, frameB, flow01, flow10, (i + 1) / 8.0))

    if not os.path.isdir(os.path.join(output_dir, id[0], id[2:])):
        os.makedirs(os.path.join(output_dir, id[0], id[2:]))

    for i in range(7):
        io.imsave(os.path.join(output_dir, id[0], id[2:], f"{str(int(id[2:]) * 8 + i + 1).zfill(5)}.jpg"), frames[i])

    return frames

def task_24to60(frameA_path, frameB_path, output_dir, id):
    frameA = io.imread(frameA_path) # (h, w, 3) RGB
    frameB = io.imread(frameB_path) # (h, w, 3) RGB

    flow01, flow10 = genFlow(frameA_path, frameB_path)

    if not os.path.isdir(os.path.join(output_dir, id[0], id[2:])):
        os.makedirs(os.path.join(output_dir, id[0], id[2:]))

    if int(id[2]) % 2 == 0: # even
        frame0 = interp_frame(frameA, frameB, flow01, flow10, 0.4)
        frame1 = interp_frame(frameA, frameB, flow01, flow10, 0.8)
        io.imsave(os.path.join(output_dir, id[0], id[2:], f"{str(int(id[2:]) * 10 + 4).zfill(5)}.jpg"), frame0)
        io.imsave(os.path.join(output_dir, id[0], id[2:], f"{str(int(id[2:]) * 10 + 8).zfill(5)}.jpg"), frame1)
    else: # odd
        frame0 = interp_frame(frameA, frameB, flow01, flow10, 0.2)
        frame1 = interp_frame(frameA, frameB, flow01, flow10, 0.6)
        io.imsave(os.path.join(output_dir, id[0], id[2:], f"{str(int(id[2:]) * 10 + 2).zfill(5)}.jpg"), frame0)
        io.imsave(os.path.join(output_dir, id[0], id[2:], f"{str(int(id[2:]) * 10 + 6).zfill(5)}.jpg"), frame1)
    return [frame0, frame1]

def interp_frame(frameA, frameB, flow01, flow10, t): # pipeline after flow generation

    # alpha = -0.01 # for my softmax splatting
    alpha = -20 # for softmax splatting
    # Z01 = alpha * np.repeat(np.mean(np.absolute(frameA - Bwarpt1(frameB, flow01, flow10, 0)), axis=2, keepdims=True), 3, axis=2)
    # Z10 = alpha * np.repeat(np.mean(np.absolute(frameB - Bwarpt0(frameA, flow01, flow10, 1)), axis=2, keepdims=True), 3, axis=2)



    ###### test softsplat ######
    tenFirst = torch.FloatTensor(np.ascontiguousarray(frameA[:, :, ::-1].transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).cuda()
    tenSecond = torch.FloatTensor(np.ascontiguousarray(frameB[:, :, ::-1].transpose(2, 0, 1)[None, :, :, :].astype(np.float32) * (1.0 / 255.0))).cuda()
    tenFlow01 = torch.FloatTensor(np.ascontiguousarray(flow01.transpose(2, 0, 1)[None, :, :, :])).cuda()
    tenFlow10 = torch.FloatTensor(np.ascontiguousarray(flow10.transpose(2, 0, 1)[None, :, :, :])).cuda()

    tenMetric01 = torch.nn.functional.l1_loss(input=tenFirst, target=backwarp(tenInput=tenSecond, tenFlow=tenFlow01), reduction='none').mean(1, True)
    tenMetric10 = torch.nn.functional.l1_loss(input=tenSecond, target=backwarp(tenInput=tenFirst, tenFlow=tenFlow10), reduction='none').mean(1, True)

    # tenAverage0t = softsplat.FunctionSoftsplat(tenInput=tenFirst, tenFlow=tenFlow01 * t, tenMetric=None, strType='average')
    # tenAverage1t = softsplat.FunctionSoftsplat(tenInput=tenSecond, tenFlow=tenFlow10 * (1-t), tenMetric=None, strType='average')
    tenSoftmax0t = softsplat.FunctionSoftsplat(tenInput=tenFirst, tenFlow=tenFlow01 * t, tenMetric=alpha * tenMetric01, strType='softmax')
    tenSoftmax1t = softsplat.FunctionSoftsplat(tenInput=tenSecond, tenFlow=tenFlow10 * (1-t), tenMetric=alpha * tenMetric10, strType='softmax')
    ###### test softsplat ######    

    # forward warping - average/softmax splatting
    # frame0t = Fwarp(frameA, flow01, t, splatting='average')       # forward warp I0 -> It
    # frame1t = Fwarp(frameB, flow10, 1.0 - t, splatting='average') # forward warp I1 -> It
    # frame0t = Fwarp(frameA, flow01, t, Z01, splatting='softmax')       # forward warp I0 -> It
    # frame1t = Fwarp(frameB, flow10, 1.0 - t, Z10, splatting='softmax') # forward warp I1 -> It

    # frame0t = tenAverage0t[0, :, :, :].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0
    # frame1t = tenAverage1t[0, :, :, :].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0
    frame0t = tenSoftmax0t[0, :, :, :].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0
    frame1t = tenSoftmax1t[0, :, :, :].cpu().numpy().transpose(1, 2, 0)[:, :, ::-1] * 255.0

    # occlusion mask - range map
    O0, O1 = occlusion(flow01, flow10) # F01 -> O1, F10 -> O0

    # hole filling
    # frame = (0.5 * frame0t + 0.5 * frame1t) * (np.tile(O0, (3, 1, 1)) * np.tile(O1, (3, 1, 1))).transpose(1, 2, 0)
    # frame[~O1] = frame1t[~O1]
    # frame[~O0] = frame0t[~O0]
    # frame[~O1] = frameB[~O1]
    # frame[~O0] = frameA[~O0]
    
    isHole = (frame0t == 0)
    frame0t[isHole] = frameB[isHole] # hole filling
    isHole = (frame1t == 0)
    frame1t[isHole] = frameA[isHole] # hole filling

    frame = 0.5 * frame0t + 0.5 * frame1t

    return frame.astype(np.uint8)