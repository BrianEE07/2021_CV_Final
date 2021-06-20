import sys, os, json
import numpy as np
import torch
import torch.nn as nn
import flow_vis
from skimage import io
from model import pwcnet
from genFlow import genFlow
from warping import warping

def task_center(frameA_path, frameB_path, output_dir, id, model):
    frameA = io.imread(frameA_path) # (h, w, 3) RGB
    frameB = io.imread(frameB_path) # (h, w, 3) RGB
    flow = genFlow(frameA, frameB, model)
    # print("flow", flow.shape)
    midframe = warping(frameA, frameB, flow, 0.5)
    if not os.path.isdir(os.path.join(output_dir, id)):
        os.makedirs(os.path.join(output_dir, id))
    io.imsave(os.path.join(output_dir, id, "frame10i11.jpg"), midframe)

    # visualize flow (for debug)
    # flow_color = flow_vis.flow_to_color(flow, convert_to_bgr=False)
    # io.imsave(os.path.join(output_dir, id, "flow.png"), flow_color)
    return

def task_30to240(frameA_path, frameB_path, output_dir, id, model):
    frameA = io.imread(frameA_path) # (h, w, 3) RGB
    frameB = io.imread(frameB_path) # (h, w, 3) RGB
    flow = genFlow(frameA, frameB, model)
    frame0 = warping(frameA, frameB, flow, 0.125)
    frame1 = warping(frameA, frameB, flow, 0.25)
    frame2 = warping(frameA, frameB, flow, 0.375)
    frame3 = warping(frameA, frameB, flow, 0.5)
    frame4 = warping(frameA, frameB, flow, 0.625)
    frame5 = warping(frameA, frameB, flow, 0.75)
    frame6 = warping(frameA, frameB, flow, 0.875)
    if not os.path.isdir(os.path.join(output_dir, id[0], id[2:])):
        os.makedirs(os.path.join(output_dir, id[0], id[2:]))
    io.imsave(os.path.join(output_dir, id[0], id[2:], f"{str(int(id[2:]) * 8 + 1).zfill(5)}.jpg"), frame0)
    io.imsave(os.path.join(output_dir, id[0], id[2:], f"{str(int(id[2:]) * 8 + 2).zfill(5)}.jpg"), frame1)
    io.imsave(os.path.join(output_dir, id[0], id[2:], f"{str(int(id[2:]) * 8 + 3).zfill(5)}.jpg"), frame2)
    io.imsave(os.path.join(output_dir, id[0], id[2:], f"{str(int(id[2:]) * 8 + 4).zfill(5)}.jpg"), frame3)
    io.imsave(os.path.join(output_dir, id[0], id[2:], f"{str(int(id[2:]) * 8 + 5).zfill(5)}.jpg"), frame4)
    io.imsave(os.path.join(output_dir, id[0], id[2:], f"{str(int(id[2:]) * 8 + 6).zfill(5)}.jpg"), frame5)
    io.imsave(os.path.join(output_dir, id[0], id[2:], f"{str(int(id[2:]) * 8 + 7).zfill(5)}.jpg"), frame6)
    return

def task_24to60(frameA_path, frameB_path, output_dir, id, model):
    frameA = io.imread(frameA_path) # (h, w, 3) RGB
    frameB = io.imread(frameB_path) # (h, w, 3) RGB
    flow = genFlow(frameA, frameB, model)
    if not os.path.isdir(os.path.join(output_dir, id[0], id[2:])):
        os.makedirs(os.path.join(output_dir, id[0], id[2:]))
    if int(id[2]) % 2 == 0: # even
        frame0 = warping(frameA, frameB, flow, 0.4)
        frame1 = warping(frameA, frameB, flow, 0.8)
        io.imsave(os.path.join(output_dir, id[0], id[2:], f"{str(int(id[2:]) * 10 + 4).zfill(5)}.jpg"), frame0)
        io.imsave(os.path.join(output_dir, id[0], id[2:], f"{str(int(id[2:]) * 10 + 8).zfill(5)}.jpg"), frame1)
    else: # odd
        frame0 = warping(frameA, frameB, flow, 0.2)
        frame1 = warping(frameA, frameB, flow, 0.6)
        io.imsave(os.path.join(output_dir, id[0], id[2:], f"{str(int(id[2:]) * 10 + 2).zfill(5)}.jpg"), frame0)
        io.imsave(os.path.join(output_dir, id[0], id[2:], f"{str(int(id[2:]) * 10 + 6).zfill(5)}.jpg"), frame1)
    return