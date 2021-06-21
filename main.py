import sys
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import math
from tqdm import tqdm
from skimage import io
from eval import psnr
from eval import ssim
from task import task_center
from task import task_30to240
from task import task_24to60

def main():

    #####################################################################################################
    # Parser
    #####################################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', help='Data split', choices=['validation', 'testing'], required=True)
    parser.add_argument('--task', help='Data tast', choices=['0_center_frame', '1_30fps_to_240fps', '2_24fps_to_60fps'], required=True)
    args = parser.parse_args()
    split = args.split
    task = args.task
    print(split, task)

    # Dataset dir
    dataset_base_dir = "./data/"
    print("data_base_dir :", dataset_base_dir)
    
    # Output dir
    output_dir = os.path.join("outputs", f"{split}", f"{task}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    
    #####################################################################################################
    # Labels
    #####################################################################################################

    labels_json = os.path.join("json", f"{split}_{task[0]}.json")

    assert os.path.isfile(labels_json), f"{labels_json} does not exist!"

    with open(labels_json, 'r') as f:
        labels = json.loads(f.read())

    #####################################################################################################
    # Load Data
    #####################################################################################################

    for label in tqdm(labels):
        frameA_path = os.path.join(dataset_base_dir, label["frameA"])
        frameB_path = os.path.join(dataset_base_dir, label["frameB"])
        print("frameA", frameA_path)
        print("frameB", frameB_path)
        
        if task == "0_center_frame":
            midframe = task_center(frameA_path, frameB_path, output_dir, label["id"])
            if split == "validation":
                frameGT_path = os.path.join(dataset_base_dir, label["frameGT"])
                frameGT = io.imread(frameGT_path) # (h, w, 3) RGB
                psnr_score = psnr(frameGT, midframe)
                ssim_score = ssim(frameGT, midframe)
                print(label["id"], psnr_score, ssim_score)
        elif task == "1_30fps_to_240fps":
            frames = task_30to240(frameA_path, frameB_path, output_dir, label["id"])
            if split == "validation":
                frameGT0_path = os.path.join(dataset_base_dir, label["frameGT0"])
                frameGT1_path = os.path.join(dataset_base_dir, label["frameGT1"])
                frameGT2_path = os.path.join(dataset_base_dir, label["frameGT2"])
                frameGT3_path = os.path.join(dataset_base_dir, label["frameGT3"])
                frameGT4_path = os.path.join(dataset_base_dir, label["frameGT4"])
                frameGT5_path = os.path.join(dataset_base_dir, label["frameGT5"])
                frameGT6_path = os.path.join(dataset_base_dir, label["frameGT6"])
        elif task == "2_24fps_to_60fps":
            frames = task_24to60(frameA_path, frameB_path, output_dir, label["id"])
            if split == "validation":
                frameGT0_path = os.path.join(dataset_base_dir, label["frameGT0"])
                frameGT1_path = os.path.join(dataset_base_dir, label["frameGT1"])

if __name__ == "__main__":
    main()

