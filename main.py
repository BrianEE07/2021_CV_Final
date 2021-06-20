import sys
import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import math
from tqdm import tqdm
from model import pwcnet
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

     # Model checkpoint to use
    saved_model = "./model/chairs_things_0.pt"
    print("model path :", saved_model)

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
    # Model
    #####################################################################################################

    assert os.path.isfile(saved_model), f"Model {saved_model} does not exist!"
    pretrained_dict = torch.load(saved_model)

    # Construct model
    model = pwcnet.PWCNet().cuda()
    model.load_state_dict(pretrained_dict)

    model.eval()

    #####################################################################################################
    # Load Data
    #####################################################################################################

    for label in tqdm(labels):
        frameA_path = os.path.join(dataset_base_dir, label["frameA"])
        frameB_path = os.path.join(dataset_base_dir, label["frameB"])
        print("frameA", frameA_path)
        print("frameB", frameB_path)
        
        if task == "0_center_frame":
            task_center(frameA_path, frameB_path, output_dir, label["id"], model)
            if split == "validation":
                frameGT_path = os.path.join(dataset_base_dir, label["frameGT"])
                # print("GT", frameGT_path)

        elif task == "1_30fps_to_240fps":
            task_30to240(frameA_path, frameB_path, output_dir, label["id"], model)
            if split == "validation":
                frameGT0_path = os.path.join(dataset_base_dir, label["frameGT0"])
                frameGT1_path = os.path.join(dataset_base_dir, label["frameGT1"])
                frameGT2_path = os.path.join(dataset_base_dir, label["frameGT2"])
                frameGT3_path = os.path.join(dataset_base_dir, label["frameGT3"])
                frameGT4_path = os.path.join(dataset_base_dir, label["frameGT4"])
                frameGT5_path = os.path.join(dataset_base_dir, label["frameGT5"])
                frameGT6_path = os.path.join(dataset_base_dir, label["frameGT6"])
                # print("GT", frameGT0_path)
                # print("GT", frameGT1_path)
                # print("GT", frameGT2_path)
                # print("GT", frameGT3_path)
                # print("GT", frameGT4_path)
                # print("GT", frameGT5_path)
                # print("GT", frameGT6_path)
        elif task == "2_24fps_to_60fps":
            task_24to60(frameA_path, frameB_path, output_dir, label["id"], model)
            if split == "validation":
                frameGT0_path = os.path.join(dataset_base_dir, label["frameGT0"])
                frameGT1_path = os.path.join(dataset_base_dir, label["frameGT1"])
                # print("GT", frameGT0_path)
                # print("GT", frameGT1_path)

if __name__ == "__main__":
    main()

