import sys, os, json
import numpy as np
import torch
import torch.nn as nn
import argparse
import math
import flow_vis
from skimage import io
from tqdm import tqdm
from model import pwcnet

def main():

    #####################################################################################################
    # Parser
    #####################################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--split', help='Data split', choices=['validation', 'testing'], required=True)
    args = parser.parse_args()
    split = args.split
    print(split)

     # Model checkpoint to use
    saved_model      = "./model/chairs_things_0.pt"

    # Dataset dir
    dataset_base_dir = "./data/"

    # Output dir
    output_dir = os.path.join("flow", f"{split}", "0_center_frame")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print("Created output dir", output_dir)
    
    #####################################################################################################
    # Labels
    #####################################################################################################

    # only for 0_center_frame now
    labels_json = os.path.join("json", f"{split}_0.json")

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

        frameA = io.imread(frameA_path) # (h, w, 3) RGB
        frameB = io.imread(frameB_path) # (h, w, 3) RGB

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
        id = label["id"]
        optical_flow_file = os.path.join(output_dir, f"{id}.npy")
        with open(optical_flow_file, 'wb') as fout:
            np.save(fout, optical_flow)

        #####################################################################################################
        # Visualize Flow (Optional)
        #####################################################################################################

        optical_flow_vis_file = os.path.join(output_dir, f"{id}.png")
        flow_color = flow_vis.flow_to_color(optical_flow, convert_to_bgr=False)
        io.imsave(optical_flow_vis_file, flow_color)

if __name__ == "__main__":
    main()