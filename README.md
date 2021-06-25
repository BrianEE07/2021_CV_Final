# 2021_CV_Final
## MTK Video Frame Interpolation
dataset can be downloaded at https://drive.google.com/file/d/1Nfn8GZaEATkFpw7Atx3EO5Ovs5_5a16u/view?usp=sharing


## Environments
1. needs GPU for torch

## Run
for single split and task:
--split='validation', 'testing'
--task='0_center_frame', '1_30fps_to_240fps', '2_24fps_to_60fps'
```
python3 main.py --split=validation --task=0_center_frame
```
run all dataset
```
sh run_main.sh
```
output images will be dumped in ./outputs/





