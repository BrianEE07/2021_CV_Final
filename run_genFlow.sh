python3 genFlow.py --split="validation" --task="0_center_frame"
python3 genFlow.py --split="validation" --task="1_30fps_to_240fps"
python3 genFlow.py --split="validation" --task="2_24fps_to_60fps"
python3 genFlow.py --split="testing" --task="0_center_frame"
python3 genFlow.py --split="testing" --task="1_30fps_to_240fps"
python3 genFlow.py --split="testing" --task="2_24fps_to_60fps"