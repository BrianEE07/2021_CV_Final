import numpy as np
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from flownet import estimate
from task import interp_frame

def createVideo(VIDEO_PATH):

    video = cv2.VideoCapture(VIDEO_PATH)
    film_h, film_w = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    film_fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    videowriter = cv2.VideoWriter("output3.avi", fourcc, film_fps, (film_w, film_h))

    # TODO: find homography per frame and apply backward warp
    pbar = tqdm(total = 228) 
    preFrame = None
    start = False
    while (video.isOpened()):
        ret, frame = video.read()
        if ret:  ## check whethere the frame is legal, i.e., there still exists a frame
            if start:
                # for i in range(3): 
                #     videowriter.write(frame)
                print(frame.shape)
                
                # frameA = io.imread(frameA_path) # (h, w, 3) RGB
                # frameB = io.imread(frameB_path) # (h, w, 3) RGB

                tsframeA = torch.FloatTensor(np.ascontiguousarray(np.array(preFrame).transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))
                tsframeB = torch.FloatTensor(np.ascontiguousarray(np.array(frame).transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)))

                flow01 = estimate(tsframeA, tsframeB)
                flow10 = estimate(tsframeB, tsframeA)

                optical_flow01 = flow01.squeeze(0).cpu().numpy().transpose(1, 2, 0) # (h, w, 2)
                optical_flow10 = flow10.squeeze(0).cpu().numpy().transpose(1, 2, 0) # (h, w, 2)

                # flow01, flow10 = genFlow(preFrame, frame)

                frames = []
                for i in range(7):
                    frames.append(interp_frame(preFrame, frame, optical_flow01, optical_flow10, (i + 1) / 8.0))

                # # if not os.path.isdir(os.path.join(output_dir, id[0], id[2:])):
                # #     os.makedirs(os.path.join(output_dir, id[0], id[2:]))

                videowriter.write(preFrame)
                for i in range(7):
                    outFrame = interp_frame(preFrame, frame, optical_flow01, optical_flow10, (i + 1) / 8.0)
                    videowriter.write(outFrame)
                videowriter.write(frame)

                # return frames

            pbar.update(1)
            preFrame = frame
            start = True

            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            break

    pbar.close()
    video.release()
    videowriter.release()
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    
    VIDEO_PATH = './hotpot.mp4'
    createVideo(VIDEO_PATH)
    