import numpy as np
import cv2

def forwarding():

    # src = np.arange(24).reshape((6, 4))
    src1 = cv2.imread("./data/testing/0_center_frame/16/input/frame10.png")
    src2 = cv2.imread("./data/testing/0_center_frame/16/input/frame11.png")
    h = src1.shape[0]
    w = src1.shape[1]
    flow = np.load("./flow/testing/0_center_frame/16.npy")
    print(src1.shape)
    print(src2.shape)
    print(flow.shape)
    print(flow)
    out = np.zeros((h, w, 3)).astype(int)
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]
    flow_coor = np.array([flow_x.flatten(), flow_y.flatten()])
    print(flow_x)
    print(flow_y)
    print(flow_coor)

    x_coor, y_coor = np.meshgrid( np.arange(w).astype(int), np.arange(h).astype(int) )
    # print(x_coor)
    # print(y_coor)
    grid_coor = np.array([x_coor.flatten(), y_coor.flatten()])
    # print(grid_coor)

    out_coor = (grid_coor + 0.5 * flow_coor).astype(int)
    mask = (out_coor[0,:] >= 0) & (out_coor[0,:] < w) & (out_coor[1,:] >= 0) & (out_coor[1,:] < h)
    out_x = out_coor[0,:] * mask
    out_y = out_coor[1,:] * mask
    out_coor = np.array([out_x, out_y])

    out[out_y,out_x] = src1[grid_coor[1,:],grid_coor[0,:]]
    print("out", out.shape)
    print(out)
    cv2.imwrite('frame10i11.jpg', out.astype(np.uint8))

if __name__ == "__main__":
    forwarding()

