import numpy as np
import cv2

def sumSplat(src, flow, t):
    h = src.shape[0]
    w = src.shape[1]

    flow_coor = np.array([flow[:, :, 0].flatten(), flow[:, :, 1].flatten()]).astype(np.float32) # (2, h * w)
    x_coor, y_coor = np.meshgrid(np.arange(w).astype(np.int32), np.arange(h).astype(np.int32))
    grid_coor = np.array([x_coor.flatten(), y_coor.flatten()]) # (2, h * w)
    flt_output_coor = grid_coor + t * flow_coor                # (2, h * w)

    int_NW_coor = np.floor(flt_output_coor).astype(np.int32)         # (2, h * w)
    int_NE_coor = np.array([int_NW_coor[0] + 1, int_NW_coor[1]])     # (2, h * w)
    int_SW_coor = np.array([int_NW_coor[0], int_NW_coor[1] + 1])     # (2, h * w)
    int_SE_coor = np.array([int_NW_coor[0] + 1, int_NW_coor[1] + 1]) # (2, h * w)

    flt_NW = np.absolute(np.prod((int_SE_coor - flt_output_coor).astype(np.float32), axis=0)) # (h * w)
    flt_NE = np.absolute(np.prod((flt_output_coor - int_SW_coor).astype(np.float32), axis=0)) # (h * w)
    flt_SW = np.absolute(np.prod((int_NE_coor - flt_output_coor).astype(np.float32), axis=0)) # (h * w)
    flt_SE = np.absolute(np.prod((flt_output_coor - int_NW_coor).astype(np.float32), axis=0)) # (h * w)

    mask_NW = (int_NW_coor[0, :] >= 0) & (int_NW_coor[0, :] < w) & (int_NW_coor[1, :] >= 0) & (int_NW_coor[1, :] < h) # (h * w)
    mask_NE = (int_NE_coor[0, :] >= 0) & (int_NE_coor[0, :] < w) & (int_NE_coor[1, :] >= 0) & (int_NE_coor[1, :] < h) # (h * w)
    mask_SW = (int_SW_coor[0, :] >= 0) & (int_SW_coor[0, :] < w) & (int_SW_coor[1, :] >= 0) & (int_SW_coor[1, :] < h) # (h * w)
    mask_SE = (int_SE_coor[0, :] >= 0) & (int_SE_coor[0, :] < w) & (int_SE_coor[1, :] >= 0) & (int_SE_coor[1, :] < h) # (h * w)

    out = np.zeros((h, w, 3))
    NW = np.multiply(src[grid_coor[1, :],grid_coor[0, :]], np.repeat(flt_NW.reshape(h * w, 1), 3, axis=1)) # (h * w, 3)
    NE = np.multiply(src[grid_coor[1, :],grid_coor[0, :]], np.repeat(flt_NE.reshape(h * w, 1), 3, axis=1)) # (h * w, 3)
    SW = np.multiply(src[grid_coor[1, :],grid_coor[0, :]], np.repeat(flt_SW.reshape(h * w, 1), 3, axis=1)) # (h * w, 3)
    SE = np.multiply(src[grid_coor[1, :],grid_coor[0, :]], np.repeat(flt_SE.reshape(h * w, 1), 3, axis=1)) # (h * w, 3)

    out[int_NW_coor[1, mask_NW], int_NW_coor[0, mask_NW]] += NW[mask_NW, :]
    out[int_NE_coor[1, mask_NE], int_NE_coor[0, mask_NE]] += NE[mask_NE, :]
    out[int_SW_coor[1, mask_SW], int_SW_coor[0, mask_SW]] += SW[mask_SW, :]
    out[int_SE_coor[1, mask_SE], int_SE_coor[0, mask_SE]] += SE[mask_SE, :]

    return out # (h, w, 3)

def Fwarp(src, flow, t, Z=None, splatting='average'):
    assert(splatting in ['summation', 'average', 'softmax'])
    assert(src.shape[0] == flow.shape[0] and src.shape[1] == flow.shape[1])

    out = None
    if (splatting == 'summation'):
        out = sumSplat(src, flow, t)
    elif (splatting == 'average'):
        normalize = sumSplat(np.ones(src.shape).astype(np.float32), flow, t)
        normalize[normalize == 0] = 1
        out = sumSplat(src, flow, t)
        out /= normalize
    elif (splatting == 'softmax'):
        assert Z is not None, "Need Z to use softmax splatting!"
        normalize = sumSplat(np.exp(Z), flow, t)
        normalize[normalize == 0] = 1
        out = sumSplat(np.exp(Z) * src, flow, t)
        out /= normalize
    
    return out.astype(np.uint8)

def Bwarpt1(src, flow01, flow10, t):
    h = src.shape[0]
    w = src.shape[1]
    out = np.zeros((h, w, 3)).astype(int)

    flow01_coor = np.array([flow01[:, :, 0].flatten(), flow01[:, :, 1].flatten()])
    flow10_coor = np.array([flow10[:, :, 0].flatten(), flow10[:, :, 1].flatten()])

    x_coor, y_coor = np.meshgrid( np.arange(w).astype(int), np.arange(h).astype(int) )
    grid_coor = np.array([x_coor.flatten(), y_coor.flatten()])

    flow_t1 = (1-t**2) * flow01_coor - t * (1-t) * flow10_coor
    out_coor = (grid_coor + flow_t1).astype(int)

    mask = (out_coor[0,:] >= 0) & (out_coor[0,:] < w) & (out_coor[1,:] >= 0) & (out_coor[1,:] < h)

    out[grid_coor[1, mask], grid_coor[0, mask]] = src[out_coor[1, mask], out_coor[0, mask]]

    return out.astype(np.uint8)

def Bwarpt0(src, flow01, flow10, t):
    h = src.shape[0]
    w = src.shape[1]
    out = np.zeros((h, w, 3)).astype(int)

    flow01_coor = np.array([flow01[:, :, 0].flatten(), flow01[:, :, 1].flatten()])
    flow10_coor = np.array([flow10[:, :, 0].flatten(), flow10[:, :, 1].flatten()])

    x_coor, y_coor = np.meshgrid( np.arange(w).astype(int), np.arange(h).astype(int) )
    grid_coor = np.array([x_coor.flatten(), y_coor.flatten()])

    flow_t0 = -(1-t) * t * flow01_coor + t**2 * flow10_coor
    out_coor = (grid_coor + flow_t0).astype(int)
    
    mask = (out_coor[0,:] >= 0) & (out_coor[0,:] < w) & (out_coor[1,:] >= 0) & (out_coor[1,:] < h)

    out[grid_coor[1, mask], grid_coor[0, mask]] = src[out_coor[1, mask], out_coor[0, mask]]

    return out.astype(np.uint8)

def occlusion(flow01, flow10):
    h = flow01.shape[0]
    w = flow01.shape[1]
    O0 = np.zeros((h, w)).astype(bool)
    O1 = np.zeros((h, w)).astype(bool)

    src = np.ones((h, w)) # one map

    flow01_x = flow01[:, :, 0]
    flow01_y = flow01[:, :, 1]
    flow01_coor = np.array([flow01_x.flatten(), flow01_y.flatten()])
    flow10_x = flow10[:, :, 0]
    flow10_y = flow10[:, :, 1]
    flow10_coor = np.array([flow10_x.flatten(), flow10_y.flatten()])

    x_coor, y_coor = np.meshgrid( np.arange(w).astype(int), np.arange(h).astype(int) )
    grid_coor = np.array([x_coor.flatten(), y_coor.flatten()])

    out1_coor = (grid_coor + flow01_coor).astype(int)
    out0_coor = (grid_coor + flow10_coor).astype(int)

    mask1 = (out1_coor[0,:] >= 0) & (out1_coor[0,:] < w) & (out1_coor[1,:] >= 0) & (out1_coor[1,:] < h)
    mask0 = (out0_coor[0,:] >= 0) & (out0_coor[0,:] < w) & (out0_coor[1,:] >= 0) & (out0_coor[1,:] < h)

    O1[out1_coor[1, mask1], out1_coor[0, mask1]] = src[grid_coor[1, mask1], grid_coor[0, mask1]]
    O0[out0_coor[1, mask0], out0_coor[0, mask0]] = src[grid_coor[1, mask0], grid_coor[0, mask0]]
    
    return O0, O1 # hole is false

if __name__ == "__main__":
    # for DEBUG
    # src1 = cv2.imread('first.png')
    # src2 = cv2.imread('second.png')
    src1 = cv2.imread('frame10.png')
    src2 = cv2.imread('frame11.png')
    # f = open('flow.flo', 'rb')
    # magic = np.fromfile(f, np.float32, count=1)
    # w = np.fromfile(f, np.int32, count=1)
    # h = np.fromfile(f, np.int32, count=1)
    # flow01 = np.fromfile(f, np.float32, count=2 * w[0]* h[0])
    # f.close()
    # flow01 = np.resize(flow01, (h[0], w[0], 2))
    flow01 = np.load('flow01_5.npy')
    flow10 = np.load('flow10_5.npy')
    O0, O1 = occlusion(flow01, flow10)
    alpha = -0.08 # for softmax splatting
    Z = alpha * np.repeat(np.mean(np.absolute(src1 - Bwarpt1(src2, flow01, flow10, 0)), axis=2, keepdims=True), 3, axis=2)
    # out = Fwarp(src1, flow01, 0.5, splatting='summation')
    out1 = Fwarp(src1, flow01, 0.5, splatting='average')
    out2 = Fwarp(src2, flow10, 0.5, splatting='average')
    # out = Fwarp(src1, flow01, 0.5, Z, splatting='softmax')
    # out = Bwarp(src2, flow01, flow10, 0.5)

    out = (0.5 * out1 + 0.5 * out2) * (np.tile(O0, (3, 1, 1)) * np.tile(O1, (3, 1, 1))).transpose(1, 2, 0)
    print(np.count_nonzero(out))
    out[~O1] = out2[~O1]
    # out[~O1] = src2[~O1]
    print(np.count_nonzero(out))
    out[~O0] = out1[~O0]
    # out[~O0] = src1[~O0]
    print(np.count_nonzero(out))
    cv2.imwrite('out5-1.png', out)
    # for DEBUG end
