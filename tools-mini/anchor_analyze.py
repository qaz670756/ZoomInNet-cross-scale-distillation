from mmdet.core import build_anchor_generator
from mmdet import cv_core
import numpy as np
import cv2

def show_anchor(input_shape_hw, stride, anchor_generator_cfg, random_n, select_n,img=None):
    try:
        img.shape
    except:
        img = np.zeros(input_shape_hw, np.uint8)
    feature_map = []
    for s in stride:
        #s = s*2 # 特征图扩大两倍
        s = s
        feature_map.append([input_shape_hw[0] // s, input_shape_hw[1] // s])
    anchor_generator = build_anchor_generator(anchor_generator_cfg)
    anchors = anchor_generator.grid_anchors(feature_map,device='cpu')  # 输出原图尺度上anchor坐标 xywh格式 左上角格式
    for _ in range(random_n):
        for idx,anchor in enumerate(anchors):
            anchor = anchor.cpu().numpy()
            orinum = anchor.shape[0]
            # 取出左上角大于0，尺寸小于图像尺寸的anchor
            index = (anchor[:, 0] > 0) & (anchor[:, 1] > 0) & (anchor[:, 2] < input_shape_hw[1]) & \
                    (anchor[:, 3] < input_shape_hw[0])
            anchor = anchor[index]
            innum = anchor.shape[0]
            anchor = np.random.permutation(anchor) # 打乱anchor
            img_ = cv_core.show_bbox(img.copy(), anchor[:select_n], thickness=1, is_show=False)
            cv2.imwrite(os.path.join(outPath, pre + str(stride[idx]) + '.jpg'), img_)
            selectnum = anchor[:select_n].shape[0]
            print('{} anchors in stride {}, {} of which are inside image, {} anchors are selected'.format(
                orinum, stride[idx], innum, selectnum))
            #disp_img.append(img_)
        #cv_core.show_img(disp_img, stride)




def demo_retinanet(input_shape_hw,img=None):
    #stride = [8, 16, 32, 64, 128]
    stride = [4, 8, 16, 32, 64]
    anchor_generator_cfg = dict(
        type='AnchorGenerator',
        octave_base_scale=8,  # 每层特征图的base anchor scale,如果变大，则整体anchor都会放大
        scales_per_octave=1,  # 每层有3个尺度 2**0 2**(1/3) 2**(2/3)
        ratios=[1.0],  # 每层的anchor有3种长宽比 故每一层每个位置有9个anchor
        strides=stride)  # 每个特征图层输出stride,故anchor范围是4x8=32,4x128x2**(2/3)=812.7
    random_n = 1
    select_n = 10
    show_anchor(input_shape_hw, stride, anchor_generator_cfg, random_n, select_n,img)


def demo_yolov3(input_shape_hw,img=None):
    stride = [4, 8, 16, 32, 64]
    anchor_generator_cfg = dict(
        type='YOLOAnchorGenerator',
        # base_sizes=[[(116, 90), (156, 198), (373, 326)],
        #             [(116, 90), (156, 198), (373, 326)],
        #             [(116, 90), (156, 198), (373, 326)],
        #             [(30, 61), (62, 45), (59, 119)],
        #             [(10, 13), (16, 30), (33, 23)]],
        base_sizes=[[(3, 4), (6, 5), (4, 9)],
                    [(7, 9), (13, 6), (11, 12)],
                    [(8, 17), (19, 10), (16, 16)],
                    [(13, 25), (30, 16), (22, 29)],
                    [(49, 27), (33, 47), (76, 62)]],
        strides=stride)

    random_n = 1
    select_n = 1000000
    show_anchor(input_shape_hw, stride, anchor_generator_cfg, random_n, select_n,img)


if __name__ == '__main__':
    import os
    imgPath = r'../data/visdrone2018/val_/0000001_08414_d_0000013.jpg'
    outPath = r'../work_dirs/debug/atss_anchor'
    idx = 1
    pre = [r'yolov_9anchor_',r'retinanet_2xfmap_768_'][idx]
    func = [demo_yolov3,demo_retinanet][idx]
    os.makedirs(outPath,exist_ok=True)
    # 1024,1024
    input_shape_hw = (768,768, 3)
    img = cv2.imread(imgPath)
    img = cv2.resize(img,(768,768))
    func(input_shape_hw,img)
