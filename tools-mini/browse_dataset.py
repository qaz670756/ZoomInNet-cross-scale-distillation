import argparse
import os
from pathlib import Path

from mmdet import cv_core
from mmdet.cv_core import Config
from mmdet.datasets.builder import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('--config', type=str,
                        default=r'/home/pc/lby/mmdetection-master/configs/mydet/retinanet_r50_fpn_2x_visdrone2018_veriloss.py',
                        help='train config file path')
    # 以下三个pipeline排除,方便可视化
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=int,
        default=0,
        help='the interval of show (ms)')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    if train_data_cfg.get('dataset', None) is not None:
        # voc数据集
        datasets = train_data_cfg['dataset']
        datasets['pipeline'] = [
            x for x in datasets.pipeline if x['type'] not in skip_type
        ]
    else:
        train_data_cfg['pipeline'] = [
            x for x in train_data_cfg.pipeline if x['type'] not in skip_type
        ]

    return cfg

import cv2
def imshow_det_bboxes(img,gtboxes,gtlabels):
    for box,label in zip(gtboxes,gtlabels):
        x1,y1,x2,y2 = [int(x) for x in box]
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),thickness=1)
        cv2.putText(img,str(int(label)),(x1,y1),cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)
    cv2.imshow('img',img)
    cv2.waitKey(0)
def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type)

    dataset = build_dataset(cfg.data.train)

    progress_bar = cv_core.ProgressBar(len(dataset))
    for item in dataset:
        filename = os.path.join(args.output_dir,
                                Path(item['filename']).name
                                ) if args.output_dir is not None else None
        # cv_core.imshow_det_bboxes(
        #     item['img'],
        #     item['gt_bboxes'],
        #     item['gt_labels'],
        #     class_names=dataset.CLASSES,
        #     show=not args.not_show,
        #     out_file=filename,
        #     wait_time=args.show_interval)
        imshow_det_bboxes(item['img'],item['gt_bboxes'],item['gt_labels'])
        progress_bar.update()


if __name__ == '__main__':
    main()
