import torch
import torch.nn as nn
import numpy as np
import argparse
import os
import os.path as osp
os.environ['CUDA_VISIBLE_DEVICES'] ='0'
import sys
import time
import json
import cv2
from mmcv import Config
from torchstat import stat
# import torchvision.models as models
# model = models.resnet18()
# stat(model, (3, 768, 1376))



from dataset import build_data_loader
from models import build_model
from models.utils import fuse_module
from utils import ResultFormat, AverageMeter



def report_speed(outputs, speed_meters):
    total_time = 0
    for key in outputs:
        if 'time' in key:
            total_time += outputs[key]
            speed_meters[key].update(outputs[key])
            print('%s: %.4f' % (key, speed_meters[key].avg))

    speed_meters['total_time'].update(total_time)
    print('FPS: %.1f' % (1.0 / speed_meters['total_time'].avg))
    return (1.0 / speed_meters['total_time'].avg)


def test(test_loader, model, cfg):
    model.eval()

    #total_fps = 0
    rf = ResultFormat(cfg.data.test.type, cfg.test_cfg.result_path)

    if cfg.report_speed:
        speed_meters = dict(
            backbone_time=AverageMeter(3000),
            neck_time=AverageMeter(3000),
            det_head_time=AverageMeter(3000),
            det_pa_time=AverageMeter(3000),
            rec_time=AverageMeter(3000),
            total_time=AverageMeter(3000)
        )

    for idx, data in enumerate(test_loader):
        print('Testing %d/%d' % (idx, len(test_loader)))
        sys.stdout.flush()

        # prepare input
        data['imgs'] = data['imgs'].cuda()

        # data['org_img'] = data['org_img'].numpy().astype('uint8')[0] #
        # text_box = data['org_img'].copy()#

        data.update(dict(
            cfg=cfg
        ))

        # forward
        with torch.no_grad():
            outputs = model(**data)

        #print(outputs)

        if cfg.report_speed:
        #     total_fps += report_speed(outputs, speed_meters)
            report_speed(outputs, speed_meters)

        # for bbox in bboxes: #
        #     cv2.drawContours(text_box, [bbox.reshape(4,2)], -1, (0, 255, 0), 2)#
        #save result
        image_name, _ = osp.splitext(osp.basename(test_loader.dataset.img_paths[idx]))
        rf.write_result(image_name, outputs)

        # text_box = cv2.resize(text_box, (text.shape[1], text.shape[0]))#
        # rf.debug(idx,test_loader.dataset.img_paths, [[text_box]], 'outputs/vis_ic15/')#


    # print('mean_total_fps: %.1f' % (total_fps / len(test_loader)))

def main(args):
    cfg = Config.fromfile(args.config)
    for d in [cfg, cfg.data.test]:
        d.update(dict(
            report_speed=args.report_speed
        ))
    print(json.dumps(cfg._cfg_dict, indent=4))
    sys.stdout.flush()

    # data loader
    data_loader = build_data_loader(cfg.data.test)
    test_loader = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    #t0 = time.time()
    # model
    model = build_model(cfg.model)
    model = model.cuda()






    if args.checkpoint is not None:
        if os.path.isfile(args.checkpoint):
            print("Loading model and optimizer from checkpoint '{}'".format(args.checkpoint))
            sys.stdout.flush()

            checkpoint = torch.load(args.checkpoint)

            d = dict()
            for key, value in checkpoint['state_dict'].items():
                tmp = key[7:]
                d[tmp] = value
            model.load_state_dict(d)
        else:
            print("No checkpoint found at '{}'".format(args.resume))
            raise

    # fuse conv and bn
    model = fuse_module(model)


    #t0 = time.time()-t0

    #print('model time: %.2f' % (t0))
    # test
    test(test_loader, model, cfg)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('config', help='config file path')
    parser.add_argument('checkpoint', nargs='?', type=str, default=None)
    parser.add_argument('--report_speed', action='store_true')
    args = parser.parse_args()

    main(args)
