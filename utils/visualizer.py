import cv2
import os

text_gt_folder = '/home/cy/PycharmProjects/pan_pp.pytorch-master/data/ICDAR2015/Challenge4/ch4_test_localization_transcription_gt'
text_gt_list = os.listdir(text_gt_folder)
text_gt_list.sort()

text_folder = '/home/cy/PycharmProjects/pan_pp.pytorch-master/outputs/submit_ic15_finetune/'
text_list = os.listdir(text_folder)
text_list.sort()

image_folder = '/home/cy/PycharmProjects/pan_pp.pytorch-master/data/ICDAR2015/Challenge4/ch4_test_images'
image_list = os.listdir(image_folder)
image_list.sort()


for i, image_name in enumerate(image_list):
    image_path = os.path.join(image_folder, image_name)
    text_gt_path = os.path.join(text_gt_folder, text_gt_list[i])
    text_path = os.path.join(text_folder, text_list[i])
    img = cv2.imread(image_path)  # 读取图片
    with open(text_gt_path, 'r', encoding='UTF-8') as gt, open(text_path, 'r', encoding='UTF-8') as tt:
        lines_gt = gt.read()  # 读取本文行
        lines = tt.read()
    for line_gt in lines_gt.split('\n'):
        pts_gt = []
        if len(line_gt) < 1:
            continue
        for ind, str in enumerate(line_gt.split(',')):
            if len(str) < 1:
                continue
            if '\ufeff' in str:
                str = str.replace('\ufeff', '')
            if ind <= 7:
                pts_gt.append(int(str))
            else:
                ann = str
        cv2.line(img, (pts_gt[0], pts_gt[1]), (pts_gt[2], pts_gt[3]), (255, 255, 255), 2)
        cv2.line(img, (pts_gt[2], pts_gt[3]), (pts_gt[4], pts_gt[5]), (255, 255, 255), 2)
        cv2.line(img, (pts_gt[4], pts_gt[5]), (pts_gt[6], pts_gt[7]), (255, 255, 255), 2)
        cv2.line(img, (pts_gt[6], pts_gt[7]), (pts_gt[0], pts_gt[1]), (255, 255, 255), 2)

    for line in lines.split('\n'):
        pts = []
        if len(line) < 1:
            continue
        for idx, str in enumerate(line.split(',')):
            if len(str) < 1:
                continue
            if idx <= 7:
                pts.append(int(str))

        cv2.line(img, (pts[0], pts[1]), (pts[2], pts[3]), (0, 255, 255), 2)
        cv2.line(img, (pts[2], pts[3]), (pts[4], pts[5]), (0, 255, 255), 2)
        cv2.line(img, (pts[4], pts[5]), (pts[6], pts[7]), (0, 255, 255), 2)
        cv2.line(img, (pts[6], pts[7]), (pts[0], pts[1]), (0, 255, 255), 2)

    cv2.imshow('img', img)
    cv2.waitKey()
