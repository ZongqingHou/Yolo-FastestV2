import albumentations as A
import matplotlib.patches as patches
import matplotlib.pyplot as plt

bbox_classes = ["0"]
trans = A.Compose([
    A.RandomCrop(320, 320),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.OneOf([
        A.GaussNoise(),  # 将高斯噪声应用于输入图像。
    ], p=0.2),  # 应用选定变换的概率
    A.OneOf([
        A.MotionBlur(p=0.2),  # 使用随机大小的内核将运动模糊应用于输入图像。
        A.MedianBlur(blur_limit=3, p=0.1),  # 中值滤波
        A.Blur(blur_limit=3, p=0.1),  # 使用随机大小的内核模糊输入图像。
    ], p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=1),
    # 随机应用仿射变换：平移，缩放和旋转输入
    A.RandomBrightnessContrast(p=0.2),  # 随机明亮对比度
    A.Resize(320, 320),
], bbox_params=A.BboxParams(format="yolo", label_fields=["bbox_classes"]))

import cv2

img = cv2.imread("/home/hdd/Desktop/val/24/00072_4.jpg")

bbox = []
with open("/home/hdd/Desktop/val/labels/00072_4.txt", "r") as buffer:
    for tmp in buffer:
        tmp_pp = [float(tmp_p) for tmp_p in tmp[:-1].split(" ")[1:]]
        # tmp_pp[0] = tmp_pp[0] * 720
        # tmp_pp[1] = tmp_pp[1] * 960
        # tmp_pp[2] = tmp_pp[2] * 720
        # tmp_pp[3] = tmp_pp[3] * 960













        bbox.append(tmp_pp)

    tt = trans(image=img, bboxes=bbox, bbox_classes=bbox_classes)
    print("done")

    fig, ax = plt.subplots(1, figsize=(15, 10))
    trans_bbox = tt["bboxes"][0]
    bbox_rect = patches.Rectangle(
        [(trans_bbox[0] - trans_bbox[2] / 2) * 320, (trans_bbox[1] - trans_bbox[3] / 2) * 320],
        trans_bbox[2] * 320,
        trans_bbox[3] * 320,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    ax.imshow(tt["image"])

    ax = plt.gca()
    ax.add_patch(bbox_rect)
    # plt.plot([trans_bbox[0]* 320, trans_bbox[1]* 320,
    #     trans_bbox[2]* 320,
    #     trans_bbox[3]* 320], linewidth=1, color='red')
    plt.show()
