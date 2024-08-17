import os
import random

import albumentations as A
import cv2
import numpy as np
import torch


def contrast_and_brightness(img):
    alpha = random.uniform(0.25, 1.75)
    beta = random.uniform(0.25, 1.75)
    blank = np.zeros(img.shape, img.dtype)
    # dst = alpha * img + beta * blank
    dst = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)
    return dst


def motion_blur(image):
    if random.randint(1,2) == 1:
        degree = random.randint(2,3)
        angle = random.uniform(-360, 360)
        image = np.array(image)
    
        # 这里生成任意角度的运动模糊kernel的矩阵， degree越大，模糊程度越高
        M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))
    
        motion_blur_kernel = motion_blur_kernel / degree
        blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    
        # convert to uint8
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        blurred = np.array(blurred, dtype=np.uint8)
        return blurred
    else:
        return image

def augment_hsv(img, hgain = 0.0138, sgain = 0.678, vgain = 0.36):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
    img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)  # no return needed
    return img


def random_resize(img):
    h, w, _ = img.shape
    rw = int(w * random.uniform(0.8, 1))
    rh = int(h * random.uniform(0.8, 1))

    img = cv2.resize(img, (rw, rh), interpolation = cv2.INTER_LINEAR) 
    img = cv2.resize(img, (w, h), interpolation = cv2.INTER_LINEAR) 
    return img

def img_aug(img):
    img = contrast_and_brightness(img)
    img = motion_blur(img)
    img = random_resize(img)
    img = augment_hsv(img)
    return img

def collate_fn(batch):
    img, label = zip(*batch)
    for i, l in enumerate(label):
        if l.shape[0] > 0:
            l[:, 0] = i
    return torch.stack(img), torch.cat(label, 0)

class TensorDataset():
    def __init__(self, path, img_size_width = 352, img_size_height = 352, imgaug = False):
        assert os.path.exists(path), "%s文件路径错误或不存在" % path

        self.path = path
        self.data_list = []
        self.img_size_width = img_size_width
        self.img_size_height = img_size_height
        self.img_formats = ['bmp', 'jpg', 'jpeg', 'png']
        self.imgaug = imgaug

        self.imgaug_operation = A.Compose([
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
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.1, rotate_limit=30, p=0.5),
            # 随机应用仿射变换：平移，缩放和旋转输入
            A.RandomBrightnessContrast(p=0.2),  # 随机明亮对比度
            A.Resize(self.img_size_width, self.img_size_height),
        ], bbox_params=A.BboxParams(format="yolo", label_fields=["bbox_classes"]))

        # 数据检查
        with open(self.path, 'r') as f:
            for line in f.readlines():
                data_path = line.strip()
                if os.path.exists(data_path):
                    img_type = data_path.split(".")[-1]
                    if img_type not in self.img_formats:
                        raise Exception("img type error:%s" % img_type)
                    else:
                        self.data_list.append(data_path)
                else:
                    raise Exception("%s is not exist" % data_path)

    def __getitem__(self, index):
        img_path = self.data_list[index]
        label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
        if os.path.exists(label_path):
            img = cv2.imread(img_path)

            if self.imgaug:
                bbox = []
                bbox_classes = []
                with open(label_path, "r") as label_buffer:
                    for tmp in label_buffer:
                        tmp_info = tmp.split(" ")
                        bbox_classes.append(tmp_info[0])

                        tmp_bbox = [float(tmp_p) for tmp_p in tmp_info[1:]]
                        for index, tmp_ in enumerate(tmp_bbox):
                            if tmp_ > 1:
                                tmp_bbox[index] = 0.99
                            elif tmp_ < 0:
                                tmp_bbox[index] = 0.01
                            else:
                                continue

                        bbox.append(tmp_bbox)
                        tmp_transform = self.imgaug_operation(image=img, bboxes=bbox, bbox_classes=bbox_classes)

                    label = []
                    for index, tmp_transformed_bbox in enumerate(tmp_transform["bboxes"]):
                        label.append([0, bbox_classes[index],
                                      tmp_transformed_bbox[0],
                                      tmp_transformed_bbox[1],
                                      tmp_transformed_bbox[2],
                                      tmp_transformed_bbox[3]])

                    img = tmp_transform["image"]
            else:
                img = cv2.resize(img, (self.img_size_width, self.img_size_height), interpolation=cv2.INTER_LINEAR)
                label = []
                with open(label_path, 'r') as f:
                    for line in f.readlines():
                        l = line.strip().split(" ")
                        label.append([0, l[0], l[1], l[2], l[3], l[4]])

            img = img.transpose(2, 0, 1)
            label = np.array(label, dtype=np.float32)

            if label.shape[0]:
                assert label.shape[1] == 6, '> 5 label columns: %s' % label_path
                #assert (label >= 0).all(), 'negative labels: %s'%label_path
                #assert (label[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels: %s'%label_path
        else:
            raise Exception("%s is not exist" % label_path)

        return torch.from_numpy(img), torch.from_numpy(label)

    def __len__(self):
        return len(self.data_list)


if __name__ == "__main__":
    data = TensorDataset("/home/xuehao/Desktop/TMP/pytorch-yolo/widerface/train.txt")
    img, label = data.__getitem__(0)
    print(img.shape)
    print(label.shape)
