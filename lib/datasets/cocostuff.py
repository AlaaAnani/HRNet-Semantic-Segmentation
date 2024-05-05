# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np

import torch
from torch.nn import functional as F
from PIL import Image

from .base_dataset import BaseDataset


class COCOStuff(BaseDataset):
    def __init__(self,
                 root,
                 list_path,
                 num_samples=None,
                 num_classes=171,
                 multi_scale=True,
                 flip=True,
                 ignore_label=-1,
                 base_size=520,
                 crop_size=(520, 520),
                 downsample_rate=1,
                 scale_factor=11,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 normalize=True):

        super(COCOStuff, self).__init__(ignore_label, base_size,
                                  crop_size, downsample_rate, scale_factor, mean, std, normalize)

        self.root = root
        self.num_classes = num_classes
        self.list_path = list_path
        self.class_weights = None
        self.normalize = normalize
        self.multi_scale = multi_scale
        self.flip = flip
        self.crop_size = crop_size
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
        if num_samples:
            self.files = self.files[:num_samples]
        # self.mapping = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 
        #             21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 
        #             40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 
        #             59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 
        #             78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 92, 93, 94, 95, 96, 
        #             97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 
        #             113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 
        #             129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 
        #             145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 
        #             161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 
        #             177, 178, 179, 180, 181, 182]
        self.mapping = [i for i in range(191)]

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, label_path = item
            image_path = 'cocostuff/' + image_path
            label_path = 'cocostuff/' + label_path
            label_path = label_path.split('.')[0] + '_labelTrainIds.png'
            name = os.path.splitext(os.path.basename(label_path))[0]
            sample = {
                'img': image_path,
                'label': label_path,
                'name': name
            }
            files.append(sample)
        return files

    def encode_label(self, labelmap):
        ret = np.ones_like(labelmap) * 255
        for idx, label in enumerate(self.mapping):
            ret[labelmap == label] = idx

        return ret

    def resize_image(self, image, label, size):
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, size, interpolation=cv2.INTER_NEAREST)
        return image, label
    
    def inference(self, config, model, image, flip=False):
        size = image.size()
        pred = model(image)

        if config.MODEL.NUM_OUTPUTS > 1:
            pred = pred[config.TEST.OUTPUT_INDEX]

        pred = F.interpolate(
            input=pred, size=size[-2:],
            mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
        )

        if flip:
            flip_img = image.numpy()[:, :, :, ::-1]
            flip_output = model(torch.from_numpy(flip_img.copy()))

            if config.MODEL.NUM_OUTPUTS > 1:
                flip_output = flip_output[config.TEST.OUTPUT_INDEX]

            flip_output = F.interpolate(
                input=flip_output, size=size[-2:],
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )

            flip_pred = flip_output.cpu().numpy().copy()
            flip_pred = torch.from_numpy(
                flip_pred[:, :, :, ::-1].copy()).cuda()
            pred += flip_pred
            pred = pred * 0.5
        return pred.exp()
    
    def multi_scale_inference_noisybatch(self, config, model, n, sigma, image_np, 
                                         normalize=True, scales=[1], flip=False, unscaled=False, 
                                         cuda_id=0, border_padding=None, size=None):
        if unscaled: assert len(scales) == 1 and scales[0] <= 1.0
        ori_height, ori_width, _ = image_np.shape
        stride_h = np.int_(self.crop_size[0] * 2.0 / 3.0)
        stride_w = np.int_(self.crop_size[1] * 2.0 / 3.0)
        final_pred = torch.zeros([n, self.num_classes,
                                    ori_height,ori_width]).to(f'cuda:{cuda_id}')
        padvalue = -1.0 * np.array(self.mean) / np.array(self.std)
        images = []
        for i in range(n):
            img = image_np + sigma * np.random.randn(*image_np.shape) 
            if normalize:
               img -= self.mean
               img /= self.std
               images.append(img)
        new_images = []
        for scale in scales:
            for img in images:
                new_img = self.multi_scale_aug(image=img,
                                           rand_scale=scale,
                                           rand_crop=False).astype(np.float32)
                height, width = new_img.shape[:-1]
                if max(height, width) <= np.min(self.crop_size):
                    new_img = self.pad_image(new_img, height, width,
                                            self.crop_size, padvalue)
                new_images.append(new_img)
            new_images_s = np.stack(new_images).transpose((0, 3, 1, 2))
            new_image_s = torch.from_numpy(new_images_s)
            if cuda_id is not None:
                new_image_s = new_image_s.to(f'cuda:{cuda_id}')
            
            if scale <= 1.0:
                preds = self.inference(config, model, new_image_s, flip)
                preds = preds[:, :, 0:height, 0:width]
                preds = F.interpolate(
                    preds, (ori_height, ori_width),
                    mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
                )
                final_pred += preds
            else:
                rows = np.int_(np.ceil(1.0 * (height - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int_(np.ceil(1.0 * (width - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([n, self.num_classes,
                                           height,width]).to(f'cuda:{cuda_id}')
                count = torch.zeros([n,1, height, width]).to(f'cuda:{cuda_id}')

                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], height)
                        w1 = min(w0 + self.crop_size[1], width)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = torch.from_numpy(new_images_s[:, :, h0:h1, w0:w1])
                        if cuda_id is not None:
                            crop_img = crop_img.to(f'cuda:{cuda_id}')
                        pred = self.inference(config, model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]
                if not unscaled:
                    preds = F.upsample(preds, (ori_height, ori_width), 
                                        mode='bilinear')
                    final_pred += preds
                else:
                    final_pred = preds
        if len(border_padding) > 0:
            border_padding_ = border_padding[0]
            final_pred = final_pred[:, :, 0:final_pred.size(2) - border_padding_[0], 0:final_pred.size(3) - border_padding_[1]]

        if final_pred.size()[-2] != size[-2] or final_pred.size()[-1] != size[-1]:
            final_pred = F.interpolate(
                final_pred, size[-2:],
                mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS
            )
        return final_pred
    
    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image_path = os.path.join(self.root, item['img'])
        label_path = os.path.join(self.root, item['label'])
        image = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )
        label = np.array(
            Image.open(label_path).convert('P')
        )
        label = self.encode_label(label)
        label = self.reduce_zero_label(label)
        size = label.shape

        if 'testval' in self.list_path:
            image, border_padding = self.resize_short_length(
                image,
                short_length=self.base_size,
                fit_stride=8,
                return_padding=True
            )
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name, border_padding

        if 'val' in self.list_path:
            image, label = self.resize_short_length(
                image,
                label=label,
                short_length=self.base_size,
                fit_stride=8
            )
            image, label = self.rand_crop(image, label)
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), label.copy(), np.array(size), name

        image, label = self.resize_short_length(image, label, short_length=self.base_size)
        image, label = self.gen_sample(image, label, self.multi_scale, self.flip)

        return image.copy(), label.copy(), np.array(size), name