# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Ke Sun (sunk@mail.ustc.edu.cn)
# ------------------------------------------------------------------------------

import os

import cv2
import numpy as np
from PIL import Image
import random
import torch
from torch.nn import functional as F

from .base_dataset import BaseDataset

class ACDC(BaseDataset):
    def __init__(self, 
                 root, 
                 list_path, 
                 num_samples=None, 
                 num_classes=19,
                 multi_scale=True, 
                 flip=True, 
                 ignore_label=-1, 
                 base_size=2048, 
                 crop_size=(512, 1024), 
                 center_crop_test=False,
                 downsample_rate=1,
                 scale_factor=16,
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225],
                normalize=True,
                jobid=1,
                numjobs=1):

        super(ACDC, self).__init__(ignore_label, base_size,
                                         crop_size, downsample_rate, scale_factor, mean, std,normalize, )

        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.class_weights = torch.FloatTensor([0.8373, 0.918, 0.866, 1.0345, 
                                        1.0166, 0.9969, 0.9754, 1.0489,
                                        0.8786, 1.0023, 0.9539, 0.9843, 
                                        1.1116, 0.9037, 1.0865, 1.0955, 
                                        1.0865, 1.1529, 1.0507]).cuda()

        self.multi_scale = multi_scale
        self.flip = flip
        self.center_crop_test = center_crop_test
        
        self.img_list = [line.strip().split() for line in open(root+list_path)]

        self.files = self.read_files()
            
        if num_samples:
            self.files = self.files[:num_samples]
        print(f'Initializing CityScapes dataset with {len(self.files)} samples')

        chunk_size = int(len(self.files)/numjobs)
        self.files = self.files[(jobid-1)*chunk_size: (jobid-1)*chunk_size + chunk_size]
        print(f'Job ({jobid}/{numjobs}): Splitting dataset into {numjobs} chunks of size {chunk_size}. Range [{(jobid-1)*chunk_size}, {(jobid-1)*chunk_size + chunk_size}] ')

        self.label_mapping = {-1: ignore_label, 0: ignore_label, 
                              1: ignore_label, 2: ignore_label, 
                              3: ignore_label, 4: ignore_label, 
                              5: ignore_label, 6: ignore_label, 
                              7: 0, 8: 1, 9: ignore_label, 
                              10: ignore_label, 11: 2, 12: 3, 
                              13: 4, 14: ignore_label, 15: ignore_label, 
                              16: ignore_label, 17: 5, 18: ignore_label, 
                              19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11,
                              25: 12, 26: 13, 27: 14, 28: 15, 
                              29: ignore_label, 30: ignore_label, 
                              31: 16, 32: 17, 33: 18}
    
    def read_files(self):
        files = []
        if 'test' in self.list_path:
            for item in self.img_list:
                image_path = item
                name = os.path.splitext(os.path.basename(image_path[0]))[0]
                files.append({
                    "img": image_path[0],
                    "name": name,
                })
        else:
            for item in self.img_list:
                image_path, label_path = item
                name = os.path.splitext(os.path.basename(label_path))[0]
                files.append({
                    "img": image_path,
                    "label": label_path,
                    "name": name,
                    "weight": 1
                })
        return files
        
    def convert_label(self, label, inverse=False):
        temp = label.copy() # seg labels 0-19
        if inverse:
            for v, k in self.label_mapping.items(): #keys [-1, 0, 1, 2, ..., 33], values=dict_values([255, 255, 255, 255, 255, 255, 255, 255, 0, 1, 255, 255, 2, 3, 4, 255, 255, 255, 5, 255, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 255, 255, 16, 17, 18])
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label
    def gen_sample(self, image, label, 
            multi_scale=True, is_flip=True, center_crop_test=False):
        if multi_scale:
            rand_scale = 0.5 + random.randint(0, self.scale_factor) / 10.0
            image, label = self.multi_scale_aug(image, label, 
                                                    rand_scale=rand_scale)

        if center_crop_test:
            image, label = self.image_resize(image, 
                                             self.base_size,
                                             label)
            image, label = self.center_crop(image, label)

        image = self.input_transform(image)
        label = self.label_transform(label)
        
        image = image.transpose((2, 0, 1))
        
        if is_flip:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        if self.downsample_rate != 1:
            label = cv2.resize(label, 
                               None, 
                               fx=self.downsample_rate,
                               fy=self.downsample_rate, 
                               interpolation=cv2.INTER_NEAREST)

        return image, label
    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = cv2.imread(os.path.join(self.root,'acdc',item["img"]),
                           cv2.IMREAD_COLOR)
        size = image.shape

        if 'test' in self.list_path:
            image = self.input_transform(image)
            image = image.transpose((2, 0, 1))

            return image.copy(), np.array(size), name

        label = cv2.imread(os.path.join(self.root,'acdc',item["label"]),
                           cv2.IMREAD_GRAYSCALE)
        label = self.convert_label(label)

        image, label = self.gen_sample(image, label, 
                                self.multi_scale, self.flip, 
                                self.center_crop_test)

        return image.copy(), label.copy(), np.array(size), name


    def scale1_noisy_inference(self, model, image, sigma, n):
        BS = 8
        out = []
        remaining = n
        with torch.no_grad():
            while (remaining) > 0:
                size = min(remaining, BS)
                dim = (size, image.size(1), image.size(2), image.size(3))
                batch = image.repeat(size, 1, 1, 1) + sigma * torch.randn(dim, device=image.device) / torch.tensor(self.std).reshape(1,3,1,1)
                out.append(self.inference(model, batch, False).argmax(1).cpu())
                remaining -= size
                print(remaining)
        out = torch.cat(out)
        print(out.size())
        return out


    def multi_scale_inference_noisybatch(self, config, model, n, sigma, image_np, normalize=True, scales=[1], flip=False, unscaled=False, cuda_id=0, border_padding=None, size=None):
        if unscaled: assert len(scales) == 1 and scales[0] <= 1.0
        ori_height, ori_width, _ = image_np.shape
        stride_h = np.int_(self.crop_size[0] * 1.0)
        stride_w = np.int_(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([n, self.num_classes,
                                    ori_height,ori_width]).to(f'cuda:{cuda_id}')
        images = []
        for i in range(n):
            img = image_np + sigma * np.random.randn(*image_np.shape) 
            if normalize:
               img -= self.mean
               img /= self.std
               images.append(img)
        for scale in scales:
            new_images = [self.multi_scale_aug(image=img,
                                           rand_scale=scale,
                                           rand_crop=False).astype(np.float32) for im in images]
            height, width = new_images[0].shape[:-1]
            new_images_s = np.stack(new_images).transpose((0, 3, 1, 2))
            new_image_s = torch.from_numpy(new_images_s)
            if cuda_id is not None:
                new_image_s = new_image_s.to(f'cuda:{cuda_id}')
            
            if scale <= 1.0:
                preds = self.inference(config, model, new_image_s, flip, unscaled=unscaled)
                if not unscaled:
                    preds = preds[:, :, 0:height, 0:width]
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
        return final_pred

    
    def multi_scale_inference(self, model, image, scales=[1], flip=False):
        batch, _, ori_height, ori_width = image.size()
        assert batch == 1, "only supporting batchsize 1."
        image = image.numpy()[0].transpose((1,2,0)).copy()
        stride_h = np.int_(self.crop_size[0] * 1.0)
        stride_w = np.int_(self.crop_size[1] * 1.0)
        final_pred = torch.zeros([1, self.num_classes,
                                    ori_height,ori_width]).cuda()
        print(stride_h, stride_w, final_pred.shape)
        for scale in scales:
            new_img = self.multi_scale_aug(image=image,
                                           rand_scale=scale,
                                           rand_crop=False)
            height, width = new_img.shape[:-1]
            if scale <= 1.0:
                new_img = new_img.transpose((2, 0, 1))
                new_img = np.expand_dims(new_img, axis=0)
                new_img = torch.from_numpy(new_img)
                preds = self.inference(model, new_img, flip)
                preds = preds[:, :, 0:height, 0:width]
            else:
                new_h, new_w = new_img.shape[:-1]
                rows = np.int_(np.ceil(1.0 * (new_h - 
                                self.crop_size[0]) / stride_h)) + 1
                cols = np.int_(np.ceil(1.0 * (new_w - 
                                self.crop_size[1]) / stride_w)) + 1
                preds = torch.zeros([1, self.num_classes,
                                           new_h,new_w]).cuda()
                count = torch.zeros([1,1, new_h, new_w]).cuda()
                print(rows, cols)
                for r in range(rows):
                    for c in range(cols):
                        h0 = r * stride_h
                        w0 = c * stride_w
                        h1 = min(h0 + self.crop_size[0], new_h)
                        w1 = min(w0 + self.crop_size[1], new_w)
                        h0 = max(int(h1 - self.crop_size[0]), 0)
                        w0 = max(int(w1 - self.crop_size[1]), 0)
                        crop_img = new_img[h0:h1, w0:w1, :]
                        crop_img = crop_img.transpose((2, 0, 1))
                        crop_img = np.expand_dims(crop_img, axis=0)
                        crop_img = torch.from_numpy(crop_img)
                        pred = self.inference(model, crop_img, flip)
                        preds[:,:,h0:h1,w0:w1] += pred[:,:, 0:h1-h0, 0:w1-w0]
                        count[:,:,h0:h1,w0:w1] += 1
                preds = preds / count
                preds = preds[:,:,:height,:width]
            preds = F.upsample(preds, (ori_height, ori_width), 
                                   mode='bilinear')
            final_pred += preds
        return final_pred

    def get_palette(self, n):
        palette = [0] * (n * 3)
        for j in range(0, n):
            lab = j
            palette[j * 3 + 0] = 0
            palette[j * 3 + 1] = 0
            palette[j * 3 + 2] = 0
            i = 0
            while lab:
                palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
                palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
                palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
                i += 1
                lab >>= 3
        return palette

    def save_pred(self, preds, sv_path, name):
        palette = self.get_palette(256)
        preds = preds.cpu().numpy().copy()
        preds = np.asarray(np.argmax(preds, axis=1), dtype=np.uint8)
        for i in range(preds.shape[0]):
            pred = self.convert_label(preds[i], inverse=True)
            save_img = Image.fromarray(pred)
            save_img.putpalette(palette)
            save_img.save(os.path.join(sv_path, name[i]+'.png'))

        
        
