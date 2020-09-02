from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import pycocotools.coco as coco


import torch.utils.data as data
import numpy as np
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import torch


class DATASET_CUSTOM(data.Dataset):


  def __init__(self, config, split):
    """

    :param opt:
    :param split: train/val
    """
    super(DATASET_CUSTOM, self).__init__()
    self.data_dir = os.path.join(config['data_dir'], config['dataset']['name'])
    self.img_dir = os.path.join(self.data_dir, 'images'.format(split))
    self.not_rand_crop = config['dataset']['not_rand_crop']
    self.keep_res = config['dataset']['keep_res']
    self.input_h = config['model']['input_h']
    self.input_w = config['model']['input_h']
    self.pad = config['model']['pad']
    self.scale = config['dataset']['scale']
    self.shift = config['dataset']['shift']
    self.flip = config['dataset']['flip']
    self.down_ratio = config['model']['down_ratio']
    self.no_color_aug = config['dataset']['no_color_aug']
    self.debug = config['train']['debug']
    self.mean = config['dataset']['mean']
    self.std = config['dataset']['std']
    self.max_objs = config['dataset']['max_obj']
    self.num_classes = config['dataset']['num_classes']


    self.annot_path = os.path.join(
          self.data_dir, 'annotations',
          '{}_cmnd.json').format(split)
    # print(self.data_dir)

    self.class_name = [
      '__background__', 'corner', 'quochuy']
    self._valid_ids = [1, 2]
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)

    self.split = split

    print('==> initializing coco 2017 {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def _coco_box_to_bbox(self, box):  # coco box [x_min, y_min, width, height] -> [x_min, y_min, x_max, y_max]
      bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],
                      dtype=np.float32)
      return bbox

  def _cocobox_to_center(self, box):
      center = np.array([box[0] + box[2] / 2, box[1] + box[3] / 2], dtype=np.float32)  # [x_center, y_center]

      return center

  def __bbox_to_center(self, bbox):
      center = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
      return center

  def _get_border(self, border, size):
      i = 1
      while size - border // i <= border // i:
          i *= 2
      return border // i

  def __getitem__(self, index):
      img_id = self.images[index]
      file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
      img_path = os.path.join(self.img_dir, file_name)
      ann_ids = self.coco.getAnnIds(imgIds=[img_id])
      anns = self.coco.loadAnns(ids=ann_ids)
      num_objs = min(len(anns), self.max_objs)
      print(img_path)
      img = cv2.imread(img_path)  # BGR

      height, width = img.shape[0], img.shape[1]
      c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
      if self.keep_res:
          input_h = (height | self.pad) + 1
          input_w = (width | self.pad) + 1
          s = np.array([input_w, input_h], dtype=np.float32)
      else:
          s = max(img.shape[0], img.shape[1]) * 1.0
          input_h, input_w = self.input_h, self.input_w

      flipped = False

      if self.split == 'train':
          if not self.not_rand_crop:
              s = s * np.random.choice(np.arange(0.6, 1.4, 0.1))
              w_border = self._get_border(128, img.shape[1])
              h_border = self._get_border(128, img.shape[0])
              c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
              c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
          else:
              sf = self.scale
              cf = self.shift
              c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
              c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
              s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

          if np.random.random() < self.flip:
              flipped = True
              img = img[:, ::-1, :]
              c[0] = width - c[0] - 1

      #
      trans_input = get_affine_transform(
          c, s, 0, [input_w, input_h])
      inp = cv2.warpAffine(img, trans_input,
                           (input_w, input_h),
                           flags=cv2.INTER_LINEAR)
      inp = (inp.astype(np.float32) / 255.)
      if self.split == 'train' and not self.no_color_aug:
          color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
      inp = (inp - self.mean) / self.std
      inp = inp.transpose(2, 0, 1)

      output_h = input_h // self.down_ratio
      output_w = input_w // self.down_ratio
      num_classes = self.num_classes
      trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

      hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)

      reg = np.zeros((self.max_objs, 2), dtype=np.float32)
      ind = np.zeros((self.max_objs), dtype=np.int64)
      reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
      cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

      # ???
      draw_gaussian = draw_umich_gaussian       # remove mse_loss ~ draw_msra_gaussian

      gt_det = []
      for k in range(num_objs):
          ann = anns[k]
          bbox = self._coco_box_to_bbox(ann['bbox'])

          cls_id = int(self.cat_ids[ann['category_id']])

          # Transform box when flip, resize
          if flipped:
              bbox[[0, 2]] = width - bbox[[2, 0]] - 1
          bbox[:2] = affine_transform(bbox[:2], trans_output)
          bbox[2:] = affine_transform(bbox[2:], trans_output)
          bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
          bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
          h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]  # height, width of box
          if h > 0 and w > 0:
              radius = 3

              ct = self.__bbox_to_center(bbox)  # center
              ct_int = ct.astype(np.int32)  # center integer
              draw_gaussian(hm[cls_id], ct_int, radius)

              ind[k] = ct_int[1] * output_w + ct_int[0]
              reg[k] = ct - ct_int
              reg_mask[k] = 1

              cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
              gt_det.append([ct[0], ct[1], 1, cls_id])  # x_center, y_center, 1, label_id

      ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'reg': reg}

      if self.debug > 0 or not self.split == 'train':
          gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
              np.zeros((1, 6), dtype=np.float32)
          meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
          ret['meta'] = meta
      return ret

  def __len__(self):
      return self.num_samples