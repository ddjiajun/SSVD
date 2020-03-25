import os
import sys
import cv2
import math
import numpy as np
import _pickle as cPickle

import torch
import torch.utils.data
from torchvision.transforms import functional as F

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from maskrcnn_benchmark.structures.bounding_box import BoxList

import warnings
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class ILSVRCSeqTest(torch.utils.data.Dataset):

    CLASSES = ('__background__',
                'airplane', 'antelope', 'bear', 'bicycle',
                'bird', 'bus', 'car', 'cattle',
                'dog', 'domestic_cat', 'elephant', 'fox',
                'giant_panda', 'hamster', 'horse', 'lion',
                'lizard', 'monkey', 'motorcycle', 'rabbit',
                'red_panda', 'sheep', 'snake', 'squirrel',
                'tiger', 'train', 'turtle', 'watercraft',
                'whale', 'zebra',)
    CLASSES_MAP = ['__background__',  # always index 0
                    'n02691156', 'n02419796', 'n02131653', 'n02834778',
                    'n01503061', 'n02924116', 'n02958343', 'n02402425',
                    'n02084071', 'n02121808', 'n02503517', 'n02118333',
                    'n02510455', 'n02342885', 'n02374451', 'n02129165',
                    'n01674464', 'n02484322', 'n03790512', 'n02324045',
                    'n02509815', 'n02411705', 'n01726692', 'n02355227',
                    'n02129604', 'n04468005', 'n01662784', 'n04530566',
                    'n02062744', 'n02391049']

    def __init__(self, data_dir, subset, data_list, vid_list, transforms=None):
        self.root = data_dir
        self.image_set = subset # VID or DET sampled list
        self.data_list = data_list

        self.cache_path = os.path.join(data_dir, 'cache')
        self.transforms = transforms
        self.is_train = False
        self._annopath = os.path.join(self.root, "Annotations", self.image_set, "%s.xml")
        self._imgpath = os.path.join(self.root, "Data", self.image_set, "%s.JPEG")
        self._image_set_index_file = os.path.join(self.root, 'ImageSets', self.data_list + '.txt')
        self._video_list_file = os.path.join(self.root, 'ImageSets', vid_list + '.txt')
        
        # flag indicates first or last frame in the sequence
        self._key_flag = -1
        # frame index in one sequence
        self._key_f = 0

        assert os.path.exists(self._image_set_index_file), 'Path does not exist: {}'.format(self._image_set_index_file)

        with open(self._image_set_index_file) as f:
            lines = [x.strip().split(' ') for x in f.readlines()]

        self.image_set_index = [x[0] for x in lines]
        self.frame_index = [int(x[1]) for x in lines]
        self.frame_seg_len = [int(x[2]) for x in lines]
        self.key_f_list = [int(x[3]) for x in lines]
        self.key_flag_list = [int(x[4]) for x in lines]

        # parse video list for sampler split
        with open(self._video_list_file) as f:
            lines = [x.strip().split(' ') for x in f.readlines()]
        self.vid_length = [int(x[3]) for x in lines]

        # self.id_to_img_map = {k: v for k, v in enumerate(self.image_set_index)}
        cls = ILSVRCSeqTest.CLASSES
        cls_map = ILSVRCSeqTest.CLASSES_MAP
        self.class_to_ind = dict(zip(cls_map, range(len(cls))))
        self.num_classes = len(ILSVRCSeqTest.CLASSES)

        # preload roidb into memory
        self._roidb = self.preload_roidb()
        # during training, filter empty frames
        if self.is_train:
            self._roidb = self.filter_roidb(self._roidb)

        # self._roidb = self.cvt2tensor(self._roidb)
        
    def preload_roidb(self):
        """
        return ground truth image regions database
        :return: imdb[image_index]['boxes', 'gt_classes', 'gt_overlaps', 'flipped']
        """
        cache_file = os.path.join(self.cache_path, self.data_list + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                gt_roidb = cPickle.load(fid)
            print('{} gt roidb loaded from {}'.format(self.data_list, cache_file))

        else:
            # parse frame list for get items

            gt_roidb = [self._preprocess_annotation(index) for index in range(0, len(self.image_set_index))]

            with open(cache_file, 'wb') as fid:
                cPickle.dump(gt_roidb, fid)

            print('wrote gt roidb to {}'.format(cache_file))


        return gt_roidb

    def cvt2tensor(self, roidb):
        cvted_roidb = []
        for anno in roidb:
            cvted_anno = dict()
            cvted_anno["boxes"] = torch.tensor(anno["boxes"], dtype=torch.float32)
            cvted_anno["labels"] = torch.tensor(anno["labels"])
            cvted_anno["trackids"] = torch.tensor(anno["trackids"])
            cvted_anno["image"] = anno["image"]
            cvted_anno["im_info"] = anno["im_info"]
            cvted_anno["key_flag"] = anno["key_flag"]
            cvted_anno["frame_id"] = anno["frame_id"]
            cvted_roidb.append(cvted_anno)
        return cvted_roidb

    def filter_roidb(self, roidb):

        def is_valid(entry):
            valid = len(entry['boxes']) > 0

            return valid
        num = len(roidb)
        filtered_roidb = [entry for entry in roidb if is_valid(entry)]
        num_after = len(filtered_roidb)
        print('filtered %d roidb entries: %d -> %d' % (num - num_after, num, num_after))

        return filtered_roidb
    
    def flip_image(self, image, target):
        image = F.hflip(image)
        target = target.transpose(0)
        return image, target

    def __getitem__(self, index):
        ######### key flag ##########
        #   0:        first frame   #
        #   1:        last frame    #
        #   2:        other frames  # 
        #############################
        # TODO :add triplet train
        roidb = self._roidb[index]

        # img_id = roidb['pattern'] % self._key_f
        img_id = roidb['image']
        img = cv2.imread(self._imgpath % img_id)
        height, width = roidb["im_info"]
        # invalid target tobe used in transform
        zeros = torch.tensor([[0, 0, 0, 0]])
        target = BoxList(zeros, (width, height), mode="xyxy")
            
        if self.transforms is not None:
            img, _ = self.transforms(img, target)

        key_flag = roidb['key_flag']
        frame_id = roidb['frame_id']
        seglen = self.frame_seg_len[index]

        return img, key_flag, frame_id, seglen


    def __len__(self):
        print('valid roidb num: ', len(self._roidb))
        return len(self._roidb)

    def get_groundtruth(self, anno):

        #anno["boxes"] = torch.tensor(anno["boxes"], dtype=torch.float32)
        #anno["labels"] = torch.tensor(anno["labels"])
        #anno["trackids"] = torch.tensor(anno["trackids"])

        height, width = anno["im_info"]
        if self.is_train:
            target = BoxList(anno["boxes"], (width, height), mode="xyxy")
            target.add_field("labels", anno["labels"])
            target.add_field("trackids", anno["trackids"])
        else:
            # invalid box for inference
            zeros = torch.tensor([[0, 0, 0, 0]])
            target = BoxList(zeros, (width, height), mode="xyxy")
        return target

    def _preprocess_annotation(self, index):

        img_id = self.image_set_index[index]
        print(self._annopath % img_id)
        anno = ET.parse(self._annopath % img_id).getroot()

        boxes = []
        gt_classes = []
        # difficult_boxes = []
        track_ids = []
        for obj in anno.iter("object"):
            name = obj.find("name").text
            if not name in self.class_to_ind:
                continue
            # difficult = int(obj.find("difficult").text) == 1
            # if not self.keep_difficult and difficult:
            #     continue
            name =  name.lower().strip()
            bb = obj.find("bndbox")
            bndbox = tuple(
                map(
                    int,
                    [
                        bb.find("xmin").text,
                        bb.find("ymin").text,
                        bb.find("xmax").text,
                        bb.find("ymax").text,
                    ],
                )
            )
            if self.image_set == 'VID':
                track_id = int(obj.find('trackid').text)
                track_ids.append(track_id)

            boxes.append(bndbox)
            gt_classes.append(self.class_to_ind[name])
            # difficult_boxes.append(difficult)

        if self.image_set == 'DET':
            track_ids = [i for i in range(len(boxes))]
        size = anno.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        res = {
            "image": img_id,
            "boxes": boxes,
            "labels": gt_classes,
            "trackids": track_ids,
            "im_info": im_info,
            "frame_id": self.frame_index[index],
            "key_f": self.key_f_list[index],
            "key_flag": self.key_flag_list[index],
        }
        return res

    def get_img_info(self, index):
        target = self._roidb[index]
        im_info = target["im_info"]
        return {"height": im_info[0], "width": im_info[1]}

    def map_class_id_to_class_name(self, class_id):
        return ILSVRCSeqTest.CLASSES[class_id]
