# A modification version from chainercv repository.
# (See https://github.com/chainer/chainercv/blob/master/chainercv/evaluations/eval_detection_voc.py)
from __future__ import division

import os
from collections import defaultdict
import numpy as np
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
import numpy as np
# import _pickle as cPickle
import pickle

def parse_vid_rec(filename, classhash, img_ids, defaultIOUthr=0.5, pixelTolerance=10):
    """
    parse imagenet vid record into a dictionary
    :param filename: xml file path
    :return: list of dict
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_dict = dict()
        obj_dict['label'] = classhash[obj.find('name').text]
        bbox = obj.find('bndbox')
        obj_dict['bbox'] = [float(bbox.find('xmin').text),
                            float(bbox.find('ymin').text),
                            float(bbox.find('xmax').text),
                            float(bbox.find('ymax').text)]
        gt_w = obj_dict['bbox'][2] - obj_dict['bbox'][0] + 1
        gt_h = obj_dict['bbox'][3] - obj_dict['bbox'][1] + 1
        thr = (gt_w*gt_h)/((gt_w+pixelTolerance)*(gt_h+pixelTolerance))
        obj_dict['thr'] = np.min([thr, defaultIOUthr])
        objects.append(obj_dict)
    return {'bbox' : np.array([x['bbox'] for x in objects]),
             'label': np.array([x['label'] for x in objects]),
             'thr'  : np.array([x['thr'] for x in objects]),
             'img_ids': img_ids}


def vid_ap(rec, prec):
    """
    average precision calculations
    [precision integrated to recall]
    :param rec: recall
    :param prec: precision
    :return: average precision
    """

    # append sentinel values at both ends
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))

    # compute precision integration ladder
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # look for recall value changes
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # sum (\delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def write_vid_results(all_boxes, output_folder, dataset, num_images):
    """
    write results files in pascal devkit path
    :param all_boxes: boxes to be processed [bbox, confidence]
    :return: None
    """
    data_list = dataset.data_list
    num_classes = dataset.num_classes
    print('Writing {} ImageNetVID results file'.format('all'))
    result_path = os.path.join(output_folder, 'result')
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    filename = os.path.join(result_path, 'det_' + data_list + '_all.txt')
    with open(filename, 'wt') as f:
        for im_ind in range(num_images):
            for cls_ind in range(1, num_classes):
                dets = all_boxes[cls_ind][im_ind]
                if len(dets) == 0:
                    continue
                # print(dets)
                # input()
                for k in range(dets.shape[0]):
                    f.write('{:d} {:d} {:.4f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.
                            format(im_ind+1, cls_ind, dets[k, -1],
                                    dets[k, 0], dets[k, 1], dets[k, 2], dets[k, 3]))

def do_python_eval(dataset, output_folder):
    """
    python evaluation wrapper
    :return: info_str
    """
    info_str = ''
    data_path = dataset.root
    data_list = dataset.data_list
    cache_path = os.path.join(data_path, 'cache')

    annopath = os.path.join(data_path, 'Annotations', '{0!s}.xml')
    imageset_file = os.path.join(data_path, 'ImageSets', data_list + '.txt')
    annocache = os.path.join(cache_path, data_list + '_annotations.pkl')
    result_path = os.path.join(output_folder, 'result')
    filename = os.path.join(result_path, 'det_' + data_list + '_all.txt')

    ap = vid_eval(0, filename, annopath, imageset_file, dataset.CLASSES_MAP, annocache, ovthresh=0.5)
    for cls_ind, cls in enumerate(dataset.CLASSES):
        if cls == '__background__':
            continue
        #print('AP for {} = {:.4f}'.format(cls, ap[cls_ind-1]))
        info_str += 'AP for {} = {:.4f}\n'.format(cls, ap[cls_ind-1])
    #print('Mean AP@0.5 = {:.4f}'.format(np.mean(ap)))
    info_str += 'Mean AP@0.5 = {:.4f}\n\n'.format(np.mean(ap))
    with open(os.path.join(result_path, 'result.txt'), 'w') as fid:
        fid.write(info_str) 
    return info_str

def do_ilsvrc_evaluation(dataset, predictions, output_folder, logger):
    num_images = len(predictions)
    num_classes = dataset.num_classes

    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    for image_id, pred_boxlist in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        if len(pred_boxlist) == 0:
            continue
        image_width = img_info["width"]
        image_height = img_info["height"]
        pred_boxlist = pred_boxlist.resize((image_width, image_height))
        pred_bbox = pred_boxlist.bbox.numpy()
        pred_label = pred_boxlist.get_field("labels").numpy()
        pred_score = pred_boxlist.get_field("scores").numpy()
        pred_score = pred_score[..., np.newaxis]
        cls_dets = np.hstack((pred_bbox, pred_score))
        for idx in range(1, dataset.num_classes):
            cls_inds = np.where(pred_label == idx)[0]
            if len(cls_inds) != 0:
                all_boxes[idx][image_id] = cls_dets[cls_inds, :]

    # write detection file list
    pkl_file = os.path.join(output_folder, 'all_boxes.pkl')
    with open(pkl_file, 'wb') as f:
            pickle.dump(all_boxes, f)

    write_vid_results(all_boxes, output_folder, dataset, num_images)
    info = do_python_eval(dataset, output_folder)
    print(info)
    logger.info(info)
    return info


def vid_eval(multifiles, detpath, annopath, imageset_file, classname_map, annocache, ovthresh=0.5):
    """
    imagenet vid evaluation
    :param detpath: detection results detpath.format(classname)
    :param annopath: annotations annopath.format(classname)
    :param imageset_file: text file containing list of images
    :param annocache: caching annotations
    :param ovthresh: overlap threshold
    :return: rec, prec, ap
    """
    with open(imageset_file, 'r') as f:
            lines = [x.strip().split(' ') for x in f.readlines()]
    img_basenames = [x[0] for x in lines]
    gt_img_ids = [int(x[1]) for x in lines]
    classhash = dict(zip(classname_map, range(0,len(classname_map))))

    # load annotations from cache
    if not os.path.isfile(annocache):
        recs = []
        for ind, image_filename in enumerate(img_basenames):
            recs.append(parse_vid_rec(annopath.format('VID/' + image_filename), classhash, gt_img_ids[ind]))
            if ind % 100 == 0:
                print('reading annotations for {:d}/{:d}'.format(ind + 1, len(img_basenames)))
        print('saving annotations cache to {:s}'.format(annocache))
        with open(annocache, 'wb') as f:
            pickle.dump(recs, f)
    else:
        with open(annocache, 'rb') as f:
            recs = pickle.load(f)

    # extract objects in :param classname:
    npos = np.zeros(len(classname_map))
    for rec in recs:
        rec_labels = rec['label']
        for x in rec_labels:
            npos[x] += 1

    # read detections
    splitlines = []
    if (multifiles == False):
        with open(detpath, 'r') as f:
            lines = f.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
    else:
        for det in detpath:
            with open(det, 'r') as f:
                lines = f.readlines()
            splitlines += [x.strip().split(' ') for x in lines]

    img_ids = np.array([int(x[0]) for x in splitlines])
    obj_labels = np.array([int(x[1]) for x in splitlines])
    obj_confs = np.array([float(x[2]) for x in splitlines])
    obj_bboxes = np.array([[float(z) for z in x[3:]] for x in splitlines])

    # sort by confidence
    if obj_bboxes.shape[0] > 0:
        sorted_inds = np.argsort(img_ids)
        img_ids = img_ids[sorted_inds]
        obj_labels = obj_labels[sorted_inds]
        obj_confs = obj_confs[sorted_inds]
        obj_bboxes = obj_bboxes[sorted_inds, :]

    num_imgs = max(max(gt_img_ids),max(img_ids)) + 1
    obj_labels_cell = [None] * num_imgs
    obj_confs_cell = [None] * num_imgs
    obj_bboxes_cell = [None] * num_imgs
    start_i = 0
    id = img_ids[0]
    for i in range(0, len(img_ids)):
        if i == len(img_ids)-1 or img_ids[i+1] != id:
            conf = obj_confs[start_i:i+1]
            label = obj_labels[start_i:i+1]
            bbox = obj_bboxes[start_i:i+1, :]
            sorted_inds = np.argsort(-conf)

            obj_labels_cell[id] = label[sorted_inds]
            obj_confs_cell[id] = conf[sorted_inds]
            obj_bboxes_cell[id] = bbox[sorted_inds, :]
            if i < len(img_ids)-1:
                id = img_ids[i+1]
                start_i = i+1


    # go down detections and mark true positives and false positives
    tp_cell = [None] * num_imgs
    fp_cell = [None] * num_imgs

    for rec in recs:
        id = rec['img_ids']
        gt_labels = rec['label']
        gt_bboxes = rec['bbox']
        gt_thr = rec['thr']
        num_gt_obj = len(gt_labels)
        gt_detected = np.zeros(num_gt_obj)

        labels = obj_labels_cell[id]
        bboxes = obj_bboxes_cell[id]

        num_obj = 0 if labels is None else len(labels)
        tp = np.zeros(num_obj)
        fp = np.zeros(num_obj)

        for j in range(0,num_obj):
            bb = bboxes[j, :]
            ovmax = -1
            kmax = -1
            for k in range(0,num_gt_obj):
                if labels[j] != gt_labels[k]:
                    continue
                if gt_detected[k] > 0:
                    continue
                bbgt = gt_bboxes[k, :]
                bi=[np.max((bb[0],bbgt[0])), np.max((bb[1],bbgt[1])), np.min((bb[2],bbgt[2])), np.min((bb[3],bbgt[3]))]
                iw=bi[2]-bi[0]+1
                ih=bi[3]-bi[1]+1
                if iw>0 and ih>0:            
                    # compute overlap as area of intersection / area of union
                    ua = (bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) + \
                           (bbgt[2] - bbgt[0] + 1.) * \
                           (bbgt[3] - bbgt[1] + 1.) - iw*ih
                    ov=iw*ih/ua
                    # makes sure that this object is detected according
                    # to its individual threshold
                    if ov >= gt_thr[k] and ov > ovmax:
                        ovmax=ov
                        kmax=k
            if kmax >= 0:
                tp[j] = 1
                gt_detected[kmax] = 1
            else:
                fp[j] = 1

        tp_cell[id] = tp
        fp_cell[id] = fp

    tp_all = np.concatenate([x for x in np.array(tp_cell)[gt_img_ids] if x is not None])
    fp_all = np.concatenate([x for x in np.array(fp_cell)[gt_img_ids] if x is not None])
    obj_labels = np.concatenate([x for x in np.array(obj_labels_cell)[gt_img_ids] if x is not None])
    confs = np.concatenate([x for x in np.array(obj_confs_cell)[gt_img_ids] if x is not None])

    sorted_inds = np.argsort(-confs)
    tp_all = tp_all[sorted_inds]
    fp_all = fp_all[sorted_inds]
    obj_labels = obj_labels[sorted_inds]

    ap = np.zeros(len(classname_map))
    for c in range(1, len(classname_map)):
        # compute precision recall
        fp = np.cumsum(fp_all[obj_labels == c])
        tp = np.cumsum(tp_all[obj_labels == c])
        rec = tp / float(npos[c])
        # avoid division by zero in case first detection matches a difficult ground ruth
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap[c] = vid_ap(rec, prec)
    ap = ap[1:]
    return ap
