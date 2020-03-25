# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os
import numpy as np
from tqdm import tqdm
from collections import deque

import torch

from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str


def compute_on_dataset(cfg, model, data_loader, device, timer=None):
    _max_range = cfg.INPUT.SUPPORT_RANGE_TEST
    sup_num = cfg.INPUT.SUPPORT_NUM_TEST
    sup_num = sup_num // 2 * 2

    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        image, key_flag, frame_id, seq_len = batch
        if timer:
            timer.tic()
        if key_flag == 0:
            # initialize memory buffer
            max_range = int(np.minimum(seq_len//2, _max_range))
            fpn_p3_list = deque(maxlen=2*max_range+1)
            fpn_p4_list = deque(maxlen=2*max_range+1)
            fpn_p5_list = deque(maxlen=2*max_range+1)
            fpn_p6_list = deque(maxlen=2*max_range+1)
            fidx_list = deque(maxlen=2*max_range+1)
            data_list = deque(maxlen=2*max_range+1)
            # extract fpn features
            data = image.to(device)
            p3_feat, p4_feat, p5_feat, p6_feat = model.get_fpn_feature(data)
            # push fpn features into memory buffers
            data_list.append(data)
            fidx_list.append(frame_id)
            fpn_p3_list.append(p3_feat)
            fpn_p4_list.append(p4_feat)
            fpn_p5_list.append(p5_feat)
            fpn_p6_list.append(p6_feat)
            while len(fidx_list) < max_range + 1:
                data_list.append(data)
                fidx_list.append(frame_id)
                fpn_p3_list.append(p3_feat)
                fpn_p4_list.append(p4_feat)
                fpn_p5_list.append(p5_feat)
                fpn_p6_list.append(p6_feat)
            if timer:
                torch.cuda.synchronize()
                timer.toc()
        elif key_flag == 2:
            # extract fpn features
            data = image.to(device)
            p3_feat, p4_feat, p5_feat, p6_feat = model.get_fpn_feature(data)
            # push fpn features into memory buffers
            data_list.append(data)
            fidx_list.append(frame_id)
            fpn_p3_list.append(p3_feat)
            fpn_p4_list.append(p4_feat)
            fpn_p5_list.append(p5_feat)
            fpn_p6_list.append(p6_feat)
            if len(fidx_list) == 2 * max_range + 1:
                # detect with memorized features
                cur_input_list, sup_input_list = prepare_data(data_list, fpn_p3_list, fpn_p4_list, fpn_p5_list, fpn_p6_list, max_range, sup_num)
                output = model.aggregate_and_detect(cur_input_list, sup_input_list)
                output = [o.to(cpu_device) for o in output]
                results_dict.update(
                    {fidx_list[max_range]: output[0]}
                )
            if timer:
                torch.cuda.synchronize()
                timer.toc()
        elif key_flag == 1:
            end_counter = 0
            data = image.to(device)
            p3_feat, p4_feat, p5_feat, p6_feat = model.get_fpn_feature(data)
            while len(fidx_list) < 2 * max_range:
                data_list.append(data)
                fidx_list.append(frame_id)
                fpn_p3_list.append(p3_feat)
                fpn_p4_list.append(p4_feat)
                fpn_p5_list.append(p5_feat)
                fpn_p6_list.append(p6_feat)
                end_counter += 1
            while end_counter < max_range + 1:
                data_list.append(data)
                fidx_list.append(frame_id)
                fpn_p3_list.append(p3_feat)
                fpn_p4_list.append(p4_feat)
                fpn_p5_list.append(p5_feat)
                fpn_p6_list.append(p6_feat)
                cur_input_list, sup_input_list = prepare_data(data_list, fpn_p3_list, fpn_p4_list, fpn_p5_list, fpn_p6_list, max_range, sup_num)
                output = model.aggregate_and_detect(cur_input_list, sup_input_list)
                output = [o.to(cpu_device) for o in output]
                results_dict.update(
                    {fidx_list[max_range]: output[0]}
                )
                end_counter += 1
            if timer:
                torch.cuda.synchronize()
                timer.toc()
       
    return results_dict

def prepare_data(data_list, fpn_p3_list, fpn_p4_list, fpn_p5_list, fpn_p6_list, max_range, num_sup):
    # feature and data of the current frame
    cur_p3 = fpn_p3_list[max_range]
    cur_p4 = fpn_p4_list[max_range]
    cur_p5 = fpn_p5_list[max_range]
    cur_p6 = fpn_p6_list[max_range]
    cur_data = data_list[max_range]
    
    sup_data_list = [cur_data.tensors]
    sup_p3_list   = [cur_p3]
    sup_p4_list   = [cur_p4]
    sup_p5_list   = [cur_p5]
    sup_p6_list   = [cur_p6]
    
    interval = 2 * max_range // num_sup
    for i in range(num_sup // 2):
        sup_p3_list.append(fpn_p3_list[max_range-(i+1)*interval])
        sup_p4_list.append(fpn_p4_list[max_range-(i+1)*interval])
        sup_p5_list.append(fpn_p5_list[max_range-(i+1)*interval])
        sup_p6_list.append(fpn_p6_list[max_range-(i+1)*interval])
        sup_data_list.append(data_list[max_range-(i+1)*interval].tensors)

        sup_p3_list.append(fpn_p3_list[max_range+(i+1)*interval])
        sup_p4_list.append(fpn_p4_list[max_range+(i+1)*interval])
        sup_p5_list.append(fpn_p5_list[max_range+(i+1)*interval])
        sup_p6_list.append(fpn_p6_list[max_range+(i+1)*interval])
        sup_data_list.append(data_list[max_range+(i+1)*interval].tensors)
    
    sup_p3 = torch.cat(sup_p3_list, dim=0)
    sup_p4 = torch.cat(sup_p4_list, dim=0)
    sup_p5 = torch.cat(sup_p5_list, dim=0)
    sup_p6 = torch.cat(sup_p6_list, dim=0)
    sup_data = torch.cat(sup_data_list, dim=0)

    sup_input_list = [sup_data, sup_p3, sup_p4, sup_p5, sup_p6]
    cur_input_list = [cur_data, cur_p3, cur_p4, cur_p5, cur_p6]

    return cur_input_list, sup_input_list


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        predictions.update(p)
    # convert a dict where the key is the index in a list
    image_ids = list(sorted(predictions.keys()))
    if len(image_ids) != image_ids[-1] + 1:
        logger = logging.getLogger("maskrcnn_benchmark.inference")
        logger.warning(
            "Number of images that were gathered from multiple processes is not "
            "a contiguous set. Some images might be missing from the evaluation"
        )

    # convert to a list
    predictions = [predictions[i] for i in image_ids]
    return predictions


def inference(
        cfg,
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset
    if not os.path.exists(os.path.join(output_folder, 'predictions.pth')):
    
        logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()
        with torch.no_grad():
            predictions = compute_on_dataset(cfg, model, data_loader, device, inference_timer)
        # wait for all processes to complete before measuring the time
        synchronize()
        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        logger.info(
            "Total run time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time * num_devices / len(dataset), num_devices
            )
        )
        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "Model inference time: {} ({} s / img per device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / len(dataset),
                num_devices,
            )
        )

        predictions = _accumulate_predictions_from_multiple_gpus(predictions)
        if not is_main_process():
            return

        if output_folder:
            torch.save(predictions, os.path.join(output_folder, "predictions.pth"))
    else:
        if not is_main_process():
            return
        predictions = torch.load(os.path.join(output_folder, 'predictions.pth'))

    extra_args = dict(
        box_only=box_only,
        iou_types=iou_types,
        expected_results=expected_results,
        expected_results_sigma_tol=expected_results_sigma_tol,
    )

    return evaluate(dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    **extra_args)
