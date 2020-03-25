import logging

from .ilsvrc_eval import do_ilsvrc_evaluation


def ilsvrc_evaluation(dataset, predictions, output_folder, box_only, **_):
    logger = logging.getLogger("ILSVRC VID inference")
    if box_only:
        logger.warning("ILSVRC evaluation doesn't support box_only, ignored.")
    logger.info("performing ILSVRC VID evaluation, ignored iou_types.")
    return do_ilsvrc_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        logger=logger,
    )
