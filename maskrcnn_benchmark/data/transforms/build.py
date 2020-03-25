from . import transforms_cv2 as cv2_T

def build_test_transforms_cv2(cfg):

    normalize_transform = cv2_T.Normalize(
        mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
    )

    _resize = cv2_T.SquareResize(cfg.INPUT.DATA_SHAPE)

    transform = cv2_T.Compose(
        [
            _resize,
            cv2_T.ToTensor(),
            normalize_transform,
        ]
    )
    return transform

