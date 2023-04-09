# ------------------------------------------------------------------------------
# Reference: https://github.com/SHI-Labs/OneFormer
# Modified by Vidit Goel (https://github.com/vidit98)
# ------------------------------------------------------------------------------

import os
import random
# fmt: off
import sys
sys.path.insert(1, './annotator/OneFormer')
# fmt: on

import imutils
import cv2
import numpy as np

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog

from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from demo.defaults import DefaultPredictor


def setup_cfg(config_file, wts):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(config_file)
    cfg.MODEL.WEIGHTS = wts
    cfg.freeze()
    return cfg


class OneformerSegmenter:
    def __init__(self, wts, config='./annotator/OneFormer/configs/coco/swin/oneformer_swin_large_bs16_100ep.yaml',confidence_thresh=0.5):
        cfg = setup_cfg(config, wts)
        metadata = MetadataCatalog.get(cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused")
        self.predictor = DefaultPredictor(cfg)
        self.metadata = metadata

    def __call__(self, img, task):
        if task == 'panoptic':
            predictions = self.predictor(img, "panoptic")
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            return panoptic_seg, segments_info
        elif task == 'semantic':
            predictions = self.predictor(img, "semantic")
            semask = predictions["sem_seg"].argmax(dim=0)
            return semask