from detectron2.data.datasets import register_coco_instances
register_coco_instances("fracture_train", {}, "../FractureDetection/data/fracture/annotations/anno_train.json", "../FractureDetection/data/fracture/train")
register_coco_instances("fracture_val", {}, "../FractureDetection/data/fracture/annotations/anno_val.json", "../FractureDetection/data/fracture/val")

import random
from detectron2.utils.visualizer import Visualizer
from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
import os

setup_logger()

if __name__ == "__main__":
    cfg = get_cfg()
    cfg.merge_from_file("./configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("fracture_train",)
    cfg.DATASETS.TEST = ("fracture_test")
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.MAX_ITER = 2500
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only one class (fracture)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
