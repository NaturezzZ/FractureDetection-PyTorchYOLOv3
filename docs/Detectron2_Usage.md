# Detectron2使用方法

1.  根据官方教程安装Detectron2

    https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md

    Liux应该可以用：

    ```bash
    pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/index.html
    ```

2.  Register the dataset to detectron2

    https://detectron2.readthedocs.io/tutorials/datasets.html

    针对COCO格式的自定义数据集，可以通过以下方式告知detectron2如何获取数据：

    ```python
    from detectron2.data.datasets import register_coco_instances
    register_coco_instances("fracture_train", {}, "./fracture/annotations/anno_train.json", "./fracture/train")
    register_coco_instances("fracture_val", {}, "./fracture/annotations/anno_val.json", "./fracture/val")
    ```

3.  数据集训练

    ```python
    import random
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data.catalog import MetadataCatalog, DatasetCatalog
    import fruitsnuts_data
    import cv2
    from detectron2.engine import DefaultTrainer
    from detectron2.config import get_cfg
    from detectron2.utils.logger import setup_logger
    import os
    setup_logger()
    
    
    if __name__ == "__main__":
        cfg = get_cfg()
        cfg.merge_from_file(
            "../../configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        )	# select from the following detection baselines
        cfg.DATASETS.TRAIN = ("fracture_train",)
        cfg.DATASETS.TEST = ("fracture_test")  
        cfg.DATALOADER.NUM_WORKERS = 2
        cfg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"  # download pretrained initialization
        cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
        cfg.SOLVER.IMS_PER_BATCH = 2
        cfg.SOLVER.BASE_LR = 0.0025
        cfg.SOLVER.MAX_ITER = 2500
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # only one class (fracture)
        
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)
        trainer.train()
    ```

    一些预训练好的COCO detection baseline，可以从上面下载config文件直接merge：

    https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md

