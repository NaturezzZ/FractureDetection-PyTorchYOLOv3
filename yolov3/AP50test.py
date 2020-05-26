# matplotlib inline
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
def AP50_standard_test():
    print("\n************** Final Test ******************")
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)
    # load ground truth
    annFile = './data/json_origin/anno_train.json'
    # annFile = './data/json_origin/anno_val.json'
    cocoGt = COCO(annFile)
    # load your results
    resFile = './data/json_origin/out.json'
    cocoDt = cocoGt.loadRes(resFile)
    imgIds = sorted(cocoGt.getImgIds())
    imgIds = imgIds[0:100]
    imgId = imgIds[np.random.randint(100)]
    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == "__main__":
    AP50_standard_test()