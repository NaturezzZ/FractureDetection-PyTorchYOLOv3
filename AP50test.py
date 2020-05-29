# matplotlib inline
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
def AP50_standard_test(anno_path, output_path, img_num):
    print("\n************** Final Test ******************")
    pylab.rcParams['figure.figsize'] = (10.0, 8.0)
    # load ground truth
    annFile = anno_path
    # annFile = './data/json_origin/anno_val.json'
    cocoGt = COCO(annFile)
    # load your results
    resFile = output_path
    cocoDt = cocoGt.loadRes(resFile)
    imgIds = sorted(cocoGt.getImgIds())
    imgIds = imgIds[0:img_num]
    imgId = imgIds[np.random.randint(img_num)]
    cocoEval = COCOeval(cocoGt,cocoDt,'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

if __name__ == "__main__":
    AP50_standard_test('data/json_origin/anno_val.json', 'data/json_origin/out.json')
