## Evaluation Metrics

Ref.

Explanation: https://github.com/rafaelpadilla/Object-Detection-Metrics

Coco evaluation metrics: http://cocodataset.org/#detection-eval

Coco evaluation API: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb

### Definitions

#### Intersection over Union (IoU)

Jaccard Index that evaluates the overlap between two bounding boxes.

<a href="https://www.codecogs.com/eqnedit.php?latex=\text{IoU}&space;=\frac{\text{area}(B_p\cap&space;B_{gt})}{\text{area}(B_p\cup&space;B_{gt})}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\text{IoU}&space;=\frac{\text{area}(B_p\cap&space;B_{gt})}{\text{area}(B_p\cup&space;B_{gt})}" title="\text{IoU} =\frac{\text{area}(B_p\cap B_{gt})}{\text{area}(B_p\cup B_{gt})}" /></a>

Threshold: Usually set to 50%, 75%, 95%.

#### Precision

The ability to identify **only** the relevant objects.

<a href="https://www.codecogs.com/eqnedit.php?latex=Precision&space;=&space;\frac{\text{True&space;Positive}}{\text{all&space;detections}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Precision&space;=&space;\frac{\text{True&space;Positive}}{\text{all&space;detections}}" title="Precision = \frac{\text{True Positive}}{\text{all detections}}" /></a>

#### Recall

The ability to find **all** the relevant cases(all the ground truth).

<a href="https://www.codecogs.com/eqnedit.php?latex=Recall&space;=&space;\frac{\text{True&space;Positive}}{\text{all&space;ground&space;truths}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Recall&space;=&space;\frac{\text{True&space;Positive}}{\text{all&space;ground&space;truths}}" title="Recall = \frac{\text{True Positive}}{\text{all ground truths}}" /></a>



### Metrics

#### Average Precision (AP)

<a href="https://www.codecogs.com/eqnedit.php?latex=AP&space;=&space;\sum_{r=0}^{1}\left(r_{n&plus;1}-r_{n}\right)&space;\rho_{i&space;n&space;t&space;e&space;r&space;p}\left(r_{n&plus;1}\right),\\&space;\rho_{i&space;n&space;t&space;e&space;r&space;p}\left(r_{n&plus;1}\right)&space;=&space;\max_{\tilde{r}:\tilde{r}\ge&space;r_{n&plus;1}}\rho(\tilde{r})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?AP&space;=&space;\sum_{r=0}^{1}\left(r_{n&plus;1}-r_{n}\right)&space;\rho_{i&space;n&space;t&space;e&space;r&space;p}\left(r_{n&plus;1}\right),\\&space;\rho_{i&space;n&space;t&space;e&space;r&space;p}\left(r_{n&plus;1}\right)&space;=&space;\max_{\tilde{r}:\tilde{r}\ge&space;r_{n&plus;1}}\rho(\tilde{r})" title="AP = \sum_{r=0}^{1}\left(r_{n+1}-r_{n}\right) \rho_{i n t e r p}\left(r_{n+1}\right),\\ \rho_{i n t e r p}\left(r_{n+1}\right) = \max_{\tilde{r}:\tilde{r}\ge r_{n+1}}\rho(\tilde{r})" /></a>

Here, <a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{r=0}^1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{r=0}^1" title="\sum_{r=0}^1" /></a> means interpolating through all points from 0 to 1. We take the precision of <a href="https://www.codecogs.com/eqnedit.php?latex=r_{n&plus;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r_{n&plus;1}" title="r_{n+1}" /></a> the maximum precision whose **recall value** is greater or equal than <a href="https://www.codecogs.com/eqnedit.php?latex=r_{n&plus;1}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r_{n&plus;1}" title="r_{n+1}" /></a>.

#### Metrics for COCO

![屏幕快照 2020-05-16 下午8.11.28](https://tva1.sinaimg.cn/large/007S8ZIlgy1geujv9qf9vj31ki0n644z.jpg)

AR is defined similarly as AP. They are both averaged over all **categories and IoUs**.



### Code

Complete code at `FractureDetection/src/test.py`

-   `evaluate()` produces `evalImgs` data structure, which measures quality per-image

-   `accumulate()` produces `eval` data structure, which measures aggregated performance, including **precision and recall** (defined above).

-   `eval` also includes `params`, where parameters used for evaluation are defined. 

    ⚠️ We will see `maxDets` in the result of `summarize()`. It is defined inn `params` and denotes thresholds on **max detections per image**.

-   `analyze()` is not included in `test.py`, since it is currently only for Matlab.

    It plots "Precision x Recall" curve of the model. Note that it takes **significant time** to run.