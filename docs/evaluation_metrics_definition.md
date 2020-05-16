## Evaluation Metrics

Ref.

[1]: https://github.com/rafaelpadilla/Object-Detection-Metrics	"Explanation"
[2]: http://cocodataset.org/#detection-eval	"Coco evaluation metrics"
[3]: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb	"Coco evaluation API"



### Definitions

#### Intersection over Union (IoU)

Jaccard Index that evaluates the overlap between two bounding boxes.
$$
\text{IoU} =\frac{\text{area}(B_p\cap B_{gt})}{\text{area}(B_p\cup B_{gt})}
$$
Threshold: Usually set to 50%, 75%, 95%.

#### Precision

The ability to identify **only** the relevant objects.
$$
Precision = \frac{\text{True Positive}}{\text{all detections}}
$$

#### Recall

The ability to find **all** the relevant cases(all the ground truth).
$$
Recall = \frac{\text{True Positive}}{\text{all ground truths}}
$$


### Metrics

#### Average Precision (AP)

$$
AP = \sum_{r=0}^{1}\left(r_{n+1}-r_{n}\right) \rho_{i n t e r p}\left(r_{n+1}\right),\\
\rho_{i n t e r p}\left(r_{n+1}\right) = \max_{\tilde{r}:\tilde{r}\ge r_{n+1}}\rho(\tilde{r})
$$

Here, $\sum_{r=0}^1$ means interpolating through all points from 0 to 1. We take the precision of $r_{n+1}$ the maximum precision whose **recall value** is greater or equal than $r_{n+1}$.

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

    Here is an example of `analyze()` graph:

    ![image-20200516202825743](https://tva1.sinaimg.cn/large/007S8ZIlgy1geukcuiuntj319a0imduc.jpg)

