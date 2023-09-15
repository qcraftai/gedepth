import os
import cv2
import numpy as np
from collections import defaultdict
from IPython import embed

class MIoUEvaluator:
    """
    Evaluate semantic segmentation accuracy in metric "miou"
    """
    def __init__(self,semantic_class,ignore_label=255,**kwargs):
        """
        Args:
            semantic_class (List): names of semantic classes
            ignore_label (int): value in semantic segmentation ground truth. Predictions for the
            corresponding pixels should be ignored.
        """
        
        self.semantic_class = semantic_class
        self._ignore_label = ignore_label
        self._num_classes = len(semantic_class)
        self._N = self._num_classes + 1

        self._conf_matrix = np.zeros((self._N, self._N), dtype=np.int64)

    def update(self,pred,gt):
        pred = pred.astype(np.int64)
        gt = gt.astype(np.int64)
        gt[gt == self._ignore_label] = self._num_classes
        self._conf_matrix += np.bincount(
            self._N * pred.reshape(-1) + gt.reshape(-1), minlength=self._N ** 2
        ).reshape(self._N, self._N)

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):
            Mean intersection-over-union averaged across classes (mIoU)
        """
        acc = np.zeros(self._num_classes, dtype=np.float64)
        iou = np.zeros(self._num_classes, dtype=np.float64)
        tp = self._conf_matrix.diagonal()[:-1].astype(np.float64)
        pos_gt = np.sum(self._conf_matrix[:-1, :-1], axis=0).astype(np.float64)
        pos_pred = np.sum(self._conf_matrix[:-1, :-1], axis=1).astype(np.float64)
        acc_valid = pos_gt > 0
        acc[acc_valid] = tp[acc_valid] / pos_gt[acc_valid]
        iou_valid = (pos_gt + pos_pred) > 0
        union = pos_gt + pos_pred - tp
        iou[acc_valid] = tp[acc_valid] / union[acc_valid]
        
        miou = (np.sum(iou) / np.sum(iou_valid))*100
        res = {}
        # res[self.semantic_class[1]] = iou[1]*100
        res["mIoU"] = miou
        for i in range(self._num_classes):
            res[self.semantic_class[i]] = iou[i]*100
        return res


def check_mask_pe_gt(split,gt_root_path,pe_root_path):
    with open(split) as f:
        for line in f:
            gt_path = line.split(' ')[1]
            if not '2011_09_26_drive_0023_sync/proj_depth/groundtruth/image_02/0000000018.png' in gt_path:
                continue
            embed()
            exit()
            print(gt_root_path+'/'+gt_path)
            if gt_path == 'None': 
                continue
            gt_depth = cv2.imread(gt_root_path+'/'+gt_path,-1)/256
            pe_depth = cv2.imread(pe_root_path+'/'+gt_path.split('/')[0].split('_')[0]+'_'+gt_path.split('/')[0].split('_')[1]+'_'+gt_path.split('/')[0].split('_')[2]+'/'+gt_path.split('/')[0]+'/pe.png',-1)/100
            pe_depth[pe_depth>85] = 0
            gt_err = np.divide(abs(pe_depth-gt_depth),gt_depth,out=np.zeros_like(gt_depth),where=gt_depth!=0)
            gt_mask = gt_err<=0.03
            gt_mask = np.logical_and(gt_mask,gt_depth)
            gt_mask = gt_mask.astype(np.uint8)
            kernel = np.ones((7, 7), 'uint8')
            gt_mask = cv2.dilate(gt_mask, kernel, iterations=1)
            out = np.zeros_like(gt_mask).astype(np.uint8)
            contours,_ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for contour in contours:
                if len(contour) > 1:
                    cv2.fillConvexPoly(out, cv2.convexHull(contour,clockwise=True,returnPoints=True), (1))
            kernel = np.ones((9, 9), 'uint8')
            gt_mask = cv2.erode(out,kernel,iterations = 1)


            


if __name__ == "__main__":
    split = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/data/kitti/kitti_eigen_test.txt"
    gt_root_path = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/data/kitti/gt_depth"
    pe_root_path = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/data/kitti/input"


    save_root_path = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/work_dirs/maskpe_baseline_CE_WEIGHT_Dilate_Erode/kitti_val_gt"
    pe_gt_root_path = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/work_dirs/maskpe_baseline_CE_WEIGHT_Dilate_Erode/kitti_val"
    
    check_mask_pe_gt(split,gt_root_path,pe_root_path)