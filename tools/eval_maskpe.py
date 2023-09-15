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


def val():
    err_list = []
    iou = MIoUEvaluator(semantic_class=['Mask Value 0 IoU','Mask Value 1 IoU'])
    
    split = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/data/kitti/kitti_eigen_test.txt"
    gt_root_path = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/data/kitti/gt_depth"
    pe_gt_root_path = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/work_dirs/maskpe_baseline_CE_WEIGHT_Dilate_Erode/kitti_val"
    pe_root_path = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/data/kitti/input"
    save_root_path = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/work_dirs/maskpe_baseline_CE_WEIGHT_Dilate_Erode/kitti_val_gt"
    with open(split) as f:
        for line in f:
            gt_path = line.split(' ')[1]
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
            pre_pe_mask = (cv2.imread(pe_gt_root_path+'/'+gt_path.split('/')[0].split('_')[0]+'_'+gt_path.split('/')[0].split('_')[1]+'_'+gt_path.split('/')[0].split('_')[2]+'_'+gt_path.split('/')[0]+'_image_02_data_'+gt_path.split('/')[-1],-1)/255)
            
            pre_pe_mask = cv2.resize(pre_pe_mask,(gt_depth.shape[1],gt_depth.shape[0]),cv2.INTER_NEAREST).astype(np.uint8)
            iou.update(pre_pe_mask,gt_mask)
            # cv2.imwrite(save_root_path+'/'+gt_path.split('/')[0].split('_')[0]+'_'+gt_path.split('/')[0].split('_')[1]+'_'+gt_path.split('/')[0].split('_')[2]+'_'+gt_path.split('/')[0]+'_image_02_data_'+gt_path.split('/')[-1],gt_mask*255)
            # pe_mask_pred = pre_pe_mask*pe_depth
            # pe_mask_gt = gt_mask*pe_depth
            # # pe_err = np.divide(abs(pe_mask_pred-gt_depth),gt_depth,out=np.zeros_like(gt_depth),where=gt_depth!=0)
            # pe_err = np.zeros((pe_mask_pred.shape[0],pe_mask_pred.shape[1]))-1
            # # embed()
            # # exit()
            # for i in range(pe_mask_pred.shape[0]):
            #     for j in range(pe_mask_pred.shape[1]):
            #         if not pe_mask_pred[i][j] == 0 and not gt_depth[i][j] == 0:
            #             # embed()
            #             # exit()
            #             pe_err[i][j] = abs(pe_mask_pred[i][j]-gt_depth[i][j])/gt_depth[i][j]
            # err_list.append(pe_err.max())
            # embed()
            # exit()
            
            
    result = iou.evaluate()
    embed()
    exit()



if __name__ == "__main__":
    iou_all = MIoUEvaluator(semantic_class=['Mask Value 0 IoU','Mask Value 1 IoU'])
    miou_90 = []
    gt_path = "/mnt/vepfs/ML/Users/mazhuang/PE/Monocular-Depth-Estimation-Toolbox/val_pemaskgt"
    pe_path = "/mnt/vepfs/ML/Users/mazhuang/PE/Monocular-Depth-Estimation-Toolbox/work_dirs/maskpe_baseline_CE_weight_backup/kitti_val_11200_518"
    gt_path_list = os.listdir(gt_path)
    gt_path_list = sorted(gt_path_list)
    for gt_name in gt_path_list:
        iou = MIoUEvaluator(semantic_class=['Mask Value 0 IoU','Mask Value 1 IoU'])

        gt_mask = cv2.imread(gt_path+"/"+gt_name,-1)/255
        gt_mask = gt_mask.astype(np.uint8)

        pe_mask = cv2.imread(pe_path+"/"+gt_name,-1)/255
        # temp_pe_mask = pe_mask.astype(np.uint8)
        height = gt_mask.shape[0]
        width = gt_mask.shape[1]
        top_margin = int(height - 352)
        left_margin = int((width - 1216) / 2)
        temp_pe_mask = np.zeros_like(gt_mask).astype(np.uint8)
        temp_pe_mask[top_margin: top_margin + 352, left_margin: left_margin + 1216] = pe_mask

        # kernel = np.ones((35,35), 'uint8')
        # random = np.random.rand()
        # if random > 0.5:
        #     pe_mask = cv2.erode(gt_mask, kernel, iterations=1)
        # else:
        #     pe_mask = cv2.dilate(gt_mask, kernel, iterations=1)
        # cv2.imwrite('./pe_mask.png',temp_pe_mask*255)
        # cv2.imwrite('./gt_mask.png',gt_mask*255)
        iou.update(temp_pe_mask,gt_mask)
        # iou_all.update(temp_pe_mask,gt_mask)
        result = iou.evaluate()
        # if result['mIoU'] > 50 and result['Mask Value 1 IoU']>0:
        if 90<=result['Mask Value 1 IoU']<100:
        # if 75<=result['Mask Value 1 IoU']:
            print(result)
            iou_all.update(temp_pe_mask,gt_mask)
            miou_90.append([pe_path+"/"+gt_name,str(result['mIoU'])])
    embed()
    exit()
    for iou_result in miou_90:
        path_split = iou_result[0].split('/')[-1].split('_')
        img_path = path_split[0]+"_"+path_split[1]+"_"+path_split[2]+"/"+path_split[3]+"_"+path_split[4]+"_"+path_split[5]+"_"+path_split[6]+"_"+path_split[7]+"_"+path_split[8]+"/"+path_split[9]+"_"+path_split[10]+"/"+path_split[11]+"/"+path_split[12]+" "+path_split[3]+"_"+path_split[4]+"_"+path_split[5]+"_"+path_split[6]+"_"+path_split[7]+"_"+path_split[8]+"/proj_depth/groundtruth/"+path_split[9]+"_"+path_split[10]+"/"+path_split[12]
        print(img_path)
    result = iou_all.evaluate()
    print(result)