import sys
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from IPython import embed
import cv2
import time
import multiprocessing
import json
def gen_rgb(value, minimum=0, maximum=255):
    minimum, maximum = float(minimum), float(maximum)
    ratio = 8 * (value-minimum) / (maximum - minimum)
    b = int(max(0, 255*(1 - ratio)))
    r = int(max(0, 255*(ratio - 1)))
    g = 255-b-r
    return r,g,b

def search_height(img_path,gt_path,calib_path,save_path,height):
    with open(calib_path,'r') as f:
        calib = f.readlines()
    ori_img = cv2.imread(img_path)
    save_name = img_path.split('/')[-1]
    gt = cv2.imread(gt_path,-1)/256
    P2 = np.matrix([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3,4)
    R0_rect = np.matrix([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3,3)
    # Add a 1 in bottom-right, reshape to 4 x 4
    R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
    R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)
    Tr_velo_to_cam = np.matrix([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3,4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)
    A = P2 * R0_rect * Tr_velo_to_cam

    R_inv = np.linalg.inv(A[0:3,0:3])
    T = A[0:3,3]
    RT =  R_inv * T

    num_points = []
    
    pe_depth = np.ones((ori_img.shape[0], ori_img.shape[1]))*200
    for i in range(ori_img.shape[1]):
        for j in range(ori_img.shape[0]):
            t = (RT[2]-height)/(R_inv[2,0]*i+R_inv[2,1]*j+R_inv[2,2])
            # t = RT[1]+1.73/(R_inv[1,0]*i+R_inv[1,1]*j+R_inv[1,2])
            if t < 0:continue
            pe_depth[j][i] = t
            if gt[j,i] > 0:
                err = (abs(pe_depth[j][i]-gt[j,i]))/(gt[j,i])
                if err < 0.03:
                    num_points.append(err)
                    cv2.circle(ori_img, (i,j), 1, gen_rgb(pe_depth[j][i]), -1)

    cv2.putText(ori_img, str(len(num_points)), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 255, 0), 2, cv2.LINE_AA)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path,save_name.replace('.png','_'+str(height)+'.png')),ori_img)

    return len(num_points)


def search_height_new(img_path,gt_path,pe_depth,save_path,height):

    ori_img = cv2.imread(img_path)
    save_name = img_path.split('/')[-1]
    gt = cv2.imread(gt_path,-1)/256

    num_points = []
    
    # for i in range(pe_depth.shape[1]):
    #     for j in range(pe_depth.shape[0]):
    #         if gt[j,i] > 0:
    #             err = (abs(pe_depth[j][i]-gt[j,i]))/(gt[j,i])
    #             if err < 0.03:
    #                 num_points.append(err)
    #                 cv2.circle(ori_img, (i,j), 1, gen_rgb(pe_depth[j][i]), -1)
    # cv2.putText(ori_img, str(len(num_points)), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 
    #                1.0, (0, 255, 0), 2, cv2.LINE_AA)
    
    gt[gt == 0] = 0.00001
    err_map = abs(pe_depth-gt)/gt
    coord = np.where(err_map<0.03)
    for i in range(len(coord[0])):
        cv2.circle(ori_img, (coord[1][i],coord[0][i]), 1, gen_rgb(pe_depth[coord[0][i],coord[1][i]]), -1)
    cv2.putText(ori_img, str(len(coord[0])), (100,100), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.0, (0, 255, 0), 2, cv2.LINE_AA)

    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path,save_name.replace('.png','_'+str(height)+'.png')),ori_img)
    # embed()
    # exit()
    return len(coord[0])




def single_process(proc_id,gt_root_list,gt_root_path):
    print("call process :{}".format(proc_id))
    
    height_list = np.arange(155,174)/100
    root_path = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/data/kitti"
    save_root_path = root_path.replace('data/kitti','data/kitti_pe_vis')
    img_root_path = os.path.join(root_path,'input')
    calib_root_path = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/data/kitti/input/2011_09_26/cam_to_world.txt"
    
    for idx,gt_root_run_list in enumerate(gt_root_list):
        run_height_avg = dict()
        print("core:{} processing img {}".format(proc_id, idx))
        gt_list_path = os.path.join(os.path.join(gt_root_path,gt_root_run_list),'proj_depth/groundtruth/image_02')
        
        gt_list = os.listdir(gt_list_path)
        run_height = []
        for gt_name in gt_list:
            gt_path = os.path.join(gt_list_path,gt_name)
            img_root_folder_name = gt_root_run_list.split('drive')[0]
            calib_path = calib_root_path.replace('2011_09_26',img_root_folder_name[0:len(img_root_folder_name)-1])
            img_root_folder_path = os.path.join(img_root_path,img_root_folder_name[0:len(img_root_folder_name)-1])
            img_path = os.path.join(os.path.join(img_root_folder_path,gt_root_run_list),'image_02/data/'+gt_name)
            
            save_path = os.path.join(save_root_path,img_root_folder_name[0:len(img_root_folder_name)-1])
            save_path = os.path.join(save_path,gt_root_run_list)
            num_points = []
            for height in height_list:
                check_exist = save_path+'/'+str(int(height*100))+'/'+img_path.split('/')[-1].replace('.png','_'+str(height)+'.png')
                # if os.path.exists(check_exist):
                    # continue
                print(save_path+'/'+str(int(height*100)))
                # num_points.append(search_height(img_path,gt_path,calib_path,save_path+'/'+str(int(height*100)),height))
                pe_depth = cv2.imread(calib_path.replace('cam_to_world.txt','pe/')+str(int(height*100))+".png",-1)/100
                num_points.append(search_height_new(img_path,gt_path,pe_depth,save_path+'/'+str(int(height*100)),height))
            num_points = np.array(num_points)
            height_id = np.argmax(num_points)
            run_height.append(height_list[height_id])
        run_height_avg[gt_root_run_list] = np.mean(np.array(run_height))
        # embed()
        # exit()
        json.dump(run_height_avg,open(save_path+'/'+save_path.split('/')[-1]+'.json', 'w'))


def err_callback(x):
    print(x)


def multi_process():    
    root_path = "/mnt/vepfs/ML/Users/mazhuang/Monocular-Depth-Estimation-Toolbox/data/kitti"
    gt_root_path = os.path.join(root_path,'gt_depth')
    gt_root_list = os.listdir(gt_root_path)
    # cpu_num = multiprocessing.cpu_count()
    cpu_num = 32
    img_split = np.array_split(gt_root_list, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(img_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    for proc_id, imgs in enumerate(img_split):
        workers.apply_async(single_process,(proc_id, imgs,gt_root_path),error_callback=err_callback)
    workers.close()
    workers.join()
    # single_process(0,["2011_09_28_drive_0035_sync"],gt_root_path)



if __name__ == "__main__":
    multi_process()
