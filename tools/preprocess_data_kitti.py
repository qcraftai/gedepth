import os
import sys
import json
import numpy as np
import cv2
import multiprocessing
from IPython import embed

data_root_path = "data/kitti/input"
R_inv_T_dict = dict()
data_root_list = os.listdir(data_root_path)
data_root_list.sort()
a_dict = dict()

for data_root_name in data_root_list:
    data_path = os.path.join(data_root_path,data_root_name)
    
    calib_cam_to_cam_path = os.path.join(data_path,'calib_cam_to_cam.txt')
    calib_velo_to_cam_path = os.path.join(data_path,'calib_velo_to_cam.txt')


    with open(calib_cam_to_cam_path,'r') as f:
        calib_cam_to_cam = f.readlines()

    with open(calib_velo_to_cam_path,'r') as f:
        calib_velo_to_cam = f.readlines()


    P2 = np.matrix([float(x) for x in calib_cam_to_cam[25].strip('\n').split(' ')[1:]]).reshape(3,4)
    R0_rect = np.matrix([float(x) for x in calib_cam_to_cam[8].strip('\n').split(' ')[1:]]).reshape(3,3)
    R0_rect = np.insert(R0_rect,3,values=[0,0,0],axis=0)
    R0_rect = np.insert(R0_rect,3,values=[0,0,0,1],axis=1)

    velo_to_cam_R = np.matrix([float(x) for x in calib_velo_to_cam[1].strip('\n').split(' ')[1:]]).reshape(3,3)
    Tr_velo_to_cam = np.insert(velo_to_cam_R,3,values=np.array(calib_velo_to_cam[2].strip('\n').split(' ')[1:]),axis=1)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam,3,values=[0,0,0,1],axis=0)

    data_folder_list = os.listdir(data_path)
    for i in data_folder_list:
        if 'sync' in i:
            break
    

    single_img_path = os.path.join(data_path,i,'image_02/data/0000000000.png')
    ori_img = cv2.imread(single_img_path)

    A = P2 * R0_rect * Tr_velo_to_cam

    R_inv_T = np.linalg.inv(A[0:3,0:3])
    T = A[0:3,3]
    RT =  R_inv_T * T    
    u,v = np.meshgrid(range(ori_img.shape[1]), range(ori_img.shape[0]), indexing='xy') 
    pe_temp = (RT[2]-1.65)/(R_inv_T[2,0]*u+R_inv_T[2,1]*v+R_inv_T[2,2])
    save_path = os.path.join(data_path,'pe')
    os.makedirs(save_path,exist_ok=True)
    np.save(save_path+"/pe_165.npy",pe_temp)


def find_k(gt_img, pe_img_comput,run_name):
    a = (-1.65)/pe_img_comput
    b = 1.65/gt_img
    k = b+a
    return k

def SingleProcessFindK(proc_id,split_txt):
    gt_path_root = "data/kitti/gt_depth"
    pe_path_root = "data/kitti/input"
    k_err_dict = {}
    slope_set = set()
    for i in range(len(split_txt)):
        
        if split_txt[i].split(' ')[1] == "None":
            continue

        gt_path = gt_path_root+"/"+split_txt[i].split(' ')[1].replace('\n','')
        save_k_path = gt_path.replace('.png','.npz')
        save_k_path = save_k_path.replace('gt_depth','slope_range_5_5_interval_1')
        pe_path = pe_path_root+"/"+split_txt[i].split(' ')[0].split('/')[0]+"/pe/pe_165.npy"
        gt_img = cv2.imread(gt_path,-1)/256
        valid_mask = gt_img == 0
        pe_img_comput = np.load(pe_path).astype(np.float32)

        k = find_k(gt_img, pe_img_comput,split_txt[i].split(' ')[0].split('/')[0])
        run_name = split_txt[i].split(' ')[0].split('/')[0]

        k = np.around(np.rad2deg(np.arctan(k)))
        k[k>5] = 5
        k[k<-5] = -5
        k[valid_mask] = 255

        os.makedirs(save_k_path.replace(save_k_path.split('/')[-1],''),exist_ok=True)
        np.savez_compressed(save_k_path, k_img=k)
        print("id:{} core:{} find img {}, K is: {}".format(i, proc_id, gt_path, str(np.unique(k))))

def err_callback(value):
    print(value)

def MultiProcessFindK():
    f = open('data/kitti/kitti_eigen_train.txt','r')
    split_txts = f.readlines()
    cpu_num = multiprocessing.cpu_count()
    # cpu_num = 115
    img_split = np.array_split(split_txts, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(img_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    for proc_id, split_txt in enumerate(img_split):
        workers.apply_async(SingleProcessFindK,
                            (proc_id,split_txt),error_callback=err_callback)
    workers.close()
    workers.join()




MultiProcessFindK()