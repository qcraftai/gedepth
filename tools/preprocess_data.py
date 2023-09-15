import os
import json
import cv2
import numpy as np
from IPython import embed


data_root_path = "data/kitti/input"

data_root_list = os.listdir(data_root_path)
data_root_list.sort()

for data_root_name in data_root_list:
    data_path = os.path.join(data_root_path,data_root_name)
    
    calib_cam_to_cam_path = os.path.join(data_path,'calib_cam_to_cam.txt')
    calib_velo_to_cam_path = os.path.join(data_path,'calib_velo_to_cam.txt')
    cam_to_world_path = os.path.join(data_path,'cam_to_world.txt')

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

    B = R0_rect * Tr_velo_to_cam
    R = B[:3,:3]
    T = B[:3,3]
    K = P2[:3,:3]

    kr_inv = np.linalg.inv(R) * np.linalg.inv(K)

    R_inv_T = np.linalg.inv(R) * (T*-1)

    data_folder_list = os.listdir(data_path)
    for i in data_folder_list:
        if 'sync' in i:
            break
    

    single_img_path = os.path.join(data_path,i,'image_02/data/0000000000.png')
    ori_img = cv2.imread(single_img_path)
    u,v = np.meshgrid(range(ori_img.shape[1]), range(ori_img.shape[0]), indexing='xy')

    pe_temp = (R_inv_T[2]-1.65)/(kr_inv[2,0]*u+kr_inv[2,1]*v+kr_inv[2,2])

    save_path = os.path.join(data_path,'GE')

    os.makedirs(save_path,exist_ok=True)
    np.save(save_path+"/ge_165.npy",pe_temp)