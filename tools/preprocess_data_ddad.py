from IPython import embed
import os
import sys
sys.path.append('/mnt/vepfs/ML/ml-users/mazhuang/dgp')
from dgp.datasets import SynchronizedSceneDataset
import numpy as np
import cv2
import multiprocessing

ddad_json_path = 'data/DDAD/ddad.json'
DATUMS = ['lidar'] + ['CAMERA_%02d' % idx for idx in [1, 5, 6, 9]]
dataset = SynchronizedSceneDataset(ddad_json_path,
    datum_names=DATUMS,
    split='train',
    generate_depth_from_datum='lidar')
scene_dir = dataset.dataset_metadata.directory

for cam_id in range(4):
    sample_idx = 0
    scene_idx, sample_idx_in_scene, datum_indices = dataset.dataset_item_index[sample_idx]
    filename = dataset.get_datum(
                    scene_idx, sample_idx_in_scene, 
                    datum_indices[cam_id]).datum.image.filename
    rgb = np.array(dataset[sample_idx][0][cam_id]['rgb'])
    camera_intrinsics = dataset[sample_idx][0][cam_id]['intrinsics']
    extrinsics = dataset[sample_idx][0][cam_id]['extrinsics'].matrix
    extrinsics = np.matrix(extrinsics)
    camera_pose = dataset[sample_idx][0][cam_id]['pose'].matrix
    lidar_pose = dataset[sample_idx][0][4]['pose'].matrix
    camera_intrinsics = np.matrix(camera_intrinsics)
    camera_intrinsics = np.insert(camera_intrinsics,3,values=[0,0,0],axis=0)
    camera_intrinsics = np.insert(camera_intrinsics,3,values=[0,0,0,1],axis=1)


    A = camera_intrinsics @ np.linalg.inv(camera_pose) @ lidar_pose
    R_inv_T = np.linalg.inv(A[:3,:3])
    T = A[0:3,3]
    RT =  R_inv_T @ T
    u,v = np.meshgrid(range(rgb.shape[1]), range(rgb.shape[0]), indexing='xy')
    l_index = 2
    pe_temp = (RT[l_index])/(R_inv_T[l_index,0]*u+R_inv_T[l_index,1]*v+R_inv_T[l_index,2]) 

    save_path = "data/DDAD/pe_public_debug/"+DATUMS[cam_id+1]+"/ddad_pe.npz"
    np.savez_compressed(save_path, pe=pe_temp)


def find_k(gt_img, pe_img_comput,h):
    a = -h/pe_img_comput
    b = h/gt_img
    k = b+a
    return k

def SingleProcessFindK(proc_id,split_txt):
    slope_set = []
    k_err_dict = {}
    for i in range(len(split_txt)):
        if split_txt[i].split(' ')[1] == "None":
            continue

        gt_path = split_txt[i].split(' ')[1].replace('\n','').replace('depth_val','depth')

        save_k_path = gt_path.replace('.npz','_slope_public_debug.npz')
        cam_num = split_txt[i].split(' ')[1].split('/')[-2]
        pe_path = os.path.join('data/DDAD/pe_public_debug',cam_num+"/ddad_pe.npz")
        gt_img = np.load(gt_path)['depth']
        pe_img_comput = np.load(pe_path)['pe']

        if 'CAMERA_01' == cam_num:
            h = 1.56
        elif 'CAMERA_05' == cam_num:
            h = 1.57
        elif 'CAMERA_06' == cam_num:
            h = 1.53
        elif 'CAMERA_09' == cam_num:
            h = 1.53 

        k = find_k(gt_img, pe_img_comput,h)
        k = np.rad2deg(np.arctan(k)).astype(np.int64)
    
        k[k>5] = 5
        k[k<-5] = -5
        k[gt_img==0] = 255
        os.makedirs(save_k_path.replace(save_k_path.split('/')[-1],''),exist_ok=True)
        np.savez_compressed(save_k_path, k_img=k)
        print("id:{} core:{} find img {}, K is: {}".format(i, proc_id, gt_path, str(np.unique(k))))

def err_callback(value):
    print(value)
    embed()
    exit()

def MultiProcessFindK():
    f = open('splits/ddad_train_split.txt','r')
    all_split_txts = f.readlines()
    f.close()

    split_txts = []
    cam_list = ['CAMERA_01','CAMERA_05','CAMERA_06','CAMERA_09']
    for i in range(len(all_split_txts)):
        if all_split_txts[i].split(' ')[1].split('/')[-2] in cam_list:
            split_txts.append(all_split_txts[i])



    cpu_num = multiprocessing.cpu_count()
    img_split = np.array_split(split_txts, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(img_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    for proc_id, split_txt in enumerate(img_split):
        workers.apply_async(SingleProcessFindK,
                            (proc_id,split_txt),error_callback=err_callback)
    workers.close()
    workers.join()

MultiProcessFindK()