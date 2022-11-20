## Note that when we are running debugger_test.py, we are in conda environment vlm_bench

import torch
import numpy as np
import cv2
import sys
from os.path import join, dirname, abspath, isfile
CURRENT_DIR = dirname(abspath(__file__))
sys.path.insert(0, join(CURRENT_DIR, '..'))
from diffuser.Dataloader import VLM_Waypoint_dataset
from utils.voxelization import VoxelGrid
from utils.helper import visualise_voxel

class Arguments:
    def __init__(self):
        self.data_dir = "/media/xihang/761C73981C7351DB/vlmbench/data"
        self.img_size = (360, 360)
        self.unused_camera_list = [None]
        self.preprocess = False
        self.use_fail_cases = False
        self.sample_method = 'waypoints'
        self.train_tasks = ['door_complex']

        self.batch_size = 1
        self.workers = 4
        self.pin_memory = True
        self.gpu = 0

        self.rotation_resolution = 5 # degree increments per axis
        self.voxel_size = 100
        self.bounds = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]


if __name__=="__main__":
    args = Arguments()
    
    ## some params to be specified
    SCENE_BOUNDS = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6] # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
    VOXEL_SIZES = [100]
    device = torch.device(args.gpu)
    IMAGE_SIZE = 360
    CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist', 'overhead']
    vox_grid = VoxelGrid(
        coord_bounds=SCENE_BOUNDS,
        voxel_size=VOXEL_SIZES[0],
        device=device,
        batch_size=args.batch_size,
        feature_size=3,
        max_num_coords=np.prod([IMAGE_SIZE, IMAGE_SIZE]) * len(CAMERAS),
    )

    ## Read waypoints data
    dataset = VLM_Waypoint_dataset(args.data_dir, 'train', img_size=args.img_size, unused_camera_list = args.unused_camera_list, preprocess = args.preprocess, 
                    use_fail_cases = args.use_fail_cases, sample_method = args.sample_method, train_tasks=args.train_tasks, mood="perceiver", args=args)
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=args.pin_memory, sampler=None, 
        drop_last=True, persistent_workers=True)
    data_iterator = iter(data_loader)
    data = next(data_iterator)
    data = next(data_iterator)
    data = next(data_iterator)
    print("data: ", data.keys())
    # data = {k: v.to(device) for k, v in data.items() if type(v) == torch.Tensor}

    ## flattened point cloud and img already in the dataset
    bounds = torch.tensor(SCENE_BOUNDS, device=device).unsqueeze(0)
    pcd_flat = data['flat_rgb_pcd'][...,3:6].to(device)
    flat_imag_features = data['flat_rgb_pcd'][...,:3].to(device)
    voxel_grid = vox_grid.coords_to_bounding_voxel_grid(pcd_flat, 
                                                    coord_features=flat_imag_features, 
                                                    coord_bounds=bounds)
    vis_voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach().cpu().numpy()
    rotation_amount = 35
    vis_gt_coord = data['trans_indicies'][0].int().numpy()
    rendered_img = visualise_voxel(vis_voxel_grid[0],
                               None,
                               None,
                               vis_gt_coord[0],
                               voxel_size=0.045,
                               rotation_amount=np.deg2rad(rotation_amount))
    cv2.imwrite('point3.png', cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB))
    a = 1