
import torch
from perceiver_io import PerceiverIO
from diffuser.Dataloader import VLM_Waypoint_training_dataset
from Q_function import QFunction
from PerceiverActorAgent import PerceiverActorAgent
import time
from utils.helper import visualise_voxel, discrete_euler_to_quaternion, get_gripper_render_pose
import numpy as np
import matplotlib.pyplot as plt
import cv2


# constants
TASK = 'open_drawer'
DATA_FOLDER ='peract_colab/data'
EPISODES_FOLDER = 'colab_dataset/open_drawer/all_variations/episodes'
EPISODE_FOLDER = 'episode%d'
CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist']
LOW_DIM_SIZE = 4   # {left_finger_joint, right_finger_joint, gripper_open, timestep}
IMAGE_SIZE =  128  # 128x128 - if you want to use higher voxel resolutions like 200^3, you might want to regenerate the dataset with larger images
VARIATION_DESCRIPTIONS_PKL = 'variation_descriptions.pkl' # the pkl file that contains language goals for each demonstration
EPISODE_LENGTH = 10 # max steps for agents
DEMO_AUGMENTATION_EVERY_N = 10 # sample n-th frame in demo
ROTATION_RESOLUTION = 5 # degree increments per axis

# settings
VOXEL_SIZES = [100] # 100x100x100 voxels
NUM_LATENTS = 512 # PerceiverIO latents
SCENE_BOUNDS = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6] # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
BATCH_SIZE = 1 
NUM_DEMOS = 8 # total number of training demonstrations to use while training PerAct
NUM_TEST = 2 # episodes to evaluate on

class Arguments:
    def __init__(self):
        self.data_dir = "/media/xihang/498ml/vlm_bench/data"
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
        self.num_latents = 512

        self.rotation_resolution = 5 # degree increments per axis
        self.voxel_size = 100
        self.bounds = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]

        self.scene_bounds = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6] # [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
        self.image_size = 360



def main():

    args = Arguments()
    device = torch.device(args.gpu)
    # torch.multiprocessing.set_start_method('spawn')
    CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist', 'overhead']
    

    # initialize PerceiverIO Transformer
    perceiver_encoder = PerceiverIO(
        depth=6,                     
        iterations=1,                
        voxel_size=args.voxel_size,   
        initial_dim=3 + 3 + 1 + 3,   
        low_dim_size=4,              
        layer=0,
        num_rotation_classes=72,     
        num_grip_classes=2,          
        num_collision_classes=2,     
        num_latents=args.num_latents,     
        latent_dim=512,              
        cross_heads=1,   
        latent_heads=8,
        cross_dim_head=64,
        latent_dim_head=64,
        weight_tie_layers=False,   
        activation='lrelu',
        input_dropout=0.1,
        attn_dropout=0.1,
        decoder_dropout=0.0,
        voxel_patch_size=5,           
        voxel_patch_stride=5,        
        final_dim=64,                
    )


    # initialize PerceiverActor
    peract_agent = PerceiverActorAgent(
        coordinate_bounds=args.scene_bounds,
        perceiver_encoder=perceiver_encoder,
        camera_names=CAMERAS,
        batch_size=args.batch_size,
        voxel_size=args.voxel_size,
        voxel_feature_size=3,
        num_rotation_classes=72,
        rotation_resolution=5,
        lr=0.0001,
        image_resolution=[args.image_size, args.image_size],
        lambda_weight_l2=0.000001,
        transform_augmentation=False,
        optimizer_type='lamb',
    )

    peract_agent.build(training=True, device=device)

    LOG_FREQ = 50
    training_epoch = 2



    start_time = time.time()

    # i=0 ## iter
    # for epoch in range(training_epoch):
    #     training_dataset = VLM_Waypoint_training_dataset(args.data_dir, 'train', img_size=args.img_size, unused_camera_list = args.unused_camera_list, preprocess = args.preprocess, 
    #                     use_fail_cases = args.use_fail_cases, sample_method = args.sample_method, train_tasks=args.train_tasks, mood="perceiver", args=args)
    #     train_data_loader = torch.utils.data.DataLoader(
    #         training_dataset, batch_size=args.batch_size, shuffle=True,
    #         num_workers=args.workers, pin_memory=args.pin_memory, sampler=None, 
    #         drop_last=True, persistent_workers=True)        
    #     for batch in train_data_loader:

    #         batch_partial_device = {k: v.to(device) for k, v in batch.items() if type(v) == torch.Tensor}
    #         # print("batch: ", batch.keys())
    #         update_dict = peract_agent.update(i, batch, batch_partial_device)

    #         if i % LOG_FREQ == 0:
    #             elapsed_time = (time.time() - start_time) / 60.0
    #             print("Total Loss: %f | Elapsed Time: %f mins" % (update_dict['total_loss'], elapsed_time))

    #         del batch_partial_device
    #         i+=1


    """ Inference and Visualization """


    test_dataset = VLM_Waypoint_training_dataset(args.data_dir, 'test', img_size=args.img_size, unused_camera_list = args.unused_camera_list, preprocess = args.preprocess, 
                    use_fail_cases = args.use_fail_cases, sample_method = args.sample_method, train_tasks=args.train_tasks, mood="perceiver", args=args)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=args.pin_memory, sampler=None, 
        drop_last=True, persistent_workers=True)
    test_data_iter = iter(test_data_loader)

    batch = next(test_data_iter)
    # lang_goal = batch['lang_goal'][0][0][0]
    lang_goal = batch['lang_goal_embs']
    batch_partial_device = {k: v.to(device) for k, v in batch.items() if type(v) == torch.Tensor}
    update_dict = peract_agent.update(1, batch, batch_partial_device, backprop=False)

    # things to visualize
    vis_voxel_grid = update_dict['voxel_grid'][0].detach().cpu().numpy()
    vis_trans_q = update_dict['q_trans'][0].detach().cpu().numpy()
    vis_trans_coord = update_dict['pred_action']['trans'][0].detach().cpu().numpy()
    vis_gt_coord = update_dict['expert_action']['action_trans'][0].detach().cpu().numpy()

    # discrete to continuous
    continuous_trans = update_dict['pred_action']['continuous_trans'][0].detach().cpu().numpy()
    continuous_quat = discrete_euler_to_quaternion(update_dict['pred_action']['rot_and_grip'][0][:3].detach().cpu().numpy(),
                                                resolution=peract_agent._rotation_resolution)
    gripper_open = bool(update_dict['pred_action']['rot_and_grip'][0][-1].detach().cpu().numpy())
    ignore_collision = bool(update_dict['pred_action']['collision'][0][0].detach().cpu().numpy())

    # gripper visualization pose
    voxel_size = 0.045
    voxel_scale = voxel_size * 100
    gripper_pose_mat = get_gripper_render_pose(voxel_scale, 
                                            SCENE_BOUNDS[:3],
                                            continuous_trans,
                                            continuous_quat)

    show_expert_action = True  #@param {type:"boolean"}
    show_q_values = True  #@param {type:"boolean"}
    render_gripper = True  #@param {type:"boolean"}
    rotation_amount = 90 #@param {type:"slider", min:-180, max:180, step:5}

    rendered_img = visualise_voxel(vis_voxel_grid,
                                vis_trans_q if show_q_values else None,
                                vis_trans_coord,
                                vis_gt_coord if show_expert_action else None,
                                voxel_size=voxel_size,
                                rotation_amount=np.deg2rad(rotation_amount),
                                render_gripper=render_gripper,
                                gripper_pose=gripper_pose_mat,
                                gripper_mesh_scale=voxel_scale)

    fig = plt.figure(figsize=(15, 15))
    cv2.imwrite('after_training.png', cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB))
    

    print(f"Lang Goal: {lang_goal}")






    #     ## Read waypoints data
    # dataset = VLM_Waypoint_dataset(args.data_dir, 'train', img_size=args.img_size, unused_camera_list = args.unused_camera_list, preprocess = args.preprocess, 
    #                 use_fail_cases = args.use_fail_cases, sample_method = args.sample_method, train_tasks=args.train_tasks, mood="perceiver", args=args)
    # data_loader = torch.utils.data.DataLoader(
    #     dataset, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=args.pin_memory, sampler=None, 
    #     drop_last=True, persistent_workers=True)
    # data_iterator = iter(data_loader)
    # data = next(data_iterator)
    # print("data: ", data.keys())
    # # data = {k: v.to(device) for k, v in data.items() if type(v) == torch.Tensor}

    # ## flattened point cloud and img already in the dataset
    # bounds = torch.tensor(SCENE_BOUNDS, device=device).unsqueeze(0)
    # pcd_flat = data['flat_rgb_pcd'][...,3:6].to(device)
    # flat_imag_features = data['flat_rgb_pcd'][...,:3].to(device)
    # voxel_grid = vox_grid.coords_to_bounding_voxel_grid(pcd_flat, 
    #                                                 coord_features=flat_imag_features, 
    #                                                 coord_bounds=bounds)
    # vis_voxel_grid = voxel_grid.permute(0, 4, 1, 2, 3).detach().cpu().numpy()
    # rotation_amount = 35
    # vis_gt_coord = data['trans_indicies'][0].int().numpy()
    # rendered_img = visualise_voxel(vis_voxel_grid[0],
    #                            None,
    #                            None,
    #                            vis_gt_coord[0],
    #                            voxel_size=0.045,
    #                            rotation_amount=np.deg2rad(rotation_amount))
    # cv2.imwrite('point4.png', cv2.cvtColor(rendered_img, cv2.COLOR_BGR2RGB))
    # a = 1


if __name__ == "__main__":
    main()