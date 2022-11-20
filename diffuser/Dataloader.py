import os
from random import sample
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
from pathlib import Path
import pickle
from cliport.utils.utils import get_fused_heightmap
pickle.DEFAULT_PROTOCOL=pickle.HIGHEST_PROTOCOL
from amsolver.observation_config import ObservationConfig
from amsolver.utils import get_stored_demos
import time
import copy
from scipy.spatial.transform import Rotation as R
from num2words import num2words
import utils.helper as helper


class VLM_dataset(Dataset):
    def __init__(self, root, setd, img_size=(360, 360), 
                    unused_camera_list = ['left_shoulder', 'right_shoulder', 'overhead','wrist'], preprocess = True, 
                    use_fail_cases = True, sample_method="waypoints", train_tasks = None, args=None, mood="diffuser"):
        self.root = root
        self.setd = setd
        self.dataset_path = Path(os.path.join(self.root, self.setd))
        self.episode_list = []
        self.variation_list = []
        self.task_list = {}
        self.fail_cases_list = []
        self.train_tasks = train_tasks
        self.read_lists()
        self.use_fail_cases = use_fail_cases
        if train_tasks is not None:
            self.episode_list = []
            for t in train_tasks:
                for n in self.task_list:
                    if t in n:
                        self.episode_list += self.task_list[n]['success']
                        self.fail_cases_list += self.task_list[n]['fail']
        if use_fail_cases:
            self.episode_list += self.fail_cases_list
        #only train selected tasks

        self.valid_episodes, self.invalid_episodes = set(),set()
        self.img_size = img_size
        self.sample_method = sample_method
        self.preprocess = preprocess

        self.obs_config = ObservationConfig()
        self.obs_config.set_all(True)
        self.obs_config.right_shoulder_camera.image_size = self.img_size
        self.obs_config.left_shoulder_camera.image_size = self.img_size
        self.obs_config.overhead_camera.image_size = self.img_size
        self.obs_config.wrist_camera.image_size = self.img_size
        self.obs_config.front_camera.image_size = self.img_size

        self.views = list(set(['left_shoulder', 'right_shoulder', 'overhead', 'wrist', 'front']) - set(unused_camera_list))

        if 'left_shoulder' in unused_camera_list:
            self.obs_config.left_shoulder_camera.set_all(False)
        if 'right_shoulder' in unused_camera_list:
            self.obs_config.right_shoulder_camera.set_all(False)
        if 'overhead' in unused_camera_list:
            self.obs_config.overhead_camera.set_all(False)
        if 'wrist' in unused_camera_list:
            self.obs_config.wrist_camera.set_all(False)
        if 'front' in unused_camera_list:
            self.obs_config.front_camera.set_all(False)
        
        # self.relative = False
        # self.renew_obs = False
        # self.add_low_lang = False
        # if args is not None:
        #     self.relative = args.relative
        #     self.renew_obs = args.renew_obs
        #     self.add_low_lang = args.add_low_lang
        self.args = args
        self.mood = mood
        if self.mood == 'diffuser':
            self.diffuser_config()


    def read_lists(self):
        print("self.dataset_path: ", self.dataset_path)
        tasks_list_path = self.dataset_path / '{}_list.pkl'.format(self.setd)
        print("tasks_list_path: ", tasks_list_path)
        if not tasks_list_path.is_file():
            self.task_list = {}
            self.variation_list =set()
            for path in self.dataset_path.rglob('low_dim_obs*'):
                path = path.relative_to(self.dataset_path)
                task_name = str(path.parents[3])
                if task_name not in self.task_list:
                    self.task_list[task_name]={'success':[], 'fail':[]}
                self.variation_list.add(path.parents[2])
                if 'fail_cases' in str(path):
                    self.fail_cases_list.append(path.parent)
                    self.task_list[task_name]['fail'].append(path.parent)
                else:
                    self.episode_list.append(path.parent)
                    self.task_list[task_name]['success'].append(path.parent)
            self.variation_list = list(self.variation_list)
            with open(tasks_list_path,'wb') as f:
                pickle.dump({'task_list': self.task_list, 
                            'episode_list': self.episode_list,
                            'fail_cases_list': self.fail_cases_list,
                            'variation_list': self.variation_list}, f)
        else:
            with open(tasks_list_path,'rb') as f:
                info_dict = pickle.load(f)
                self.task_list = info_dict['task_list']
                self.episode_list = info_dict['episode_list']
                self.variation_list = info_dict['variation_list']
                self.fail_cases_list = info_dict['fail_cases_list']

    def __getitem__(self, index):
        if index in self.invalid_episodes:
            index = sample(self.valid_episodes, 1)[0]
        episode = self.episode_list[index]

        low_dim_obs = self.dataset_path/episode/"low_dim_obs.pkl"
        with open(low_dim_obs, 'rb') as f:
            demo_temple = pickle.load(f)
        
        if self.mood == 'diffuser':
            output_dict = self.get_diffuser_gt(demo_temple._observations)
        # elif self.mood == 'perceiver':
        #     obs_select_inds = self.sample_steps(demo_temple)

        #     variation_path = episode.parents[1]
        #     task_name = episode.parents[2]
        #     fail_cases = 'fail_cases' in str(episode)
        #     episode_name = episode.name
        #     variation_number = int(variation_path.name.replace('variation',''))
        #     demos = get_stored_demos(1, False, self.dataset_path, variation_number, 
        #                             task_name, self.obs_config, episode_name, fail_cases, obs_select_inds)
        
        if output_dict['valid']:
            self.valid_episodes.add(index)
        else:
            self.invalid_episodes.add(index)
            if len(self.valid_episodes) == 0:
                other_indexs = list(set(range(self.__len__())) - self.invalid_episodes)
                valid_index = sample(other_indexs, 1)[0]
            else:
                valid_index = sample(self.valid_episodes, 1)[0]
            output_dict = self.__getitem__(valid_index)
        return output_dict

    def diffuser_config(self):
        self.horizon_timestep = 64
        self.obs_dim = 8

        conditions = [
        ([0], 0), ## first
        ([-1], 0), ## last
        ([0,-1], 1), ## first and last
        ]
        conditions_k, conditions_p = zip(*conditions)
        self.conditions_k = np.array(conditions_k, dtype=np.object)
        self.conditions_p = np.array(conditions_p) / sum(conditions_p)
    
    def get_diffuser_gt(self, data):

        object_info = data[0].object_informations
        waypoint_name="waypoint{}"
        step_points = []
        i=0
        while True:
            try:
                wp_info = object_info[waypoint_name.format(i)]
                wp_type = wp_info['waypoint_type']
                if not ("pre" in wp_type or "post" in wp_type):
                    step_points.append(waypoint_name.format(i))
                i+=1
            except:
                break
        select_index = [0]
        wp_index = 0
        joint_positions, ee_matrices = [], []
        begin=False
        for i, obs in enumerate(data):
            if obs.current_waypoint_name==step_points[wp_index]:
                begin=True
            elif obs.current_waypoint_name!=step_points[wp_index] and begin:
                begin=False
                select_index.append(i)
                wp_index+=1
            joint_positions.append(obs.joint_positions)
            ee_matrices.append(obs.gripper_matrix)
        joint_positions = np.array(joint_positions)
        ee_matrices = np.array(joint_positions)
        select_index.append(len(data))
        select_index = list(zip(select_index[:-1], select_index[1:]))
        interp_all_joints, masks, step_nums = [], [], []
        for i, (start, end) in enumerate(select_index):
            time_steps = np.linspace(start, end-1, self.horizon_timestep)
            step = (end-start)/self.horizon_timestep
            time_steps[1:-1] += np.random.normal(scale=step/2)
            xp = list(range(start, end))
            fp = joint_positions[start:end]
            interp_joints = [np.interp(time_steps, xp, fp[:, i]) for i in range(joint_positions.shape[1])]
            interp_joints = np.stack(interp_joints, axis=1)
            cond = np.random.choice(self.conditions_k, p=self.conditions_p)
            mask = np.zeros_like(interp_joints[..., -1])
            for t in cond:
                mask[t] = 1
            interp_all_joints.append(interp_joints)
            masks.append(mask)
            step_nums.append(i+1)
        normalized_interp_all_joints = self.normalize(np.array(interp_all_joints))
        step_nums = np.array(step_nums)
        step_nums = step_nums/step_nums.max()
        step_nums = step_nums*2-1
        step_nums = np.repeat(step_nums[:, np.newaxis, np.newaxis], self.horizon_timestep, axis=1)
        normalized_interp_all_joints = np.concatenate((normalized_interp_all_joints, step_nums), -1)
        masks = np.array(masks)
        output_dict = {
            "interp_joints":normalized_interp_all_joints,
            "masks":masks,
            "original_joints":joint_positions,
            "original_ee_poses":ee_matrices,
            'valid':1
        }
        return output_dict

    def sample_steps(self, demo_temple):
        sequence_length = len(demo_temple._observations)
        obs_select_inds = np.arange(sequence_length)
        if self.sample_method == 'random':
            obs_select_inds = np.sort(np.random.choice(obs_select_inds, self.sample_numbers, replace=False))
        elif self.sample_method == 'waypoints':
            obs_select_inds = [0]
            previous_waypoint="waypoint0"
            all_waypoints = [previous_waypoint]
            for i, obs in enumerate(demo_temple._observations):
                if obs.current_waypoint_name == previous_waypoint:
                    continue
                else:
                    previous_waypoint = obs.current_waypoint_name
                    all_waypoints.append(previous_waypoint)
                    obs_select_inds.append(i)
        elif self.sample_method.isnumeric():
            obs_select_inds = obs_select_inds[0:int(self.sample_numbers)]
        return obs_select_inds
    @staticmethod
    def normalize(joints):
        joint_intervals = np.array([[-2.8973000049591064, 5.794600009918213], 
        [-1.7627999782562256, 3.525599956512451], 
        [-2.8973000049591064, 5.794600009918213], 
        [-3.0717999935150146, 3.002000093460083], 
        [-2.8973000049591064, 5.794600009918213], 
        [-0.017500000074505806, 3.7699999809265137], 
        [-2.8973000049591064, 5.794600009918213]])
        normalized_joints = (joints - joint_intervals[:, 0])/joint_intervals[:,1]
        normalized_joints = normalized_joints*2 - 1
        return normalized_joints
    
    @staticmethod
    def unnormalize(normalized_joints):
        joint_intervals = np.array([[-2.8973000049591064, 5.794600009918213], 
        [-1.7627999782562256, 3.525599956512451], 
        [-2.8973000049591064, 5.794600009918213], 
        [-3.0717999935150146, 3.002000093460083], 
        [-2.8973000049591064, 5.794600009918213], 
        [-0.017500000074505806, 3.7699999809265137], 
        [-2.8973000049591064, 5.794600009918213]])

        joints = np.clip(normalized_joints, -1, 1)
        joints = (joints+1)*0.5
        joints = joints*joint_intervals[:,1] + joint_intervals[:, 0]
        return joints

    @staticmethod
    def depth2normal(d_im):
        d_im = d_im.astype("float32")
        # zy, zx = np.gradient(d_im)
        # You may also consider using Sobel to get a joint Gaussian smoothing and differentation
        # to reduce noise
        zx = cv2.Sobel(d_im, cv2.CV_64F, 1, 0, ksize=3)
        zy = cv2.Sobel(d_im, cv2.CV_64F, 0, 1, ksize=3)
        normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
        n = np.linalg.norm(normal, axis=2)
        normal[:, :, 0] /= n
        normal[:, :, 1] /= n
        normal[:, :, 2] /= n
        # offset and rescale values to be in 0-1
        normal += 1
        normal /= 2
        return normal

    @staticmethod
    def extract_bboxes(mask):
        """Compute bounding boxes from masks.
        mask: [height, width, num_instances]. Mask pixels are either 1 or 0.
        Returns: bbox array [num_instances, (y1, x1, y2, x2)].
        """
        horizontal_indicies = np.where(np.any(mask, axis=0))[0]
        vertical_indicies = np.where(np.any(mask, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        return np.array([x1, y1, x2, y2])

    def __len__(self):
        return len(self.episode_list)

class VLM_Waypoint_dataset(VLM_dataset):
    def __init__(self, root, setd, img_size=(360, 360), unused_camera_list=['left_shoulder', 'right_shoulder', 'overhead', 'wrist'], preprocess=True, use_fail_cases=True, sample_method="waypoints", train_tasks=None, args=None, mood="diffuser"):
        sample_method="waypoints"
        super().__init__(root, setd, img_size, unused_camera_list, preprocess, use_fail_cases, sample_method, train_tasks, args, mood)
        self.split_episode_to_waypoints(force_resample=False)

    def split_episode_to_waypoints(self, force_resample=False):
        waypoint_list_path = self.dataset_path / '{}_waypoints.pkl'.format(self.setd)
        if waypoint_list_path.is_file() and not force_resample:
            with open(waypoint_list_path,'rb') as f:
                all_episode_wp_pairs = pickle.load(f)
        else:
            all_episode_wp_pairs = {}
            for key_t, value_t in self.task_list.items():
                all_episode_wp_pairs[key_t] = []
                for episode in value_t['success']:
                    low_dim_obs = self.dataset_path/episode/"low_dim_obs.pkl"
                    with open(low_dim_obs, 'rb') as f:
                        demo_temple = pickle.load(f)
                    waypoint_inds = self.sample_steps(demo_temple)
                    for i, wp in enumerate(waypoint_inds):
                        if i+1<len(waypoint_inds):
                            all_episode_wp_pairs[key_t].append([episode, i, wp, waypoint_inds[i+1]])
                        else:
                            all_episode_wp_pairs[key_t].append([episode, i, wp, len(waypoint_inds)])
            with open(waypoint_list_path,'wb') as f:
                pickle.dump(all_episode_wp_pairs, f)
        
        self.all_waypoints = []
        for t in self.train_tasks:
            for n in all_episode_wp_pairs:
                if t in n:
                    self.all_waypoints += all_episode_wp_pairs[n]

        return None
    
    def __len__(self):
        return len(self.all_waypoints)
    
    def __getitem__(self, index):
        if index in self.invalid_episodes:
            index = sample(self.valid_episodes, 1)[0]
        episode, wp_number, current_idx, target_idx = self.all_waypoints[index] ## what is all_waypoints
        variation_path = episode.parents[1]
        task_name = episode.parents[2]
        fail_cases = 'fail_cases' in str(episode)
        episode_name = episode.name
        variation_number = int(variation_path.name.replace('variation',''))
        demos = get_stored_demos(1, False, self.dataset_path, variation_number, 
                                task_name, self.obs_config, episode_name, fail_cases, [current_idx])
        output_dict = self.get_perceiver_gt(demos[0], current_idx, target_idx)
        output_dict['episode'] = str(episode)
        output_dict['frame'] = current_idx
        if output_dict['valid']:
            self.valid_episodes.add(index)
        else:
            self.invalid_episodes.add(index)
            if len(self.valid_episodes) == 0:
                other_indexs = list(set(range(self.__len__())) - self.invalid_episodes)
                valid_index = sample(other_indexs, 1)[0]
            else:
                valid_index = sample(self.valid_episodes, 1)[0]
            output_dict = self.__getitem__(valid_index)

        return output_dict

    def _norm_rgb(self, x):
        return (x.astype(np.float32) / 255.0) * 2.0 - 1.0



    def get_perceiver_gt(self, data, currend_idx, target_idx):
        observation = data._observations[currend_idx]
        all_views = []
        for view in self.views:
            rgb = getattr(observation, f"{view}_rgb")
            pcd = getattr(observation, f"{view}_point_cloud").astype(np.float32)
            rgb = self._norm_rgb(rgb)
            all_views.append(np.concatenate((rgb, pcd), axis=-1).reshape((-1, 6)))
        all_views = np.concatenate(all_views, axis=0)

        bounds = np.array(self.args.bounds)
        trans_indicies, rot_grip_indicies = [], []

        observation = data._observations[currend_idx+1]
        quat = observation.gripper_pose[3:]
        if quat[-1]<0:
            quat = -quat
        disc_rot = helper.quaternion_to_discrete_euler(quat, self.args.rotation_resolution)
        attention_coordinate = observation.gripper_pose[:3]
        index = helper.point_to_voxel_index(attention_coordinate, self.args.voxel_size, bounds)
        trans_indicies.append(index)
        disc_rot = disc_rot.tolist()
        disc_rot.extend([int(observation.gripper_open)])
        rot_grip_indicies.append(np.array(disc_rot))

        low_dim_state = np.array([[observation.gripper_open, observation.gripper_joint_positions[0], observation.gripper_joint_positions[0], currend_idx/len(data._observations)]])
        low_dim_state = torch.from_numpy(low_dim_state).float()

        lang_goal_embs = data.high_level_instructions

        output_dict = {
            "flat_rgb_pcd":all_views,
            "valid":True,
            "trans_indicies":trans_indicies,
            "rot_grip_indicies":rot_grip_indicies,
            "lang_goal_embs": data.high_level_instructions,
            "low_dim_state": low_dim_state
        }

        return output_dict