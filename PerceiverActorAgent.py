from utils.helper import Lamb
from utils.helper import stack_on_channel
import numpy as np
import os
import sys
import shutil
import pickle
import torch
from torch import nn, einsum
from utils.voxelization import VoxelGrid
from Q_function import QFunction
# from utils.helper import _preprocess_inputs
from cliport.models.core.clip import build_model, load_clip, tokenize
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt

class PerceiverActorAgent():
    def __init__(self,
                coordinate_bounds: list,
                perceiver_encoder: nn.Module,
                camera_names: list,
                batch_size: int,
                voxel_size: int,
                voxel_feature_size: int,
                num_rotation_classes: int,
                rotation_resolution: float,
                lr: float = 0.0001,
                image_resolution: list = None,
                lambda_weight_l2: float = 0.0,
                transform_augmentation: bool = True,
                transform_augmentation_xyz: list = [0.0, 0.0, 0.0],
                transform_augmentation_rpy: list = [0.0, 0.0, 180.0],
                transform_augmentation_rot_resolution: int = 5,
                optimizer_type: str = 'lamb'):

        self._coordinate_bounds = coordinate_bounds
        self._perceiver_encoder = perceiver_encoder
        self._camera_names = camera_names
        self._batch_size = batch_size
        self._voxel_size = voxel_size
        self._voxel_feature_size = voxel_feature_size
        self._num_rotation_classes = num_rotation_classes
        self._rotation_resolution = rotation_resolution
        self._lr = lr
        self._image_resolution = image_resolution
        self._lambda_weight_l2 = lambda_weight_l2
        self._transform_augmentation = transform_augmentation
        self._transform_augmentation_xyz = transform_augmentation_xyz
        self._transform_augmentation_rpy = transform_augmentation_rpy
        self._transform_augmentation_rot_resolution = transform_augmentation_rot_resolution
        self._optimizer_type = optimizer_type

        self._cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def build(self, training: bool, device: torch.device = None):
        self._training = training
        self._device = device
        IMAGE_SIZE = 360
        CAMERAS = ['front', 'left_shoulder', 'right_shoulder', 'wrist', 'overhead']

        vox_grid = VoxelGrid(
            coord_bounds=self._coordinate_bounds,
            voxel_size=self._voxel_size,
            device=device,
            batch_size=self._batch_size,
            feature_size=self._voxel_feature_size,
            max_num_coords=np.prod([IMAGE_SIZE, IMAGE_SIZE]) * len(CAMERAS),
        )
        self._vox_grid = vox_grid

        self._q = QFunction(self._perceiver_encoder,
                            vox_grid,
                            self._rotation_resolution,
                            device,
                            training).to(device).train(training)

        self._coordinate_bounds = torch.tensor(self._coordinate_bounds,
                                               device=device).unsqueeze(0)

        if self._optimizer_type == 'lamb':
            # From: https://github.com/cybertronai/pytorch-lamb/blob/master/pytorch_lamb/lamb.py
            self._optimizer = Lamb(
                self._q.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
                betas=(0.9, 0.999),
                adam=False,
            )
        elif self._optimizer_type == 'adam':
            self._optimizer = torch.optim.Adam(
                self._q.parameters(),
                lr=self._lr,
                weight_decay=self._lambda_weight_l2,
            )
        else:
            raise Exception('Unknown optimizer')

    def _softmax_q(self, q):
        q_shape = q.shape
        return F.softmax(q.reshape(q_shape[0], -1), dim=1).reshape(q_shape)
    
    def _get_one_hot_expert_actions(self,  # You don't really need this function since GT labels are already in the right format. This is some leftover code from my experiments with label smoothing.
                                    batch_size,
                                    action_trans,
                                    action_rot_grip,
                                    action_ignore_collisions,
                                    device):
        bs = batch_size

        # initialize with zero tensors
        action_trans_one_hot = torch.zeros((bs, self._voxel_size, self._voxel_size, self._voxel_size), dtype=int, device=device)
        action_rot_x_one_hot = torch.zeros((bs, self._num_rotation_classes), dtype=int, device=device)
        action_rot_y_one_hot = torch.zeros((bs, self._num_rotation_classes), dtype=int, device=device)
        action_rot_z_one_hot = torch.zeros((bs, self._num_rotation_classes), dtype=int, device=device)
        action_grip_one_hot  = torch.zeros((bs, 2), dtype=int, device=device)
        action_collision_one_hot = torch.zeros((bs, 2), dtype=int, device=device) 

        # fill one-hots
        for b in range(bs):
          # translation
          gt_coord = action_trans[b, :]
          action_trans_one_hot[b, gt_coord[0], gt_coord[1], gt_coord[2]] = 1

          # rotation
          gt_rot_grip = action_rot_grip[b, :]
          action_rot_x_one_hot[b, gt_rot_grip[0]] = 1
          action_rot_y_one_hot[b, gt_rot_grip[1]] = 1
          action_rot_z_one_hot[b, gt_rot_grip[2]] = 1
          action_grip_one_hot[b, gt_rot_grip[3]] = 1

          # ignore collision
          gt_ignore_collisions = action_ignore_collisions[b, :]
          action_collision_one_hot[b, gt_ignore_collisions[0].long()] = 1
        
        # flatten trans
        action_trans_one_hot = action_trans_one_hot.view(bs, -1) 

        return action_trans_one_hot, \
               action_rot_x_one_hot, \
               action_rot_y_one_hot, \
               action_rot_z_one_hot, \
               action_grip_one_hot,  \
               action_collision_one_hot

    def encode_text(self, x):
        with torch.no_grad():
            tokens = tokenize(x).to(self._device)
            clip_rn50,_ = load_clip('RN50', device=self._device)
            text_feat, text_emb = clip_rn50.encode_text_with_embeddings(tokens)

        text_mask = torch.where(tokens==0, tokens, 1)  # [1, max_token_len]
        return text_feat, text_emb, text_mask


    def update(self, step: int, replay_sample: dict, replay_sample_partial_on_device: dict, backprop: bool = True) -> dict:
        # sample
        # print("replay_sample: ", replay_sample.keys())


        action_trans = replay_sample['trans_indicies'][0].int()
        action_rot_grip = replay_sample['rot_grip_indicies'][0].int()
        # action_ignore_collisions = replay_sample['ignore_collisions'][:, -1].int()
        action_ignore_collisions = np.zeros((self._batch_size, 1))
        action_ignore_collisions = torch.tensor(action_ignore_collisions) 
        # action_gripper_pose = replay_sample['gripper_pose'][:, -1]

        a = replay_sample['lang_goal_embs']
        l_enc, lang_goal_embs, l_mask = self.encode_text(replay_sample['lang_goal_embs'][0])
        # lang_goal_embs = replay_sample['lang_goal_embs'][:, -1].float()
        
        # metric scene bounds
        bounds = bounds_tp1 = self._coordinate_bounds

        # inputs
        proprio = stack_on_channel(replay_sample_partial_on_device['low_dim_state'])
        # obs, pcd = _preprocess_inputs(replay_sample)

        # TODO: data augmentation by applying SE(3) pertubations to obs and actions

        # Q function
        q_trans, rot_grip_q, collision_q, voxel_grid = self._q(proprio,
                                                               replay_sample_partial_on_device['flat_rgb_pcd'],
                                                               lang_goal_embs,
                                                               bounds)
        
        # one-hot expert actions
        bs = self._batch_size
        action_trans_one_hot, action_rot_x_one_hot, \
        action_rot_y_one_hot, action_rot_z_one_hot, \
        action_grip_one_hot, action_collision_one_hot = self._get_one_hot_expert_actions(bs,
                                                                                         action_trans,
                                                                                         action_rot_grip,
                                                                                         action_ignore_collisions,
                                                                                         device=self._device)
        total_loss = 0.
        if backprop:
            # cross-entropy loss
            trans_loss = self._cross_entropy_loss(q_trans.view(bs, -1), 
                                                  action_trans_one_hot.argmax(-1))
            
            rot_grip_loss = 0.
            rot_grip_loss += self._cross_entropy_loss(rot_grip_q[:, 0*self._num_rotation_classes:1*self._num_rotation_classes], 
                                                      action_rot_x_one_hot.argmax(-1))
            rot_grip_loss += self._cross_entropy_loss(rot_grip_q[:, 1*self._num_rotation_classes:2*self._num_rotation_classes], 
                                                      action_rot_y_one_hot.argmax(-1))
            rot_grip_loss += self._cross_entropy_loss(rot_grip_q[:, 2*self._num_rotation_classes:3*self._num_rotation_classes], 
                                                      action_rot_z_one_hot.argmax(-1))
            rot_grip_loss += self._cross_entropy_loss(rot_grip_q[:, 3*self._num_rotation_classes:],
                                                      action_grip_one_hot.argmax(-1))
            
            collision_loss = self._cross_entropy_loss(collision_q,
                                                      action_collision_one_hot.argmax(-1))
            
            total_loss = trans_loss + rot_grip_loss + collision_loss
            total_loss = total_loss.mean()

            # backprop
            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()

            total_loss = total_loss.item()

        # choose best action through argmax
        coords_indicies, rot_and_grip_indicies, ignore_collision_indicies = self._q.choose_highest_action(q_trans,
                                                                                                          rot_grip_q,
                                                                                                          collision_q)
        
        # discrete to continuous translation action
        res = (bounds[:, 3:] - bounds[:, :3]) / self._voxel_size
        continuous_trans = bounds[:, :3] + res * coords_indicies.int() + res / 2
        
        return {
            'total_loss': total_loss,
            'voxel_grid': voxel_grid,
            'q_trans': self._softmax_q(q_trans),
            'pred_action': {
                'trans': coords_indicies,
                'continuous_trans': continuous_trans,
                'rot_and_grip': rot_and_grip_indicies,
                'collision': ignore_collision_indicies
            },
            'expert_action': {
                'action_trans': action_trans
            }
        }