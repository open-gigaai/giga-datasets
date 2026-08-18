import math
import os
from decimal import ROUND_HALF_UP, Decimal
from functools import reduce

import cv2
import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.nuscenes import NuScenes
from nuscenes.utils import splits as nus_splits
from PIL import Image
from pyquaternion import Quaternion
from shapely import affinity, ops
from shapely.geometry import LineString, MultiLineString, MultiPoint, MultiPolygon, box
from tqdm import tqdm

from giga_datasets import BaseProcessor, Dataset, LmdbWriter, PklWriter, boxes3d_utils, load_dataset


class NuScenesConverter:
    def __init__(self, data_dir, version, save_path, save_version='v0.0.1', num_sweeps=0):
        assert version in ['v1.0-mini', 'v1.0-trainval', 'v1.0-test']
        self.version = version
        self.save_path = save_path
        self.save_version = save_version
        self.num_sweeps = num_sweeps
        self.test = 'test' in version
        self.splits = ['train', 'val'] if not self.test else ['test']
        self.cam_types = [
            'CAM_FRONT',
            'CAM_FRONT_LEFT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        self.class2label = {
            'road_divider': 0,
            'lane_divider': 0,
            'ped_crossing': 1,
            'contours': 2,
            'others': -1,
        }
        self.label2class = {
            0: 'road_lane_divider',
            1: 'ped_crossing',
            2: 'contours',
        }
        self.map_names = [
            'boston-seaport',
            'singapore-onenorth',
            'singapore-hollandvillage',
            'singapore-queenstown',
        ]
        self.nusc = NuScenes(version=version, dataroot=data_dir, verbose=True)
        self.nusc_maps = dict()
        self.map_explorer = dict()
        for map_name in self.map_names:
            self.nusc_maps[map_name] = NuScenesMap(dataroot=data_dir, map_name=map_name)
            self.map_explorer[map_name] = NuScenesMapExplorer(self.nusc_maps[map_name])
        # save paths
        # lidar
        self.save_lidar_label_paths = dict()
        self.save_lidar_image_paths = dict()
        self.save_lidar_point_paths = dict()
        for split in self.splits:
            self.save_lidar_label_paths[split] = os.path.join(save_path, 'lidar_' + split, save_version, 'labels')
            self.save_lidar_point_paths[split] = os.path.join(save_path, 'lidar_' + split, save_version, 'points')
            self.save_lidar_image_paths[split] = dict()
            for cam_type in self.cam_types:
                cam_name = cam_type.lower()[4:]
                self.save_lidar_image_paths[split][cam_type] = os.path.join(save_path, 'lidar_' + split, save_version, 'images_' + cam_name)
        # cam
        self.save_cam_label_paths = dict()
        self.save_cam_image_paths = dict()
        self.save_cam_hdmap_paths = dict()
        for split in self.splits:
            self.save_cam_label_paths[split] = os.path.join(save_path, 'cam_' + split, save_version, 'labels')
            self.save_cam_image_paths[split] = os.path.join(save_path, 'cam_' + split, save_version, 'images')
            self.save_cam_hdmap_paths[split] = os.path.join(save_path, 'cam_' + split, save_version, 'hdmaps')
        # cam all
        self.save_cam_all_label_paths = dict()
        self.save_cam_all_image_paths = dict()
        self.save_cam_all_hdmap_paths = dict()
        self.save_cam_all_lane_paths = dict()
        for split in self.splits:
            self.save_cam_all_label_paths[split] = os.path.join(save_path, 'cam_all_' + split, save_version, 'labels')
            self.save_cam_all_image_paths[split] = os.path.join(save_path, 'cam_all_' + split, save_version, 'images')
            self.save_cam_all_hdmap_paths[split] = os.path.join(save_path, 'cam_all_' + split, save_version, 'hdmaps')
            self.save_cam_all_lane_paths[split] = os.path.join(save_path, 'cam_all_' + split, save_version, 'lanes')

    def __call__(self):
        self.convert_cam()
        self.convert_cam_all()
        self.convert_lidar()

    def convert_lidar(self):
        self.convert_lidar_labels()
        self.convert_lidar_images()
        self.convert_lidar_points()
        for split in self.splits:
            datasets = [
                load_dataset(self.save_lidar_label_paths[split]),
                load_dataset(self.save_lidar_point_paths[split]),
            ]
            for cam_type in self.cam_types:
                datasets.append(load_dataset(self.save_lidar_image_paths[split][cam_type]))
            dataset = Dataset(datasets)
            dataset.save(os.path.join(self.save_path, 'lidar_' + split, self.save_version))

    def convert_cam(self):
        self.convert_cam_labels()
        self.convert_cam_images()
        self.convert_cam_hdmaps()
        for split in self.splits:
            datasets = [
                load_dataset(self.save_cam_label_paths[split]),
                load_dataset(self.save_cam_image_paths[split]),
                load_dataset(self.save_cam_hdmap_paths[split]),
            ]
            dataset = Dataset(datasets)
            dataset.save(os.path.join(self.save_path, 'cam_' + split, self.save_version))

    def convert_cam_all(self):
        self.convert_cam_all_labels()
        self.convert_cam_all_images()
        self.convert_cam_all_hdmaps()
        self.convert_cam_all_lanes()
        for split in self.splits:
            datasets = [
                load_dataset(self.save_cam_all_label_paths[split]),
                load_dataset(self.save_cam_all_image_paths[split]),
                load_dataset(self.save_cam_all_hdmap_paths[split]),
            ]
            dataset = Dataset(datasets)
            dataset.save(os.path.join(self.save_path, 'cam_all_' + split, self.save_version))

    def _get_scenes_token(self):
        if self.version == 'v1.0-trainval':
            train_scenes = nus_splits.train
            val_scenes = nus_splits.val
        elif self.version == 'v1.0-test':
            train_scenes = nus_splits.test
            val_scenes = []
        elif self.version == 'v1.0-mini':
            train_scenes = nus_splits.mini_train
            val_scenes = nus_splits.mini_val
        else:
            assert False
        train_scenes_token = []
        val_scenes_token = []
        for scene in self.nusc.scene:
            if scene['name'] in train_scenes:
                train_scenes_token.append(scene['token'])
            elif scene['name'] in val_scenes:
                val_scenes_token.append(scene['token'])
            else:
                assert False
        return train_scenes_token, val_scenes_token

    def _get_lidar_calib(self, lidar_token, extra=False):
        lidar_record = self.nusc.get('sample_data', lidar_token)
        cs_record = self.nusc.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', lidar_record['ego_pose_token'])
        l2e_r = Quaternion(cs_record['rotation']).rotation_matrix
        l2e_t = cs_record['translation']
        e2g_r = Quaternion(pose_record['rotation']).rotation_matrix
        e2g_t = pose_record['translation']
        l2e = transform_matrix(l2e_r, l2e_t)
        e2g = transform_matrix(e2g_r, e2g_t)
        calib = {
            'lidar2ego': l2e,
            'ego2global': e2g,
            'lidar2ego_quat': (cs_record['rotation'], cs_record['translation']),
            'ego2global_quat': (pose_record['rotation'], pose_record['translation']),
        }
        if extra:
            calib.update(
                {
                    'lidar_token': lidar_token,
                    'lidar_timestamp': lidar_record['timestamp'],
                }
            )
        return calib

    def _get_cam_calib(self, cam_token, extra=False):
        cam_path, _, cam_intrinsic = self.nusc.get_sample_data(cam_token)
        cam_intrinsic_homo = np.zeros([4, 4], dtype=np.float32)
        cam_intrinsic_homo[:3, :3] = cam_intrinsic
        cam_intrinsic_homo[3, 3] = 1.0
        cam_record = self.nusc.get('sample_data', cam_token)
        cs_record = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
        l2e_r_s = Quaternion(cs_record['rotation']).rotation_matrix
        l2e_t_s = cs_record['translation']
        e2g_r_s = Quaternion(pose_record['rotation']).rotation_matrix
        e2g_t_s = pose_record['translation']
        l2e_s = transform_matrix(l2e_r_s, l2e_t_s)
        e2g_s = transform_matrix(e2g_r_s, e2g_t_s)
        calib = {
            'cam_intrinsic': cam_intrinsic_homo,
            'cam2ego': l2e_s,
            'ego2global': e2g_s,
            'cam2ego_quat': (cs_record['rotation'], cs_record['translation']),
            'ego2global_quat': (pose_record['rotation'], pose_record['translation']),
        }
        if extra:
            calib.update(
                {
                    'cam_token': cam_token,
                    'cam_timestamp': cam_record['timestamp'],
                }
            )
        return calib

    def _get_cam2lidar_calib(self, lidar_token, cam_token):
        lidar_record = self.nusc.get('sample_data', lidar_token)
        cs_record = self.nusc.get('calibrated_sensor', lidar_record['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', lidar_record['ego_pose_token'])
        l2e_r = Quaternion(cs_record['rotation']).rotation_matrix
        l2e_t = cs_record['translation']
        e2g_r = Quaternion(pose_record['rotation']).rotation_matrix
        e2g_t = pose_record['translation']
        cam_record = self.nusc.get('sample_data', cam_token)
        cs_record = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
        l2e_r_s = Quaternion(cs_record['rotation']).rotation_matrix
        l2e_t_s = cs_record['translation']
        e2g_r_s = Quaternion(pose_record['rotation']).rotation_matrix
        e2g_t_s = pose_record['translation']
        l2e_s = transform_matrix(l2e_r_s, l2e_t_s)
        e2g_s = transform_matrix(e2g_r_s, e2g_t_s)
        e2l = transform_matrix(l2e_r, l2e_t, inverse=True)
        g2e = transform_matrix(e2g_r, e2g_t, inverse=True)
        s2l_s = reduce(np.dot, [e2l, g2e, e2g_s, l2e_s])
        return s2l_s

    def _get_calib(self, sample, extra=False):
        calib = dict()
        lidar_type = 'LIDAR_TOP'
        lidar_token = sample['data'][lidar_type]
        calib[lidar_type.lower()] = self._get_lidar_calib(lidar_token, extra=extra)
        for cam_type in self.cam_types:
            cam_token = sample['data'][cam_type]
            calib[cam_type.lower()] = self._get_cam_calib(cam_token, extra=extra)
            calib[cam_type.lower()]['cam2lidar'] = self._get_cam2lidar_calib(lidar_token, cam_token)
        return calib

    def _get_hdmap(self, cam_token, scene_token, return_image=True):
        cam_record = self.nusc.get('sample_data', cam_token)
        pose_record = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
        cs_record = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        imsize = (cam_record['width'], cam_record['height'])
        ego2global_translation = np.array(pose_record['translation'])
        rotation = Quaternion(pose_record['rotation'])
        map_pose = ego2global_translation[:2]
        patch_box = (map_pose[0], map_pose[1], 102.4, 102.4)  # TODO 102.4 is hardcode
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        location = self.nusc.get('log', self.nusc.get('scene', scene_token)['log_token'])['location']
        nusc_map = self.nusc_maps[location]
        map_explorer = self.map_explorer[location]
        line_geom = get_map_geom(patch_box, patch_angle, ['road_divider', 'lane_divider'], nusc_map, map_explorer)
        line_vector_dict = line_geoms_to_vectors(line_geom)  # line to points
        ped_geom = get_map_geom(patch_box, patch_angle, ['ped_crossing'], nusc_map, map_explorer)
        ped_vector_list = line_geoms_to_vectors(ped_geom)['ped_crossing']
        polygon_geom = get_map_geom(patch_box, patch_angle, ['road_segment', 'lane'], nusc_map, map_explorer)
        poly_bound_list = poly_geoms_to_vectors(polygon_geom)
        vectors = []
        for line_type, vects in line_vector_dict.items():
            for line, length in vects:
                vectors.append((line.astype(float), length, self.class2label.get(line_type, -1)))
        for ped_line, length in ped_vector_list:
            vectors.append((ped_line.astype(float), length, self.class2label.get('ped_crossing', -1)))
        for contour, length in poly_bound_list:
            vectors.append((contour.astype(float), length, self.class2label.get('contours', -1)))
        filtered_vectors = []
        for pts, pts_num, type in vectors:
            if type != -1:
                filtered_vectors.append({'pts': pts, 'pts_num': pts_num, 'type': type})
        for vector in filtered_vectors:
            pts = vector['pts']
            vector['pts'] = np.concatenate((pts, np.zeros((pts.shape[0], 1))), axis=1)
        for vector in filtered_vectors:
            assert vector['pts'][:, 2].sum() == 0
            this_pts = vector['pts'].T
            this_pts = this_pts - np.array(cs_record['translation']).reshape((-1, 1))
            this_pts = np.dot(Quaternion(cs_record['rotation']).rotation_matrix.T, this_pts)
            this_pts, this_depth = view_points_depth(this_pts, cam_intrinsic, normalize=True)
            this_pts = this_pts[:, this_depth > 1e-3]
            vector['pts_num'] -= (this_depth <= 1e-3).sum()
            this_depth = this_depth[this_depth > 1e-3]
            this_pts = this_pts[:2, :]
            vector['pts'] = this_pts.T
            vector['depth'] = this_depth
        if return_image:
            map_canvas_size = [imsize[1], imsize[0]]
            semantic_masks = preprocess_map(filtered_vectors, map_canvas_size, max_channel=3, thickness=10)
            color_base_map = 255 * np.ones((imsize[1], imsize[0], 3), dtype=np.uint8)
            color_base_map[..., 0] *= ~semantic_masks[0]
            color_base_map[..., 1] *= ~semantic_masks[1]
            color_base_map[..., 2] *= ~semantic_masks[2]
            color_base_map = 255 - color_base_map
            color_base_map = color_base_map[:, :, ::-1]
            color_base_map = Image.fromarray(color_base_map)
            return color_base_map
        else:
            lanes = []
            lane_labels = []
            lane_depths = []
            for vector in filtered_vectors:
                assert len(vector['pts']) == len(vector['depth']) == vector['pts_num']
                if vector['pts_num'] <= 1:
                    continue
                if vector['type'] < 0:
                    continue
                lanes.append(vector['pts'])
                lane_labels.append(self.label2class[vector['type']])
                lane_depths.append(vector['depth'])
            data_dict = {
                'lanes': lanes,
                'lane_labels': lane_labels,
                'lane_depths': lane_depths,
            }
            return data_dict

    def _get_cam_label(self, cam_token):
        cam_record = self.nusc.get('sample_data', cam_token)
        cs_record = self.nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        pose_record = self.nusc.get('ego_pose', cam_record['ego_pose_token'])
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])
        width, height = cam_record['width'], cam_record['height']
        cam_boxes = self.nusc.get_boxes(cam_token)
        boxes3d = []
        names = []
        velocities = []
        attributes = []
        for cam_box in cam_boxes:
            cam_box.translate(-np.array(pose_record['translation']))
            cam_box.rotate(Quaternion(pose_record['rotation']).inverse)
            cam_box.translate(-np.array(cs_record['translation']))
            cam_box.rotate(Quaternion(cs_record['rotation']).inverse)
            corners = cam_box.corners().T.astype(np.float32)
            if not boxes3d_utils.corners3d_in_image(corners[None], cam_intrinsic, (height, width))[0]:
                continue
            # box3d
            loc = cam_box.center
            dim = cam_box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            rot = -np.array(cam_box.orientation.yaw_pitch_roll[0]).reshape(1)
            zero = np.array([0])
            box3d = np.concatenate([loc, dim, zero, rot, zero], axis=0)
            # velo
            global_velo2d = self.nusc.box_velocity(cam_box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_record['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_record['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2]
            if math.isnan(velo[0]):
                velo[:] = 0.0
            # attr
            ann_token = self.nusc.get('sample_annotation', cam_box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = self.nusc.get('attribute', ann_token[0])['name']
            boxes3d.append(box3d[None])
            names.append(cam_box.name)
            velocities.append(velo[None])
            attributes.append(attr_name)
        if len(boxes3d) > 0:
            boxes3d = np.concatenate(boxes3d, axis=0).astype(np.float32)
            velocities = np.concatenate(velocities, axis=0).astype(np.float32)
        else:
            boxes3d = np.zeros((0, 9), dtype=np.float32)
            velocities = np.zeros((0, 2), dtype=np.float32)
        label_dict = {
            'boxes3d': boxes3d,
            'labels3d': names,
            'velocities': velocities,
            'attributes': attributes,
        }
        return label_dict

    def convert_lidar_labels(self):
        train_scenes_token, val_scenes_token = self._get_scenes_token()
        label_writers = dict()
        for split in self.splits:
            label_writers[split] = PklWriter(self.save_lidar_label_paths[split])
        for idx in tqdm(range(len(self.nusc.sample)), desc='Convert Lidar Labels'):
            sample = self.nusc.sample[idx]
            calib = self._get_calib(sample, extra=True)
            label_dict = {
                'data_index': idx,
                'scene_token': sample['scene_token'],
                'sample_token': sample['token'],
                'sample_timestamp': sample['timestamp'],
                'calib': calib,
            }
            if not self.test:
                lidar_calib = calib['lidar_top']
                lidar_token = sample['data']['LIDAR_TOP']
                boxes = self.nusc.get_sample_data(lidar_token)[1]
                l2e_r = Quaternion(lidar_calib['lidar2ego_quat'][0]).rotation_matrix
                e2g_r = Quaternion(lidar_calib['ego2global_quat'][0]).rotation_matrix
                locs = np.array([b.center for b in boxes]).reshape(-1, 3)
                dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
                rots = np.array([b.orientation.yaw_pitch_roll[0] for b in boxes]).reshape(-1, 1)
                zeros = np.zeros((locs.shape[0], 2))
                all_boxes3d = np.concatenate([locs, dims[:, [1, 0, 2]], zeros, rots], axis=1)
                boxes3d = []
                names = []
                velocities = []
                num_lidar_pts = []
                num_radar_pts = []
                for i in range(len(boxes)):
                    box_name = boxes[i].name
                    boxes3d.append(all_boxes3d[i, None])
                    names.append(box_name)
                    token = sample['anns'][i]
                    velo = self.nusc.box_velocity(token)[:2]
                    velo = np.array([*velo, 0.0])
                    velo = reduce(np.dot, [velo, np.linalg.inv(e2g_r).T, np.linalg.inv(l2e_r).T])
                    if math.isnan(velo[0]):
                        velo[:] = 0.0
                    velocities.append(velo[None, :2])
                    anno = self.nusc.get('sample_annotation', token)
                    num_lidar_pts.append(anno['num_lidar_pts'])
                    num_radar_pts.append(anno['num_radar_pts'])
                if len(boxes3d) > 0:
                    boxes3d = np.concatenate(boxes3d, axis=0).astype(np.float32)
                    velocities = np.concatenate(velocities, axis=0).astype(np.float32)
                    num_lidar_pts = np.array(num_lidar_pts, dtype=np.int32)
                    num_radar_pts = np.array(num_radar_pts, dtype=np.int32)
                else:
                    boxes3d = np.zeros((0, 9), dtype=np.float32)
                    velocities = np.zeros((0, 2), dtype=np.float32)
                    num_lidar_pts = np.zeros((0,), dtype=np.int32)
                    num_radar_pts = np.zeros((0,), dtype=np.int32)
                label_dict.update(
                    {
                        'boxes3d': boxes3d,
                        'labels3d': names,
                        'velocities': velocities,
                        'num_lidar_pts': num_lidar_pts,
                        'num_radar_pts': num_radar_pts,
                    }
                )
                if sample['scene_token'] in train_scenes_token:
                    label_writers['train'].write_dict(label_dict)
                elif sample['scene_token'] in val_scenes_token:
                    label_writers['val'].write_dict(label_dict)
                else:
                    assert False
            else:
                label_writers['test'].write_dict(label_dict)
        for split in self.splits:
            label_writers[split].write_config()
            label_writers[split].close()

    def convert_lidar_images(self):
        for split in self.splits:
            label_dataset = load_dataset(self.save_lidar_label_paths[split])
            for cam_type in self.cam_types:
                cam_name = cam_type.lower()[4:]
                image_writer = LmdbWriter(self.save_lidar_image_paths[split][cam_type])
                for i in tqdm(range(len(label_dataset)), desc='Convert Lidar Images {}'.format(cam_type)):
                    label_dict = label_dataset[i]
                    data_index = label_dict['data_index']
                    cam_token = label_dict['calib'][cam_type.lower()]['cam_token']
                    data_path = self.nusc.get_sample_data_path(cam_token)
                    image_writer.write_image(data_index, data_path)
                image_writer.write_config(data_name='image_' + cam_name)
                image_writer.close()

    def convert_lidar_points(self):
        for split in self.splits:
            label_dataset = load_dataset(self.save_lidar_label_paths[split])
            point_writer = LmdbWriter(self.save_lidar_point_paths[split])
            for i in tqdm(range(len(label_dataset)), desc='Convert Lidar Points'):
                label_dict = label_dataset[i]
                data_index = label_dict['data_index']
                lidar_token = label_dict['calib']['lidar_top']['lidar_token']
                data_path = self.nusc.get_sample_data_path(lidar_token)
                points = load_pointclouds(data_path)
                point_writer.write_numpy(data_index, points)
            point_writer.write_config(data_name='points3d')
            point_writer.close()

    def convert_cam_labels(self):
        train_scenes_token, val_scenes_token = self._get_scenes_token()
        label_writers = dict()
        for split in self.splits:
            label_writers[split] = PklWriter(self.save_cam_label_paths[split])
        idx = 0
        for i in tqdm(range(len(self.nusc.sample)), desc='Convert Cam Labels'):
            sample = self.nusc.sample[i]
            for cam_type in self.cam_types:
                cam_token = sample['data'][cam_type]
                cam_calib = self._get_cam_calib(cam_token)
                cam_record = self.nusc.get('sample_data', cam_token)
                label_dict = {
                    'data_index': idx,
                    'scene_token': sample['scene_token'],
                    'sample_token': sample['token'],
                    'sample_timestamp': sample['timestamp'],
                    'cam_type': cam_type.lower(),
                    'cam_token': cam_token,
                    'cam_timestamp': cam_record['timestamp'],
                    'calib': cam_calib,
                }
                idx += 1
                if not self.test:
                    label_dict.update(self._get_cam_label(cam_token))
                    if sample['scene_token'] in train_scenes_token:
                        label_writers['train'].write_dict(label_dict)
                    elif sample['scene_token'] in val_scenes_token:
                        label_writers['val'].write_dict(label_dict)
                    else:
                        assert False
                else:
                    label_writers['test'].write_dict(label_dict)
        for split in self.splits:
            label_writers[split].write_config()
            label_writers[split].close()

    def convert_cam_images(self):
        for split in self.splits:
            label_dataset = load_dataset(self.save_cam_label_paths[split])
            image_writer = LmdbWriter(self.save_cam_image_paths[split])
            for i in tqdm(range(len(label_dataset)), desc='Convert Cam Images'):
                label_dict = label_dataset[i]
                data_index = label_dict['data_index']
                cam_token = label_dict['cam_token']
                data_path = self.nusc.get_sample_data_path(cam_token)
                image_writer.write_image(data_index, data_path)
            image_writer.write_config()
            image_writer.close()

    def convert_cam_hdmaps(self):
        for split in self.splits:
            label_dataset = load_dataset(self.save_cam_label_paths[split])
            hdmap_writer = LmdbWriter(self.save_cam_hdmap_paths[split])
            for i in tqdm(range(len(label_dataset)), desc='Convert Cam Hdmaps'):
                label_dict = label_dataset[i]
                data_index = label_dict['data_index']
                cam_token = label_dict['cam_token']
                scene_token = label_dict['scene_token']
                image_hdmap = self._get_hdmap(cam_token, scene_token)
                hdmap_writer.write_image(data_index, image_hdmap)
            hdmap_writer.write_config(data_name='image_hdmap')
            hdmap_writer.close()

    def convert_cam_all_labels(self):
        train_scenes_token, val_scenes_token = self._get_scenes_token()
        label_writers = dict()
        for split in self.splits:
            label_writers[split] = PklWriter(self.save_cam_all_label_paths[split])
        idx = 0
        for scene_token in tqdm(train_scenes_token + val_scenes_token, desc='Convert Cam All Labels'):
            scene = self.nusc.get('scene', scene_token)
            first_sample_token = scene['first_sample_token']
            first_sample_record = self.nusc.get('sample', first_sample_token)
            for cam_type in self.cam_types:
                cam_token = first_sample_record['data'][cam_type]
                while cam_token != '':
                    cam_record = self.nusc.get('sample_data', cam_token)
                    label_dict = {
                        'data_index': idx,
                        'scene_token': scene_token,
                        'sample_token': cam_record['sample_token'],
                        'cam_type': cam_type.lower(),
                        'cam_token': cam_token,
                        'cam_timestamp': cam_record['timestamp'],
                        'calib': self._get_cam_calib(cam_token),
                    }
                    idx += 1
                    if not self.test:
                        label_dict.update(self._get_cam_label(cam_token))
                        if scene_token in train_scenes_token:
                            label_writers['train'].write_dict(label_dict)
                        elif scene_token in val_scenes_token:
                            label_writers['val'].write_dict(label_dict)
                        else:
                            assert False
                    else:
                        label_writers['test'].write_dict(label_dict)
                    cam_token = cam_record['next']
        for split in self.splits:
            label_writers[split].write_config()
            label_writers[split].close()

    def convert_cam_all_images(self):
        for split in self.splits:
            label_dataset = load_dataset(self.save_cam_all_label_paths[split])
            image_writer = LmdbWriter(self.save_cam_all_image_paths[split])
            for i in tqdm(range(len(label_dataset)), desc='Convert Cam All Images'):
                label_dict = label_dataset[i]
                data_index = label_dict['data_index']
                cam_token = label_dict['cam_token']
                data_path = self.nusc.get_sample_data_path(cam_token)
                image_writer.write_image(data_index, data_path)
            image_writer.write_config()
            image_writer.close()

    def convert_cam_all_hdmaps(self):
        for split in self.splits:
            label_dataset = load_dataset(self.save_cam_all_label_paths[split])
            hdmap_writer = LmdbWriter(self.save_cam_all_hdmap_paths[split])
            for i in tqdm(range(len(label_dataset)), desc='Convert Cam All Hdmaps'):
                label_dict = label_dataset[i]
                data_index = label_dict['data_index']
                cam_token = label_dict['cam_token']
                scene_token = label_dict['scene_token']
                image_hdmap = self._get_hdmap(cam_token, scene_token)
                hdmap_writer.write_image(data_index, image_hdmap)
            hdmap_writer.write_config(data_name='image_hdmap')
            hdmap_writer.close()

    def convert_cam_all_lanes(self):
        for split in self.splits:
            label_dataset = load_dataset(self.save_cam_all_label_paths[split])
            lane_writer = LmdbWriter(self.save_cam_all_lane_paths[split])
            for i in tqdm(range(len(label_dataset)), desc='Convert Cam All Lanes'):
                label_dict = label_dataset[i]
                data_index = label_dict['data_index']
                cam_token = label_dict['cam_token']
                scene_token = label_dict['scene_token']
                lanes = self._get_hdmap(cam_token, scene_token, return_image=False)
                lane_writer.write_dict(data_index, lanes)
            lane_writer.write_config()
            lane_writer.close()


class NuScenesProcessor(BaseProcessor):
    def __init__(self, save_path):
        self.name_maps = {
            'animal': ['animal'],
            'human.pedestrian.adult': ['pedestrian', 'adult'],
            'human.pedestrian.child': ['pedestrian', 'child'],
            'human.pedestrian.construction_worker': ['pedestrian', 'construction_worker'],
            'human.pedestrian.police_officer': ['pedestrian', 'police_officer'],
            'human.pedestrian.personal_mobility': None,
            'human.pedestrian.stroller': None,
            'human.pedestrian.wheelchair': None,
            'vehicle.bicycle': ['non_vehicle', 'bicycle'],
            'vehicle.motorcycle': ['non_vehicle', 'motorcycle'],
            'static_object.bicycle_rack': ['non_vehicle', 'bicycle_rack'],
            'vehicle.car': ['vehicle', 'car'],
            'vehicle.truck': ['vehicle', 'truck'],
            'vehicle.bus.bendy': ['vehicle', 'bus'],
            'vehicle.bus.rigid': ['vehicle', 'bus'],
            'vehicle.construction': ['vehicle', 'construction'],
            'vehicle.emergency.ambulance': ['vehicle', 'ambulance'],
            'vehicle.emergency.police': ['vehicle', 'police'],
            'vehicle.trailer': ['vehicle', 'trailer'],
            'movable_object.barrier': ['traffic_barrier'],
            'movable_object.trafficcone': ['traffic_cone'],
            'movable_object.debris': None,
            'movable_object.pushable_pullable': None,
        }
        self.views = {
            'cam_front': ['front_view'],
            'cam_front_left': ['side_view', 'left_front'],
            'cam_front_right': ['side_view', 'right_front'],
            'cam_back': ['back_view'],
            'cam_back_left': ['side_view', 'left_back'],
            'cam_back_right': ['side_view', 'right_back'],
        }
        self.writer = PklWriter(save_path)

    def __call__(self, data_dict):
        assert len(data_dict['boxes3d']) == len(data_dict['labels3d'])
        cam_intrinsic = data_dict['calib']['cam_intrinsic']
        labels = []
        for label in data_dict['labels3d']:
            labels.append(self.name_maps[label])
        if len(labels) > 0:
            boxes3d = data_dict['boxes3d']
            corners3d = boxes3d_utils.boxes3d_to_corners3d(boxes3d, rot_axis=1)
            assert (corners3d[..., 2].mean(-1) > 0).all()
            corners3d = boxes3d_utils.crop_corners3d(corners3d)
            corners = boxes3d_utils.corners3d_to_corners2d(corners3d, cam_intrinsic=cam_intrinsic)
        else:
            corners = np.zeros((0, 8, 2), dtype=np.float32)
        data_dict['ori_labels3d'] = data_dict.pop('labels3d')
        data_dict['labels3d'] = labels
        data_dict['corners'] = corners
        data_dict['image_labels'] = {
            'ISP': 'nuscenes',
            'view': self.views[data_dict['cam_type']],
        }
        return data_dict

    def process(self, data_dict):
        self.writer.write_dict(data_dict)

    def close(self):
        self.writer.write_config()
        self.writer.close()


def transform_matrix(rotation, translation, inverse=False):
    tm = np.eye(4, dtype='float32')
    if inverse:
        rot_inv = rotation.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation
        tm[:3, 3] = np.transpose(np.array(translation))
    return tm


def load_pointclouds(file_path, point_dim=5):
    assert os.path.exists(file_path)
    if file_path.endswith('.npy'):
        points = np.load(file_path)
    else:
        points = np.fromfile(file_path, dtype=np.float32)
    points = points.reshape((-1, point_dim))
    return points


def post_process_coords(corner_coords, imsize):
    polygon_from_2d_box = MultiPoint(corner_coords.tolist()).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])
    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array([coord for coord in img_intersection.exterior.coords])
        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])
        return np.array([min_x, min_y, max_x, max_y], dtype=np.float32)
    else:
        return None


def quaternion_yaw(q: Quaternion):
    """Calculate the yaw angle from a quaternion.

    Note that this only works for a quaternion that represents a box in lidar or global coordinate frame. It does not work for a box in the camera
    frame.
    :param q: Quaternion of interest.
    :return: Yaw angle in radians.
    """
    # Project into xy plane.
    v = np.dot(q.rotation_matrix, np.array([1, 0, 0]))
    # Measure yaw using arctan.
    yaw = np.arctan2(v[1], v[0])
    return yaw


def get_map_geom(patch_box, patch_angle, layer_names, nusc_map, map_explorer):
    map_geom = []
    for layer_name in layer_names:
        if layer_name in ['road_divider', 'lane_divider']:
            geoms = map_explorer._get_layer_line(patch_box, patch_angle, layer_name)
            map_geom.append((layer_name, geoms))
        elif layer_name in ['road_segment', 'lane']:
            geoms = map_explorer._get_layer_polygon(patch_box, patch_angle, layer_name)
            map_geom.append((layer_name, geoms))
        elif layer_name in ['ped_crossing']:
            geoms = get_ped_crossing_line(patch_box, patch_angle, nusc_map, map_explorer)
            map_geom.append((layer_name, geoms))
    return map_geom


def get_ped_crossing_line(patch_box, patch_angle, nusc_map, map_explorer):
    def add_line(poly_xy, idx, patch, patch_angle, patch_x, patch_y, line_list):
        points = [(p0, p1) for p0, p1 in zip(poly_xy[0, idx : idx + 2], poly_xy[1, idx : idx + 2])]
        line = LineString(points)
        line = line.intersection(patch)
        if not line.is_empty:
            line = affinity.rotate(line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
            line = affinity.affine_transform(line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
            line_list.append(line)

    patch_x = patch_box[0]
    patch_y = patch_box[1]
    patch = NuScenesMapExplorer.get_patch_coord(patch_box, patch_angle)
    line_list = []
    records = nusc_map.ped_crossing
    for record in records:
        polygon = map_explorer.extract_polygon(record['polygon_token'])
        poly_xy = np.array(polygon.exterior.xy)
        dist = np.square(poly_xy[:, 1:] - poly_xy[:, :-1]).sum(0)
        x1, x2 = np.argsort(dist)[-2:]
        add_line(poly_xy, x1, patch, patch_angle, patch_x, patch_y, line_list)
        add_line(poly_xy, x2, patch, patch_angle, patch_x, patch_y, line_list)
    return line_list


def line_geoms_to_vectors(line_geom):
    line_vectors_dict = dict()
    for line_type, a_type_of_lines in line_geom:
        one_type_vectors = one_type_line_geom_to_vectors(a_type_of_lines)
        line_vectors_dict[line_type] = one_type_vectors
    return line_vectors_dict


def one_type_line_geom_to_vectors(line_geom):
    line_vectors = []
    for line in line_geom:
        if not line.is_empty:
            if line.geom_type == 'MultiLineString':
                for line_i in line:
                    line_vectors.append(sample_pts_from_line(line_i))
            elif line.geom_type == 'LineString':
                line_vectors.append(sample_pts_from_line(line))
            else:
                raise NotImplementedError
    return line_vectors


def sample_pts_from_line(line):
    distances = np.arange(0, line.length, 1)
    sampled_points = np.array([list(line.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
    num_valid = len(sampled_points)
    return sampled_points, num_valid


def poly_geoms_to_vectors(polygon_geom):
    roads = polygon_geom[0][1]
    lanes = polygon_geom[1][1]
    union_roads = ops.unary_union(roads)
    union_lanes = ops.unary_union(lanes)
    union_segments = ops.unary_union([union_roads, union_lanes])
    max_x = 102.4 / 2  # FIXME 102.4 hardcode
    max_y = 102.4 / 2
    local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
    exteriors = []
    interiors = []
    if union_segments.geom_type != 'MultiPolygon':
        union_segments = MultiPolygon([union_segments])
    for poly in union_segments:
        exteriors.append(poly.exterior)
        for inter in poly.interiors:
            interiors.append(inter)
    results = []
    for ext in exteriors:
        if ext.is_ccw:
            ext.coords = list(ext.coords)[::-1]
        lines = ext.intersection(local_patch)
        if isinstance(lines, MultiLineString):
            lines = ops.linemerge(lines)
        results.append(lines)
    for inter in interiors:
        if not inter.is_ccw:
            inter.coords = list(inter.coords)[::-1]
        lines = inter.intersection(local_patch)
        if isinstance(lines, MultiLineString):
            lines = ops.linemerge(lines)
        results.append(lines)
    return one_type_line_geom_to_vectors(results)


def preprocess_map(vectors, canvas_size, max_channel, thickness):
    confidence_levels = [-1]
    vector_num_list = {}
    for i in range(max_channel):
        vector_num_list[i] = []
    for vector in vectors:
        if vector['pts_num'] >= 2:
            vector_num_list[vector['type']].append(LineString(vector['pts'][: vector['pts_num']]))
    idx = 1
    filter_masks = []
    instance_masks = []
    for i in range(max_channel):
        map_mask, idx = line_geom_to_mask(vector_num_list[i], confidence_levels, canvas_size, thickness, idx)
        instance_masks.append(map_mask)
        filter_mask, _ = line_geom_to_mask(vector_num_list[i], confidence_levels, canvas_size, thickness + 4, 1)
        filter_masks.append(filter_mask)
    filter_masks = np.stack(filter_masks)
    instance_masks = np.stack(instance_masks)
    instance_masks = overlap_filter(instance_masks, filter_masks)
    semantic_masks = instance_masks != 0
    return semantic_masks


def line_geom_to_mask(layer_geom, confidence_levels, canvas_size, thickness, idx, type='index', angle_class=36):
    patch = box(0, 0, canvas_size[1], canvas_size[0])
    map_mask = np.zeros(canvas_size, np.uint8)
    for line in layer_geom:
        if isinstance(line, tuple):
            line, confidence = line
        else:
            confidence = None
        new_line = line.intersection(patch)
        if not new_line.is_empty:
            confidence_levels.append(confidence)
            if new_line.geom_type == 'MultiLineString':
                for new_single_line in new_line:
                    map_mask, idx = mask_for_lines(new_single_line, map_mask, thickness, idx, type, angle_class)
            else:
                map_mask, idx = mask_for_lines(new_line, map_mask, thickness, idx, type, angle_class)
    return map_mask, idx


def overlap_filter(mask, filter_mask):
    C, _, _ = mask.shape
    for c in range(C - 1, -1, -1):
        filter = np.repeat((filter_mask[c] != 0)[None, :], c, axis=0)
        mask[:c][filter] = 0
    return mask


def round_dec(n, d):
    s = '0.' + '0' * d
    return Decimal(str(n)).quantize(Decimal(s), ROUND_HALF_UP)


def mask_for_lines(lines, mask, thickness, idx, type='index', angle_class=36):
    lines = lines.coords
    lines = np.array(lines)
    coords = np.zeros_like(lines)
    for i in range(lines.shape[0]):
        for j in range(lines.shape[1]):
            lin = lines[i][j]
            lr = round_dec(lin, 0)
            coords[i][j] = lr
    coords = coords.astype(np.int32)
    if len(coords) < 2:
        return mask, idx
    if type == 'backward':
        coords = np.flip(coords, 0)
    if type == 'index':
        cv2.polylines(mask, [coords], False, color=idx, thickness=thickness)
        idx += 1
    else:
        for i in range(len(coords) - 1):
            cv2.polylines(
                mask,
                [coords[i:]],
                False,
                color=get_discrete_degree(coords[i + 1] - coords[i], angle_class=angle_class),
                thickness=thickness,
            )
    return mask, idx


def get_discrete_degree(vec, angle_class=36):
    deg = np.mod(np.degrees(np.arctan2(vec[1], vec[0])), 360)
    deg = (int(deg / (360 / angle_class) + 0.5) % angle_class) + 1
    return deg


def view_points_depth(points, view, normalize):
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3
    viewpad = np.eye(4)
    viewpad[: view.shape[0], : view.shape[1]] = view
    nbr_points = points.shape[1]
    # Do operation in homogenous coordinates.
    points = np.concatenate((points, np.ones((1, nbr_points))))
    points = np.dot(viewpad, points)
    points = points[:3, :]
    depth = points[2, :]
    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)
    return points, depth


def main():
    version = 'v1.0-mini'  # v1.0-mini, v1.0-trainval
    data_dir = f'./public_datasets/nuscenes/{version}/'
    save_path = f'./giga_datasets/public_data/nuscenes/{version}/'
    NuScenesConverter(data_dir=data_dir, version=version, save_path=save_path, save_version='v0.0.1')()

    # data_path = './nuscenes/v1.0-mini/v0.0.1/labels/'
    # save_path = data_path.replace('v0.0.1', 'v0.0.2')
    # dataset = load_dataset(data_path)
    # processor = NuScenesProcessor(save_path)
    # dataset.process(processor, num_workers=4)


if __name__ == '__main__':
    main()
