"""Script to convert LeRobot dataset to the LeRobot dataset v2.1 format."""

import dataclasses
from pathlib import Path
from typing import Dict, List, Literal

import h5py
import numpy as np
import torch
import tqdm
import tyro

from giga_datasets.datasets.lerobot_dataset import FastLeRobotDataset


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 0.0001
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = None


DEFAULT_DATASET_CONFIG = DatasetConfig()


def find_last_valid_frame(array):
    false_length = array.shape[0]
    end_value = array[0]
    end_index = 0
    for i in range(1, false_length):
        if (np.abs(array[i] - end_value) > 1e-2).any():
            end_index = i
            end_value = array[i]
    return end_index + 1


def find_first_valid_frame(array):
    false_length = array.shape[0]
    start_value = array[0]
    start_index = 0
    for i in range(1, false_length):
        if (np.abs(array[i] - start_value) > 1e-2).any():
            start_index = i
            break

    return start_index


def create_empty_dataset(
    out_dir: Path,
    repo_id: str,
    robot_type: str,
    mode: Literal['video', 'image'] = 'image',
    *,
    is_mobile: bool = False,
    has_velocity: bool = False,
    has_effort: bool = False,
    has_depth_images: bool = False,
    dataset_config: DatasetConfig = DEFAULT_DATASET_CONFIG,
) -> FastLeRobotDataset:
    motors = [
        'left_waist',
        'left_shoulder',
        'left_elbow',
        'left_forearm_roll',
        'left_wrist_angle',
        'left_wrist_rotate',
        'left_gripper',
        'right_waist',
        'right_shoulder',
        'right_elbow',
        'right_forearm_roll',
        'right_wrist_angle',
        'right_wrist_rotate',
        'right_gripper',
    ]
    cameras = [
        'cam_high',
        'cam_left_wrist',
        'cam_right_wrist',
    ]

    features = {
        'observation.state': {
            'dtype': 'float32',
            'shape': (len(motors),),
            'names': motors,
        },
        'action': {
            'dtype': 'float32',
            'shape': (len(motors) + 2,) if is_mobile else (len(motors),),
            'names': (motors + ['linear_x', 'angular_z']) if is_mobile else motors,
        },
    }

    if has_velocity:
        features['observation.velocity'] = {
            'dtype': 'float32',
            'shape': (len(motors),),
            'names': motors,
        }
    if has_effort:
        features['observation.effort'] = {
            'dtype': 'float32',
            'shape': (len(motors),),
            'names': motors,
        }

    for cam in cameras:
        features[f'observation.images.{cam}'] = {
            'dtype': mode,
            'shape': (3, 480, 640),
            'names': [
                'channels',
                'height',
                'width',
            ],
        }

    if has_depth_images:
        for cam in cameras:
            features[f'observation.depth_images.{cam}'] = {
                'dtype': mode,
                'shape': (3, 480, 640),
                'names': [
                    'channels',
                    'height',
                    'width',
                ],
            }

    return FastLeRobotDataset.create(
        root=out_dir,
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )


def get_cameras(hdf5_files: List[Path]) -> List[str]:
    with h5py.File(hdf5_files[0], 'r') as ep:
        return [key for key in ep['/observations/images'].keys() if 'depth' not in key]  # noqa: SIM118


def has_velocity(hdf5_files: List[Path]) -> bool:
    with h5py.File(hdf5_files[0], 'r') as ep:
        return '/observations/qvel' in ep


def has_effort(hdf5_files: List[Path]) -> bool:
    with h5py.File(hdf5_files[0], 'r') as ep:
        return '/observations/effort' in ep


def load_raw_images_per_camera(ep: h5py.File, cameras: List[str]) -> Dict[str, np.ndarray]:
    imgs_per_cam = {}
    for camera in cameras:
        uncompressed = ep[f'/observations/images/{camera}'].ndim == 4

        if uncompressed:
            imgs_array = ep[f'/observations/images/{camera}'][:]
        else:
            import cv2

            imgs_array = []
            for data in ep[f'/observations/images/{camera}']:
                imgs_array.append(cv2.cvtColor(cv2.imdecode(data, 1), cv2.COLOR_BGR2RGB))
            imgs_array = np.array(imgs_array)

        imgs_per_cam[camera] = imgs_array
    return imgs_per_cam


def load_raw_depth_images_per_camera(ep: h5py.File, cameras: List[str], depth_units_per_meter: float | None = None) -> Dict[str, np.ndarray]:
    depth_imgs_per_cam = {}
    if '/observations/images_depth' not in ep:
        return None
    for camera in cameras:
        depth_imgs_array = ep[f'/observations/images_depth/{camera}'][:]

        if depth_units_per_meter is not None and depth_units_per_meter > 0:
            depth_imgs_array = depth_imgs_array / depth_units_per_meter
        depth_imgs_per_cam[camera] = np.clip(depth_imgs_array.astype(np.float32) * 100, 0, 255).astype(np.uint8)[:, :, :, None].repeat(3, axis=-1)
    return depth_imgs_per_cam


def load_raw_episode_data(
    ep_path: Path,
    is_mobile: bool = False,
    depth_units_per_meter: float | None = None,
) -> tuple[dict[str, np.ndarray], torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    with h5py.File(ep_path, 'r') as ep:
        qpos_data = ep['/observations/qpos'][:]

        state = torch.from_numpy(qpos_data)
        action = torch.from_numpy(ep['/action'][:])

        if is_mobile:
            base_action = torch.from_numpy(ep['/base_action'][:])
            action = torch.cat((action, base_action), dim=1)
        velocity = None
        effort = None

        imgs_per_cam = load_raw_images_per_camera(
            ep,
            [
                'cam_high',
                'cam_left_wrist',
                'cam_right_wrist',
            ],
        )
        for cam in imgs_per_cam:
            imgs_per_cam[cam] = imgs_per_cam[cam]

        depth_imgs_per_cam = load_raw_depth_images_per_camera(
            ep,
            [
                'cam_high',
                'cam_left_wrist',
                'cam_right_wrist',
            ],
            depth_units_per_meter=depth_units_per_meter,
        )
        if depth_imgs_per_cam is not None:
            for cam in depth_imgs_per_cam:
                depth_imgs_per_cam[cam] = depth_imgs_per_cam[cam][:]
    return imgs_per_cam, state, action, velocity, effort, depth_imgs_per_cam


def populate_dataset(
    dataset: FastLeRobotDataset,
    hdf5_files: list[Path],
    task: str,
    episodes: list[int] | None = None,
    is_mobile: bool = False,
    depth_units_per_meter: float | None = None,
) -> FastLeRobotDataset:
    if episodes is None:
        episodes = list(range(len(hdf5_files)))

    for ep_idx in tqdm.tqdm(episodes):
        ep_path = hdf5_files[ep_idx]

        try:
            imgs_per_cam, state, action, velocity, effort, depth_imgs_per_cam = load_raw_episode_data(
                ep_path, is_mobile=is_mobile, depth_units_per_meter=depth_units_per_meter
            )
            num_frames = state.shape[0]

            for i in range(num_frames):
                frame = {
                    'observation.state': state[i],
                    'action': action[i],
                }

                for camera, img_array in imgs_per_cam.items():
                    frame[f'observation.images.{camera}'] = img_array[i]

                if velocity is not None:
                    frame['observation.velocity'] = velocity[i]
                if effort is not None:
                    frame['observation.effort'] = effort[i]

                if depth_imgs_per_cam is not None:
                    for camera, depth_array in depth_imgs_per_cam.items():
                        frame[f'observation.depth_images.{camera}'] = depth_array[i]

                dataset.add_frame(frame, task)
            dataset.save_episode()

        except Exception as e:
            print(f'Error processing episode {ep_idx}: {e}')
            continue

    return dataset


def convert_lerobot(
    raw_dirs: list[str],
    tasks: list[str],
    out_dir: str,
    repo_id: str,
    episodes: list[list[int]] | None = None,
    is_mobile: bool = False,
    mode: Literal['image', 'video'] = 'video',
    has_depth_images: bool = False,
    depth_units_per_meter: float | None = None,
) -> None:
    """Convert raw hdf5 data to LeRobot dataset.

    Args:
        raw_dirs: List of raw data directories.
        tasks: List of task names. The number of tasks must match the number of raw directories.
        out_dir: Output directory for the dataset.
        repo_id: Repository ID to use for the dataset.
        episodes: Optional list of episode indices to include, one list per directory.
        is_mobile: Whether the robot is mobile.
        mode: Mode to use for the dataset. One of ['image', 'video'].
        has_depth_images: Whether to include depth images.
        depth_units_per_meter: The depth in meters.
    """

    if len(raw_dirs) != len(tasks):
        raise ValueError('Number of raw directories must match number of tasks')

    if episodes is not None and len(raw_dirs) != len(episodes):
        raise ValueError('Number of raw directories must match number of episode lists')

    if mode not in ['image', 'video']:
        raise ValueError('Mode must be one of ["image", "video"]')

    dataset = create_empty_dataset(
        out_dir,
        repo_id,
        robot_type='mobile_aloha' if is_mobile else 'aloha',
        mode=mode,
        is_mobile=is_mobile,
        has_effort=False,
        has_velocity=False,
        has_depth_images=has_depth_images,
    )

    for i, raw_dir in enumerate(raw_dirs):
        task = tasks[i]
        raw_dir = Path(raw_dir)
        if not raw_dir.exists():
            raise ValueError(f'Directory {raw_dir} does not exist')

        hdf5_files = sorted(raw_dir.glob('episode_*.hdf5'), key=lambda x: int(x.stem.split('_')[-1].split('.')[0]))

        ep_list = None
        if episodes is not None:
            ep_list = episodes[i]

        dataset = populate_dataset(
            dataset,
            hdf5_files,
            task=task,
            episodes=ep_list,
            is_mobile=is_mobile,
            depth_units_per_meter=depth_units_per_meter,
        )


if __name__ == '__main__':
    tyro.cli(convert_lerobot)
