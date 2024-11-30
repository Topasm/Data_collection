import panda_py
from panda_py import libfranka

import time
import threading
import transforms3d
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
from multiprocessing.managers import SharedMemoryManager
from utils.camera.multi_cam import MultiRealsense, SingleRealsense
import numpy as np
from utils.robot.panda_interpolation_controller import PandaInterpolationController
from typing import Optional
import math
import pathlib
import shutil
from utils.camera.timestamp_accumulator import (
    TimestampObsAccumulator,
    TimestampActionAccumulator,
    align_timestamps,
)
from utils.replay_buffer import ReplayBuffer
from utils.cv2_util import (
    get_image_transform,
    optimal_row_cols,
)
from utils.multi_camera_visualizer import MultiCameraVisualizer
from utils.camera.video_recorder import VideoRecorder
import torch
import os


class RealEnv:
    def __init__(self,
                 task_config=None,
                 WH=[640, 480],
                 capture_fps=15,
                 obs_fps=15,
                 n_obs_steps=1,
                 enable_color=True,
                 enable_depth=True,
                 process_depth=False,
                 use_robot=True,
                 verbose=False,
                 gripper_enable=False,
                 speed=50,
                 wrist=None,
                 ):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.WH = WH
        self.capture_fps = capture_fps
        self.obs_fps = obs_fps
        self.n_obs_steps = n_obs_steps
        if wrist is None:
            print('No wrist camera. Using default camera id.')
            self.WRIST = '311322300308'
        else:
            self.WRIST = wrist

        base_path = os.path.dirname(
            os.path.dirname(os.path.realpath(__file__)))
        self.vis_dir = os.path.join(base_path, 'dump/vis_real_world')

        self.serial_numbers = SingleRealsense.get_connected_devices_serial()
        if self.WRIST is not None and self.WRIST in self.serial_numbers:
            print('Found wrist camera.')
            self.serial_numbers.remove(self.WRIST)
            self.serial_numbers = self.serial_numbers + \
                [self.WRIST]  # put the wrist camera at the end
            self.n_fixed_cameras = len(self.serial_numbers) - 1
        else:
            self.n_fixed_cameras = len(self.serial_numbers)
        print(f'Found {self.n_fixed_cameras} fixed cameras.')

        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        self.realsense = MultiRealsense(
            serial_numbers=self.serial_numbers,
            shm_manager=self.shm_manager,
            resolution=(self.WH[0], self.WH[1]),
            capture_fps=self.capture_fps,
            enable_color=enable_color,
            enable_depth=enable_depth,
            process_depth=process_depth,
            verbose=verbose)
        self.realsense.set_exposure(exposure=100, gain=60)
        self.realsense.set_white_balance()
        self.last_realsense_data = None
        self.enable_color = enable_color
        self.enable_depth = enable_depth
        self.use_robot = use_robot

        if self.use_robot:
            self.robot = PandaInterpolationController(
                shm_manager=self.shm_manager,
                robot_ip='172.16.0.2',
                frequency=100,
                verbose=False,
                speed=speed,
                gripper_enable=gripper_enable
            )

    # ======== start-stop API =============

    @property
    def is_ready(self):
        return self.realsense.is_ready and self.robot.is_ready

    def start(self, wait=True):
        self.realsense.start(wait=False)
        self.robot.start(wait=False)
        if wait:
            self.start_wait()

    def stop(self, wait=False):
        self.end_episode()
        self.robot.stop(wait=False)
        self.realsense.stop(wait=False)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.realsense.start_wait()
        self.robot.start_wait()

    def stop_wait(self):
        self.robot.stop_wait()
        self.realsense.stop_wait()

    # ========= context manager ===========

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= async env API ===========
    def get_obs(self) -> dict:
        "observation dict"
        assert self.is_ready

        # get data
        k = math.ceil(self.n_obs_steps * (self.capture_fps / self.obs_fps))
        self.last_realsense_data = self.realsense.get(
            k=k,
            out=self.last_realsense_data
        )
        robot_obs = dict()
        if self.use_robot:
            robot_obs['joint_position'] = self.robot.get_state()
            robot_obs['EE_pose'] = self.robot.get_EE_pose()
            robot_obs['gripper_state'] = self.robot.get_gripper_state()

            # 125 hz, robot_receive_timestamp
            last_robot_data = self.robot.get_all_state()
            # both have more than n_obs_steps data

            # align camera obs timestamps
            dt = 1 / self.frequency
            last_timestamp = np.max([x['timestamp'][-1]
                                    for x in self.last_realsense_data.values()])
            obs_align_timestamps = last_timestamp - \
                (np.arange(self.n_obs_steps)[::-1] * dt)

            camera_obs = dict()
            for camera_idx, value in self.last_realsense_data.items():
                this_timestamps = value['timestamp']
                this_idxs = list()
                for t in obs_align_timestamps:
                    is_before_idxs = np.nonzero(this_timestamps < t)[0]
                    this_idx = 0
                    if len(is_before_idxs) > 0:
                        this_idx = is_before_idxs[-1]
                    this_idxs.append(this_idx)
                # remap key
                camera_obs[f'camera_{camera_idx}'] = value['color'][this_idxs]
                # camera_obs['image'] = value['color'][this_idxs]

            # align robot obs
            robot_timestamps = last_robot_data['robot_receive_timestamp']
            this_timestamps = robot_timestamps
            this_idxs = list()
            for t in obs_align_timestamps:
                is_before_idxs = np.nonzero(this_timestamps < t)[0]
                this_idx = 0
                if len(is_before_idxs) > 0:
                    this_idx = is_before_idxs[-1]
                this_idxs.append(this_idx)

            robot_obs_raw = dict()
            for k, v in last_robot_data.items():
                if k in self.obs_key_map:
                    robot_obs_raw[self.obs_key_map[k]] = v

            robot_obs = dict()
            for k, v in robot_obs_raw.items():
                robot_obs[k] = v[this_idxs]

            # accumulate obs
            if self.obs_accumulator is not None:
                self.obs_accumulator.put(
                    robot_obs_raw,
                    robot_timestamps
                )

            # return obs
            obs_data = dict(camera_obs)
            obs_data.update(robot_obs)
            obs_data['timestamp'] = obs_align_timestamps
            return obs_data

    def get_robot_state(self):
        """
        Get the real robot state.
        """
        gripper_state = self.gripper.read_once()
        gripper_qpos = gripper_state.width

        robot_qpos = np.concatenate(
            [self.panda.get_log()["q"][-1], [gripper_qpos / 2.0]]
        )

        obs = np.concatenate(
            [self.panda.get_position(), self.panda.get_orientation(), robot_qpos],
            dtype=np.float32,  # 15
        )

        assert obs.shape == (15,), f"incorrect obs shape, {obs.shape}"

        return obs

    def log_pose(self, verbose=False):
        while True:
            start_time = time.time()

            pose = np.ascontiguousarray(
                self.panda.get_pose()).astype(np.float32)
            init_time = time.time()

            data = {
                "pose": pose,
                "timestamp": init_time,
            }

            self.pose_buffer.put(data)

            elapsed_time = time.time() - start_time
            if elapsed_time < 0.001:
                time.sleep(0.001 - elapsed_time)

    def step(self, action, visualize=False):
        """
        Step robot in the real.
        """
        # Simple motion in cartesian space
        gripper = action[-1] * 0.08
        euler = action[3:-1]  # Euler angle
        quat = transforms3d.euler.euler2quat(*euler)

        pose = np.concatenate([action[:3], quat], axis=0)
        print(pose)

        try:
            results = self.planner.plan_screw(
                pose, self.agent.get_qpos(), time_step=0.1
            )
            waypoints = results["position"][..., np.newaxis]

            self.panda.move_to_joint_position(
                waypoints=waypoints, speed_factor=0.1)
            self.gripper.move(width=gripper, speed=0.3)

            q_pose = np.zeros((9,))
            q_pose[:-2] = self.panda.q
            q_pose[-2] = gripper / 2.0
            q_pose[-1] = gripper / 2.0

            self.agent.set_qpos(q_pose)
        except Exception as e:
            print(e)
            print("Failed to generate valid waypoints.")

        return self.get_obs(visualize=visualize)

    def get_robot_state(self):
        return self.robot.get_state()

    # recording API
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # prepare recording stuff
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.realsense.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))

        # start recording on realsense
        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(
            video_path=video_paths, start_time=start_time)

        # create accumulators
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1/self.frequency
        )
        print(f'Episode {episode_id} started!')

    def end_episode(self):
        "Stop recording"
        assert self.is_ready

        # stop video recorder
        self.realsense.stop_recording()

        if self.obs_accumulator is not None:
            # recording
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            # Since the only way to accumulate obs and action is by calling
            # get_obs and exec_actions, which will be in the same thread.
            # We don't need to worry new data come in here.
            obs_data = self.obs_accumulator.data
            obs_timestamps = self.obs_accumulator.timestamps

            actions = self.action_accumulator.actions
            action_timestamps = self.action_accumulator.timestamps
            stages = self.stage_accumulator.actions
            n_steps = min(len(obs_timestamps), len(action_timestamps))
            if n_steps > 0:
                episode = dict()
                episode['timestamp'] = obs_timestamps[:n_steps]
                episode['action'] = actions[:n_steps]
                episode['stage'] = stages[:n_steps]
                for key, value in obs_data.items():
                    episode[key] = value[:n_steps]
                self.replay_buffer.add_episode(episode, compressors='disk')
                episode_id = self.replay_buffer.n_episodes - 1
                print(f'Episode {episode_id} saved!')

            self.obs_accumulator = None
            self.action_accumulator = None
            self.stage_accumulator = None

    def drop_episode(self):
        self.end_episode()
        self.replay_buffer.drop_episode()
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        if this_video_dir.exists():
            shutil.rmtree(str(this_video_dir))
        print(f'Episode {episode_id} dropped!')
