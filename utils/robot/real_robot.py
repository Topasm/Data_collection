import panda_py
from panda_py import libfranka

import time
import threading
import panda_py.controllers
import transforms3d
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R
import multiprocessing as mp
from multiprocessing import Process, Queue, Event
from multiprocessing.managers import SharedMemoryManager
from utils.camera.multi_cam import MultiRealsense, SingleRealsense
import numpy as np

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
from utils.inputs.spacemouse_shared_memory import Spacemouse
import multiprocessing as mp
import pickle
from utils.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer,
)

cameras = {
    "wrist_cam": "",
    "right_cam": "",
    "left_cam": "",
}
SPEED = 0.1  # [m/s]
FORCE = 20.0  # [N]
MOVE_INCREMENT = 0.005


class RealEnv(mp.Process):
    def __init__(self,
                 output_dir='./data',
                 robot_ip=None,
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
                 gripper_enable=True,
                 speed=50,
                 wrist=None,
                 init_joints=False,
                 ):
        super().__init__()
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.WH = WH
        self.capture_fps = capture_fps
        self.obs_fps = obs_fps
        self.n_obs_steps = n_obs_steps
        self.panda = panda_py.Panda(robot_ip)
        self.gripper = libfranka.Gripper(robot_ip)
        self.panda.enable_logging(int(10))

        Robot_state = {
            # [x,y,z, qx,qy,qz,qw, gripper]
            'EEF_state': np.zeros(8, dtype=np.float32),
            'timestamp': np.array([time.time()], dtype=np.float64)
        }

        self.serial_numbers = SingleRealsense.get_connected_devices_serial()

        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()

        self.Robot_state_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=self.shm_manager,
            examples=Robot_state,
            get_max_k=30,  # Store last 30 states
            get_time_budget=0.2,
            put_desired_frequency=self.obs_fps
        )

        self.space_mouse = Spacemouse(
            shm_manager=self.shm_manager, deadzone=0.3)
        self.space_mouse.start()
        self.ready_event = mp.Event()

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

        multi_cam_vis = MultiCameraVisualizer(
            realsense=self.realsense,
            row=self.WH[0],
            col=self.WH[1],
            rgb_to_bgr=False
        )
        self.multi_cam_vis = multi_cam_vis

        self.output_dir = output_dir
        self.init_joints = init_joints
        self.current_episode = []
        self.episode_counter = 0  # To generate unique filenames
        self.is_recording = False
        os.makedirs(self.output_dir, exist_ok=True)

    # ======== start-stop API =============

    def put_state(self, state_data):
        """Put new state data into ring buffer"""
        self.Robot_state_buffer.put(state_data)

    def run(self):

        try:
            self.init_robot()
            running = True
            self.ready_event.set()
            current_rotation = self.panda.get_orientation()
            current_translation = self.panda.get_position()
            sm_state = self.space_mouse.get_motion_state_transformed()

            ctx = self.panda.create_context(frequency=1000)
            controller = panda_py.controllers.CartesianImpedance()
            self.panda.start_controller(controller)
            time.sleep(1)

            while ctx.ok() and running:
                start_time = time.perf_counter()

                sm_state = self.space_mouse.get_motion_state_transformed()
                dpos = sm_state[:3] * MOVE_INCREMENT
                drot_xyz = sm_state[3:] * MOVE_INCREMENT * 3
                drot_xyz[:] = 0
                state_data = self.get_state()

                current_translation += np.array([dpos[0], dpos[1], dpos[2]])
                if drot_xyz is not None:
                    delta_rotation = R.from_euler('xyz', drot_xyz)
                    current_rotation = (
                        delta_rotation * R.from_quat(current_rotation)).as_quat()

                controller.set_control(current_translation, current_rotation)

                # Handle gripper state changes
                if self.space_mouse.is_button_pressed(0):
                    success = self.gripper.grasp(0.03, speed=SPEED, force=FORCE,
                                                 epsilon_inner=0.05, epsilon_outer=0.05)
                    if success:
                        print("Grasp successful")
                    else:
                        print("Grasp failed")

                elif self.space_mouse.is_button_pressed(1):
                    success = self.gripper.move(0.08, speed=SPEED)
                    if success:
                        print("Release successful")
                    else:
                        print("Release failed")

                # Sleep to maintain loop frequency of 1000 Hz
                end_time = time.perf_counter()
                self.put_state(state_data)

        except Exception as e:
            print(e)
            running = False

    def init_robot(self):
        joint_pose = [-0.01588696, -0.25534376, 0.18628714, -
                      2.28398158, 0.0769999, 2.02505396, 0.07858208]

        self.panda.move_to_joint_position(joint_pose)
        # self.gripper.move(width=0.8, speed=0.1)

        # replicate in sim
        action = np.zeros((9,))
        action[:-2] = joint_pose

    # ========= context manager ===========

    @property
    def is_ready(self):
        return self.realsense.is_ready

    def start(self, wait=True, exposure_time=5):
        self.realsense.start(
            wait=False, put_start_time=time.time() + exposure_time)
        self.multi_cam_vis.start(wait=False)
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.realsense.stop(wait=False)
        self.multi_cam_vis.stop(wait=False)
        self.join()
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.realsense.start_wait()
        self.multi_cam_vis.start_wait()
        self.ready_event.wait()

    def stop_wait(self):
        self.realsense.stop_wait()
        self.multi_cam_vis.stop_wait()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    def get_robot_state(self):
        """Get latest state from ring buffer"""
        try:
            state = self.Robot_state_buffer.get()
            return state
        except Exception as e:
            print(f"Error getting state from buffer: {e}")
            return None

    def get_state(self):
        obs = dict()
        gripper_state = self.gripper.read_once()
        gripper_qpos = gripper_state.is_grasped

        obs = np.concatenate(
            [self.panda.get_position(), self.panda.get_orientation(),
             [gripper_qpos]],
            dtype=np.float32,  # 15
        )

        # joint_log = self.panda.get_log().get("q", [])
        # if not joint_log:
        #     raise ValueError(
        #         "Joint state log is empty. Unable to retrieve joint state.")

        obs = {'EEF_state': obs
               }

        return obs

    def get_obs(self, get_color=True, get_depth=False) -> dict:
        assert self.is_ready

        # get data
        k = math.ceil(self.n_obs_steps * (self.capture_fps / self.obs_fps))
        self.last_realsense_data = self.realsense.get(
            k=k,
            out=self.last_realsense_data
        )

        robot_obs = self.get_robot_state()

        dt = 1 / self.obs_fps
        timestamp_list = [x['timestamp'][-1]
                          for x in self.last_realsense_data.values()]
        last_timestamp = np.max(timestamp_list)
        obs_align_timestamps = last_timestamp - \
            (np.arange(self.n_obs_steps)[::-1] * dt)
        # the last timestamp is the latest one

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
            if get_color:
                assert self.enable_color
                camera_obs[f'color_img_{camera_idx}'] = value['color'][this_idxs]
            if get_depth and isinstance(camera_idx, int):
                assert self.enable_depth
                camera_obs[f'depth_{camera_idx}'] = value['depth'][this_idxs] / 1000.0

        # return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data['timestamp'] = obs_align_timestamps

        if self.is_recording:
            self.current_episode.append(obs_data)

        return obs_data

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

    # recording API

    def move_to(self, positions, orientations):
        self.panda.move_to_pose(positions, orientations)

    def grasp(self):
        self.gripper.grasp(0.03, speed=SPEED, force=FORCE,
                           epsilon_inner=0.3, epsilon_outer=0.3)

    def release(self):
        self.gripper.move(width=0.08, speed=0.1)

    def start_episode(self):
        """Start recording an episode."""
        self.is_recording = True
        self.current_episode = []
        print("Recording started.")

    def end_episode(self):
        """End recording and save the episode data."""
        self.is_recording = False
        episode_file = os.path.join(
            self.output_dir, f'episode_{self.episode_counter}.pkl')
        with open(episode_file, 'wb') as f:
            pickle.dump(self.current_episode, f)
        print(f"Recording stopped. Episode saved to {episode_file}")
        self.episode_counter += 1

    def drop_episode(self):
        """Delete the most recently saved episode."""
        if self.episode_counter > 0:
            self.episode_counter -= 1
            episode_file = os.path.join(
                self.output_dir, f'episode_{self.episode_counter}.pkl')
            if os.path.exists(episode_file):
                os.remove(episode_file)
                print(f"Episode {self.episode_counter} deleted.")
            else:
                print("No episode file found to delete.")
        else:
            print("No episodes to delete.")
