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

from utils.camera.multi_cam import MultiRealsense
from utils.shared_memory.shared_memory_queue import SharedMemoryQueue

import numpy as np

from utils.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer

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
from utils.video_recorder import VideoRecorder

# Define default observation key map
DEFAULT_OBS_KEY_MAP = {
    # robot
    'ActualTCPPose': 'robot_eef_pose',
    'ActualTCPSpeed': 'robot_eef_pose_vel',
    'ActualQ': 'robot_joint',
    'ActualQd': 'robot_joint_vel',
    # timestamps
    'step_idx': 'step_idx',
    'timestamp': 'timestamp'
}


class RealRobot:
    def __init__(self,
                 robot_ip,
                 output_dir,
                 # env params
                 frequency=10,
                 n_obs_steps=2,
                 # obs
                 obs_image_resolution=(640, 480),
                 max_obs_buffer_size=30,
                 camera_serial_numbers=None,
                 obs_key_map=DEFAULT_OBS_KEY_MAP,
                 obs_float32=False,
                 # action
                 max_pos_speed=0.015,
                 max_rot_speed=0.03,
                 # robot
                 tcp_offset=0.13,
                 init_joints=True,
                 # video capture params
                 video_capture_fps=30,
                 video_capture_resolution=(1280, 720),
                 # saving params
                 record_raw_video=True,
                 thread_per_video=2,
                 video_crf=21,
                 # vis params
                 enable_multi_cam_vis=True,
                 multi_cam_vis_resolution=(1280, 720),
                 shm_manager=None):
        print("Initializing robot ...")

        # Initialize robot connection and libraries
        self.panda = panda_py.Panda(robot_ip)
        self.gripper = libfranka.Gripper(robot_ip)
        self.panda.enable_logging(int(10))

        # Initialize output directories and replay buffer
        output_dir = pathlib.Path(output_dir)
        assert output_dir.parent.is_dir()
        video_dir = output_dir.joinpath('videos')
        video_dir.mkdir(parents=True, exist_ok=True)
        zarr_path = str(output_dir.joinpath('replay_buffer.zarr').absolute())
        self.replay_buffer = ReplayBuffer.create_from_path(
            zarr_path=zarr_path, mode='a')

        # Shared memory manager
        if shm_manager is None:
            self.shm_manager = SharedMemoryManager()
            self.shm_manager.start()
        else:
            self.shm_manager = shm_manager

        # Camera initialization
        if camera_serial_numbers is None:
            camera_serial_numbers = self.get_connected_camera_serial_numbers()

        self.camera_serial_numbers = camera_serial_numbers
        self.obs_image_resolution = obs_image_resolution
        self.video_capture_resolution = video_capture_resolution
        self.video_capture_fps = video_capture_fps
        self.record_raw_video = record_raw_video
        self.frequency = frequency
        self.obs_float32 = obs_float32
        self.enable_multi_cam_vis = enable_multi_cam_vis
        self.multi_cam_vis_resolution = multi_cam_vis_resolution
        self.thread_per_video = thread_per_video
        self.video_crf = video_crf
        self.n_obs_steps = n_obs_steps
        self.max_obs_buffer_size = max_obs_buffer_size
        self.obs_key_map = obs_key_map
        self.output_dir = output_dir
        self.video_dir = video_dir
        self.init_joints = init_joints
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed

        # Initialize other components
        self.init_robot()
        self.init_cameras()
        self.init_visualizer()
        self.init_video_recorder()

        # Accumulators for recording
        self.obs_accumulator = None
        self.action_accumulator = None
        self.stage_accumulator = None

        # Variables to track state
        self.last_realsense_data = None
        self.start_time = None
        self.is_started = False

        print("Finished initializing robot.")

    # ======== Start-Stop API =============
    @property
    def is_ready(self):
        # Check if both robot and cameras are ready
        return self.panda is not None and self.realsense.is_ready

    def start(self, wait=True):
        # Start cameras and robot
        self.realsense.start(wait=False)
        if self.enable_multi_cam_vis:
            self.multi_cam_vis.start(wait=False)
        self.is_started = True
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        # Stop cameras and robot
        self.end_episode()  # Ensure any ongoing episode is ended
        if self.enable_multi_cam_vis:
            self.multi_cam_vis.stop(wait=False)
        self.realsense.stop(wait=False)
        self.is_started = False
        if wait:
            self.stop_wait()

    def start_wait(self):
        # Wait for components to be ready
        self.realsense.start_wait()
        if self.enable_multi_cam_vis:
            self.multi_cam_vis.start_wait()

    def stop_wait(self):
        # Wait for components to fully stop
        self.realsense.stop_wait()
        if self.enable_multi_cam_vis:
            self.multi_cam_vis.stop_wait()

    # ========= Context Manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========== Initialization Methods ==========
    def init_robot(self):
        # Initialize robot to starting position
        if not self.init_joints:
            joint_pose = None
        else:
            joint_pose = [
                0.00000000e00,
                -3.19999993e-01,
                0.00000000e00,
                -2.61799383e00,
                0.00000000e00,
                2.23000002e00,
                7.85398185e-01,
            ]

        if joint_pose is not None:
            self.panda.move_to_joint_position(joint_pose)
        self.gripper.move(width=0.0, speed=0.1)  # Close gripper

    def init_cameras(self):
        # Initialize cameras and shared memory
        color_tf = get_image_transform(
            input_res=self.video_capture_resolution,
            output_res=self.obs_image_resolution,
            bgr_to_rgb=True
        )

        if self.obs_float32:
            def color_transform(x): return color_tf(x).astype(
                np.float32) / 255
        else:
            color_transform = color_tf

        def transform(data):
            data['color'] = color_transform(data['color'])
            return data

        # Visualization transform
        rw, rh, col, row = optimal_row_cols(
            n_cameras=len(self.camera_serial_numbers),
            in_wh_ratio=self.obs_image_resolution[0] /
            self.obs_image_resolution[1],
            max_resolution=self.multi_cam_vis_resolution
        )
        vis_color_transform = get_image_transform(
            input_res=self.video_capture_resolution,
            output_res=(rw, rh),
            bgr_to_rgb=False
        )

        def vis_transform(data):
            data['color'] = vis_color_transform(data['color'])
            return data

        recording_transfrom = None
        recording_fps = self.video_capture_fps
        recording_pix_fmt = 'bgr24'
        if not self.record_raw_video:
            recording_transfrom = transform
            recording_fps = self.frequency
            recording_pix_fmt = 'rgb24'

        self.video_recorder = VideoRecorder.create_h264(
            fps=recording_fps,
            codec='h264',
            input_pix_fmt=recording_pix_fmt,
            crf=self.video_crf,
            thread_type='FRAME',
            thread_count=self.thread_per_video
        )

        self.realsense = MultiRealsense(
            serial_numbers=self.camera_serial_numbers,
            shm_manager=self.shm_manager,
            resolution=self.video_capture_resolution,
            capture_fps=self.video_capture_fps,
            put_fps=self.video_capture_fps,
            put_downsample=False,
            record_fps=recording_fps,
            enable_color=True,
            enable_depth=False,
            enable_infrared=False,
            get_max_k=self.max_obs_buffer_size,
            transform=transform,
            vis_transform=vis_transform,
            recording_transform=recording_transfrom,
            video_recorder=self.video_recorder,
            verbose=False
        )

    def init_visualizer(self):
        # Initialize multi-camera visualizer
        if self.enable_multi_cam_vis:
            rw, rh, col, row = optimal_row_cols(
                n_cameras=len(self.camera_serial_numbers),
                in_wh_ratio=self.obs_image_resolution[0] /
                self.obs_image_resolution[1],
                max_resolution=self.multi_cam_vis_resolution
            )
            self.multi_cam_vis = MultiCameraVisualizer(
                realsense=self.realsense,
                row=row,
                col=col,
                rgb_to_bgr=False
            )
        else:
            self.multi_cam_vis = None

    def init_video_recorder(self):
        # Video recorder is initialized in init_cameras
        pass

    def get_connected_camera_serial_numbers(self):
        # Placeholder method to get connected camera serial numbers
        # You may need to implement this based on your setup
        return ['camera_serial_1', 'camera_serial_2']

    # ========= Episode Management ===========
    def start_episode(self, start_time=None):
        "Start recording and return first obs"
        if start_time is None:
            start_time = time.time()
        self.start_time = start_time

        assert self.is_ready

        # Prepare recording directories
        episode_id = self.replay_buffer.n_episodes
        this_video_dir = self.video_dir.joinpath(str(episode_id))
        this_video_dir.mkdir(parents=True, exist_ok=True)
        n_cameras = self.realsense.n_cameras
        video_paths = list()
        for i in range(n_cameras):
            video_paths.append(
                str(this_video_dir.joinpath(f'{i}.mp4').absolute()))

        # Start recording on realsense
        self.realsense.restart_put(start_time=start_time)
        self.realsense.start_recording(
            video_path=video_paths, start_time=start_time)

        # Create accumulators
        self.obs_accumulator = TimestampObsAccumulator(
            start_time=start_time,
            dt=1 / self.frequency
        )
        self.action_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1 / self.frequency
        )
        self.stage_accumulator = TimestampActionAccumulator(
            start_time=start_time,
            dt=1 / self.frequency
        )
        print(f'Episode {episode_id} started!')

    def end_episode(self):
        "Stop recording"
        if not self.is_started:
            return  # No episode to end

        assert self.is_ready

        # Stop video recorder
        self.realsense.stop_recording()

        if self.obs_accumulator is not None:
            # Recording
            assert self.action_accumulator is not None
            assert self.stage_accumulator is not None

            # Collect data from accumulators
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

    # ========= Observation and Action Methods ===========
    def get_obs(self) -> dict:
        "Observation dict"
        assert self.is_ready

        # Get data from cameras
        k = max(
            self.n_obs_steps,
            int(self.n_obs_steps * (self.video_capture_fps / self.frequency))
        )
        self.last_realsense_data = self.realsense.get(
            k=k,
            out=self.last_realsense_data)

        # Get robot data
        robot_pose = self.get_robot_state()
        robot_timestamps = np.array([time.time()])

        # Align camera obs timestamps
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
            # Collect camera observations
            camera_obs[f'camera_{camera_idx}'] = value['color'][this_idxs]

        # Align robot obs
        this_timestamps = robot_timestamps
        this_idxs = list()
        for t in obs_align_timestamps:
            is_before_idxs = np.nonzero(this_timestamps < t)[0]
            this_idx = 0
            if len(is_before_idxs) > 0:
                this_idx = is_before_idxs[-1]
            this_idxs.append(this_idx)

        robot_obs_raw = {
            'ActualTCPPose': robot_pose,
            'timestamp': robot_timestamps
        }

        robot_obs = dict()
        for k, v in robot_obs_raw.items():
            if k in self.obs_key_map:
                robot_obs[self.obs_key_map[k]] = v

        # Accumulate observations
        if self.obs_accumulator is not None:
            self.obs_accumulator.put(
                robot_obs_raw,
                robot_timestamps
            )

        # Return obs
        obs_data = dict(camera_obs)
        obs_data.update(robot_obs)
        obs_data['timestamp'] = obs_align_timestamps
        return obs_data

    def exec_actions(self,
                     actions: np.ndarray,
                     timestamps: np.ndarray,
                     stages: np.ndarray = None):
        assert self.is_ready
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        if not isinstance(timestamps, np.ndarray):
            timestamps = np.array(timestamps)
        if stages is None:
            stages = np.zeros_like(timestamps, dtype=np.int64)
        elif not isinstance(stages, np.ndarray):
            stages = np.array(stages, dtype=np.int64)

        # Execute actions
        for i in range(len(actions)):
            action = actions[i]
            target_time = timestamps[i]

            # Extract gripper action and pose
            gripper = action[-1] * 0.08
            euler = action[3:-1]  # Euler angles
            quat = transforms3d.euler.euler2quat(*euler)
            pose = np.concatenate([action[:3], quat], axis=0)

            # Move robot to the pose
            try:
                # Directly move to the pose (you may need to adjust this)
                self.panda.move_to_pose(pose, speed=0.1)
                self.gripper.move(width=gripper, speed=0.3)
            except Exception as e:
                print(e)
                print("Failed to execute action.")

            # Record actions
            if self.action_accumulator is not None:
                self.action_accumulator.put(
                    action[np.newaxis, :],
                    np.array([target_time])
                )
            if self.stage_accumulator is not None:
                self.stage_accumulator.put(
                    stages[i:i+1],
                    np.array([target_time])
                )

    def get_robot_state(self):
        """
        Get the real robot state.
        """
        gripper_state = self.gripper.read_once()
        gripper_qpos = gripper_state.width

        robot_qpos = self.panda.get_q()
        robot_qvel = self.panda.get_dq()

        tcp_pose = np.ascontiguousarray(
            self.panda.get_pose()).astype(np.float32)

        robot_state = {
            'q': robot_qpos,
            'dq': robot_qvel,
            'tcp_pose': tcp_pose,
            'gripper_width': gripper_qpos
        }

        return robot_state

    # ========= Additional Methods ===========
    def get_tcp_pose(self):
        return np.ascontiguousarray(self.panda.get_pose()).astype(np.float32)

    def log_pose(self, verbose=False):
        while self.is_started:
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

    def test_sequence(self):
        """
        Test sequence of actions to test the robot.
        """
        for i in range(10):
            joint_pose = [
                0.00000000e00,
                -3.19999993e-01,
                0.00000000e00,
                -2.61799383e00,
                0.00000000e00,
                2.23000002e00,
                7.85398185e-01,
            ]

            self.panda.move_to_joint_position(joint_pose, speed_factor=0.1)

            obs = self.get_obs()

            self.panda.move_to_start(speed_factor=0.1)

    def end(self):
        self.stop()

    # ======= Main execution for testing =======
if __name__ == "__main__":
    # Example usage
    robot_ip = '172.160.0.2'  # Replace with your robot's IP
    output_dir = '/path/to/output'  # Replace with your desired output directory
    # Replace with your camera serial numbers
    cameras = ['serial_number_1', 'serial_number_2']

    with RealRobot(robot_ip=robot_ip, output_dir=output_dir, camera_serial_numbers=cameras) as robot:
        robot.start_episode()
        try:
            # Your code to interact with the robot
            obs = robot.get_obs()
            print("Initial observation:", obs)

            # Example action
            action = np.zeros(7)  # Replace with your action
            timestamp = np.array([time.time() + 1.0])  # Execute after 1 second
            robot.exec_actions(
                actions=action[np.newaxis, :], timestamps=timestamp)

            # Wait and get new observation
            time.sleep(1.0)
            obs = robot.get_obs()
            print("Next observation:", obs)

        finally:
            robot.end_episode()
