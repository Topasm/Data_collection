from typing import List, Optional, Union, Dict, Callable
import numbers
import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import pyrealsense2 as rs
from utils.camera.single_cam import SingleRealsense


class MultiRealsense:
    def __init__(self,
                 serial_numbers: Optional[List[str]] = None,
                 shm_manager: Optional[SharedMemoryManager] = None,
                 resolution=(1280, 720),
                 capture_fps=30,
                 put_fps=None,
                 put_downsample=True,
                 enable_color=True,
                 enable_depth=False,
                 process_depth=False,
                 enable_infrared=False,
                 get_max_k=30,
                 advanced_mode_config: Optional[Union[dict,
                                                      List[dict]]] = None,
                 transform: Optional[Union[Callable[[
                     Dict], Dict], List[Callable]]] = None,
                 verbose=False,
                 pose_buffer=None
                 ):
        if shm_manager is None:
            shm_manager = SharedMemoryManager()
            shm_manager.start()
        if serial_numbers is None:
            serial_numbers = SingleRealsense.get_connected_devices_serial()
        n_cameras = len(serial_numbers)

        advanced_mode_config = repeat_to_list(
            advanced_mode_config, n_cameras, dict)
        transform = repeat_to_list(
            transform, n_cameras, Callable)

        cameras = dict()
        for i, serial in enumerate(serial_numbers):
            cameras[serial] = SingleRealsense(
                shm_manager=shm_manager,
                serial_number=serial,
                resolution=resolution,
                capture_fps=capture_fps,
                put_fps=put_fps,
                put_downsample=put_downsample,
                enable_color=enable_color,
                enable_depth=enable_depth,
                process_depth=process_depth,
                enable_infrared=enable_infrared,
                get_max_k=get_max_k,
                advanced_mode_config=advanced_mode_config[i],
                transform=transform[i],
                verbose=verbose
            )

        self.cameras = cameras
        self.serial_numbers = serial_numbers
        self.shm_manager = shm_manager
        self.pose_buffer = pose_buffer

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def n_cameras(self):
        return len(self.cameras)

    @property
    def is_ready(self):
        is_ready = True
        for camera in self.cameras.values():
            if not camera.is_ready:
                is_ready = False
        return is_ready

    def start(self, wait=True, put_start_time=None):
        if put_start_time is None:
            put_start_time = time.time()
        for camera in self.cameras.values():
            camera.start(wait=False, put_start_time=put_start_time)

        if wait:
            self.start_wait()

    def stop(self, wait=True):
        for camera in self.cameras.values():
            camera.stop(wait=False)

        if wait:
            self.stop_wait()

    def start_wait(self):
        for camera in self.cameras.values():
            camera.start_wait()

    def stop_wait(self):
        for camera in self.cameras.values():
            camera.join()

    def get(self, k=None, index=None, out=None) -> Dict[int, Dict[str, np.ndarray]]:
        """
        Return order T,H,W,C
        {
            0: {
                'rgb': (T,H,W,C),
                'timestamp': (T,)
            },
            1: ...
        }
        """
        if index is not None:
            this_out = None
            this_out = self.cameras[self.serial_numbers[index]].get(
                k=k, out=this_out)
            return this_out
        if out is None:
            out = dict()
        for i, camera in enumerate(self.cameras.values()):
            this_out = None
            if i in out:
                this_out = out[i]
            this_out = camera.get(k=k, out=this_out)
            out[i] = this_out
        return out

    def set_color_option(self, option, value):
        n_camera = len(self.cameras)
        value = repeat_to_list(value, n_camera, numbers.Number)
        for i, camera in enumerate(self.cameras.values()):
            camera.set_color_option(option, value[i])

    def set_exposure(self, exposure=None, gain=None):
        """150nit. (0.1 ms, 1/10000s)
        gain: (0, 128)
        """

        if exposure is None and gain is None:
            # auto exposure
            self.set_color_option(rs.option.enable_auto_exposure, 1.0)
        else:
            # manual exposure
            self.set_color_option(rs.option.enable_auto_exposure, 0.0)
            if exposure is not None:
                self.set_color_option(rs.option.exposure, exposure)
            if gain is not None:
                self.set_color_option(rs.option.gain, gain)

    def set_white_balance(self, white_balance=None):
        if white_balance is None:
            self.set_color_option(rs.option.enable_auto_white_balance, 1.0)
        else:
            self.set_color_option(rs.option.enable_auto_white_balance, 0.0)
            self.set_color_option(rs.option.white_balance, white_balance)

    def get_intrinsics(self):
        return np.array([c.get_intrinsics() for c in self.cameras.values()])

    def get_depth_scale(self):
        return np.array([c.get_depth_scale() for c in self.cameras.values()])

    def restart_put(self, start_time):
        for camera in self.cameras.values():
            camera.restart_put(start_time)

    # def run(self):
    #     threadpool_limits(16)
    #     cv2.setNumThreads(16)
    #     iter_idx = 0
    #     while not self.stop_event.is_set():

    #         out = dict()
    #         # get camera data
    #         for i, camera in enumerate(self.cameras.values()):
    #             this_out = None
    #             if i in out:
    #                 this_out = out[i]

    #             this_out = camera.get(out=this_out)

    #             out[i] = this_out

    #         # process camera data
    #         tmp = dict()
    #         for i, camera in enumerate(self.cameras.values()):
    #             v = out[i]["vertices"]
    #             tex_coords = out[i]["tex"]
    #             color = out[i]["color"]

    #             colors = get_color_from_tex_coords(tex_coords, color)
    #             colors = np.ascontiguousarray(colors)

    #             points, colors = filter_vectors(v, colors)

    #             if camera.name == "wrist_cam":
    #                 poses = self.pose_buffer.get_last_k(k=50)

    #                 pose_times = np.abs(
    #                     poses["timestamp"] - out[i]["timestamp"])
    #                 pose_idx = np.argmin(pose_times)

    #                 diff = poses["timestamp"][pose_idx] - out[i]["timestamp"]
    #                 tcp_T_base = poses["pose"][pose_idx]

    #                 wristcam_T_base = tcp_T_base @ camera.transform
    #                 wristcam_T_base_T = np.ascontiguousarray(
    #                     wristcam_T_base.T[:3, :3])
    #                 points = points @ wristcam_T_base_T + \
    #                     wristcam_T_base[:3, 3]

    #             else:
    #                 points = points @ camera.transform_T + \
    #                     camera.transform[:3, 3]

    #             tmp[camera.name] = np.concatenate(
    #                 [points.reshape(-1, 3), colors.reshape(-1, 3) / 255.0], axis=-1
    #             )

    #         pcds = np.concatenate(
    #             [x for x in tmp.values()], axis=0)  # merge pcds
    #         mask = pcd_filter_bound(pcds[..., :3])
    #         pcds = pcds[mask]

    #         pcds = pcds[uniform_sampling(pcds, npoints=16384)]

    #         # commented out for data collection. Uncomment for testing

    #         fps_sampling_idx = fpsample.fps_sampling(pcds[..., :3], 5000)

    #         pcds = pcds[fps_sampling_idx]

    #         pcds = pcds[
    #             dbscan_outlier_removal_idx(
    #                 pcds[..., :3], eps=0.3, min_samples=300)
    #         ]
    #         pcds = pcds[
    #             dbscan_outlier_removal_idx(
    #                 pcds[..., :3], eps=0.02, min_samples=5)
    #         ]
    #         pcds = pcds[uniform_sampling(pcds, npoints=4096)]

    #         pcds = {"pcds": pcds}

    #         self.pcd_ring_buffer.put(pcds)

    #         if iter_idx == 0:
    #             self.pcd_process_ready_event.set()

    #         iter_idx += 1


def repeat_to_list(x, n: int, cls):
    if x is None:
        x = [None] * n
    if isinstance(x, cls):
        x = [x] * n
    assert len(x) == n
    return x
