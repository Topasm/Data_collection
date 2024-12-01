import time
import multiprocessing as mp
import numpy as np
import cv2
from threadpoolctl import threadpool_limits
from utils.camera.multi_cam import MultiRealsense


class MultiCameraVisualizer(mp.Process):
    def __init__(self,
                 realsense: MultiRealsense,
                 row, col,
                 window_name='Multi Cam Vis',
                 vis_fps=60,
                 fill_value=0,
                 rgb_to_bgr=True
                 ):
        super().__init__()
        self.row = row
        self.col = col
        self.window_name = window_name
        self.vis_fps = vis_fps
        self.fill_value = fill_value
        self.rgb_to_bgr = rgb_to_bgr
        self.realsense = realsense
        # shared variables
        self.stop_event = mp.Event()

    def start(self, wait=False):
        super().start()

    def stop(self, wait=False):
        self.stop_event.set()
        if wait:
            self.stop_wait()

    def start_wait(self):
        pass

    def stop_wait(self):
        self.join()

    def run(self):
        cv2.setNumThreads(1)
        threadpool_limits(1)
        channel_slice = slice(None)
        if self.rgb_to_bgr:
            channel_slice = slice(None, None, -1)

        vis_data = None
        vis_img = None
        while not self.stop_event.is_set():
            vis_data = self.realsense.get(out=vis_data)
            color = vis_data[0]['color']
            H, W, C = color.shape
            assert C == 3
            vis_img = color  # Assign the color image to vis_img
            cv2.imshow(self.window_name, vis_img)
            cv2.pollKey()
            time.sleep(1 / self.vis_fps)
