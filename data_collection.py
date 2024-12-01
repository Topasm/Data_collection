import numpy as np
import time
import threading
import queue
import panda_py.controllers
from scipy.spatial.transform import Rotation as R
import panda_py
from utils.inputs.spacemouse_shared_memory import Spacemouse
import cv2
from multiprocessing.managers import SharedMemoryManager
from utils.robot.real_robot import RealEnv
from utils.precise_sleep import precise_wait
import pickle
import os
from pathlib import Path

from utils.inputs.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
import scipy.spatial.transform as st
import click
MOVE_INCREMENT = 0.005


@click.command()
@click.option('--output', '-o', default="./data", required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', default="172.16.0.2", required=True, help="Franka's IP address e.g. 172.16.0.2")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SpaceMouse command to executing on Robot in Sec.")
def main(output, robot_ip, init_joints, frequency, command_latency):
    dt = 1 / frequency
    output_path = Path(output)
    observations = []  # List to store observations

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
                RealEnv(
                    output_dir=output,
                    robot_ip=robot_ip,
                    # recording resolution
                    obs_fps=frequency,
                    init_joints=init_joints,
        ) as env:
            # super.__init__()

            cv2.setNumThreads(1)

            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=3900)
            # env.robot.start()

            time.sleep(2.0)
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            while not stop:

                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()
                observations.append(obs)

                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        # Start recording
                        print('Start recording!')
                        env.start_episode(
                            t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
                        key_counter.clear()
                        is_recording = True
                        print('Recording!')
                    elif key_stroke == KeyCode(char='s'):
                        # Stop recording
                        env.end_episode()
                        key_counter.clear()
                        is_recording = False
                        print('Stopped.')
                    elif key_stroke == Key.backspace:
                        # Delete the most recent recorded episode
                        if click.confirm('Are you sure to drop an episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                        # delete
                stage = key_counter[Key.space]

                # get teleop command
                precise_wait(t_sample)

                precise_wait(t_cycle_end)
                iter_idx += 1

            # Save to pickle file
            pkl_path = output_path / 'observations.pkl'
            with open(pkl_path, 'wb') as f:
                pickle.dump(observations, f)
            print(f"Saved observations to {pkl_path}")


            # %%
if __name__ == '__main__':
    main()
