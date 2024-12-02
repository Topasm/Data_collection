import numpy as np
import time
import cv2
from multiprocessing.managers import SharedMemoryManager
from utils.robot.real_robot import RealEnv
from utils.precise_sleep import precise_wait
import pickle
import os
from pathlib import Path
import click
from pynput import keyboard
import threading

# Initialize the KeyListener class


class KeyListener:
    def __init__(self):
        self.is_recording = False
        self.stop_program = False
        self.save_data = False
        self.init_robot_flag = False  # Add this line
        self.lock = threading.Lock()
        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        with self.lock:
            try:
                if key.char == 'q':
                    self.stop_program = True
                    print("Exiting program.")
                elif key.char == 'c':
                    self.is_recording = True
                    print("Started recording.")
                elif key.char == 's':
                    self.is_recording = False
                    self.save_data = True
                    print("Stopped recording and will save data.")
                elif key.char == 'h':
                    self.init_robot_flag = True  # Set the flag
                    print("Initializing robot.")
                elif key.char == '\x08':  # Backspace key
                    # Implement drop episode functionality if needed
                    pass
            except AttributeError:
                pass  # Handle special keys if needed

    def stop(self):
        self.listener.stop()


@click.command()
@click.option('--output', '-o', default="./data", required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', default="172.16.0.2", required=True, help="Franka's IP address e.g. 172.16.0.2")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving commands and executing on Robot in Sec.")
def main(output, robot_ip, init_joints, frequency, command_latency):
    dt = 1 / frequency
    output_path = Path(output)
    observations = []  # List to store observations

    # Ensure the output directory exists
    output_path.mkdir(parents=True, exist_ok=True)

    # Check for existing observation files and set episode_counter accordingly
    existing_files = list(output_path.glob('observations_*.pkl'))
    if existing_files:
        # Extract episode numbers from existing filenames
        existing_indices = []
        for f in existing_files:
            try:
                index = int(f.stem.split('_')[1])
                existing_indices.append(index)
            except (IndexError, ValueError):
                continue  # Skip files that don't match the expected pattern
        if existing_indices:
            episode_counter = max(existing_indices) + 1
        else:
            episode_counter = 0
    else:
        episode_counter = 0  # Start from 0 if no files exist

    print(f"Starting from episode {episode_counter}")

    # Initialize the KeyListener
    key_listener = KeyListener()

    with SharedMemoryManager() as shm_manager:
        with RealEnv(
                output_dir=output,
                robot_ip=robot_ip,
                obs_fps=frequency,
                init_joints=init_joints,
        ) as env:

            cv2.setNumThreads(1)

            # Realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            # Realsense white balance
            env.realsense.set_white_balance(white_balance=3900)
            # env.robot.start()

            time.sleep(2.0)
            t_start = time.monotonic()
            iter_idx = 0

            while not key_listener.stop_program:

                # Calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # Pump obs
                obs = env.get_obs()

                # If recording, append obs to observations
                with key_listener.lock:
                    if key_listener.is_recording:
                        observations.append(obs)

                    if key_listener.init_robot_flag:
                        # env.init_robot()
                        key_listener.init_robot_flag = False  # Reset the flag

                    if key_listener.save_data:
                        # Save observations to file
                        pkl_path = output_path / \
                            f'observations_{episode_counter}.pkl'

                        # Ensure we do not overwrite existing files
                        while pkl_path.exists():
                            print(
                                f"File {pkl_path} already exists. Incrementing episode_counter to avoid overwrite.")
                            episode_counter += 1
                            pkl_path = output_path / \
                                f'observations_{episode_counter}.pkl'

                        with open(pkl_path, 'wb') as f:
                            pickle.dump(observations, f)
                        print(f"Saved observations to {pkl_path}")
                        observations = []  # Clear observations after saving
                        episode_counter += 1  # Increment for the next episode
                        key_listener.save_data = False  # Reset the flag

                iter_idx += 1

            # Clean up
            key_listener.stop()


if __name__ == '__main__':
    main()
