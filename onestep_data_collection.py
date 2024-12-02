import numpy as np
import time
import cv2
from multiprocessing.managers import SharedMemoryManager
from utils.robot.real_robot import RealEnv
from utils.precise_sleep import precise_wait
import os
from pathlib import Path
import click
from pynput import keyboard
import threading
# Import for quaternion to rotation vector conversion
from scipy.spatial.transform import Rotation as R

# Initialize the KeyListener class


class KeyListener:
    def __init__(self):
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
                elif key.char == 's':
                    self.save_data = True
                    print("Stopped and will save data.")
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
@click.option('--output', '-o', default="./data/onestep", required=True, help="Directory to save demonstration dataset.")
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
    existing_files = list(output_path.glob('episode_*'))
    if existing_files:
        existing_indices = []
        for f in existing_files:
            try:
                index = int(f.name.split('_')[1])
                existing_indices.append(index)
            except (IndexError, ValueError):
                continue  # Skip files that don't match the expected pattern
        if existing_indices:
            episode_counter = max(existing_indices) + 1
        else:
            episode_counter = 0
    else:
        episode_counter = 0  # Start from 0 if no folders exist

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

                with key_listener.lock:
                    if key_listener.init_robot_flag:
                        env.init_robot()
                        key_listener.init_robot_flag = False  # Reset the flag

                    if key_listener.save_data:
                        # Find the next available index
                        existing_images = list(output_path.glob('*.png'))
                        existing_states = list(output_path.glob('*.txt'))
                        existing_files = existing_images + existing_states
                        if existing_files:
                            indices = []
                            for f in existing_files:
                                try:
                                    # Get first 3 chars as number
                                    index = int(f.stem[:3])
                                    indices.append(index)
                                except ValueError:
                                    continue
                            next_index = max(indices) + 1 if indices else 1
                        else:
                            next_index = 1

                        # Format index as 3 digits
                        index_str = f"{next_index:03d}"

                        # Save image
                        for cam_key, image in obs.items():
                            if 'color_img_0' in cam_key:
                                try:
                                    if image.ndim == 4:
                                        image = np.squeeze(image, axis=0)

                                    if image.ndim == 3 and image.shape[2] == 3:
                                        image_filename = output_path / \
                                            f"{index_str}.png"
                                        cv2.imwrite(str(image_filename), image)
                                except Exception as e:
                                    print(f"Error saving image: {e}")

                        # Save state
                        current_state = obs.get('EEF_state', None)
                        if current_state is not None:
                            translation = current_state[:3]
                            quaternion = current_state[3:7]
                            gripper_state = current_state[7]
                            rotation_vector = R.from_quat(
                                quaternion).as_rotvec()
                            state_combined = np.concatenate(
                                (translation, rotation_vector, [gripper_state]))

                            state_filename = output_path / f"{index_str}.txt"
                            with open(state_filename, 'w') as f:
                                np.savetxt(f, [state_combined], fmt='%.6f')

                            print(
                                f"Saved image and state with index {index_str}")
                            key_listener.save_data = False

                iter_idx += 1

    # Clean up
    key_listener.stop()


if __name__ == '__main__':
    main()
