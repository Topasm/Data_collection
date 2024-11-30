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
from utils.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from utils.robot.real_robot import RealRobot
from utils.precise_sleep import precise_wait

from utils.inputs.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)

import click


@click.command()
@click.option('--output', '-o', default="./data", required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', default="172.16.0.2", required=True, help="Franka's IP address e.g. 172.16.0.2")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SpaceMouse command to executing on Robot in Sec.")
def main(output, robot_ip, frequency, command_latency, init_joints):
    dt = 1 / frequency

    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
                Spacemouse(shm_manager=shm_manager) as sm, \
                RealRobot(robot_ip=robot_ip, output_dir=output, shm_manager=shm_manager
                          ) as env:

            cv2.setNumThreads(1)

            # Initialization is handled within RealRobot's __init__
            time.sleep(1)

            stop = False
            iter_idx = 0
            t_start = time.monotonic()
            is_recording = False  # Initialize recording flag

            # Initialize target_pose with current robot TCP pose
            target_tcp_pose = env.get_tcp_pose()  # Returns position + quaternion
            # Convert quaternion to rotation vector (axis-angle)
            rotvec = R.from_quat(target_tcp_pose[3:]).as_rotvec()
            target_pose = np.concatenate([target_tcp_pose[:3], rotvec])

            while not stop:
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # Get observations
                obs = env.get_obs()

                # Handle key presses
                press_events = key_counter.get_press_events()
                for key_stroke in press_events:
                    if key_stroke == KeyCode(char='q'):
                        # Exit program
                        stop = True
                    elif key_stroke == KeyCode(char='c'):
                        # Start recording
                        print('Start recording!')
                        env.start_episode(
                            start_time=t_start + (iter_idx + 2) * dt - time.monotonic() + time.time())
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
                        if click.confirm('Are you sure to drop the last episode?'):
                            env.drop_episode()
                            key_counter.clear()
                            is_recording = False
                stage = key_counter[Key.space]

                precise_wait(t_sample)

                # Get teleoperation command from the SpaceMouse
                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3] * env.max_pos_speed
                drot_xyz = sm_state[3:] * env.max_rot_speed

                # Determine movement mode based on button presses
                if not sm.is_button_pressed(0):
                    # Translation mode
                    drot_xyz[:] = 0
                else:
                    dpos[:] = 0
                if not sm.is_button_pressed(1):
                    # 2D translation mode
                    dpos[2] = 0

                # Compute rotation increment
                drot_vec = drot_xyz  # Assuming small angle approximation

                # Update target_pose
                target_pose[:3] += dpos
                drot = R.from_rotvec(drot_vec)
                current_rot = R.from_rotvec(target_pose[3:])
                new_rot = drot * current_rot
                target_pose[3:] = new_rot.as_rotvec()

                # Prepare action (position, rotation vector, gripper action)
                gripper_action = 0.0  # Adjust gripper action if needed
                action = np.concatenate([target_pose, [gripper_action]])

                # Execute action
                env.exec_actions(
                    actions=[action],
                    timestamps=[t_command_target -
                                time.monotonic() + time.time()],
                    stages=[stage])
                precise_wait(t_cycle_end)
                iter_idx += 1


if __name__ == '__main__':
    main()
