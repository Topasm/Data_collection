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

from utils.inputs.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
import scipy.spatial.transform as st
import click


@click.command()
@click.option('--output', '-o', default="./data", required=True, help="Directory to save demonstration dataset.")
@click.option('--robot_ip', '-ri', default="172.16.0.2", required=True, help="Franka's IP address e.g. 172.16.0.2")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SpaceMouse command to executing on Robot in Sec.")
def main(output, robot_ip, init_joints, frequency, command_latency):
    dt = 1 / frequency
    with SharedMemoryManager() as shm_manager:
        with KeystrokeCounter() as key_counter, \
                Spacemouse(shm_manager=shm_manager) as sm, \
                RealEnv(
                    output_dir=output,
                    robot_ip=robot_ip,
                    # recording resolution
                    obs_image_resolution=(640, 480),
                    frequency=frequency,
                    init_joints=init_joints,
                    enable_multi_cam_vis=True,
                    record_raw_video=True,
                    # number of threads per camera view for video recording (H.264)
                    thread_per_video=3,
                    # video recording quality, lower is better (but slower).
                    video_crf=21,
                    shm_manager=shm_manager
        ) as env:

            cv2.setNumThreads(1)

            # realsense exposure
            env.realsense.set_exposure(exposure=120, gain=0)
            # realsense white balance
            env.realsense.set_white_balance(white_balance=3900)
            # env.robot.start()
            state = env.get_robot_state()
            target_pose = state['ActualTCPPose']
            print("ActualTCPPose", target_pose)

            time.sleep(2.0)
            t_start = time.monotonic()
            iter_idx = 0
            stop = False
            is_recording = False
            while not stop:

                state = env.get_robot_state()

                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                t_command_target = t_cycle_end + dt

                # pump obs
                obs = env.get_obs()

                # handle key presses
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

                # # visualize
                # vis_img = obs[f'camera_{vis_camera_idx}'][-1,
                #                                           :, :, ::-1].copy()
                # episode_id = env.replay_buffer.n_episodes
                # text = f'Episode: {episode_id}, Stage: {stage}'
                # if is_recording:
                #     text += ', Recording!'
                # cv2.putText(
                #     vis_img,
                #     text,
                #     (10, 30),
                #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                #     fontScale=1,
                #     thickness=2,
                #     color=(255, 255, 255)
                # )

                # cv2.imshow('default', vis_img)
                # cv2.pollKey()

                precise_wait(t_sample)
                # get teleop command
                sm_state = sm.get_motion_state_transformed()
                # print(sm_state)
                dpos = sm_state[:3] * (env.max_pos_speed)
                drot_xyz = sm_state[3:] * (env.max_rot_speed)

                drot_xyz[:] = 0
                if sm.is_button_pressed(0):
                    env.robot.panda.grip()

                elif sm.is_button_pressed(1):
                    env.robot.panda.release()

                drot = st.Rotation.from_euler('xyz', drot_xyz)

                target_pose[:3] += dpos
                target_pose[3:] = (drot * st.Rotation.from_rotvec(
                    target_pose[3:])).as_rotvec()
                # sprint("target_pose", target_pose)

                # execute teleop command
                env.step(
                    actions=[target_pose],
                    timestamps=[t_command_target -
                                time.monotonic() + time.time()],
                    stages=[stage])
                precise_wait(t_cycle_end)
                iter_idx += 1


# %%
if __name__ == '__main__':
    main()
