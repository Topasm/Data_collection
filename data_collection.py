import numpy as np
import time
import panda_py.controllers
from scipy.spatial.transform import Rotation as R
import panda_py
from utils.spacemouse import Spacemouse
from spatialmath import *
import roboticstoolbox as rtb
import threading
import cv2
from utils.video_recorder import VideoRecorder

# Constants
MOVE_INCREMENT = 0.0002
SPEED = 0.05  # [m/s]
FORCE = 20.0  # [N]


# Initialize robot and gripper
hostname = '172.16.0.2'
robot = panda_py.Panda(hostname)
gripper = panda_py.libfranka.Gripper(hostname)


robot.recover()

# Get initial pose

robot.move_to_start()
gripper.homing()
current_translation = robot.get_position()
current_rotation = robot.get_orientation()
defaultq = robot.q

print("Initial pose:", defaultq)


panda_rtb = rtb.models.Panda()


def main():
    # Initialize video capture and recorder
    video_capture = cv2.VideoCapture(0)  # Adjust the camera index if needed
    video_recorder = VideoRecorder.create_h264(
        fps=30,
        input_pix_fmt='bgr24',
        crf=18,
        thread_type='FRAME',
        thread_count=1)
    recording = False
    joint_data = []

    print("Use Spacemouse to control the robot in Cartesian space.")
    print("Press button 0 to toggle gripper state (open/close).")
    print("Press button 1 to grasp an object and button 2 to release.")
    print("Press button 3 to start/stop recording.")

    with Spacemouse(deadzone=0.3) as sm, robot.create_context(frequency=1000) as ctx:
        running = True
        controller = panda_py.controllers.CartesianImpedance()
        robot.start_controller(controller)
        time.sleep(1)

        while ctx.ok() and running:
            start_time = time.perf_counter()

            # Get Spacemouse input
            sm_state = sm.get_motion_state_transformed()
            dpos = sm_state[:3] * MOVE_INCREMENT
            drot_xyz = sm_state[3:] * MOVE_INCREMENT * 3
            drot_xyz = np.array([drot_xyz[0], drot_xyz[1], drot_xyz[2]])

            # Update current pose
            current_translation += np.array([dpos[0], dpos[1], dpos[2]])
            if drot_xyz is not None:
                delta_rotation = R.from_euler('xyz', drot_xyz)
                current_rotation = (
                    delta_rotation * R.from_quat(current_rotation)).as_quat()

            # Handle gripper state changes
            if sm.is_button_pressed(0):
                success = gripper.grasp(0.01, speed=SPEED, force=FORCE,
                                        epsilon_inner=0.005, epsilon_outer=0.005)
                if success:
                    print("Grasp successful")
                else:
                    print("Grasp failed")
            elif sm.is_button_pressed(1):
                success = gripper.move(0.08, speed=SPEED)
                if success:
                    print("Release successful")
                else:
                    print("Release failed")

            # Handle recording state toggle
            if sm.is_button_pressed(3):
                if recording:
                    recording = False
                    video_recorder.stop()
                    np.save('joint_data.npy', np.array(joint_data))
                    joint_data = []
                    print("Recording stopped.")
                else:
                    recording = True
                    video_recorder.start('video_output.mp4')
                    joint_data = []
                    print("Recording started.")

            if recording:
                # Record joint data
                timestamp = time.time()
                robot_q = robot.q.copy()
                joint_data.append((timestamp, robot_q))

                # Capture and record video frame
                ret, frame = video_capture.read()
                if ret:
                    video_recorder.write_frame(frame)

            # Set robot control
            controller.set_control(current_translation, current_rotation)
            end_time = time.perf_counter()
            loop_duration = end_time - start_time

    # Release video capture resource
    video_capture.release()


if __name__ == "__main__":
    main()
