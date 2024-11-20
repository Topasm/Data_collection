import numpy as np
import time
import threading
import queue
import panda_py.controllers
from scipy.spatial.transform import Rotation as R
import panda_py
from utils.spacemouse import Spacemouse
import cv2

# Constants
MOVE_INCREMENT = 0.0002
SPEED = 0.05  # [m/s]
FORCE = 20.0  # [N]

# Initialize robot and gripper
hostname = '172.16.0.2'
robot = panda_py.Panda(hostname)
gripper = panda_py.libfranka.Gripper(hostname)
robot.recover()
robot.move_to_start()
gripper.homing()

# Global variables
running = True
recording = False
current_translation = robot.get_position()
current_rotation = robot.get_orientation()

# Create a thread-safe queue for data sharing
data_queue = queue.Queue(maxsize=1000)


def data_recording():
    global running, recording
    # Initialize video capture
    video_capture = cv2.VideoCapture(4)  # Adjust camera index if needed

    # Set camera properties to limit FPS and resolution
    video_capture.set(cv2.CAP_PROP_FPS, 30)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Small delay to ensure camera is initialized
    time.sleep(0.5)
    video_writer = None

    # Desired frame interval in seconds
    frame_interval = 1 / 30  # For 30 FPS
    last_frame_time = time.time()

    while running:
        current_time = time.time()
        elapsed = current_time - last_frame_time
        if elapsed >= frame_interval:
            last_frame_time = current_time
            ret, frame = video_capture.read()
            if ret:
                # Display the frame
                cv2.imshow('Recording', frame)
                # Check for keyboard input
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    if not recording:
                        recording = True
                        video_writer = cv2.VideoWriter(
                            'video_output.avi',
                            cv2.VideoWriter_fourcc(*'XVID'),
                            30,
                            (frame.shape[1], frame.shape[0])
                        )
                        print("Recording started.")
                elif key == ord('s'):
                    if recording:
                        recording = False
                        if video_writer is not None:
                            video_writer.release()
                        print("Recording stopped.")
                elif key == ord('q'):
                    running = False
                    print("Exiting program.")

                if recording:
                    # Write video frame
                    video_writer.write(frame)
                    # Retrieve data from queue and save
                    while not data_queue.empty():
                        try:
                            timestamp, robot_q, ee_pose = data_queue.get(
                                timeout=0.01)
                            # Save data as needed
                        except queue.Empty:
                            pass
            else:
                print("Failed to read frame from camera.")
        else:
            # Sleep for the remaining time to match the desired frame rate
            time.sleep(frame_interval - elapsed)
    # Clean up
    if recording and video_writer is not None:
        video_writer.release()
    video_capture.release()
    cv2.destroyAllWindows()


def main():
    global running, recording, current_translation, current_rotation
    # Start data recording thread
    data_thread = threading.Thread(target=data_recording)
    data_thread.start()
    # Delay to ensure data recording thread initializes
    time.sleep(1.0)

    with Spacemouse(deadzone=0.3) as sm, robot.create_context(frequency=1000) as ctx:
        controller = panda_py.controllers.CartesianImpedance()
        robot.start_controller(controller)
        time.sleep(1)

        while ctx.ok() and running:
            # Start time for loop timing
            start_time = time.perf_counter()

            # Get Spacemouse input
            sm_state = sm.get_motion_state_transformed()
            dpos = sm_state[:3] * MOVE_INCREMENT
            drot_xyz = sm_state[3:] * MOVE_INCREMENT * 3

            # Update current pose
            current_translation += dpos
            delta_rotation = R.from_euler('xyz', drot_xyz)
            current_rotation = (
                delta_rotation * R.from_quat(current_rotation)
            ).as_quat()

            # Handle gripper state changes
            if sm.is_button_pressed(0):
                gripper.grasp(0.01, speed=SPEED, force=FORCE,
                              epsilon_inner=0.005, epsilon_outer=0.005)
            elif sm.is_button_pressed(1):
                gripper.move(0.08, speed=SPEED)

            # Collect data and send to recording thread
            if recording:
                timestamp = time.time()
                robot_q = robot.q.copy()
                ee_pose = robot.get_pose()
                try:
                    data_queue.put_nowait((timestamp, robot_q, ee_pose))
                except queue.Full:
                    print("Data queue is full. Data is being dropped.")

            # Set robot control
            controller.set_control(current_translation, current_rotation)

            # Loop timing to maintain control frequency
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            sleep_time = max(0, (1 / 1000) - elapsed)
            time.sleep(sleep_time)

    data_thread.join()


if __name__ == "__main__":
    main()
