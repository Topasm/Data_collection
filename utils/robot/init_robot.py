"""
Connect to the panda robot test. 
"""

import os
import panda_py
from panda_py import libfranka
import logging

# TODO change the hostname, username, and password to the correct values
hostname = "172.16.0.2"


logging.basicConfig(level=logging.INFO)

# Use the desk client to connect to the web-application
# running on the control unit to unlock brakes
# and activate FCI for robot torque control

# desk = panda_py.Desk(hostname, username, password)
# desk.unlock()
# desk.activate_fci()

# Connect to the robot using the Panda class.
panda = panda_py.Panda(hostname)
gripper = libfranka.Gripper(hostname)

# panda.move_to_start()

pose = panda.q
print(f"Current joint position: {pose}")


joint_pose = [-0.01588696, -0.25534376, 0.18628714, -
              2.28398158, 0.0769999, 2.02505396, 0.07858208]

panda.move_to_joint_position(joint_pose, speed_factor=0.1)

# gripper.move(0.0, 0.1)
