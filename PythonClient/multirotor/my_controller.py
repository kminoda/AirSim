
import setup_path 
import airsim

import pandas as pd
import numpy as np
import os
import tempfile
import pprint
import cv2
import time
import random
import argparse

def move_pos(client, x, y, z, vel):
    print("move: ({0}, {1}, {2}, {3})".format(x, y, z, vel))
    client.moveToPositionAsync(x, y, z, vel).join()

def move_yaw(client, yaw):
    print("yaw: ({0})".format(yaw))
    client.rotateToYawAsync(yaw, 10, 10).join()

parser = argparse.ArgumentParser()
parser.add_argument("csv", type=str, help="launch file")
parser.add_argument("--launch_file", type=str, default="/home/minoda/git/lis_vio/airsim/airsim_record.launch", help="launch file")
parser.add_argument("--bag_path", type=str, default="/home/minoda/AirSim_results/dataset_airsim", help="rosbag output path")
parser.add_argument("--disable_ros", action="store_true")
args = parser.parse_args()

if not args.disable_ros:
    import roslaunch 
    import rospy
    rospy.init_node('en_Mapping', anonymous=True)
    uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
    roslaunch.configure_logging(uuid)

    cli_args = [args.launch_file, 'rosbag_path:='+args.bag_path]
    roslaunch_args = cli_args[1:]
    roslaunch_file = [(roslaunch.rlutil.resolve_launch_arguments(cli_args)[0], roslaunch_args)]
    launch = roslaunch.parent.ROSLaunchParent(uuid, roslaunch_file)
    launch.start()

# load waypoints
df = pd.read_csv(args.csv)

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()

try:
    for index, row in df.iterrows():
        if row['pos_or_yaw'] == 'pos':
            move_pos(client, row.x, row.y, row.z, row.vel)
        elif row['pos_or_yaw'] == 'yaw':
            move_yaw(client, row.yaw)
        else:
            print('invalid waypoint in row {0}'.format(index))
except KeyboardInterrupt:
    client.armDisarm(False)
    client.reset()
    client.enableApiControl(False)

client.hoverAsync().join()

time.sleep(3)
if not args.disable_ros:
    launch.shutdown()
print("finished")

client.armDisarm(False)
client.reset()

# that's enough fun for now. let's quit cleanly
client.enableApiControl(False)
