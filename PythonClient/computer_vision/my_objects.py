# In settings.json first activate computer vision mode: 
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

import setup_path 
import airsim
import time
import numpy as np

import pprint
import argparse
client = airsim.VehicleClient()
client.confirmConnection()

parser = argparse.ArgumentParser()
parser.add_argument("object", type=str, help="mesh name")
args = parser.parse_args()

# search object by name: 
prev_pose = None
while True:
    pose1 = client.simGetObjectPose(args.object)
    print("%s - Position: %s, Orientation: %s" % (args.object, pprint.pformat(pose1.position),
        pprint.pformat(pose1.orientation)))
    print(type(pose1.position.x_val))
    if prev_pose is not None:
        print(prev_pose == pose1)
    prev_pose = pose1
    time.sleep(1)