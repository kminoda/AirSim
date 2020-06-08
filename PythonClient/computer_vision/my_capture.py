# In settings.json first activate computer vision mode: 
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

import sys
sys.path.append('/home/minoda/git/AirSim/PythonClient')
sys.path.append('/home/minoda/git/AirSim/PythonClient/computer_vision')
import setup_path 
import airsim
import rosbag
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

import pprint
import tempfile
import os
import time
import argparse
import shutil
import numpy as np
import tqdm

class CapturePoint:
    def __init__(self, idx, pos, quat, tsecs, tnsecs):
        self.idx = idx
        self.pos = pos
        self.quat = quat
        self.tsecs = tsecs
        self.tnsecs = tnsecs


def get_odom_list_from_bag(bag, topic_gt):
    odom_pos_list = []
    odom_quat_list = []
    odom_time_list = []
    for topic, msg, t in bag:
        if topic == topic_gt:
            pos = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ]
            odom_pos_list.append(pos)

            quat = [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ]
            odom_quat_list.append(quat)

            odom_time_list.append(t.secs + t.nsecs * pow(10, -9))
    return {'pos': odom_pos_list, 'quat': odom_quat_list, 'time': odom_time_list}


def get_capture_points(bag, odom_list, sync_topic):
    capture_points = []
    cur_idx = 0
    for topic, _, t in bag:
        if topic == sync_topic:
            time = t.secs + t.nsecs*pow(10, -9)
            nearest_idx = (np.abs(odom_list['time'] - time)).argmin()
            pos = odom_list['pos'][nearest_idx]
            quat = odom_list['quat'][nearest_idx]
            point = CapturePoint(cur_idx, pos, quat, t.secs, t.nsecs)
            capture_points.append(point)
            cur_idx += 1
    return capture_points

def copy_bag(inbag, outbag):
    for topic, msg, t in inbag.read_messages():
        outbag.write(topic, msg, t)
    return outbag

def record_segment_imgs(capture_points, tmp_dir):
    for capture_point in tqdm.tqdm(capture_points): # do few times
        if os.path.exists(os.path.join(tmp_dir, '{0:05d}.png'.format(capture_point.idx))):
            continue
        pos = capture_point.pos
        quat = capture_point.quat
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pos[0], pos[1], pos[2]),
                                 airsim.Quaternionr(quat[0], quat[1], quat[2], quat[3])),
                                 True)
        time.sleep(0.01)

        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene),
            airsim.ImageRequest("1", airsim.ImageType.Scene),
            airsim.ImageRequest("2", airsim.ImageType.Scene)])

        for i, response in enumerate(responses):
            if response.pixels_as_float:
                airsim.write_pfm(os.path.normpath(os.path.join(tmp_dir, '{0:05d}.pfm'.format(capture_point.idx))), airsim.get_pfm_array(response))
            else:
                airsim.write_file(os.path.normpath(os.path.join(tmp_dir, '{0:05d}.png'.format(capture_point.idx))), response.image_data_uint8)
    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)

def write_segment_imgs(outbag, tmp_dir):
    bridge = CvBridge()
    for capture_point in capture_points:
        img_pth = os.path.join(args.img_path, '{0:05d}.png'.format(capture_point.idx))
        if not (img_pth.split('.')[-1] == 'jpg' or img_pth.split('.')[-1] == 'png'):
            print(img_pth, "is not an image.")
            continue
        img = cv2.imread(img_pth)
        # print(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_msg = bridge.cv2_to_imgmsg(img.astype(np.uint8), encoding='rgb8')
        img_msg.header.stamp.secs = capture_point.tsecs
        img_msg.header.stamp.nsecs = capture_point.tnsecs
        img_pth = os.path.join(args.img_path, '{0:05d}.png'.format(capture_point.idx))
        outbag.write("/airsim/drone/segmentation/data", img_msg)
        if capture_point.idx==0:
            print("Image size: ({0}, {1}, {2})".format(img.shape[0], img.shape[1], img.shape[2]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rosbag_path", type=str, help="path to rosbag file")
    parser.add_argument("--rosbag_out_path", type=str, help="path to output rosbag file")
    parser.add_argument("--topic_gt", type=str, help="topic name for ground truth")
    parser.add_argument("--freq", type=float, default=10, help="frequency for capturing image")
    parser.add_argument("--img_path", type=str, default='', help="Where to save images as a temp")
    parser.add_argument("--topic_sync", type=str, default='', help="topic name for image, which will be used for synchronization")
    args = parser.parse_args()

    client = airsim.VehicleClient()
    if args.img_path != '':
        print("WARNING!!!! This will erase all the images in " + args.img_path)
        airsim.wait_key('Press any key to confirm erasing these.')
        if os.path.exists(args.img_path):
            shutil.rmtree(args.img_path)
        os.makedirs(args.img_path)

    print("Loading rosbag...")
    inbag = rosbag.Bag(args.rosbag_path)
    
    # Start modifying images
    if args.rosbag_path == args.rosbag_out_path:
        print("Are you sure you don't want to take a copy of an original rosbag?")
        # airsim.wait_key('Press any key to confirm this')
    
    outbag = rosbag.Bag(args.rosbag_out_path, 'w')
    outbag = copy_bag(inbag, outbag)

    odom_list = get_odom_list_from_bag(inbag, args.topic_gt)
    capture_points = get_capture_points(inbag, odom_list, sync_topic = args.topic_image)

    if len(args.img_path)==0:
        tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
    else:
        tmp_dir = args.img_path
    print ("Saving images to %s" % tmp_dir)

    record_segment_imgs(capture_points, tmp_dir)
    write_segment_imgs(outbag, tmp_dir)
    outbag.close()

    # capture_points = [] 
    # t_prev = -1
    # cur_idx = 0
    # for topic, msg, t in inbag.read_messages():
    #     msg.header.stamp = t
    #     outbag.write(topic, msg)
    #     # print(t.secs, t.nsecs)
    #     # time = msg
    #     msg_time = t
    #     if topic != args.topic_name:
    #         continue
    #     if t_prev>=0 and t_prev + 1.0/args.freq > msg_time.secs + msg_time.nsecs*pow(10,-9):# Not at nHz, continue
    #         continue
    #     if t_prev < 0:
    #         t_prev = msg_time.secs + msg_time.nsecs * pow(10, -9)
    #     # print(t_prev)
    #     t_prev = msg_time.secs + msg_time.nsecs*pow(10, -9)
    #     pos = [msg.pose.pose.position.x,
    #            msg.pose.pose.position.y,
    #            msg.pose.pose.position.z] 
    #     quat = [msg.pose.pose.orientation.x,
    #             msg.pose.pose.orientation.y,
    #             msg.pose.pose.orientation.z,
    #             msg.pose.pose.orientation.w] 
    #     point = CapturePoint(cur_idx, pos, quat, msg_time.secs, msg_time.nsecs)
    #     cur_idx += 1
    #     capture_points.append(point)
    # print("Finished loading rosbag. There are {0} capture points".format(len(capture_points)))


    # airsim.wait_key('Press any key to get camera parameters')
    # for camera_id in range(2):
    #     camera_info = client.simGetCameraInfo(str(camera_id))
        # print("CameraInfo %d: %s" % (camera_id, pp.pprint(camera_info)))

    # airsim.wait_key('Press any key to get images')
    # if len(args.img_path)==0:
    #     tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
    # else:
    #     tmp_dir = args.img_path
    # print ("Saving images to %s" % tmp_dir)

    # for capture_point in tqdm.tqdm(capture_points): # do few times
    #     if os.path.exists(os.path.join(tmp_dir, '{0:05d}.png'.format(capture_point.idx))):
    #         continue
    #     pos = capture_point.pos
    #     quat = capture_point.quat
    #     client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pos[0], pos[1], pos[2]),
    #                              airsim.Quaternionr(quat[0], quat[1], quat[2], quat[3])),
    #                              True)
    #     time.sleep(0.01)

    #     responses = client.simGetImages([
    #         airsim.ImageRequest("0", airsim.ImageType.Scene),
    #         airsim.ImageRequest("1", airsim.ImageType.Scene),
    #         airsim.ImageRequest("2", airsim.ImageType.Scene)])

    #     for i, response in enumerate(responses):
    #         if response.pixels_as_float:
    #             # print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_float), pprint.pformat(response.camera_position)))
    #             airsim.write_pfm(os.path.normpath(os.path.join(tmp_dir, '{0:05d}.pfm'.format(capture_point.idx))), airsim.get_pfm_array(response))
    #         else:
    #             # print("Type %d, size %d, pos %s" % (response.image_type, len(response.image_data_uint8), pprint.pformat(response.camera_position)))
    #             airsim.write_file(os.path.normpath(os.path.join(tmp_dir, '{0:05d}.png'.format(capture_point.idx))), response.image_data_uint8)
            
    #     pose = client.simGetVehiclePose()
    #     # pp.pprint(pose)

    #     # time.sleep(0.01)

    # # currently reset() doesn't work in CV mode. Below is the workaround
    # client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)

    # bridge = CvBridge()
    # for capture_point in capture_points:
    #     img_pth = os.path.join(args.img_path, '{0:05d}.png'.format(capture_point.idx))
    #     if not (img_pth.split('.')[-1] == 'jpg' or img_pth.split('.')[-1] == 'png'):
    #         print(img_pth, "is not an image.")
    #         continue
    #     img = cv2.imread(img_pth)
    #     # print(img)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img_msg = bridge.cv2_to_imgmsg(img.astype(np.uint8), encoding='rgb8')
    #     img_msg.header.stamp.secs = capture_point.tsecs
    #     img_msg.header.stamp.nsecs = capture_point.tnsecs
    #     img_pth = os.path.join(args.img_path, '{0:05d}.png'.format(capture_point.idx))
    #     outbag.write("/airsim/image_raw", img_msg)
    #     if capture_point.idx==0:
    #         print("Image size: ({0}, {1}, {2})".format(img.shape[0], img.shape[1], img.shape[2]))
    # outbag.close()