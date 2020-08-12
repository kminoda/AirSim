# In settings.json first activate computer vision mode: 
# https://github.com/Microsoft/AirSim/blob/master/docs/image_apis.md#computer-vision-mode

import sys
sys.path.append('/home/minoda/git/AirSim/PythonClient')
sys.path.append('/home/minoda/git/AirSim/PythonClient/computer_vision')
import setup_path 
import airsim
import rosbag
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import cv2
from cv_bridge import CvBridge

import pprint
import tempfile
import os
import time
import argparse
import shutil
import numpy as np
import pandas as pd
import tqdm
import copy

from scipy.spatial.transform import Rotation as R

class CapturePoint:
    def __init__(self, idx, pos, quat, tsecs, tnsecs):
        self.idx = idx
        self.pos = pos
        self.quat = quat
        self.tsecs = tsecs
        self.tnsecs = tnsecs
        self.debug_actual_pos = None
        self.debug_actual_quat = None

def get_capture_point_from_gazebo_model_states(idx, odom_msg, odom_timestamp):
    pos = [
        odom_msg.pose[2].position.y,
        odom_msg.pose[2].position.x,
        -odom_msg.pose[2].position.z
    ]
    quat = [
        odom_msg.pose[2].orientation.y,
        odom_msg.pose[2].orientation.x,
        -odom_msg.pose[2].orientation.z,
        odom_msg.pose[2].orientation.w
    ]
    capture_point = CapturePoint(idx, pos, quat, odom_timestamp.secs, odom_timestamp.nsecs)
    return capture_point

def get_capture_point_from_navmsgs_odom(idx, odom_msg, odom_timestamp):
    pos = [
        odom_msg.pose.pose.position.y,
        odom_msg.pose.pose.position.x,
        -odom_msg.pose.pose.position.z
    ]
    quat = [
        odom_msg.pose.pose.orientation.y,
        odom_msg.pose.pose.orientation.x,
        -odom_msg.pose.pose.orientation.z,
        odom_msg.pose.pose.orientation.w
    ]
    capture_point = CapturePoint(idx, pos, quat, odom_timestamp.secs, odom_timestamp.nsecs)
    return capture_point

def get_odom_list_from_bag(bag, topic_gt):
    odom_pos_list = []
    odom_quat_list = []
    odom_time_list = []
    odom_stamp_list = []
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

            odom_stamp_list.append(t)
            odom_time_list.append(t.secs + t.nsecs * pow(10, -9))
    return {'pos': odom_pos_list, 'quat': odom_quat_list, 'time': odom_time_list, 'stamp': odom_stamp_list}

def get_capture_points_gazebo(bag, odom_topic='/gazebo/model_states', sync_topic='/mavros/imu/data_raw', camera_freq=20, sync_topic_freq=100, method='every'):
    """
    method(string): method for sampling capturing points. 
        'every': Sample IMU for every n msgs, and then capture odometry msg which has the closest timestamp. This requires the existence of odom_msg for every imu_msg.
    """
    odom_msg_list = []
    odom_time_list = []
    odom_stamp_list = []
    capture_time_list = []
    sync_topic_num = 0
    for topic, msg, t in bag:
        if topic==odom_topic:
            odom_msg_list.append(msg)
            odom_time_list.append(t.to_time())
            odom_stamp_list.append(copy.deepcopy(t))

    for topic, msg, t in bag:
        if topic==sync_topic:
            if odom_time_list[0] > t.to_time():
                continue
            if sync_topic_num % (int(sync_topic_freq/camera_freq)) == 0:
                capture_time_list.append(t.to_time())
            sync_topic_num += 1
    assert len(odom_msg_list)==len(odom_time_list) and len(odom_msg_list)==len(odom_stamp_list), 'length of odom_(msg/time/stamp)_list is not equal.'

    # start sampling odometry
    capture_points = []
    curr_odom_idx = 0
    for idx, capture_time in enumerate(capture_time_list):
        # take an odometry msg which has the timestamp closest to capture_time
        if capture_time < min(odom_time_list):
            continue

        while abs(capture_time - odom_time_list[curr_odom_idx]) >= 5*10**(-5):
            curr_odom_idx += 1
            if curr_odom_idx >= len(odom_time_list): 
                break
        if curr_odom_idx >= len(odom_time_list): 
            break
        if odom_topic=='/gazebo/gazebo_states':
            capture_point = get_capture_point_from_gazebo_model_states(idx, odom_msg_list[curr_odom_idx], odom_stamp_list[curr_odom_idx])
        elif odom_topic=='/odometry':
            capture_point = get_capture_point_from_navmsgs_odom(idx, odom_msg_list[curr_odom_idx], odom_stamp_list[curr_odom_idx])
        capture_points.append(capture_point) 
    return capture_points

def get_capture_points(bag, odom_list, sync_topic=None, freq=None):
    capture_points = []
    cur_idx = 0
    if (sync_topic is not None) and (freq is None):
        for topic, _, t in bag:
            if topic == sync_topic:
                time = t.secs + t.nsecs*pow(10, -9)
                nearest_idx = (np.abs(odom_list['time'] - time)).argmin()
                pos = odom_list['pos'][nearest_idx]
                quat = odom_list['quat'][nearest_idx]
                point = CapturePoint(cur_idx, pos, quat, t.secs, t.nsecs)
                capture_points.append(point)
                cur_idx += 1
        
    elif (freq is not None) and (sync_topic is None):
        idx_list = undersample(odom_list['time'], freq)
        # pos_list = np.array(odom_list['pos'])[idx_list]
        # quat_list = np.array(odom_list['quat'])[idx_list]
        # stamp_list = np.array(odom_list['stamp'])[idx_list]
        for i, idx in enumerate(idx_list):
            point = CapturePoint(
                i, 
                odom_list['pos'][idx], 
                odom_list['quat'][idx], 
                odom_list['stamp'][idx].secs, 
                odom_list['stamp'][idx].nsecs
            )
            capture_points.append(point)
    else:
        print("WARNING: sync_topic and freq is both given in get_capture_points")

    return capture_points

def get_capture_points_sync(bag, gt_topic, sync_topic, sync_topic_freq=500, target_freq=20):
    time_list = []
    stamp_list = []
    msg_list = []
    topic_name_list = []
    for topic, msg, t in bag:
        if topic == gt_topic or topic==sync_topic:
            time_list.append(t.secs + t.nsecs*pow(10, -9))
            topic_name_list.append(topic)
            msg_list.append(msg)
            stamp_list.append(t)
    # Sort with time, just to make sure that there is no time disordered msgs.
    idx_sort = np.argsort(time_list)

    # skipping this number of imu msgs.
    skip_num = int(sync_topic_freq/target_freq)
    skipped_num = 0

    capture_points = []
    for idx in idx_sort:
        if topic_name_list[idx] == sync_topic:
            skipped_num += 1
        
        if skipped_num >= skip_num and topic_name_list[idx] == gt_topic:
            msg = msg_list[idx]
            t = stamp_list[idx]

            pos = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ]

            quat = [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ]

            cur_idx = capture_points[-1].idx + 1 if len(capture_points)>0 else 0
            point = CapturePoint(cur_idx, pos, quat, t.secs, t.nsecs)
            capture_points.append(point)
            
            skipped_num = 0
    return capture_points

def get_capture_points_by_odom(bag, gt_topic, gt_topic_freq=500, target_freq=20):
    time_list = []
    stamp_list = []
    msg_list = []
    topic_name_list = []
    for topic, msg, t in bag:
        if topic == gt_topic:
            time_list.append(t.secs + t.nsecs*pow(10, -9))
            topic_name_list.append(topic)
            msg_list.append(msg)
            stamp_list.append(t)
    # Sort with time, just to make sure that there is no time disordered msgs.
    idx_sort = np.argsort(time_list)

    # skipping this number of imu msgs.
    skip_num = int(gt_topic_freq/target_freq) 
    skipped_num = 0

    capture_points = []
    for idx in idx_sort:
        if topic_name_list[idx] == gt_topic:
            skipped_num += 1
        
        if skipped_num >= skip_num and topic_name_list[idx] == gt_topic:
            msg = msg_list[idx]
            t = stamp_list[idx]

            pos = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z
            ]

            quat = [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w
            ]

            cur_idx = capture_points[-1].idx + 1 if len(capture_points)>0 else 0
            point = CapturePoint(cur_idx, pos, quat, t.secs, t.nsecs)
            capture_points.append(point)
            
            skipped_num = 0
    return capture_points

def undersample(t_origin, freq):
    idx_undersampled = []
    t_prev = -1
    for i, t in enumerate(t_origin):
        if t_prev < 0 or t >= t_prev + 1.0/freq:
            idx_undersampled.append(i)
            t_prev = t
        else:
            continue
    return idx_undersampled

def set_segment_id(client, df):
    for mesh2id in df.iterrows():
        success = client.simSetSegmentationObjectID(mesh2id[1][0], mesh2id[1][1])
        if success:
            print('Succeeded in seting {0} to ID:{1}'.format(mesh2id[1][0], mesh2id[1][1]))

def copy_bag(inbag, outbag):
    for topic, msg, t in inbag.read_messages():
        outbag.write(topic, msg, t)
    return outbag

def get_stereo_camera_pose(pos, quat, dist=0.05):
    """
    pos: position of monocular camera
    quat: quaternion of monocular camera
    dist: distance between two cameras
    """
    r = R.from_quat(quat)
    rotation_matrix = r.as_dcm()
    pos_stereo_base = np.array([0, dist, 0])
    pos_stereo_world = pos + np.dot(rotation_matrix, pos_stereo_base)
    return pos_stereo_world, quat

def record_imgs(client, capture_points, tmp_dir, df_object_list = None, stereo=False, debug=False):
    if df_object_list is not None:
        dict_objects_poses = {}
    
    for idx, capture_point in tqdm.tqdm(enumerate(capture_points)): # do few times
        if os.path.exists(os.path.join(tmp_dir, '{0:05d}.png'.format(capture_point.idx))):
            continue
        pos = capture_point.pos
        quat = capture_point.quat
        if debug:
            print('position:', pos)
            print('quaternion:', quat)
            print('===============')
        client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(pos[0], pos[1], pos[2]),
                                 airsim.Quaternionr(quat[0], quat[1], quat[2], quat[3])),
                                 True)
        client.simPause(True)
        responses = client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.Scene),
            airsim.ImageRequest("0", airsim.ImageType.Segmentation)
        ])
        for i, response in enumerate(responses):
            if response.pixels_as_float:
                airsim.write_pfm(os.path.normpath(os.path.join(tmp_dir, 'id{0}_{1:05d}.pfm'.format(i, capture_point.idx))), airsim.get_pfm_array(response))
            else:
                airsim.write_file(os.path.normpath(os.path.join(tmp_dir, 'id{0}_{1:05d}.png'.format(i, capture_point.idx))), response.image_data_uint8)
   
       
       # Take stereo image
        if stereo:
            stereo_pos, stereo_quat = get_stereo_camera_pose(pos, quat)
            client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(stereo_pos[0], stereo_pos[1], stereo_pos[2]),
                            airsim.Quaternionr(stereo_quat[0], stereo_quat[1], stereo_quat[2], stereo_quat[3])),
                            True)
            responses = client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene),
                airsim.ImageRequest("0", airsim.ImageType.Segmentation)
            ])

            for i, response in enumerate(responses):
                if response.pixels_as_float:
                    airsim.write_pfm(os.path.normpath(os.path.join(tmp_dir, 'id{0}_{1:05d}_stereo.pfm'.format(i, capture_point.idx))), airsim.get_pfm_array(response))
                else:
                    airsim.write_file(os.path.normpath(os.path.join(tmp_dir, 'id{0}_{1:05d}_stereo.png'.format(i, capture_point.idx))), response.image_data_uint8)

        client.simPause(False)       

        # capture objects pose
        if df_object_list is not None:
            dict_objects_poses[capture_point.idx] = []
            for mesh2id in df_object_list.iterrows():
                pose = client.simGetObjectPose(mesh2id[1][0])
                if np.isnan(pose.position.x_val):
                    continue
                tmp = {
                    'pose': pose,
                    'segID': mesh2id[1][1],
                    'object_name': mesh2id[1][0]
                }
                dict_objects_poses[capture_point.idx].append(tmp)
    if df_object_list is None:
        return False
    else:
        return dict_objects_poses


    client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(0, 0, 0)), True)

def is_same_pose(pose1, pose2):
    """ The "=" operator for AirSim poses
    """
    pos_x_same = pose1.position.x_val == pose2.position.x_val
    pos_y_same = pose1.position.y_val == pose2.position.y_val
    pos_z_same = pose1.position.z_val == pose2.position.z_val
    ori_x_same = pose1.orientation.x_val == pose2.orientation.x_val
    ori_y_same = pose1.orientation.y_val == pose2.orientation.y_val
    ori_z_same = pose1.orientation.z_val == pose2.orientation.z_val
    ori_w_same = pose1.orientation.w_val == pose2.orientation.w_val
    return pos_x_same and pos_y_same and pos_z_same and ori_x_same and ori_y_same and ori_z_same and ori_w_same 

def filter_out_static_objects(objects_poses):
    """ filter out static objects poses. Might be better to use "|x1-x2|<eps" rather than "x1=x2"
    """
    if not objects_poses:
        return objects_poses
    first_poses = objects_poses[0]
    last_poses = objects_poses[len(objects_poses)-1]

    first_id2pose_dict = {v['segID']: v['pose'] for v in first_poses}
    last_id2pose_dict = {v['segID']: v['pose'] for v in last_poses}
    ids = [v['segID'] for v in first_poses]
    static_ids = []
    for idx in ids:
        if is_same_pose(first_id2pose_dict[idx], last_id2pose_dict[idx]):
            static_ids.append(idx)
    
    new_objects_poses = {}
    for k, v in objects_poses.items():
        tmp = []
        for pose_info in v:
            if pose_info['segID'] in static_ids:
                continue
            tmp.append(pose_info)
        new_objects_poses[k] = tmp
    return new_objects_poses

def get_img_msg(img_pth, bridge):
    img = cv2.imread(img_pth)
    try:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except cv2.error as e:
        print("Got cv2.error here. Image path is {0}".format(img_pth))
    img_msg = bridge.cv2_to_imgmsg(img.astype(np.uint8), encoding='rgb8')
    return img_msg

def load_images_to_bag(outbag,
                       tmp_dir, 
                       objects_poses=False,
                       stereo=False,
                       debug=False):
    
    # if stereo:
    #     topic_images=['/airsim_node/drone/cam0/camera/image_raw', '/airsim_node/drone/cam0/segmentation/image_raw']
    # else:
    #     topic_images=['/airsim_node/drone/camera/image_raw', '/airsim_node/drone/segmentation/image_raw']
    topic_images=['/cam0/image_raw', '/cam0/segmentation']
    
    bridge = CvBridge()
    for capture_point in capture_points:
        for i, topic_name in enumerate(topic_images):
            img_pth = os.path.join(args.img_path, 'id{0}_{1:05d}.png'.format(i, capture_point.idx))
            if not (img_pth.split('.')[-1] == 'jpg' or img_pth.split('.')[-1] == 'png'):
                print(img_pth, "is not an image.")
                continue
            if not os.path.exists(img_pth):
                continue
            img_msg = get_img_msg(img_pth, bridge)
            img_msg.header.stamp.secs = capture_point.tsecs
            img_msg.header.stamp.nsecs = capture_point.tnsecs
            # img_pth = os.path.join(args.img_path, '{0:05d}.png'.format(capture_point.idx))
            outbag.write(topic_name, img_msg, img_msg.header.stamp)

            if stereo:
                img_pth = os.path.join(args.img_path, 'id{0}_{1:05d}_stereo.png'.format(i, capture_point.idx))
                if not (img_pth.split('.')[-1] == 'jpg' or img_pth.split('.')[-1] == 'png'):
                    print(img_pth, "is not an image.")
                    continue
                img_msg = get_img_msg(img_pth, bridge)
                img_msg.header.stamp.secs = capture_point.tsecs
                img_msg.header.stamp.nsecs = capture_point.tnsecs
                # img_pth = os.path.join(args.img_path, '{0:05d}.png'.format(capture_point.idx))
                outbag.write(topic_name.replace('0', '1'), img_msg, img_msg.header.stamp)

        if debug:
            ### write "/cam0/odometry" in bag for debug. This should be equal to /odometry ###
            print(capture_point.debug_actual_pos, capture_point.debug_actual_quat)
            odom_msg = Odometry()
            odom_msg.pose.pose.position.x = capture_point.debug_actual_pos[1]
            odom_msg.pose.pose.position.y = capture_point.debug_actual_pos[0]
            odom_msg.pose.pose.position.z = -capture_point.debug_actual_pos[2]
            odom_msg.pose.pose.orientation.x = capture_point.debug_actual_quat[0]
            odom_msg.pose.pose.orientation.y = capture_point.debug_actual_quat[1]
            odom_msg.pose.pose.orientation.z = capture_point.debug_actual_quat[2]
            odom_msg.pose.pose.orientation.w = capture_point.debug_actual_quat[3]
            odom_msg.header.stamp.secs = capture_point.tsecs
            odom_msg.header.stamp.nsecs = capture_point.tnsecs
            outbag.write('/cam0/odometry', odom_msg, odom_msg.header.stamp)


        # write other object_poses
        if objects_poses:
            for object_pose in objects_poses[capture_point.idx]:
                odom_msg = Odometry()
                odom_msg.pose.pose.position.x = object_pose['pose'].position.x_val
                odom_msg.pose.pose.position.y = object_pose['pose'].position.y_val
                odom_msg.pose.pose.position.z = object_pose['pose'].position.z_val
                odom_msg.pose.pose.orientation.x = object_pose['pose'].orientation.x_val
                odom_msg.pose.pose.orientation.y = object_pose['pose'].orientation.y_val
                odom_msg.pose.pose.orientation.z = object_pose['pose'].orientation.z_val
                odom_msg.pose.pose.orientation.w = object_pose['pose'].orientation.z_val
                odom_msg.header.stamp.secs = capture_point.tsecs
                odom_msg.header.stamp.nsecs = capture_point.tnsecs
                topic_name = '/objects/{0}/odom_local_ned'.format(object_pose['segID'])
                outbag.write(topic_name, odom_msg, odom_msg.header.stamp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rosbag_path", type=str, help="path to rosbag file")
    parser.add_argument("--rosbag_out_path", type=str, help="path to output rosbag file")
    parser.add_argument("--topic_gt", type=str, default='/airsim_node/drone/odom_local_ned', help="topic name for ground truth")
    parser.add_argument("--img_path", type=str, default='', help="Where to save images as a temp")
    parser.add_argument("--topic_sync", type=str, default='/airsim_node/drone/imu/imu', help="topic name for image, which will be used for synchronization")
    parser.add_argument("--topic_sync_freq", type=int, default=500, help="synchronized topic freq")
    parser.add_argument("--freq", type=float, default=None, help="frequency for images")
    parser.add_argument("--seg_ID", type=str, default=None, help="segment IDs")
    parser.add_argument("--stereo", action="store_true", help='record stereo if True')
    parser.add_argument("--use_gazebo", action="store_true", help='use gazebo dataset if True')
    parser.add_argument("--debug", action="store_true", help='debug mode if True')
    args = parser.parse_args()

    if args.stereo:
        print("USE STEREO!!")

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

    if args.use_gazebo:
        print('USE GAZEBO ROSBAG')
        capture_points = get_capture_points_gazebo(
            inbag, 
            odom_topic=args.topic_gt, 
            sync_topic=args.topic_sync, 
            camera_freq=20 if not args.debug else 1, 
            sync_topic_freq=100,
            method='every'
        ) 

    else: 
        # capture_points = get_capture_points_sync(inbag, args.topic_gt, args.topic_sync, args.topic_sync_freq, args.freq)
        capture_points = get_capture_points_by_odom(inbag,
                                                    args.topic_gt, 
                                                    gt_topic_freq=args.topic_sync_freq, 
                                                    target_freq=args.freq)
    print('{0} capture points'.format(len(capture_points)))
    if len(args.img_path)==0:
        tmp_dir = os.path.join(tempfile.gettempdir(), "airsim_drone")
    else:
        tmp_dir = args.img_path
    print ("Saving images to %s" % tmp_dir)
    airsim.wait_key('Press any key to confirm erasing these.')
    print("Recording images from AirSim...")
    client = airsim.VehicleClient()
    df_segment_id = pd.read_csv(args.seg_ID, header=None)
    set_segment_id(client, df_segment_id)
    objects_poses = record_imgs(client, 
                                capture_points,
                                tmp_dir, 
                                df_object_list=df_segment_id,
                                stereo=args.stereo,
                                debug=args.debug)
    objects_poses = filter_out_static_objects(objects_poses)
    print("Writing images into output bag...")
    load_images_to_bag(outbag, 
                       tmp_dir, 
                       objects_poses,
                       stereo=args.stereo,
                       debug=args.debug)
    print("Finished generating bag: "+ args.rosbag_out_path)
    outbag.close()
