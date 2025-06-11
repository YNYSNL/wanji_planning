#!/usr/bin/env python3
import sys
sys.path.append('/home/wanji/HIL/devel/lib/python3/dist-packages')
import rospy
import rosbag
from common_msgs.msg import actuator
from common_msgs.msg import hdmap
from common_msgs.msg import hdroutetoglobal
from common_msgs.msg import roadpoint
from common_msgs.msg import sensorgps
from common_msgs.msg import sensorobject
from common_msgs.msg import sensorobjects
from common_msgs.msg import planningmotion
from common_msgs.msg import decisionbehavior
import copy
import math
import numpy as np
import matplotlib.pyplot as plt
import cvxpy
from MotionPlanning.Control.MPC_XY_Frame import P, Node
from transform import lon_lat_to_xy, xy_to_lon_lat
import logging
from planner import CACS_plan, bag_plan
import os
import json
import time

logging.basicConfig(filename='test_flow_online.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
global pub_ego_plan
global pub_ego_decision
global reference_line_received
reference_line_received = False

global bag_time
bag_time = 0
global bag_data
# file_path = '/media/wanji/ssd2T/20250211-bag/2025-02-11-15-31-30.bag'
# file_path = '/home/admin/Downloads/20250211-bag/2025-02-11-15-31-30.bag'
# 打开bag文件
# bag = rosbag.Bag(file_path, 'r')

guidespeed = []
guideangle = []
timestamp = []
roadpints = []
# 遍历bag文件中的每条消息
# for topic, msg, t in bag.read_messages(topics='/planningmotion'):
#     # 读取 /planningmotion 话题中的数据
#     guidespeed.append(msg.guidespeed)   
#     guideangle.append(msg.guideangle)  
#     timestamp.append(msg.timestamp)
#     # 解析 /planningmotion 中的每个 roadpoint
#     planning_points = []
#     for point in msg.points:
#         p = roadpoint()
#         p.x = point.x
#         p.y = point.y
#         p.speed = point.speed
#         p.heading = point.heading
#         # p.jerk = point.jerk
#         # p.lanewidth = point.lanewidth
#         p.s = point.s
#         p.gx = point.gx
#         p.gy = point.gy
#         p.roadtype = point.roadtype
#         p.a = point.a
#         p.lanetype = point.lanetype
#         p.turnlight = point.turnlight
#         # p.mergelanetype = point.mergelanetype
#         # p.sensorlanetype = point.sensorlanetype
#         p.curvature = point.curvature
#         p.relativetime = point.relativetime
#         # p.dkappa = point.dkappa
#         # p.ddkappa = point.ddkappa
#         # p.sideroadwidth = point.sideroadwidth
#         # p.leftlanewidth = point.leftlanewidth
#         # p.rightlanewidth = point.rightlanewidth
#         # p.laneswitch = point.laneswitch
#         # p.lanenum = point.lanenum
#         # p.lanesite = point.lanesite
#         planning_points.append(p)   
#     roadpints.append(planning_points)
        
#     # 关闭bag文件
# bag.close()

bag_data = {
    "guidespeed": guidespeed, 
    "guideangle": guideangle,   
    "timestamp": timestamp,
    "roadpoints": roadpints    
}

# 缓存每帧数据
frame_data = {
    "sensorgps": None,
    "objectTrack": None,
    "actuator": None,
    "hdroutetoglobal": None
}

# 数据状态标志
data_status = {
    "sensorgps": False,
    "objectTrack": False,
    "actuator": False,
    "hdroutetoglobal": False
}

# 障碍物处理类
class ObstacleInfo:
    def __init__(self, obs_id, x, y, width=2.0, length=5.0):
        self.obs_id = obs_id
        self.x = x
        self.y = y
        self.width = width
        self.length = length
        self.positions = [[np.array([x, y])]]  # 静态障碍物的位置 [t][v_id]

# 处理障碍物信息
def process_obstacles(sensorobjects_msg):
    """
    处理障碍物信息并更新MPC控制器中的障碍物
    注意：此函数不再直接使用，而是被cal_action中的代码取代，
    保留此函数仅作为参考。cal_action中的代码对坐标变换进行了更详细的处理。
    
    参数:
        sensorobjects_msg: 传感器检测到的障碍物消息
    """
    obstacles = {}
    
    if sensorobjects_msg and hasattr(sensorobjects_msg, 'obs') and len(sensorobjects_msg.obs) > 0:
        # 清空之前的障碍物
        P.obstacles.clear()
        
        # 更新障碍物数量
        P.num_obstacles = len(sensorobjects_msg.obs)
        
        # 处理每个障碍物
        for i, obj in enumerate(sensorobjects_msg.obs):
            # 提取障碍物ID和位置
            obs_id = obj.id
            
            # 获取障碍物的全局位置（米为单位）
            # 注意：需要将传感器数据中的厘米转换为米
            x_global = obj.x / 100.0  # 厘米转米
            y_global = obj.y / 100.0  # 厘米转米
            
            # 使用检测到的宽度和长度，或者使用默认值（单位转换为米）
            width = obj.width / 100.0 if obj.width > 0.1 else 2.0  # 默认宽度2.0m
            length = obj.length / 100.0 if obj.length > 0.1 else 5.0  # 默认长度5.0m
            
            # 将障碍物位置转换到车辆坐标系
            # 此处假设车辆位于原点(0,0)，并且朝向是车辆坐标系的正前方
            # 实际应用中需要根据车辆当前位置和朝向进行坐标变换
            
            # 创建障碍物信息对象，使用局部坐标
            obstacle = ObstacleInfo(obs_id, x_global, y_global, width, length)
            
            # 将障碍物添加到P.obstacles中
            P.obstacles[obs_id] = obstacle
            
        rospy.loginfo(f"更新了 {P.num_obstacles} 个障碍物")
    else:
        # 清空障碍物
        P.obstacles.clear()
        P.num_obstacles = 0
        rospy.loginfo("没有检测到障碍物或障碍物消息为空")
    
    return obstacles

def update_frame_data(topic, msg):
    if topic == "/sensorgps":
        frame_data["sensorgps"] = msg
        data_status["sensorgps"] = True
        rospy.loginfo("sensorgps received!")  # 使用普通的print替代rospy.loginfo
        
    elif topic == "/objectTrack/track_results":
        frame_data["objectTrack"] = msg
        data_status["objectTrack"] = True
        rospy.loginfo("objectTrack received!")
        
    elif topic == "/actuator":
        frame_data["actuator"] = msg
        data_status["actuator"] = True
        rospy.loginfo("actuator received!")
    elif topic == "/hdroutetoglobal":
        frame_data["hdroutetoglobal"] = msg
        data_status["hdroutetoglobal"] = True
        rospy.loginfo("hdroutetoglobal received!")
        
# 规划动作
def cal_action(sensor_data, reference_data):
    ego_plan = planningmotion()
    ego_decision = decisionbehavior()
    ego_plan.points = roadpoint()
    use_mpc = True
    
    # 处理障碍物信息
    if "objectTrack" in frame_data and frame_data["objectTrack"] is not None:
        sensorobjects_msg = frame_data["objectTrack"]
        
        # 初始化障碍物参数
        if hasattr(sensorobjects_msg, 'obs'):
            P.init(num_obstacles=len(sensorobjects_msg.obs),
                  obstacle_horizon=20,
                  num_modes=1,
                  treat_obstacles_as_static=True)
            
            # 清空之前的障碍物
            P.obstacles.clear()
            
            # 如果有障碍物，处理它们
            if len(sensorobjects_msg.obs) > 0:
                rospy.loginfo(f"检测到 {len(sensorobjects_msg.obs)} 个障碍物")
                
                # 处理每个障碍物
                for i, obj in enumerate(sensorobjects_msg.obs):
                    # 提取障碍物ID
                    obs_id = i
                    
                    # 获取障碍物的全局位置（米为单位）
                    obs_x_global = obj.x 
                    obs_y_global = obj.y 
                    
                    # # 障碍物相对于车辆的位置（全局坐标系）
                    # dx = obs_x_global - ego_x
                    # dy = obs_y_global - ego_y
                    
                    # # 将障碍物位置从全局坐标系转换到车辆局部坐标系
                    # # 1. 平移（使车辆位置为原点）
                    # # 2. 旋转（使车辆朝向为x轴正方向）
                    # obs_x_local = dx * np.cos(-ego_heading) - dy * np.sin(-ego_heading)
                    # obs_y_local = dx * np.sin(-ego_heading) + dy * np.cos(-ego_heading)
                    
                    # 使用检测到的宽度和长度，或者使用默认值（单位转换为米）
                    width = obj.width 
                    length = obj.length
                    # rospy.loginfo(f"障碍物 {obs_id} 局部坐标: x={obs_x_local:.2f}, y={obs_y_local:.2f}, w={width:.2f}, l={length:.2f}")
                    
                    # 创建障碍物信息对象，使用局部坐标
                    obstacle = ObstacleInfo(obs_id, obs_x_global, obs_y_global, width, length)
                    
                    # 将障碍物添加到P.obstacles中
                    P.obstacles[obs_id] = obstacle

        else:
            # 如果没有障碍物信息，则清空障碍物
            P.init(num_obstacles=0)
            rospy.loginfo("没有检测到障碍物，清空障碍物列表")
    
    # 调用规划算法
    ego_plan, ego_decision = CACS_plan(sensor_data, reference_data, ego_plan, ego_decision, use_mpc)
# 
    # global bag_data
    # global bag_time

    # ego_plan, ego_decision = bag_plan(bag_data, bag_time, ego_plan, ego_decision)
    # bag_time += 1

    return ego_plan, ego_decision

# 重置数据状态标志
def reset_data_status():
    global data_status
    for key in data_status:
        data_status[key] = False
        data_status["hdroutetoglobal"] = True

# 检查数据是否齐全，且仅在数据齐全时处理
def check_and_process_data():

    if all(data_status.values()):
        rospy.loginfo("All data received, processing planning...")     
        # 经纬度转为笛卡尔坐标
        lon_lat_converter_state = lon_lat_to_xy([frame_data["sensorgps"]])
        state_dict = lon_lat_converter_state.get_pos()

        # lon_lat_converter_ref = lon_lat_to_xy_map([frame_data["hdroutetoglobal"]])
        # ref_dict = lon_lat_converter_ref.get_pos()
        # state_dict = None
        ref_dict = {
            "hdroutetoglobal": frame_data["hdroutetoglobal"]
        }

        start_time = time.time()

        # 进行规划
        ego_plan, ego_decision = cal_action(state_dict, ref_dict)

        end_time = time.time()
        execution_time = end_time - start_time
        rospy.loginfo(f"Planning and decision execution time: {execution_time:.2f}s")

        # 发布规划结果
        pub_ego_plan.publish(ego_plan)
        pub_ego_decision.publish(ego_decision)
        rospy.loginfo("Planning and decision published!")

        # 重置数据状态标志
        reset_data_status()

# 回调函数
def callback_sensorgps(data):
    update_frame_data("/sensorgps", data)
    check_and_process_data()

def callback_objectTrack(data):
    update_frame_data("/objectTrack/track_results", data)
    check_and_process_data()

def callback_actuator(data):
    update_frame_data("/actuator", data)
    check_and_process_data()

def callback_hdroutetoglobal(data):
    """处理参考线数据"""
    if not frame_data.get("hdroutetoglobal"):
        # 第一次收到参考线数据时初始化
        frame_data["hdroutetoglobal"] = data
        data_status["hdroutetoglobal"] = True
        reference_data = {"hdroutetoglobal": data}
        from planner import init_reference_path
        init_reference_path(reference_data)
        check_and_process_data()
    else:
        rospy.loginfo("hdroutetoglobal already received!")  

def offline_test():
    """离线测试模式"""
    global bag_data, frame_data, data_status
    
    # 打开bag文件
    # file_path = '/home/admin/Downloads/20250211-bag/2025-02-11-15-31-30.bag'
    file_path = '/home/admin/Downloads/2025-06-06-09-18-01.bag' 
    bag = rosbag.Bag(file_path, 'r')
    
    # 按时间戳组织数据
    frame_timestamps = {}  # 存储每个时间戳对应的数据
    
    # 读取所有相关话题的消息
    topics = ['/sensorgps', '/objectTrack/track_results', 
             '/actuator', '/hdroutetoglobal']
    
    print("Reading bag file...")
    frame_count = 0
    max_frames = 610  # 只保存前600帧数据
    
    for topic, msg, t in bag.read_messages(topics=topics):
        timestamp = t.to_sec()
        # 将时间戳四舍五入到最近的0.1秒，以便对齐不同话题的数据
        rounded_timestamp = round(timestamp * 10) / 10
        
        if rounded_timestamp not in frame_timestamps:
            frame_timestamps[rounded_timestamp] = {}
            frame_count += 1
            
            # 只保存前600帧数据
            if frame_count > max_frames:
                print(f"Reached maximum frames limit ({max_frames}), stopping data collection...")
                break
                
        frame_timestamps[rounded_timestamp][topic] = msg
    
    bag.close()
    print(f"Found {len(frame_timestamps)} frames in bag file")
    
    # 按时间戳顺序处理每一帧
    sorted_timestamps = sorted(frame_timestamps.keys())
    
    # 创建结果发布器
    pub_ego_plan = rospy.Publisher("/planningmotion", planningmotion, queue_size=5)
    pub_ego_decision = rospy.Publisher("/behaviordecision", decisionbehavior, queue_size=5)
    
    # 处理每一帧数据
    for i, timestamp in enumerate(sorted_timestamps):

        frame = frame_timestamps[timestamp]
        print(f"\nProcessing frame {i+1}/{len(sorted_timestamps)}")
        
        # 更新frame_data
        if '/sensorgps' in frame:
            update_frame_data("/sensorgps", frame['/sensorgps'])
        if '/objectTrack/track_results' in frame:
            update_frame_data("/objectTrack/track_results", frame['/objectTrack/track_results'])
        if '/actuator' in frame:
            update_frame_data("/actuator", frame['/actuator'])
        if '/hdroutetoglobal' in frame:
            update_frame_data("/hdroutetoglobal", frame['/hdroutetoglobal'])

        print(data_status.values())       
        # 检查数据是否完整并进行规划
        if i < 400 or i > 610:
            continue
        
        if all(data_status.values()):
            print("Processing planning for frame...")
            
            # 经纬度转为笛卡尔坐标
            lon_lat_converter_state = lon_lat_to_xy([frame_data["sensorgps"]])
            state_dict = lon_lat_converter_state.get_pos()
            
            ref_dict = {
                "hdroutetoglobal": frame_data["hdroutetoglobal"]
            }
            
            # 进行规划
            ego_plan, ego_decision = cal_action(state_dict, ref_dict)

            print('---')
            
            # 重置数据状态标志
            reset_data_status()
        else:
            print("Incomplete data for this frame, skipping...")

def main():
    # 选择运行模式
    if len(sys.argv) > 1 and sys.argv[1] == '--offline': # sys.argv[0] == filename.py sys.argv[1] == '--offline'
        print("Running in offline mode...")
        rospy.init_node("ros_topic_processor_offline", anonymous=True)
        offline_test()
    else:
        print("Running in online mode...")
        # 初始化ROS节点
        rospy.init_node("ros_topic_processor", anonymous=True)
        
        global pub_ego_plan
        global pub_ego_decision
        
        pub_ego_plan = rospy.Publisher("/planningmotion", planningmotion, queue_size=3)
        pub_ego_decision = rospy.Publisher("/behaviordecision", decisionbehavior, queue_size=3)
        
        # 订阅传感器数据
        rospy.Subscriber("/objectTrack/track_results", sensorobjects, callback_objectTrack, queue_size=3)
        rospy.Subscriber("/actuator", actuator, callback_actuator, queue_size=3)
        rospy.Subscriber("/hdroutetoglobal", hdroutetoglobal, callback_hdroutetoglobal, queue_size=1)
        rospy.Subscriber("/sensorgps", sensorgps, callback_sensorgps, queue_size=3)
        
        rospy.loginfo("ROS node initialized and waiting for data...")
        rate = rospy.Rate(100) # 100Hz  
        rospy.spin() # keep the node running

if __name__ == "__main__":
    main()
