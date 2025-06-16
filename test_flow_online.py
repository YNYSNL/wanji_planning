#!/usr/bin/env python3
"""
自动驾驶规划系统 - 在线测试版本

修改说明：
- 添加了线程锁保护，解决高频传感器数据导致的数据竞争问题
- 实现数据快照机制，确保规划算法使用的是同一时刻的一致性数据
- 添加数据新鲜度验证，避免使用过时数据进行规划
- 优化了数据处理流程，提高系统稳定性和安全性
- 添加系统预热机制，解决JIT编译器冷启动问题
"""
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
from MotionPlanning.Control.MPC_XY_Frame import P, Node
from transform import lon_lat_to_xy, xy_to_lon_lat
import logging
from planner import CACS_plan, bag_plan
import os
import json
import time
import threading  # 添加线程模块

logging.basicConfig(filename='test_flow_online.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# 添加数据锁保护
data_lock = threading.Lock()

# 全局变量声明
pub_ego_plan = None
pub_ego_decision = None

bag_time = 0
bag_data = {}
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
    def __init__(self, obs_id, x, y, vx, vy, width=2.0, length=5.0):
        self.obs_id = obs_id
        self.x = x
        self.y = y
        self.vx = vx  # 绝对速度x分量 (全局坐标系)
        self.vy = vy  # 绝对速度y分量 (全局坐标系)
        self.v = np.sqrt(vx**2 + vy**2)  # 绝对速度大小
        self.width = width
        self.length = length
        self.positions = [[np.array([x, y])]]  # 障碍物的位置 [t][v_id]
        
        # 动静态判断参数
        self.speed_threshold = 0.8  # 动静态判断阈值 (m/s)
        self.is_dynamic = self.v > self.speed_threshold
        
        # 障碍物类型
        if self.is_dynamic:
            self.type = "dynamic"
        else:
            self.type = "static"

def validate_obstacle_speed_conversion(ego_speed, ego_heading_deg, rel_vx_body, rel_vy_body, abs_vx_global, abs_vy_global):
    """
    验证障碍物速度转换是否正确（修正版）
    
    Args:
        ego_speed: 主车速度 (m/s)
        ego_heading_deg: 主车航向角 (度)
        rel_vx_body, rel_vy_body: 车身坐标系下的相对速度分量
        abs_vx_global, abs_vy_global: 全局坐标系下的绝对速度分量
    """
    ego_heading_rad = np.radians(ego_heading_deg)
    
    # 主车在车身坐标系下的速度
    ego_vx_body = 0.0
    ego_vy_body = ego_speed
    
    # 障碍物在车身坐标系下的绝对速度
    expected_abs_vx_body = rel_vx_body + ego_vx_body
    expected_abs_vy_body = rel_vy_body + ego_vy_body
    
    # 转换到全局坐标系
    expected_abs_vx_global = expected_abs_vx_body * np.cos(ego_heading_rad + np.pi/2) - expected_abs_vy_body * np.sin(ego_heading_rad + np.pi/2)
    expected_abs_vy_global = expected_abs_vx_body * np.sin(ego_heading_rad + np.pi/2) + expected_abs_vy_body * np.cos(ego_heading_rad + np.pi/2)
    
    vx_error = abs(abs_vx_global - expected_abs_vx_global)
    vy_error = abs(abs_vy_global - expected_abs_vy_global)
    
    if vx_error > 0.01 or vy_error > 0.01:
        rospy.logwarn(f"速度转换可能有误: vx_error={vx_error:.3f}, vy_error={vy_error:.3f}")
        rospy.logwarn(f"主车: speed={ego_speed:.2f}, heading={ego_heading_deg:.1f}°")
        rospy.logwarn(f"车身系相对速度: ({rel_vx_body:.2f}, {rel_vy_body:.2f})")
        rospy.logwarn(f"全局系转换后: ({abs_vx_global:.2f}, {abs_vy_global:.2f})")
        rospy.logwarn(f"全局系期望值: ({expected_abs_vx_global:.2f}, {expected_abs_vy_global:.2f})")


def update_frame_data(topic, msg):
    with data_lock:
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
def cal_action(sensor_data, reference_data, frame_data_snapshot=None):
    ego_plan = planningmotion()
    ego_decision = decisionbehavior()
    ego_plan.points = roadpoint()
    use_mpc = True
    
    # 使用快照数据而不是全局变量，确保数据一致性
    if frame_data_snapshot is None:
        frame_data_snapshot = frame_data  # 向后兼容
    
    # 获取主车当前状态信息
    ego_speed = 0.0  # 主车速度
    ego_heading = 0.0  # 主车航向角（弧度）
    
    # 从sensor_data获取主车状态
    if sensor_data and '0' in sensor_data:
        ego_speed = sensor_data['0'].get("v", 0.0)  # 主车速度 (m/s)
        ego_heading_deg = sensor_data['0'].get("heading", 0.0)  # 主车航向角（度）
        ego_heading = np.radians(ego_heading_deg)  # 转换为弧度
    
    # 主车在车身坐标系下的速度分量
    # 车身坐标系：x方向为横向（垂直车身），y方向为纵向（沿车身前进方向）
    ego_vx_body = 0.0  # 主车横向速度为0（不侧滑）
    ego_vy_body = ego_speed  # 主车纵向速度等于车速
    
    # 处理障碍物信息（使用快照数据确保一致性）
    if "objectTrack" in frame_data_snapshot and frame_data_snapshot["objectTrack"] is not None:
        sensorobjects_msg = frame_data_snapshot["objectTrack"]
        
        # 初始化障碍物参数
        if hasattr(sensorobjects_msg, 'obs'):
            P.init(num_obstacles=len(sensorobjects_msg.obs),
                  obstacle_horizon=20,
                  num_modes=1,
                  treat_obstacles_as_static=False)
            
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
                    
                    # 使用检测到的宽度和长度，或者使用默认值（单位转换为米）
                    width = obj.width 
                    length = obj.length
                    
                    # 获取障碍物相对速度信息（车身坐标系）
                    rel_vx_body = getattr(obj, 'relspeedx', 0.0)  # 相对速度x分量（横向）
                    rel_vy_body = getattr(obj, 'relspeedy', 0.0)  # 相对速度y分量（纵向）
                    
                    # 将相对速度转换为车身坐标系下的绝对速度
                    # 绝对速度 = 相对速度 + 主车速度（均在车身坐标系下）
                    abs_vx_body = rel_vx_body + ego_vx_body  # 横向绝对速度
                    abs_vy_body = rel_vy_body + ego_vy_body  # 纵向绝对速度
                    
                    # 将车身坐标系下的绝对速度转换为全局坐标系
                    # 全局坐标系转换：考虑主车航向角
                    abs_vx_global = abs_vx_body * np.cos(ego_heading + np.pi/2) - abs_vy_body * np.sin(ego_heading + np.pi/2)
                    abs_vy_global = abs_vx_body * np.sin(ego_heading + np.pi/2) + abs_vy_body * np.cos(ego_heading + np.pi/2)
                    
                    # 验证速度转换是否正确
                    validate_obstacle_speed_conversion(ego_speed, ego_heading_deg, rel_vx_body, rel_vy_body, abs_vx_global, abs_vy_global)
                    
                    # 创建障碍物信息对象，传递全局坐标系下的绝对速度信息
                    obstacle = ObstacleInfo(obs_id, obs_x_global, obs_y_global, abs_vx_body, abs_vy_body, width, length)
                    
                    # 判断障碍物类型并输出调试信息
                    v_magnitude = obstacle.v
                    obstacle_type = obstacle.type
                    
                    rospy.loginfo(f"障碍物 {obs_id}: {obstacle_type}, 位置=({obs_x_global:.1f},{obs_y_global:.1f})")
                    
                    rospy.loginfo(f"  车身系相对速度=({rel_vx_body:.2f},{rel_vy_body:.2f})")
                    rospy.loginfo(f"  车身系绝对速度=({abs_vx_body:.2f},{abs_vy_body:.2f})")
                    rospy.loginfo(f"  全局系绝对速度=({abs_vx_global:.2f},{abs_vy_global:.2f}), |v|={v_magnitude:.2f}")
                    rospy.loginfo(f"  主车速度={ego_speed:.2f}m/s, 航向={ego_heading_deg:.1f}°, 动态判断={obstacle.is_dynamic}")
                    
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

# 验证数据新鲜度
def validate_data_freshness(frame_data_snapshot):
    """验证数据是否新鲜，避免使用过时数据"""
    try:
        current_time = rospy.Time.now()
        max_age = 0.5  # 最大允许数据年龄（秒）
        
        for key, msg in frame_data_snapshot.items():
            if msg is None:
                continue
                
            # 检查消息是否有时间戳
            if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
                age = (current_time - msg.header.stamp).to_sec()
                if age > max_age:
                    rospy.logwarn(f"{key} data is {age:.2f}s old, considered stale")
                    return False
            elif hasattr(msg, 'timestamp'):
                # 某些消息可能直接有timestamp字段
                age = current_time.to_sec() - msg.timestamp
                if age > max_age:
                    rospy.logwarn(f"{key} data is {age:.2f}s old, considered stale")
                    return False
        
        return True
    except Exception as e:
        rospy.logwarn(f"Error validating data freshness: {e}")
        return True  # 如果验证失败，仍然允许处理

# 检查数据是否齐全，且仅在数据齐全时处理
def check_and_process_data():
    # 创建数据快照，避免在处理过程中被修改
    global pub_ego_plan, pub_ego_decision
    frame_data_snapshot = None
    
    with data_lock:
        if all(data_status.values()):
            rospy.loginfo("All data received, creating data snapshot...")
            # 创建数据快照，确保数据一致性
            frame_data_snapshot = copy.deepcopy(frame_data)
            rospy.logdebug(f"Data snapshot created with keys: {list(frame_data_snapshot.keys())}")
            # 立即重置状态，允许新数据进入
            reset_data_status()
    
    # 在锁外进行计算，使用快照数据
    if frame_data_snapshot is not None:
        rospy.loginfo("Processing planning with snapshot data...")
        
        # 验证数据新鲜度（可选）
        if validate_data_freshness(frame_data_snapshot):
            # 经纬度转为笛卡尔坐标
            lon_lat_converter_state = lon_lat_to_xy([frame_data_snapshot["sensorgps"]])
            state_dict = lon_lat_converter_state.get_pos()

            ref_dict = {
                "hdroutetoglobal": frame_data_snapshot["hdroutetoglobal"]
            }

            start_time = time.time()

            # 进行规划（使用快照数据，确保数据一致性）
            ego_plan, ego_decision = cal_action(state_dict, ref_dict, frame_data_snapshot)

            end_time = time.time()
            execution_time = end_time - start_time
            rospy.loginfo(f"Planning and decision execution time: {execution_time:.2f}s")

            # 发布规划结果
            pub_ego_plan.publish(ego_plan)
            pub_ego_decision.publish(ego_decision)
            rospy.loginfo("Planning and decision published!")
        else:
            rospy.logwarn("Data snapshot is stale, skipping planning cycle")

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

        check_and_process_data()
    else:
        rospy.loginfo("hdroutetoglobal already received!")  

def offline_test():
    """离线测试模式"""
    global bag_data, frame_data, data_status, pub_ego_plan, pub_ego_decision
    
    # 打开bag文件
    # file_path = '/home/admin/Downloads/20250211-bag/2025-02-11-15-31-30.bag'2025-06-11-17-19-41.bag
    # file_path = '/home/admin/Downloads/2025-06-11-17-23-05.bag' 
    file_path = '/home/admin/Downloads/2025-06-14-11-03-41/2025-06-14-11-03-41.bag' 
    bag = rosbag.Bag(file_path, 'r')
    
    # 按时间戳组织数据
    frame_timestamps = {}  # 存储每个时间戳对应的数据
    
    # 读取所有相关话题的消息
    topics = ['/sensorgps', '/objectTrack/track_results', 
             '/actuator', '/hdroutetoglobal']
    
    print("Reading bag file...")
    frame_count = 0
    max_frames = 500  # 只保存前600帧数据
    
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
    
    # 创建结果发布器（使用全局变量）
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
        if i < 280 or i > 610:
            continue

        start_time = time.time()
        
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

        end_time = time.time()
        execution_time = end_time - start_time
        rospy.loginfo(f"Planning and decision execution time: {execution_time:.2f}s")

# 系统预热函数
def warm_up_planning_system():
    """
    系统预热：预先触发所有耗时的初始化过程
    
    这个函数会预先触发以下耗时操作的JIT编译和初始化：
    1. NumPy/SciPy BLAS/LAPACK初始化
    2. Numba JIT编译 (planner.calc_distance_batch)
    """
    print("启动规划系统预热...")
    warm_start_time = time.time()
    
    try:
        # 1. 预热NumPy/SciPy - 触发BLAS/LAPACK库初始化
        print("预热NumPy/SciPy BLAS/LAPACK...")
        dummy_large = np.random.rand(1000, 1000)
        _ = np.linalg.inv(dummy_large @ dummy_large.T + np.eye(1000) * 0.1)
        _ = np.fft.fft(np.random.rand(4096))  # 预热FFT
        
        # 2. 预热Numba JIT编译器 - 这是最耗时的部分 (60-80%)
        print("预热Numba JIT编译器...")
        try:
            from planner import calc_distance_batch
            # 多次调用确保JIT完全编译
            dummy_coords = np.array([116.0, 116.1, 116.2])
            dummy_lats = np.array([40.0, 40.1, 40.2])
            for _ in range(3):  # 多次调用确保编译完成
                _ = calc_distance_batch(dummy_coords, dummy_lats, dummy_coords, dummy_lats)
            print("Numba JIT编译完成")
        except Exception as e:
            print(f"Numba预热失败: {e}")
        
        warm_end_time = time.time()
        warm_duration = warm_end_time - warm_start_time
        print(f"系统预热完成！总耗时: {warm_duration:.2f}s")

        
    except Exception as e:
        print(f"预热过程中遇到错误: {e}")
        print("系统仍可正常运行，但首次规划可能较慢")

def main():
    global pub_ego_plan, pub_ego_decision  # 在函数开头统一声明所有需要的全局变量
    
    # 选择运行模式
    if len(sys.argv) > 1 and sys.argv[1] == '--offline': # sys.argv[0] == filename.py sys.argv[1] == '--offline'
        print("Running in offline mode...")
        rospy.init_node("ros_topic_processor_offline", anonymous=True)
        
        # 离线模式也执行预热
        warm_up_planning_system()
        
        offline_test()
    else:
        print("Running in online mode...")
        # 初始化ROS节点
        rospy.init_node("ros_topic_processor", anonymous=True)
        
        # 系统预热（在订阅话题之前进行）
        warm_up_planning_system()
        
        pub_ego_plan = rospy.Publisher("/planningmotion", planningmotion, queue_size=3)
        pub_ego_decision = rospy.Publisher("/behaviordecision", decisionbehavior, queue_size=3)
        
        # 订阅传感器数据
        rospy.Subscriber("/objectTrack/track_results", sensorobjects, callback_objectTrack, queue_size=3)
        rospy.Subscriber("/actuator", actuator, callback_actuator, queue_size=3)
        rospy.Subscriber("/hdroutetoglobal", hdroutetoglobal, callback_hdroutetoglobal, queue_size=1)
        rospy.Subscriber("/sensorgps", sensorgps, callback_sensorgps, queue_size=3)
        
        rospy.loginfo("ROS node initialized and waiting for data...")
        rospy.spin() # keep the node running

if __name__ == "__main__":
    main()
