#!/usr/bin/env python3
import sys
sys.path.append('/home/wanji/HIL/devel/lib/python3/dist-packages')
import rospy
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

logging.basicConfig(filename='test_flow_online.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
global pub_ego_plan
global pub_ego_decision

# 缓存每帧数据
frame_data = {
    "sensorgps": None,
    "objectTrack": None,
    "actuator": None
}

# 数据状态标志
data_status = {
    "sensorgps": False,
    "objectTrack": False,
    "actuator": False
}

# 更新缓存数据并检查是否准备好进行规划
def update_frame_data(topic, msg):
    if topic == "/sensorgps":
        frame_data["sensorgps"] = msg
        data_status["sensorgps"] = True
    elif topic == "/objectTrack/track_results8CornerForRVIZ":
        frame_data["objectTrack"] = msg
        data_status["objectTrack"] = True
    elif topic == "/actuator":
        frame_data["actuator"] = msg
        data_status["actuator"] = True
    # rospy.loginfo(f"Received data from topic {topic}: {msg}")

# 回调函数
def callback_sensorgps(data):

    update_frame_data("/sensorgps", data)
    check_and_process_data()

def callback_objectTrack(data):
    update_frame_data("/objectTrack/track_results8CornerForRVIZ", data)
    check_and_process_data()

def callback_actuator(data):
    update_frame_data("/actuator", data)
    check_and_process_data()

def cal_action(data):
    ego_plan = planningmotion()
    ego_decision = decisionbehavior()
    ego_plan.points = roadpoint()

    DataFromEgo = copy.deepcopy(data)
    # print('DataFromEgo', DataFromEgo)

    # Get the current sensor data
    sensorgps_data = frame_data.get("sensorgps")

    v0 = sensorgps_data.velocity
    heading = sensorgps_data.heading


    ego_state = Node(
        x=DataFromEgo['0'].get("x", 0.0), 
        y=DataFromEgo['0'].get("y", 0.0), 
        yaw=heading, 
        v=DataFromEgo['0'].get("v", 0.0), 
        direct=1.0
    )
    # Acceleration and time steps for the motion plan
    a = 1  # Constant acceleration
    t_max = 10  # Total time for the trajectory
    time_steps = 50  # Number of time steps to break down the trajectory
    dt = t_max / time_steps  # Time step for each update

    # Generate the trajectory using the Node's kinematic update
    trajectory_points = []
    trajectory_points_x = []
    trajectory_points_y = []
    for t in range(time_steps):
        # Update the vehicle's state using the Node's update method
        # Create a roadpoint from the updated state
        point = roadpoint()
        point.x = ego_state.x  # Updated x-coordinate
        point.y = ego_state.y  # Updated y-coordinate
        point.gx = ego_state.x  # 经度
        point.gy = ego_state.y  # 纬度

        point.speed = ego_state.v  # Updated velocity
        point.heading = heading  # Updated heading
        point.a = a
        point.jerk = 0  # Assuming no jerk (smooth motion)
        point.lanewidth = 3.5  # Default lane width, adjust if needed
        point.s = ego_state.x  # s position along the trajectory (simplified)
        ego_state.update(a, delta=0.0, direct=1.0)  # Assume no steering (delta=0) for simplicity
        # xy_converter = xy_to_lon_lat([ego_state], [state_dict])te(100)
        # transformed_data = xy_converter.transform()

        # 获取转换后的经纬度
        # lat, lon = transformed_data[0]["lat"], transformed_data[0]["lon"]

        # 将经纬度赋值给roadpoint

        trajectory_points_x.append(point.x)
        trajectory_points_y.append(point.y)

        trajectory_points.append(point)

    logging.info(f"trajectory_points_x: {trajectory_points_x}")


    # # Plot the trajectory
    # plt.figure(figsize=(8, 6))
    # plt.plot(trajectory_points_x, trajectory_points_y, marker='o', label='Trajectory Points')
    # plt.title('Trajectory of Vehicle')
    # plt.xlabel('X (meters)')
    # plt.ylabel('Y (meters)')
    # plt.grid(True)
    # plt.legend()
    # # Save the figure
    # plt.savefig('trajectory_plot.png')

    # Update the ego_plan with the generated trajectory points
    ego_plan.points = trajectory_points
    # print('ego_plan.points', ego_plan.points)

    # plt.plot(ego_plan.points.x, ego_plan.points.y, marker='o', label='Trajectory Points')
    # plt.savefig('trajectory_plot.png')
    ego_plan.guidespeed = ego_state.v  # Final velocity
    ego_plan.guideangle = ego_state.yaw  # Final heading
    ego_plan.timestamp = int(rospy.Time.now().to_sec())

    # Decision-making (example behavior)
    ego_decision.drivebehavior = 2  # Drive behavior, this could be adjusted

    rospy.loginfo(f"Planning result: ego_plan={ego_plan}, ego_decision={ego_decision}")

    return ego_plan, ego_decision
# 检查数据是否齐全，且仅在数据齐全时处理
def check_and_process_data():
    if all(data_status.values()):
        rospy.loginfo("All data received, processing planning...")

        # 经纬度转为笛卡尔坐标
        lon_lat_converter = lon_lat_to_xy([frame_data["sensorgps"]])
        state_dict = lon_lat_converter.get_pos()

        # 进行规划
        ego_plan, ego_decision = cal_action(state_dict)

        # 发布规划结果
        pub_ego_plan.publish(ego_plan)
        pub_ego_decision.publish(ego_decision)
        rospy.loginfo("Planning and decision published!")

        # 重置数据状态标志
        reset_data_status()

# 重置数据状态标志
def reset_data_status():
    global data_status
    for key in data_status:
        data_status[key] = False

def main():
    # 初始化ROS节点
    rospy.init_node("ros_topic_processor", anonymous=True)

    global pub_ego_plan
    global pub_ego_decision

    pub_ego_plan = rospy.Publisher("/planningmotion", planningmotion, queue_size=5)
    pub_ego_decision = rospy.Publisher("/behaviordecision0", decisionbehavior, queue_size=5)

    # 订阅传感器数据
    rospy.Subscriber("/objectTrack/track_results8CornerForRVIZ", sensorobjects, callback_objectTrack, queue_size=3)
    rospy.Subscriber("/actuator", actuator, callback_actuator, queue_size=3)
    rospy.Subscriber("/sensorgps", sensorgps, callback_sensorgps, queue_size=3)

    rospy.loginfo("ROS node initialized and waiting for data...")
    rate = rospy.Rate(100)
    rospy.spin()

if __name__ == "__main__":
    main()
