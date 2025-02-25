import copy
import numpy as np
import matplotlib.pyplot as plt
import csv
import rospy
from path_handling import load_splines, SplinePath, find_best_s, get_path_obj
from MotionPlanning.Control.MPC_XY_Frame import P, Node
from common_msgs.msg import actuator
from common_msgs.msg import hdmap
from common_msgs.msg import hdroutetoglobal
from common_msgs.msg import roadpoint
from common_msgs.msg import sensorgps
from common_msgs.msg import sensorobject
from common_msgs.msg import sensorobjects
from common_msgs.msg import planningmotion
from common_msgs.msg import decisionbehavior
from pyproj import Proj, Transformer
import math

# 全局变量存储参考线信息
ref_lons = []
ref_lats = []
ref_s = []
vehicle_init_pos = None  # 存储车辆初始位置的全局变量

def yaw_to_heading(yaw):
    # yaw 转 heading
    # 1. 先将yaw转换为以北为0的角度（逆时针为正）
    north_angle = np.pi/2 - yaw
    # 2. 转换为顺时针为正
    heading = -north_angle
    # 3. 归一化到0-2π范围
    heading = heading % (2 * np.pi)
    # 4. 转换为角度
    heading_deg = math.degrees(heading)
    return heading_deg

def heading_to_yaw(heading_deg):
    # heading 转 yaw
    # 1. 转换为弧度
    heading = math.radians(heading_deg)
    # 2. 转换为逆时针为正
    north_angle = -heading
    # 3. 转换为以x轴为0的角度
    yaw = np.pi/2 - north_angle
    # 4. 归一化到-π到π范围
    yaw = (yaw + np.pi) % (2 * np.pi) - np.pi
    return yaw

def init_reference_path(reference_data):
    """初始化参考线信息"""
    global ref_lons, ref_lats, ref_s, vehicle_init_pos
    
    try:
        # 因为传入的是 ref_dict，所以需要先获取 hdroutetoglobal 数据
        hd_route = reference_data["hdroutetoglobal"]
        for hdmap_msg in hd_route.map:
            for point in hdmap_msg.point:
                ref_lons.append(point.lon)
                ref_lats.append(point.lat)
                
    except Exception as e:
        print("Error processing reference_data:", e)
        print("reference_data:", reference_data)
        return None
    
    # 计算参考线的累计距离，但先不赋值给ref_s
    temp_s = [0]
    for i in range(1, len(ref_lons)):
        dist = calc_distance(ref_lons[i-1], ref_lats[i-1], ref_lons[i], ref_lats[i])
        temp_s.append(temp_s[-1] + dist)
    
    # rospy.loginfo("参考线初始化完成，共 %d 个点", len(ref_lons))
    return True

def calc_distance(lon1, lat1, lon2, lat2):
    """计算两点间大地距离"""
    R = 6371000  # 地球半径(米)
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi/2) * np.sin(delta_phi/2) + \
        np.cos(phi1) * np.cos(phi2) * \
        np.sin(delta_lambda/2) * np.sin(delta_lambda/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def xy_to_latlon(lon_origin, lat_origin, x, y):

    proj_string = "+proj=tmerc +lon_0=" + str(lon_origin) + " +lat_0=" + str(lat_origin) + " +ellps=WGS84"

    proj = Proj(proj_string)

    transformer_inv = Transformer.from_crs(proj_string, "epsg:4326", always_xy=True)

    lon_target, lat_target = transformer_inv.transform(x + proj(lon_origin, lat_origin)[0], 
                                                       y + proj(lon_origin, lat_origin)[1])

    return lon_target, lat_target

def CACS_plan(state_data, reference_data, ego_plan, ego_decision):
    global ref_lons, ref_lats, ref_s, vehicle_init_pos
    DataFromEgo = copy.deepcopy(state_data)
    # print('DataFromEgo', DataFromEgo)

    # Get the current sensor data
    # sensorgps_data = frame_data.get("sensorgps")

    # v0 = sensorgps_data.velocity
    # heading = sensorgps_data.heading
    if not ref_lons and reference_data:
        # print("reference_data type:", type(reference_data))
        init_reference_path(reference_data)

    ego_state = Node(
        x=DataFromEgo['0'].get("x", 0.0) * 100, 
        y=DataFromEgo['0'].get("y", 0.0) * 100, 
        yaw=DataFromEgo['0'].get("heading", 0.0), 
        v=DataFromEgo['0'].get("v", 0.0) * 100, 
        gx=DataFromEgo['0'].get("gx", 0.0), 
        gy=DataFromEgo['0'].get("gy", 0.0), 
        direct=1.0
    )
    current_pos = (ego_state.gx, ego_state.gy)
    
    # 如果是第一次运行，初始化车辆初始位置
    if vehicle_init_pos is None and ref_lons:
        vehicle_init_pos = current_pos
        # 找到距离初始位置最近的参考线点
        distances = []
        for i in range(len(ref_lons)):
            dist = calc_distance(vehicle_init_pos[0], vehicle_init_pos[1], ref_lons[i], ref_lats[i])
            distances.append(dist)
        init_idx = np.argmin(distances)
        
        # 预先计算相邻点之间的距离
        point_distances = []
        for j in range(len(ref_lons)-1):
            dist = calc_distance(ref_lons[j], ref_lats[j], ref_lons[j+1], ref_lats[j+1])
            point_distances.append(dist)
            
        ref_s = [0] * len(ref_lons)  # 预分配列表空间
        
        # 计算init_idx之前的点的累积距离（负值）
        curr_dist = 0
        for i in range(init_idx - 1, -1, -1):
            curr_dist += point_distances[i]
            ref_s[i] = -curr_dist
            
        # 计算init_idx之后的点的累积距离（正值）
        curr_dist = 0
        for i in range(init_idx, len(ref_lons)-1):
            curr_dist += point_distances[i]
            ref_s[i+1] = curr_dist
          
    # Acceleration and time steps for the motion plan
    a = 2  # Constant acceleration
    # t_max = 3  # Total time for the trajectory
    time_steps = 50  # Number of time steps to break down the trajectory

    # Generate the trajectory using the Node's kinematic update
    trajectory_points = []
    trajectory_points_x = []
    trajectory_points_y = []

    ego_state.yaw = np.pi/2  # Convert to radians
    for t in range(time_steps):
        # Update the vehicle's state using the Node's update method
        # Create a roadpoint from the updated state
        point = roadpoint()
        point.x = ego_state.x  # Updated x-coordinate
        point.y = ego_state.y  # Updated y-coordinate    relativetime: 0.0
        # print('gx',ego_state.gx,'gy',ego_state.gy)
        point.gx, point.gy = xy_to_latlon(ego_state.gx,ego_state.gy,point.x/100,point.y/100)

        point.speed = 15 # ego_state.v / 100  # Updated velocity
        point.heading = 320  # Updated heading
        point.roadtype = 2
        point.turnlight = 0
        # print(f'heading is {point.heading}')
        point.a = a  
        point.jerk = 0  # Assuming no jerk (smooth motion)
        point.lanewidth = 0  # Default lane width, adjust if needed
        ego_state.update(200, delta=0.0, direct=1.0)  # Assume no steering (delta=0) for simplicity

        if ref_lons:
            distances = []
            for i in range(len(ref_lons)):
                dist = calc_distance(point.gx, point.gy, ref_lons[i], ref_lats[i])
                distances.append(dist)
            
            nearest_idx = np.argmin(distances)
            # 使用重新计算的ref_s
            point.s = ref_s[nearest_idx]
            
            # 如果需要，可以加入到当前位置的精确距离修正
            if nearest_idx > 0:
                # 计算到最近点的精确距离
                exact_dist = calc_distance(point.gx, point.gy, ref_lons[nearest_idx], ref_lats[nearest_idx])
                # 根据车辆位置在参考线左右侧决定距离正负
                # 这里需要根据实际情况补充判断逻辑
                point.s += exact_dist
        else:
            point.s = t * 0.1 * ego_state.v

        # xy_converter = xy_to_lon_lat([ego_state], [state_dict])te(100)
        # transformed_data = xy_converter.transform()
        trajectory_points.append(point)
        trajectory_points_x.append(point.x)
        trajectory_points_y.append(point.y)

    # path_obj = get_path_obj(np.array(trajectory_points_x), np.array(trajectory_points_y))

    # s_his = 0
    # for t, point in enumerate(trajectory_points):

    #     q = [point.x, point.y, 0, 0, s_his]
    #     point.s = find_best_s(q, path_obj, enable_global_search=True)
    #     s_his = point.s
    #     print(f's at time{t} is {point.s}')
      

    # logging.info(f"trajectory_points_x: {trajectory_points_x}")

    with open('trajectory_points.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Write the header
        writer.writerow(['x', 'y'])
        
        # Write the data
        for x, y in zip(trajectory_points_x, trajectory_points_y):
            writer.writerow([x, y])

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
    # plt.close()

    # Update the ego_plan with the generated trajectory points
    ego_plan.points = trajectory_points
    # print('ego_plan.points', ego_plan.points)

    # plt.plot(ego_plan.points.x, ego_plan.points.y, marker='o', label='Trajectory Points')
    # plt.savefig('trajectory_plot.png')
    ego_plan.guidespeed = 20  # Final velocity
    ego_plan.guideangle = 320 # Final heading
    ego_plan.timestamp = int(rospy.Time.now().to_sec()*1000)

    # Decision-making (example behavior)
    ego_decision.drivebehavior = 1  # Drive behavior, this could be adjusted
    ego_decision.guidespeed = 20
    ego_decision.carworkstatus = 0
    ego_decision.timestamp = int(rospy.Time.now().to_sec()*1000)

    # rospy.loginfo(f"Planning result: ego_plan={ego_plan}, ego_decision={ego_decision}")

    return ego_plan, ego_decision


def bag_plan(bag_data, t, ego_plan, ego_decision):

    # print(bag_data['guidespeed'][t])
    # Update the ego_plan with the generated trajectory points
    ego_plan.points = bag_data['roadpoints'][t]
    print(len(bag_data['roadpoints'][t]))
    # print('ego_plan.points', ego_plan.points.s)

    # plt.plot(ego_plan.points.x, ego_plan.points.y, marker='o', label='Trajectory Points')
    # plt.savefig('trajectory_plot.png')
    ego_plan.guidespeed = bag_data['guidespeed'][t]  # Final velocity
    ego_plan.guideangle = bag_data['guideangle'][t]  # Final heading
    ego_plan.timestamp = int(rospy.Time.now().to_sec()*1000)

    ego_decision.drivebehavior = 2  # Drive behavior, this could be adjusted

    return ego_plan, ego_decision