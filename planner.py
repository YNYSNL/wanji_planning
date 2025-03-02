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
from scipy.spatial import cKDTree
import time
from functools import lru_cache
import numba

@lru_cache(maxsize=1)
def get_transformer(lon_origin, lat_origin):
    proj_string = f"+proj=tmerc +lon_0={lon_origin} +lat_0={lat_origin} +ellps=WGS84"
    proj = Proj(proj_string)
    return Transformer.from_crs(proj_string, "epsg:4326", always_xy=True), proj

def xy_to_latlon_batch(lon_origin, lat_origin, x_array, y_array):
    """批量转换坐标"""
    transformer_inv, proj = get_transformer(lon_origin, lat_origin)
    x0, y0 = proj(lon_origin, lat_origin)
    
    # 批量转换
    lon_array, lat_array = transformer_inv.transform(
        x_array + x0,
        y_array + y0
    )
    return lon_array, lat_array

# 优化距离计算
@numba.jit(nopython=True)
def calc_distance_batch(lon1_array, lat1_array, lon2_array, lat2_array):
    """使用Numba加速批量计算距离"""
    R = 6371000  # 地球半径(米)
    phi1 = np.radians(lat1_array)
    phi2 = np.radians(lat2_array)
    delta_phi = np.radians(lat2_array - lat1_array)
    delta_lambda = np.radians(lon2_array - lon1_array)
    
    a = np.sin(delta_phi/2)**2 + \
        np.cos(phi1) * np.cos(phi2) * \
        np.sin(delta_lambda/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c
# 全局变量存储参考线信息
ref_lons = []
ref_lats = []
ref_s = []
vehicle_init_pos = None  # 存储车辆初始位置的全局变量

def coordinate_transform(x, y, target_heading=320):
    """
    将坐标系进行旋转变换
    Args:
        x: 原始x坐标数组
        y: 原始y坐标数组
        yaw: 原始偏航角数组（弧度）
        target_heading: 目标航向角（度）
    Returns:
        new_x: 变换后的x坐标数组
        new_y: 变换后的y坐标数组
        new_heading: 变换后的航向角数组（度）
    """
    # 将目标航向角转换为弧度
    theta = np.radians(target_heading)
    
    # 创建旋转矩阵
    rotation_matrix = np.array([
        [np.sin(theta), np.cos(theta)],
        [np.cos(theta), -np.sin(theta)]
    ])
    
    # 确保输入是numpy数组
    x = np.asarray(x)
    y = np.asarray(y)
    
    # 进行坐标变换
    points = np.column_stack((x - x[0], y - y[0]))
    rotated_points = np.dot(points, rotation_matrix.T)
    new_x = rotated_points[:, 0] + x[0]
    new_y = rotated_points[:, 1] + y[0]

    
    return new_x, new_y

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
        yaw=0, 
        v=DataFromEgo['0'].get("v", 0.0) * 100, 
        gx=DataFromEgo['0'].get("gx", 0.0), 
        gy=DataFromEgo['0'].get("gy", 0.0), 
        direct=1.0,
        heading=DataFromEgo['0'].get("heading", 0.0)
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
    delta = 0
    # t_max = 3  # Total time for the trajectory
    time_steps = 50  # Number of time steps to break down the trajectory
    x_array = np.zeros(time_steps)
    y_array = np.zeros(time_steps)
    yaw_array = np.zeros(time_steps)
    v_array = np.zeros(time_steps)

    x_array[0] = ego_state.x
    y_array[0] = ego_state.y
    yaw_array[0] = ego_state.yaw
    v_array[0] = ego_state.v

    for i in range(1,time_steps):
        ego_state.update(a*100, delta, 1.0)
        # 存储每个时间步的状态
        x_array[i] = ego_state.x
        y_array[i] = ego_state.y
        yaw_array[i] = ego_state.yaw
        v_array[i] = ego_state.v

    # 预先创建KD树用于快速查找最近点
    if ref_lons and ref_lats:
        ref_points = np.column_stack((ref_lons, ref_lats))
        kdtree = cKDTree(ref_points)

    base_point = roadpoint()
    base_point.speed = 15
    base_point.roadtype = 2
    base_point.turnlight = 0
    base_point.a = a
    base_point.jerk = 0
    base_point.lanewidth = 0


    x_array_global, y_array_global= coordinate_transform(x_array, y_array, target_heading=ego_state.heading)


    # 批量创建轨迹点
    raw_trajectory_points = []
    # 批量创建轨迹点
    if ref_lons:
        start_time1 = time.time()
        # 批量转换所有点的经纬度
        gx_array, gy_array = xy_to_latlon_batch(
            ego_state.gx, ego_state.gy,
            x_array_global/100, y_array_global/100)
        end_time1 = time.time()
        print(f"转换经纬度耗时：{end_time1 - start_time1} s")
        # 批量查找最近点 (使用更大的leaf_size可能会更快)
        kdtree = cKDTree(np.column_stack((ref_lons, ref_lats)), leafsize=100)
        _, nearest_indices = kdtree.query(
            np.column_stack((gx_array, gy_array)),
            workers=-1  # 使用多线程
        )
        start_time2 = time.time()
        # 批量计算exact_dist
        exact_dists = calc_distance_batch(
            gx_array, gy_array,
            np.array(ref_lons)[nearest_indices],
            np.array(ref_lats)[nearest_indices]
        )
        end_time2 = time.time()
        print(f"计算exact_dist耗时：{end_time2 - start_time2} s")
        s_values = np.array(ref_s)[nearest_indices] + exact_dists
    # else:
    #     s_values = t_array * 0.1 * ego_state.v
    #     gx_array, gy_array = xy_to_latlon_batch(
    #         ego_state.gx, ego_state.gy,
    #         x_array/100, y_array/100)

    # 使用列表推导式创建轨迹点
    yaw_deg_local = np.degrees(yaw_array)
    heading_glob = ego_state.heading - yaw_deg_local  # 一种典型的符号关系
    # 保持结果在 0~360 范围内(可选)
    heading_glob = (heading_glob + 360) % 360
    raw_trajectory_points = [
        roadpoint(
            x=x, y=y, gx=gx, gy=gy,
            speed=base_point.speed,
            heading=yaw,
            roadtype=base_point.roadtype,
            turnlight=base_point.turnlight,
            a=base_point.a,
            jerk=base_point.jerk,
            lanewidth=base_point.lanewidth,
            s=s
        ) for x, y, gx, gy, s, yaw in zip(x_array, y_array, gx_array, gy_array, s_values, heading_glob)
    ]
    points_array = np.column_stack((x_array, y_array))
    diff = np.diff(points_array, axis=0)
    distances = np.sqrt(np.sum(diff**2, axis=1))
    
    trajectory_points = []
   
    # 预先计算所有需要插值的点的索引
    interpolation_indices = np.where(distances > 20)[0]

    start_time3 = time.time()
    
    if len(interpolation_indices) > 0:
        # 预分配存储空间
        all_x_interp = []
        all_y_interp = []
        all_ratios = []
        total_points = 0
        
        # 一次性计算所有插值点的x,y坐标
        for idx in interpolation_indices:
            current_point = raw_trajectory_points[idx]
            next_point = raw_trajectory_points[idx + 1]
            num_points = int(np.ceil(distances[idx]/20))
            
            # ratios = np.linspace(1/num_points, 1-1/num_points, num_points-1)
            ratios = np.linspace(0, 1, num_points+1)[1:]
            x_interp = current_point.x + (next_point.x - current_point.x) * ratios
            y_interp = current_point.y + (next_point.y - current_point.y) * ratios
            
            all_x_interp.extend(x_interp)
            all_y_interp.extend(y_interp)
            all_ratios.extend(ratios)
            total_points += len(ratios)
        
        # 将所有插值点转换为numpy数组
        all_x_interp = np.array(all_x_interp)
        all_y_interp = np.array(all_y_interp)

        all_x_interp_global, all_y_interp_global, _ = coordinate_transform(all_x_interp, all_y_interp, 0, target_heading=ego_state.heading)
        
        # 一次性批量转换所有插值点的经纬度
        gx_interp_all, gy_interp_all = xy_to_latlon_batch(
            ego_state.gx, ego_state.gy,
            all_x_interp_global/100, all_y_interp_global/100)
        
        # 创建基础点对象作为模板
        # base_point = roadpoint()
        # base_point.speed = raw_trajectory_points[0].speed
        # base_point.roadtype = raw_trajectory_points[0].roadtype
        # base_point.turnlight = raw_trajectory_points[0].turnlight
        # base_point.a = raw_trajectory_points[0].a
        # base_point.jerk = raw_trajectory_points[0].jerk
        # base_point.lanewidth = raw_trajectory_points[0].lanewidth
        
        # 重新组织插值点
        point_idx = 0
        for idx in interpolation_indices:
            current_point = raw_trajectory_points[idx]
            next_point = raw_trajectory_points[idx + 1]
            num_points = int(np.ceil(distances[idx]/20)) - 1
            
            trajectory_points.append(current_point)
            ratios = np.linspace(1/num_points, 1-1/num_points, num_points)
            heading_interp = current_point.heading + \
                           (next_point.heading - current_point.heading) * ratios        
            # 批量创建这段的插值点
            for j in range(num_points):
                new_point = roadpoint()
                new_point.x = all_x_interp[point_idx]
                new_point.y = all_y_interp[point_idx]
                new_point.gx = gx_interp_all[point_idx]
                new_point.gy = gy_interp_all[point_idx]
                new_point.speed = base_point.speed
                new_point.heading = heading_interp[j]
                new_point.roadtype = base_point.roadtype
                new_point.turnlight = base_point.turnlight
                new_point.a = base_point.a
                new_point.jerk = base_point.jerk
                new_point.lanewidth = base_point.lanewidth
                new_point.s = current_point.s + (next_point.s - current_point.s) * (j+1)/(num_points+1)
                
                trajectory_points.append(new_point)
                point_idx += 1
            
            if idx == interpolation_indices[-1]:
                trajectory_points.append(next_point)
    else:
        # 如果没有需要插值的点，直接使用原始点
        trajectory_points = raw_trajectory_points

    end_time3 = time.time()
    print(f"插值耗时：{end_time3 - start_time3} s")

    # Update the ego_plan with the generated trajectory points
    ego_plan.points = trajectory_points
    # print('ego_plan.points', ego_plan.points)

    # plt.plot(ego_plan.points.x, ego_plan.points.y, marker='o', label='Trajectory Points')
    # plt.savefig('trajectory_plot.png')
    ego_plan.guidespeed = 20  # Final velocity
    ego_plan.guideangle = 0 # Final heading
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
