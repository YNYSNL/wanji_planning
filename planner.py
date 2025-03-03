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
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
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

def init_reference_path(reference_data):
    """初始化参考线信息并计算累计距离s值"""
    global ref_lons, ref_lats, ref_s, vehicle_init_pos
    
    # 清空之前的数据
    ref_lons = []
    ref_lats = []
    ref_s = []
    
    try:
        # 因为传入的是 ref_dict，所以需要先获取 hdroutetoglobal 数据
        hd_route = reference_data["hdroutetoglobal"]
        
        # 遍历地图数据，保持与原始代码相同的循环结构
        for hdmap_msg in hd_route.map:
            for point in hdmap_msg.point:
                # 添加经纬度
                ref_lons.append(point.lon)
                ref_lats.append(point.lat)
                
                # 计算s值
                if len(ref_s) == 0:
                    # 第一个点的s值为0
                    ref_s.append(0.0)
                else:
                    # 获取前一个点的索引
                    prev_idx = len(ref_lons) - 2
                    
                    # 计算与前一点的距离
                    dist = calc_distance(
                        ref_lons[prev_idx], ref_lats[prev_idx],
                        point.lon, point.lat
                    )
                    
                    # 累加s值
                    ref_s.append(ref_s[-1] + dist)
        
        rospy.loginfo(f"Reference path initialized with {len(ref_lons)} points")
        
    except Exception as e:
        rospy.logerr(f"Error processing reference_data: {e}")
        rospy.logerr(f"reference_data: {reference_data}")
        return None

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
# def calc_distance(lon1, lat1, lon2, lat2):
#     """计算两点之间的距离"""
#     # 使用Haversine公式计算球面距离
#     R = 6371000  # 地球半径，单位米
    
#     # 转换为弧度
#     lat1_rad = np.radians(lat1)
#     lon1_rad = np.radians(lon1)
#     lat2_rad = np.radians(lat2)
#     lon2_rad = np.radians(lon2)
    
#     # Haversine公式
#     dlon = lon2_rad - lon1_rad
#     dlat = lat2_rad - lat1_rad
#     a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
#     c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
#     distance = R * c
    
    return distance
def xy_to_latlon(lon_origin, lat_origin, x, y):

    proj_string = "+proj=tmerc +lon_0=" + str(lon_origin) + " +lat_0=" + str(lat_origin) + " +ellps=WGS84"

    proj = Proj(proj_string)

    transformer_inv = Transformer.from_crs(proj_string, "epsg:4326", always_xy=True)

    lon_target, lat_target = transformer_inv.transform(x + proj(lon_origin, lat_origin)[0], 
                                                       y + proj(lon_origin, lat_origin)[1])

    return lon_target, lat_target


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

def CACS_plan(state_data, reference_data, ego_plan, ego_decision):
    """
    基于当前状态数据和参考线数据生成规划轨迹和决策
    
    Args:
        state_data: 车辆当前状态数据
        reference_data: 参考线数据
        ego_plan: 规划轨迹消息对象
        ego_decision: 决策消息对象
        
    Returns:
        ego_plan: 更新后的规划轨迹消息
        ego_decision: 更新后的决策消息
    """
    global ref_lons, ref_lats, ref_s, vehicle_init_pos
    
    # 深拷贝输入数据，避免修改原始数据
    DataFromEgo = copy.deepcopy(state_data)
    
    # 如果参考线数据为空且有新的参考线数据，则初始化参考线
    if not ref_lons and reference_data:
        init_reference_path(reference_data)
    
    # 创建车辆状态对象
    ego_state = Node(
        x=DataFromEgo['0'].get("x", 0.0) * 100, 
        y=DataFromEgo['0'].get("y", 0.0) * 100, 
        yaw=np.pi/2, 
        v=DataFromEgo['0'].get("v", 0.0) * 100, 
        gx=DataFromEgo['0'].get("gx", 0.0), 
        gy=DataFromEgo['0'].get("gy", 0.0), 
        direct=1.0,
        heading=DataFromEgo['0'].get("heading", 0.0)
    )
    current_pos = (ego_state.gx, ego_state.gy)
    
    # 初始化车辆初始位置（仅在第一次规划时）
    if vehicle_init_pos is None:
        vehicle_init_pos = current_pos
        rospy.loginfo(f"Vehicle initial position set to: {vehicle_init_pos}")
    
    # 1. 基于车辆坐标系生成粗线条轨迹
    time_steps = 50
    a = 2  # 加速度
    delta = 0  # 转向角
    
    x_local, y_local, yaw_local, v_local = generate_trajectory(ego_state, time_steps, a, delta)
    
    # 2. 在车辆坐标系下对轨迹进行加密
    x_local_dense, y_local_dense, yaw_local_dense = densify_trajectory(x_local, y_local, yaw_local, max_distance=20)
    
    # 3. 将加密后的轨迹从车辆坐标系旋转到全局坐标系
    x_global_dense, y_global_dense = coordinate_transform(x_local_dense, y_local_dense, target_heading=ego_state.heading)
    
    # 计算全局航向角
    yaw_deg_local_dense = np.degrees(yaw_local_dense)
    heading_global_dense = ego_state.heading - yaw_deg_local_dense + 90
    heading_global_dense = (heading_global_dense + 360) % 360
    
    # 4. 一次性将所有点转换为经纬度坐标
    gx_dense, gy_dense = xy_to_latlon_batch(
        ego_state.gx, ego_state.gy,
        x_global_dense/100, y_global_dense/100)
    
    # 5. 计算s值（相对于仿真初始位置的累计行驶距离）
    if ref_lons and ref_lats and ref_s:
        # 创建KD树用于快速查找最近点
        ref_points = np.column_stack((ref_lons, ref_lats))
        kdtree = cKDTree(ref_points, leafsize=100)
        
        # 找到车辆初始位置在参考线上的投影点（仅在第一次规划时计算）
        if not hasattr(CACS_plan, 'init_s_offset'):
            init_pos_array = np.array([[vehicle_init_pos[0], vehicle_init_pos[1]]])
            _, init_nearest_idx = kdtree.query(init_pos_array, k=1)
            init_nearest_idx = init_nearest_idx[0]
            
            # 计算初始位置到最近参考点的精确距离
            init_exact_dist = calc_distance(
                vehicle_init_pos[0], vehicle_init_pos[1],
                ref_lons[init_nearest_idx], ref_lats[init_nearest_idx]
            )
            
            # 初始位置的s值
            CACS_plan.init_s_offset = ref_s[init_nearest_idx] + init_exact_dist
            rospy.loginfo(f"Initial s offset set to: {CACS_plan.init_s_offset}")
        
        # 批量查找轨迹点最近的参考线点
        _, nearest_indices = kdtree.query(
            np.column_stack((gx_dense, gy_dense)),
            workers=-1
        )
        
        # 批量计算轨迹点到参考线的精确距离
        exact_dists = calc_distance_batch(
            gx_dense, gy_dense,
            np.array(ref_lons)[nearest_indices],
            np.array(ref_lats)[nearest_indices]
        )
        
        # 计算参考线上的s值
        ref_s_values = np.array(ref_s)[nearest_indices] + exact_dists
        
        # 计算相对于仿真初始位置的s值
        s_values = ref_s_values - CACS_plan.init_s_offset
    else:
        # 如果没有参考线，使用累计距离作为s值
        s_values = np.zeros(len(x_global_dense))
        for i in range(1, len(x_global_dense)):
            s_values[i] = s_values[i-1] + np.sqrt(
                (x_global_dense[i] - x_global_dense[i-1])**2 + 
                (y_global_dense[i] - y_global_dense[i-1])**2
            )
    
    # 6. 创建基础点属性
    base_point = roadpoint()
    base_point.speed = 15
    base_point.roadtype = 2
    base_point.turnlight = 0
    base_point.a = a
    base_point.jerk = 0
    base_point.lanewidth = 0
    
    # 8. 创建最终轨迹点列表
    trajectory_points = []
    for i in range(len(x_global_dense)):
        point = roadpoint()
        point.x = x_local_dense[i]
        point.y = y_local_dense[i]
        point.gx = gx_dense[i]
        point.gy = gy_dense[i]
        point.speed = base_point.speed
        point.heading = heading_global_dense[i]
        point.roadtype = base_point.roadtype
        point.turnlight = base_point.turnlight
        point.a = base_point.a
        point.jerk = base_point.jerk
        point.lanewidth = base_point.lanewidth
        point.s = s_values[i]
        trajectory_points.append(point)
    
    # 9. 设置规划消息
    ego_plan.points = trajectory_points
    ego_plan.guidespeed = 20  # 目标速度
    ego_plan.guideangle = 0   # 目标航向角
    ego_plan.timestamp = int(rospy.Time.now().to_sec() * 1000)
    
    # 10. 设置决策消息
    ego_decision.drivebehavior = 1  # 驾驶行为
    ego_decision.guidespeed = 20    # 目标速度
    ego_decision.carworkstatus = 0  # 车辆工作状态
    ego_decision.timestamp = int(rospy.Time.now().to_sec() * 1000)
    
    rospy.loginfo("Generated trajectory with {} points".format(len(trajectory_points)))
    
    return ego_plan, ego_decision

# 辅助函数

def generate_trajectory(ego_state, time_steps=50, a=2, delta=0):
    """生成基于车辆坐标系的粗线条轨迹"""
    # 创建车辆状态的副本，避免修改原始状态
    vehicle_state = copy.deepcopy(ego_state)
    
    # 初始化轨迹数组
    x = np.zeros(time_steps)
    y = np.zeros(time_steps)
    yaw = np.zeros(time_steps)
    v = np.zeros(time_steps)
    
    # 设置初始状态
    x[0] = vehicle_state.x
    y[0] = vehicle_state.y
    yaw[0] = vehicle_state.yaw
    v[0] = vehicle_state.v
    
    # 生成轨迹点
    for i in range(1, time_steps):
        vehicle_state.update(a*100, delta, 1.0)
        x[i] = vehicle_state.x
        y[i] = vehicle_state.y
        yaw[i] = vehicle_state.yaw
        v[i] = vehicle_state.v
    
    return x, y, yaw, v

def densify_trajectory(x, y, yaw, max_distance=20):
    """对轨迹进行加密，确保相邻点距离不超过指定值（向量化实现）"""
    # 计算所有相邻点之间的距离
    points = np.column_stack((x, y))
    diff = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(diff**2, axis=1))
    
    # 如果没有需要插值的点，直接返回原始轨迹
    if np.all(distances <= max_distance):
        return x, y, yaw
    
    # 计算每段需要插入的点数
    num_segments = len(distances)
    num_points_to_insert = np.maximum(0, np.ceil(distances / max_distance).astype(int) - 1)
    total_points = num_segments + 1 + np.sum(num_points_to_insert)  # 原始点数 + 插值点数
    
    # 预分配结果数组
    x_dense = np.zeros(total_points)
    y_dense = np.zeros(total_points)
    yaw_dense = np.zeros(total_points)
    
    # 填充第一个点
    x_dense[0] = x[0]
    y_dense[0] = y[0]
    yaw_dense[0] = yaw[0]
    
    # 当前填充位置索引
    current_idx = 1
    
    # 批量处理所有段
    for i in range(num_segments):
        # 当前段的起点和终点
        x1, y1, yaw1 = x[i], y[i], yaw[i]
        x2, y2, yaw2 = x[i+1], y[i+1], yaw[i+1]
        
        # 需要插入的点数
        n_insert = num_points_to_insert[i]
        
        if n_insert > 0:
            # 计算插值比例
            ratios = np.linspace(0, 1, n_insert + 2)[1:-1]
            
            # 线性插值
            x_interp = x1 + (x2 - x1) * ratios
            y_interp = y1 + (y2 - y1) * ratios
            yaw_interp = yaw1 + (yaw2 - yaw1) * ratios
            
            # 填充插值点
            x_dense[current_idx:current_idx + n_insert] = x_interp
            y_dense[current_idx:current_idx + n_insert] = y_interp
            yaw_dense[current_idx:current_idx + n_insert] = yaw_interp
            
            current_idx += n_insert
        
        # 填充当前段的终点（除了最后一段，因为最后一点会单独处理）
        if i < num_segments - 1:
            x_dense[current_idx] = x2
            y_dense[current_idx] = y2
            yaw_dense[current_idx] = yaw2
            current_idx += 1
    
    # 填充最后一个点
    x_dense[current_idx] = x[-1]
    y_dense[current_idx] = y[-1]
    yaw_dense[current_idx] = yaw[-1]
    
    # 如果预分配的空间有多余，裁剪掉
    if current_idx + 1 < total_points:
        x_dense = x_dense[:current_idx + 1]
        y_dense = y_dense[:current_idx + 1]
        yaw_dense = yaw_dense[:current_idx + 1]
    
    return x_dense, y_dense, yaw_dense