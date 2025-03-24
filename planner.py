import copy
import numpy as np
import matplotlib.pyplot as plt
import csv
import rospy
from path_handling import load_splines, SplinePath, find_best_s, get_path_obj
from MotionPlanning.Control.MPC_XY_Frame import *
import MotionPlanning.CurvesGenerator.cubic_spline as cs
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
mpc_controller = None

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

def CACS_plan(state_data, reference_data, ego_plan, ego_decision, use_mpc=False):
    """
    CACS规划算法主函数
    
    Args:
        state_data: 车辆当前状态数据
        reference_data: 参考线数据
        ego_plan: 规划结果
        ego_decision: 决策结果
        use_mpc: 是否使用MPC生成轨迹，默认为False
    """
    global ref_lons, ref_lats, ref_s, vehicle_init_pos, mpc_controller
    
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
   
    # 根据选择的方法生成局部坐标系下的轨迹
    if use_mpc and ref_lons and ref_lats and ref_s:
        # 将参考线转换为局部坐标系
        start_time1 = time.time()
        local_ref_path = convert_reference_to_local(ego_state, ref_lons, ref_lats)
        end_time1 = time.time()
        rospy.loginfo(f"convert_reference_to_local time: {end_time1 - start_time1} seconds")
        
        # 使用MPC生成局部坐标系下的轨迹
        start_time2 = time.time()
        x_local, y_local, yaw_local, v_local = generate_mpc_trajectory(ego_state, local_ref_path)
        end_time2 = time.time()
        rospy.loginfo(f"generate_mpc_trajectory time: {end_time2 - start_time2} seconds")
    else:
        # 使用简单的匀加速模型生成局部坐标系下的轨迹
        time_steps = 50
        a = 2  # 加速度
        delta = 0  # 转向角
        x_local, y_local, yaw_local, v_local = generate_trajectory(ego_state, time_steps, a, delta)
    
    # 统一进行轨迹加密
    x_local_dense, y_local_dense, yaw_local_dense = densify_trajectory(x_local, y_local, yaw_local, max_distance=20)
    
    # 统一将加密后的轨迹从车辆坐标系旋转到全局坐标系
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
    base_point.a = 2
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
    
def convert_reference_to_local(ego_state, ref_lons, ref_lats):
    """
    将全局经纬度参考线转换为车辆局部坐标系下的参考线
    
    参数:
        ego_state: 车辆当前状态
        ref_lons: 参考线经度列表
        ref_lats: 参考线纬度列表
    
    返回:
        local_ref_path: 局部坐标系下的参考路径对象
    """
    
    # 创建局部坐标系下的参考线点
    ref_x_local = []
    ref_y_local = []
    
    # 获取车辆当前位置的投影变换器
    transformer, proj = get_transformer(ego_state.gx, ego_state.gy)
    x0, y0 = proj(ego_state.gx, ego_state.gy)
    ref_lons_array = np.array(ref_lons)
    ref_lats_array = np.array(ref_lats)
    
    # 将经纬度参考线转换为平面坐标系（与xy_to_latlon_batch的逆过程）
    x_global, y_global = transformer.transform(ref_lons_array, ref_lats_array)
    
    # 转换为numpy数组后再进行减法操作
    x_global = np.array(x_global) - x0
    y_global = np.array(y_global) - y0
    
    # 将平面坐标转换为车辆局部坐标系（与coordinate_transform的逆过程）
    theta = np.radians(ego_state.heading)
    
    # 创建旋转矩阵（逆向旋转）
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # 进行坐标变换
    for i in range(len(x_global)):
        # 相对于车辆位置的偏移
        dx = x_global[i] * 100  # 转换为厘米
        dy = y_global[i] * 100  # 转换为厘米
        
        # 旋转到车辆坐标系
        point = np.array([dx, dy])
        rotated_point = np.dot(rotation_matrix, point)
        
        ref_x_local.append(rotated_point[0])
        ref_y_local.append(rotated_point[1])
    
    # 选择参考线上的一段合适长度的路径（避免处理过长的参考线）
    # 找到距离车辆最近的点
    distances = [math.sqrt(x**2 + y**2) for x, y in zip(ref_x_local, ref_y_local)]
    min_idx = distances.index(min(distances))
    
    # 选择前后各100个点，总共最多201个点
    start_idx = max(0, min_idx - 100)
    end_idx = min(len(ref_x_local), min_idx + 100)
    
    selected_x = ref_x_local[start_idx:end_idx]
    selected_y = ref_y_local[start_idx:end_idx]
    
    # 确保有足够的点进行样条曲线拟合
    if len(selected_x) < 4:
        rospy.logerr("Not enough points for spline fitting")
        # 创建一个简单的直线路径
        cx = [0, 100, 200, 300]  # 沿车辆前方的直线
        cy = [0, 0, 0, 0]        # 车辆中心线
        cyaw = [0, 0, 0, 0]      # 与车辆方向一致
        ck = [0, 0, 0, 0]        # 无曲率
        local_ref_path = PATH(cx, cy, cyaw, ck)
        return local_ref_path
    
    # 使用样条曲线平滑参考线
    try:
        # 确保点的间距不为零
        valid_indices = []
        last_valid_x = selected_x[0]
        last_valid_y = selected_y[0]
        valid_indices.append(0)
        
        for i in range(1, len(selected_x)):
            dist = math.sqrt((selected_x[i] - last_valid_x)**2 + (selected_y[i] - last_valid_y)**2)
            if dist > 1.0:  # 最小间距为1厘米
                valid_indices.append(i)
                last_valid_x = selected_x[i]
                last_valid_y = selected_y[i]
        
        # 如果有效点太少，添加一些人工点
        if len(valid_indices) < 4:
            rospy.logwarn(f"Too few valid points ({len(valid_indices)}), adding artificial points")
            cx = [0, 100, 200, 300]  # 沿车辆前方的直线
            cy = [0, 0, 0, 0]        # 车辆中心线
            cyaw = [0, 0, 0, 0]      # 与车辆方向一致
            ck = [0, 0, 0, 0]        # 无曲率
        else:
            # 使用有效点计算样条曲线
            valid_x = [selected_x[i] for i in valid_indices]
            valid_y = [selected_y[i] for i in valid_indices]
            
            # 计算样条曲线
            cx, cy, cyaw, ck, s = cs.calc_spline_course(
                valid_x, valid_y, ds=P.d_dist)
            
            # 检查结果是否包含NaN值
            if np.isnan(np.sum(cx)) or np.isnan(np.sum(cy)) or np.isnan(np.sum(cyaw)) or np.isnan(np.sum(ck)):
                rospy.logwarn("Spline calculation produced NaN values, using simple path")
                cx = [0, 100, 200, 300]  # 沿车辆前方的直线
                cy = [0, 0, 0, 0]        # 车辆中心线
                cyaw = [0, 0, 0, 0]      # 与车辆方向一致
                ck = [0, 0, 0, 0]        # 无曲率
    
    except Exception as e:
        rospy.logerr(f"Error calculating spline course: {e}")
        # 如果样条曲线计算失败，使用简单路径
        cx = [0, 100, 200, 300]  # 沿车辆前方的直线
        cy = [0, 0, 0, 0]        # 车辆中心线
        cyaw = [0, 0, 0, 0]      # 与车辆方向一致
        ck = [0, 0, 0, 0]        # 无曲率
    
    # 创建参考路径对象
    local_ref_path = PATH(cx, cy, cyaw, ck)
    
    return local_ref_path

def generate_mpc_trajectory(ego_state, local_ref_path):
    """
    使用MPC控制器生成局部坐标系下的轨迹
    
    参数:
        ego_state: 车辆当前状态
        local_ref_path: 局部坐标系下的参考路径
    
    返回:
        x_local: 局部坐标系下的x坐标
        y_local: 局部坐标系下的y坐标
        yaw_local: 局部坐标系下的航向角
        v_local: 速度
    """
    global mpc_controller
    
    # 在局部坐标系中，车辆位置为原点
    initial_state = [0, 0, ego_state.yaw, ego_state.v]
    
    # 如果MPC控制器未初始化，则初始化
    if mpc_controller is None:    
        # 初始化MPC控制器
        mpc_controller = MPCController(P.target_speed, initial_state)
        rospy.loginfo("MPC controller initialized")
    
    # 使用MPC控制器生成轨迹
    try:
        target_ind, x_opt, y_opt, yaw_opt, v_opt = mpc_controller.update(local_ref_path, initial_state)
        
        if x_opt is None or y_opt is None or yaw_opt is None or v_opt is None:
            rospy.logwarn("MPC optimization failed, falling back to simple trajectory")
            # 如果MPC优化失败，使用简单轨迹
            return fallback_trajectory(ego_state)
        
        # 将MPC预测轨迹转换为数组格式
        x_local = np.array(x_opt)
        y_local = np.array(y_opt)
        yaw_local = np.array(yaw_opt)
        v_local = np.array(v_opt)
        
        return x_local, y_local, yaw_local, v_local
        
    except Exception as e:
        rospy.logerr(f"Error in MPC trajectory generation: {e}")
        return fallback_trajectory(ego_state)

def fallback_trajectory(ego_state):
    """
    当MPC失败时的备用轨迹生成方法
    
    返回局部坐标系下的轨迹
    """
    time_steps = 50
    a = 2  # 加速度
    delta = 0  # 转向角
    
    x_local, y_local, yaw_local, v_local = generate_trajectory(ego_state, time_steps, a, delta)
    
    return x_local, y_local, yaw_local, v_local

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