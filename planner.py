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

# 全局变量存储MPC轨迹生成时间
mpc_time_records = []
step_counter = 0

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
    global ref_lons, ref_lats, ref_s, vehicle_init_pos, mpc_controller, mpc_time_records, step_counter
    
    # 步数计数器
    step_counter += 1
    
    # 深拷贝输入数据，避免修改原始数据
    DataFromEgo = copy.deepcopy(state_data)
    
    # 如果参考线数据为空且有新的参考线数据，则初始化参考线
    if not ref_lons and reference_data:
        init_reference_path(reference_data)
    
    # 创建车辆状态对象
    ego_state = Node(
        x=DataFromEgo['0'].get("x", 0.0), 
        y=DataFromEgo['0'].get("y", 0.0), 
        yaw=np.pi/2, 
        v=DataFromEgo['0'].get("v", 0.0), 
        gx=DataFromEgo['0'].get("gx", 0.0), 
        gy=DataFromEgo['0'].get("gy", 0.0), 
        direct=1.0,
        heading=DataFromEgo['0'].get("heading", 0.0)
    )
    current_pos = (ego_state.gx, ego_state.gy)
    print('current_heading',ego_state.heading)
    
    # 初始化车辆初始位置（仅在第一次规划时）
    if vehicle_init_pos is None:
        vehicle_init_pos = current_pos
        rospy.loginfo(f"Vehicle initial position set to: {vehicle_init_pos}") 
   
    # 根据选择的方法生成局部坐标系下的轨迹
    if use_mpc and ref_lons and ref_lats and ref_s:
        # 将参考线转换为局部坐标系
        start_time1 = time.time()
        local_ref_path = convert_reference_to_local(ego_state, ref_lons, ref_lats)
        # plt.plot(local_ref_path.cx, local_ref_path.cy, marker='o', label='Trajectory Points')
        
        # plt.show()
        end_time1 = time.time()
        rospy.loginfo(f"convert_reference_to_local time: {end_time1 - start_time1} seconds")
        
        # 使用MPC生成局部坐标系下的轨迹
        start_time2 = time.time()
        x_local, y_local, yaw_local, v_local = generate_mpc_trajectory(ego_state, local_ref_path)
        # print(x_local[0],y_local[0])
        # plt.plot(local_ref_path.cx, local_ref_path.cy, label='Trajectory Points',color='blue')
        # plt.plot(x_local, y_local, label='Trajectory Points',color='red')

        # plt.scatter(ego_state.x, ego_state.y, marker='x', label='Vehicle Position',s=80,color='red')

        # plt.draw()
        # plt.pause(1)
        # plt.close()

        end_time2 = time.time()
        mpc_generation_time = end_time2 - start_time2
        
        # 记录MPC轨迹生成时间
        mpc_time_records.append({
            'step': step_counter,
            'time': mpc_generation_time,
            'timestamp': time.time()
        })
        
        rospy.loginfo(f"generate_mpc_trajectory time: {mpc_generation_time} seconds")
        
        # 每10步或达到100步时绘制时间变化曲线
        print('step_counter',step_counter)
        # if step_counter == 150:
        #     plot_mpc_time_curve()
        
    else:
        # 使用简单的匀加速模型生成局部坐标系下的轨迹
        time_steps = 50
        a = 2  # 加速度
        delta = 0  # 转向角
        x_local, y_local, yaw_local, v_local = generate_trajectory(ego_state, time_steps, a, delta)

    
    # 统一进行轨迹加密
    x_local_dense, y_local_dense, yaw_local_dense, v_local_dense = densify_trajectory(x_local, y_local, yaw_local, v_local, max_distance=0.2)
    
    # 统一将加密后的轨迹从车辆坐标系旋转到全局坐标系
    x_global_dense, y_global_dense = coordinate_transform(x_local_dense, y_local_dense, target_heading=ego_state.heading)
    
    # 计算全局航向角
    yaw_deg_local_dense = np.degrees(yaw_local_dense)
    heading_global_dense = ego_state.heading - yaw_deg_local_dense + 90
    heading_global_dense = (heading_global_dense + 360) % 360
    
    # 4. 一次性将所有点转换为经纬度坐标
    gx_dense, gy_dense = xy_to_latlon_batch(
        ego_state.gx, ego_state.gy,
        x_global_dense, y_global_dense)
    
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
        point.x = x_local_dense[i] * 100
        point.y = y_local_dense[i] * 100
        point.gx = gx_dense[i]
        point.gy = gy_dense[i]
        # 使用加密后的速度数据，如果没有则使用基础速度
        # if 'v_local_dense' in locals() and i < len(v_local_dense):
        #     point.speed = float(v_local_dense[i])
        # else:
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
    
    # 定期保存MPC时间数据（每50步保存一次）
    # if step_counter == 150:
    #     save_mpc_time_data()
    #     rospy.loginfo(f"已完成第 {step_counter} 步规划")
    # rospy.loginfo(f"step_counter: {step_counter}")
    
    return ego_plan, ego_decision
    
def convert_reference_to_local(ego_state, ref_lons, ref_lats):
    """
    convert global latitude and longitude reference line to vehicle local coordinate system reference line
    
    Args:
        ego_state: current vehicle state
        ref_lons: reference line longitude list
        ref_lats: reference line latitude list
    
    Returns:
        local_ref_path: reference path object in local coordinate system
    """
     
    # 确保有足够的参考点
    # if len(ref_lons) < 10 or len(ref_lats) < 10:
    #     rospy.logwarn(f"Not enough reference points: {len(ref_lons)}")
    #     return create_default_path()
    
    # 1. 经纬度坐标系 -> 大地坐标系
    # 创建以车辆位置为中心的投影
    proj_string = f"+proj=tmerc +lon_0={ego_state.gx} +lat_0={ego_state.gy} +ellps=WGS84"
    proj = Proj(proj_string)
    
    # 获取车辆位置在投影坐标系中的坐标
    x0, y0 = proj(ego_state.gx, ego_state.gy)
    
    # 将参考线点批量转换为投影坐标系
    x_global, y_global = proj(np.array(ref_lons), np.array(ref_lats))
    
    # 相对于车辆位置的坐标
    x_global = x_global - x0
    y_global = y_global - y0
    
    # 2. 大地坐标系 -> 车身局部坐标系
    # 注意：heading是车辆朝向与正北方向的夹角
    # theta = np.radians(90 - (360 - ego_state.heading) % 360)  
    # theta = np.radians((90 + ego_state.heading) % 360)
    # theta = np.radians((450 - ego_state.heading) % 360)
    theta = np.radians(ego_state.heading)
    
    # 创建旋转矩阵
    rotation_matrix_inv = np.array([
        [np.cos(theta), -np.sin(theta)], 
        [np.sin(theta), np.cos(theta)]
    ])
    
    # 批量进行坐标变换（转换为厘米）
    points = np.column_stack((x_global, y_global))
    rotated_points = np.dot(points, rotation_matrix_inv.T)
    
    ref_x_local = rotated_points[:, 0]
    ref_y_local = rotated_points[:, 1]
    
    # 选择车辆前方的点（y > 0）
    front_indices = np.where(ref_y_local > 0)[0]
    if len(front_indices) < 10:
        rospy.logwarn(f"Not enough points in front of vehicle: {len(front_indices)}")
        return create_default_path()
    
    # 在前方点中找到最近的点
    front_x = ref_x_local[front_indices]
    front_y = ref_y_local[front_indices]
    front_distances = np.sqrt(front_x**2 + front_y**2)
    min_front_idx = np.argmin(front_distances)
    
    # 选择前后各250个点
    start_idx = max(0, front_indices[min_front_idx] - 100)
    end_idx = min(len(ref_x_local), front_indices[min_front_idx] + 1500)
    
    # 确保有足够的点
    if end_idx - start_idx < 10:
        rospy.logwarn(f"Not enough points in range: {end_idx - start_idx}")
        return create_default_path()
    
    # 提取选定的点
    selected_x = ref_x_local[start_idx:end_idx]
    selected_y = ref_y_local[start_idx:end_idx]
    
    # 降采样
    sample_rate = max(1, len(selected_x) // 100)
    selected_x = selected_x[::sample_rate]
    selected_y = selected_y[::sample_rate]
    
    # 确保点的间距不为零
    valid_x, valid_y = filter_points(selected_x, selected_y)
    
    # 如果有效点太少，使用默认路径
    # if len(valid_x) < 4:
    #     rospy.logwarn(f"Too few valid points ({len(valid_x)}), using default path")
    #     return create_default_path()
    
    # 计算样条曲线
    cx, cy, cyaw, ck, s = cs.calc_spline_course(valid_x, valid_y, ds=P.d_dist)

    
    # 创建参考路径对象
    local_ref_path = PATH(cx, cy, cyaw, ck)
    
    return local_ref_path

def create_default_path():
    """create default path"""
    from MotionPlanning.Control.MPC_XY_Frame import PATH
    
    # 沿车辆前方的直线
    cx = np.array([0, 0, 0, 0, 0, 0, 0])
    cy = np.array([0, 50, 100, 150, 200, 250, 300])
    cyaw = np.array([np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2, np.pi/2])
    ck = np.array([0, 0, 0, 0, 0, 0, 0])
    
    return PATH(cx, cy, cyaw, ck)

def filter_points(x, y, min_dist=10.0):
    """filter points with small distance"""
    valid_x = []
    valid_y = []
    
    if len(x) > 0:
        valid_x.append(x[0])
        valid_y.append(y[0])
        
        for i in range(1, len(x)):
            dist = np.sqrt((x[i] - valid_x[-1])**2 + (y[i] - valid_y[-1])**2)
            if dist > min_dist:
                valid_x.append(x[i])
                valid_y.append(y[i])
    
    return np.array(valid_x), np.array(valid_y)

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
        mpc_controller = MPCController(P.target_speed, initial_state=initial_state)
        rospy.loginfo("MPC controller initialized")
    
    print('initial_state',initial_state)
    
    # 使用MPC控制器生成轨迹
    # try:
    target_ind, x_opt, y_opt, yaw_opt, v_opt = mpc_controller.update(local_ref_path, initial_state, consider_obstacles=True, acc_mode='accelerate',show_plot=False,use_linear_constraints=True)
    # if x_opt is None or y_opt is None or yaw_opt is None or v_opt is None:
    #     rospy.logwarn("MPC optimization failed, falling back to simple trajectory")
    #     # 如果MPC优化失败，使用简单轨迹
    #     return fallback_trajectory(ego_state)
    
    # 将MPC预测轨迹转换为数组格式
    x_local = np.array(x_opt)
    y_local = np.array(y_opt)
    yaw_local = np.array(yaw_opt)
    v_local = np.array(v_opt)
    
    return x_local, y_local, yaw_local, v_local
        
    # except Exception as e:
    #     rospy.logerr(f"Error in MPC trajectory generation: {e}")
    #     return fallback_trajectory(ego_state)

def fallback_trajectory(ego_state):
    """
    fallback trajectory generation method when MPC fails
    
    return local coordinate trajectory
    """
    time_steps = 50
    a = 2  # 加速度
    delta = 0  # 转向角
    
    x_local, y_local, yaw_local, v_local = generate_trajectory(ego_state, time_steps, a, delta)
    
    return x_local, y_local, yaw_local, v_local

def generate_trajectory(ego_state, time_steps=50, a=2, delta=0):
    """generate coarse trajectory based on vehicle coordinate system"""
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

def densify_trajectory(x, y, yaw, v=None, max_distance=0.2):
    """densify trajectory to ensure the distance between adjacent points does not exceed a specified value (vectorized implementation)"""
    # 计算所有相邻点之间的距离
    points = np.column_stack((x, y))
    diff = np.diff(points, axis=0)
    distances = np.sqrt(np.sum(diff**2, axis=1))
    
    # 如果没有需要插值的点，直接返回原始轨迹
    if np.all(distances <= max_distance):
        if v is not None:
            return x, y, yaw, v
        else:
            return x, y, yaw
    
    # 计算每段需要插入的点数
    num_segments = len(distances)
    num_points_to_insert = np.maximum(0, np.ceil(distances / max_distance).astype(int) - 1)
    total_points = num_segments + 1 + np.sum(num_points_to_insert)  # 原始点数 + 插值点数
    
    # 预分配结果数组
    x_dense = np.zeros(total_points)
    y_dense = np.zeros(total_points)
    yaw_dense = np.zeros(total_points)
    if v is not None:
        v_dense = np.zeros(total_points)
    
    # 填充第一个点
    x_dense[0] = x[0]
    y_dense[0] = y[0]
    yaw_dense[0] = yaw[0]
    if v is not None:
        v_dense[0] = v[0]
    
    # 当前填充位置索引
    current_idx = 1
    
    # 批量处理所有段
    for i in range(num_segments):
        # 当前段的起点和终点
        x1, y1, yaw1 = x[i], y[i], yaw[i]
        x2, y2, yaw2 = x[i+1], y[i+1], yaw[i+1]
        if v is not None:
            v1, v2 = v[i], v[i+1]
        
        # 需要插入的点数
        n_insert = num_points_to_insert[i]
        
        if n_insert > 0:
            # 计算插值比例
            ratios = np.linspace(0, 1, n_insert + 2)[1:-1]
            
            # 线性插值
            x_interp = x1 + (x2 - x1) * ratios
            y_interp = y1 + (y2 - y1) * ratios
            yaw_interp = yaw1 + (yaw2 - yaw1) * ratios
            if v is not None:
                v_interp = v1 + (v2 - v1) * ratios
            
            # 填充插值点
            x_dense[current_idx:current_idx + n_insert] = x_interp
            y_dense[current_idx:current_idx + n_insert] = y_interp
            yaw_dense[current_idx:current_idx + n_insert] = yaw_interp
            if v is not None:
                v_dense[current_idx:current_idx + n_insert] = v_interp
            
            current_idx += n_insert
        
        # 填充当前段的终点（除了最后一段，因为最后一点会单独处理）
        if i < num_segments - 1:
            x_dense[current_idx] = x2
            y_dense[current_idx] = y2
            yaw_dense[current_idx] = yaw2
            if v is not None:
                v_dense[current_idx] = v2
            current_idx += 1
    
    # 填充最后一个点
    x_dense[current_idx] = x[-1]
    y_dense[current_idx] = y[-1]
    yaw_dense[current_idx] = yaw[-1]
    if v is not None:
        v_dense[current_idx] = v[-1]
    
    # 如果预分配的空间有多余，裁剪掉
    if current_idx + 1 < total_points:
        x_dense = x_dense[:current_idx + 1]
        y_dense = y_dense[:current_idx + 1]
        yaw_dense = yaw_dense[:current_idx + 1]
        if v is not None:
            v_dense = v_dense[:current_idx + 1]
    
    if v is not None:
        return x_dense, y_dense, yaw_dense, v_dense
    else:
        return x_dense, y_dense, yaw_dense

def plot_mpc_time_curve():
    """绘制MPC轨迹生成时间的变化曲线"""
    global mpc_time_records
    
    if len(mpc_time_records) < 2:
        return
    
    try:
        # 提取数据
        steps = [record['step'] for record in mpc_time_records]
        times = [record['time'] for record in mpc_time_records]
        
        # 创建图形
        plt.figure(figsize=(12, 8))
        
        # 绘制时间变化曲线
        plt.subplot(2, 1, 1)
        plt.plot(steps, times, 'b-', linewidth=2, label='MPC trajectory generation time')
        plt.plot(steps, times, 'ro', markersize=4)
        plt.xlabel('Planning steps')
        plt.ylabel('Time (seconds)')
        plt.title('MPC trajectory generation time change curve')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 添加统计信息
        avg_time = np.mean(times)
        max_time = np.max(times)
        min_time = np.min(times)
        std_time = np.std(times)
        
        plt.axhline(y=avg_time, color='g', linestyle='--', alpha=0.7, label=f'Average time: {avg_time:.4f}s')
        plt.axhline(y=max_time, color='r', linestyle='--', alpha=0.7, label=f'Maximum time: {max_time:.4f}s')
        plt.axhline(y=min_time, color='orange', linestyle='--', alpha=0.7, label=f'Minimum time: {min_time:.4f}s')
        plt.legend()
        
        # 绘制时间直方图
        plt.subplot(2, 1, 2)
        plt.hist(times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency')
        plt.title('MPC trajectory generation time distribution histogram')
        plt.grid(True, alpha=0.3)
        
        # 添加统计文本
        stats_text = f'Statistics:\nAverage time: {avg_time:.4f}s\nStandard deviation: {std_time:.4f}s\nMaximum time: {max_time:.4f}s\nMinimum time: {min_time:.4f}s\nTotal steps: {len(times)}'
        plt.text(0.98, 0.98, stats_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # 保存图片
        timestamp = int(time.time())
        filename = f'mpc_time_analysis_{timestamp}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        rospy.loginfo(f"MPC time analysis image has been saved: {filename}")
        
        # 显示图片（可选，在某些环境下可能不工作）
        # plt.show()
        
        plt.close()
        
    except Exception as e:
        rospy.logerr(f"绘制MPC时间曲线时出错: {e}")

def save_mpc_time_data():
    """将MPC时间数据保存到CSV文件"""
    global mpc_time_records
    
    if not mpc_time_records:
        return
    
    try:
        timestamp = int(time.time())
        filename = f'mpc_time_data_{timestamp}.csv'
        
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['step', 'time', 'timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for record in mpc_time_records:
                writer.writerow(record)
        
        rospy.loginfo(f"MPC时间数据已保存到: {filename}")
        
    except Exception as e:
        rospy.logerr(f"保存MPC时间数据时出错: {e}")

def get_mpc_time_statistics():
    """获取MPC时间统计信息"""
    global mpc_time_records
    
    if not mpc_time_records:
        return None
    
    times = [record['time'] for record in mpc_time_records]
    
    stats = {
        'count': len(times),
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'median': np.median(times),
        'q25': np.percentile(times, 25),
        'q75': np.percentile(times, 75)
    }
    
    return stats

def export_mpc_analysis():
    """导出完整的MPC时间分析"""
    global mpc_time_records
    
    if not mpc_time_records:
        rospy.logwarn("没有MPC时间记录数据")
        return
    
    try:
        # 保存数据到CSV
        save_mpc_time_data()
        
        # 绘制最终的时间分析图
        plot_mpc_time_curve()
        
        # 打印统计信息
        stats = get_mpc_time_statistics()
        if stats:
            rospy.loginfo("=== MPC时间统计信息 ===")
            rospy.loginfo(f"总规划步数: {stats['count']}")
            rospy.loginfo(f"平均时间: {stats['mean']:.4f}s")
            rospy.loginfo(f"标准差: {stats['std']:.4f}s")
            rospy.loginfo(f"最小时间: {stats['min']:.4f}s")
            rospy.loginfo(f"最大时间: {stats['max']:.4f}s")
            rospy.loginfo(f"中位数: {stats['median']:.4f}s")
            rospy.loginfo(f"25%分位数: {stats['q25']:.4f}s")
            rospy.loginfo(f"75%分位数: {stats['q75']:.4f}s")
            rospy.loginfo("====================")
            
    except Exception as e:
        rospy.logerr(f"导出MPC分析时出错: {e}")