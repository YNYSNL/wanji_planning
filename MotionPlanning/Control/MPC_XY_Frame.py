"""
Linear MPC controller (X-Y frame)
author: huiming zhou
"""

import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import casadi as ca
import logging
from datetime import datetime
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import Control.draw as draw
import CurvesGenerator.reeds_shepp as rs
import CurvesGenerator.cubic_spline as cs

class P:

    # System config
    NX = 4  # state vector: z = [x, y, v, phi]
    NU = 2  # input vector: u = [acceleration, steer]
    T = 30  # finite time horizon length that is the same as PLAN_HORIZON

    # MPC config
    Q = np.diag([1.0, 1.0, 1.0, 1.0])  # penalty for states
    Qf = np.diag([1.0, 1.0, 1.0, 1.0])  # penalty for end state
    R = np.diag([0.01, 0.1])  # penalty for inputs
    Rd = np.diag([0.01, 0.1])  # penalty for change of inputs

    dist_stop = 1  # stop permitted when dist to goal < dist_stop
    speed_stop = 0.5 / 3.6  # stop permitted when speed < speed_stop
    time_max = 500.0  # max simulation time
    iter_max = 5  # max iteration
    target_speed = 40.0 / 3.6  # target speed
    N_IND = 50  # search index number
    dt = 0.25  # time step
    d_dist = 1.0  # dist step
    du_res = 0.2  # threshold for stopping iteration

    # vehicle config
    RF = 3.3  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
    W = 2.4  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 2.8  # [m] Wheel base
    TR = 0.44  # [m] Tyre radius
    TW = 0.7  # [m] Tyre width

    steer_max = np.deg2rad(45.0)  # max steering angle [rad]
    steer_change_max = np.deg2rad(30.0)  # maximum steering speed [rad/s]
    speed_max = 55.0 / 3.6  # maximum speed [m/s]
    speed_min = 0.0 / 3.6  # minimum speed [m/s]
    acceleration_max = 3.2  # maximum acceleration [m/s2]

    num_obstacles = 1  # number of obstacles
    obstacles = dict()  # obstacles: {id: [x, y, w, l]}
    obstacle_horizon = 15  # horizon length for obstacle avoidance
    num_modes = 1  # number of modes for Reeds-Shepp path
    treat_obstacles_as_static = False  # treat obstacles as dynamic
    use_linear_obstacle_constraints = True  # use linear constraints for obstacles (faster computation)

    @classmethod
    def init(cls, num_obstacles=1, obstacle_horizon=20, num_modes=1, treat_obstacles_as_static=True, use_linear_obstacle_constraints=False):
        cls.num_obstacles = num_obstacles
        cls.obstacles = dict()
        cls.obstacle_horizon = obstacle_horizon
        cls.num_modes = num_modes
        cls.treat_obstacles_as_static = treat_obstacles_as_static
        cls.use_linear_obstacle_constraints = use_linear_obstacle_constraints

class Node:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0, direct=1.0, gx=0, gy=0, heading=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.direct = direct
        self.gx = gx
        self.gy = gy
        self.heading = heading

    def update(self, a, delta, direct):
        delta = self.limit_input_delta(delta)
        self.x += self.v * math.cos(self.yaw) * P.dt
        self.y += self.v * math.sin(self.yaw) * P.dt
        self.yaw += self.v / P.WB * math.tan(delta) * P.dt
        self.direct = direct
        self.v += self.direct * a * P.dt
        self.v = self.limit_speed(self.v)

    @staticmethod
    def limit_input_delta(delta):
        if delta >= P.steer_max:
            return P.steer_max

        if delta <= -P.steer_max:
            return -P.steer_max

        return delta

    @staticmethod
    def limit_speed(v):
        if v >= P.speed_max:
            return P.speed_max

        if v <= P.speed_min:
            return P.speed_min

        return v


class PATH:
    def __init__(self, cx, cy, cyaw, ck):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ck = ck
        self.length = len(cx)
        self.ind_old = 0
    
    def update(self, cx, cy, cyaw, ck):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ck = ck
        self.ind_old = 0


    def nearest_index(self, node):
        """
        calc index of the nearest node in N steps
        :param node: current information
        :return: nearest index, lateral distance to ref point
        """

        dx = [node.x - x for x in self.cx[self.ind_old:self.ind_old+P.N_IND]]
        dy = [node.y - y for y in self.cy[self.ind_old:self.ind_old+P.N_IND]]
        dist = np.hypot(dx, dy)

        ind_in_N = int(np.argmin(dist))
        ind = self.ind_old + ind_in_N
        self.ind_old = ind

        rear_axle_vec_rot_90 = np.array([[math.cos(node.yaw + math.pi / 2.0)],
                                         [math.sin(node.yaw + math.pi / 2.0)]])

        vec_target_2_rear = np.array([[dx[ind_in_N]],
                                      [dy[ind_in_N]]])

        er = np.dot(vec_target_2_rear.T, rear_axle_vec_rot_90)
        er = er[0][0]

        return ind, er

def calc_ref_trajectory_in_T_step(node, ref_path, sp, acc_mode='accelerate'):
    """
    根据指定模式生成参考轨迹
    acc_mode: 'accelerate' - 匀加速运动
              'decelerate' - 匀减速运动
              'maintain'   - 匀速运动
    """
    z_ref = np.zeros((P.NX, P.T + 1))
    length = ref_path.length
    
    # 设置舒适加速度和减速度（可调整）
    comfort_acc = 3.0  # m/s^2
    comfort_dec = -3.0  # m/s^2
    
    # 获取当前速度
    v0 = abs(node.v)
    
    if acc_mode == 'accelerate':
        t_zero_index =P.T + 1
        # 计算匀加速运动末端速度
        vT = v0 + comfort_acc * P.dt * P.T
        
        # 判断是否需要调整加速度
        if vT > P.target_speed:
            # 如果末端速度超过目标速度，反推所需加速度
            vT = P.target_speed
            acc = (vT - v0) / (P.dt * P.T) if P.dt * P.T > 0 else comfort_acc
        else:
            # 使用舒适加速度
            acc = comfort_acc
            
    elif acc_mode == 'decelerate':
        # 使用舒适减速度
        acc = comfort_dec
        
        # 计算末端速度
        vT = v0 + acc * P.dt * P.T
        
        # 如果末端速度小于最小速度，调整减速度
        if vT < P.speed_min:
            # 计算何时达到最小速度
            t_zero = (P.speed_min - v0) / acc if acc != 0 else float('inf')
            t_zero_index = int(t_zero / P.dt)
        else:
            t_zero_index = P.T + 1  # 不会达到最小速度
            
    else:  # 'maintain'
        # 匀速运动
        acc = 0.0
        vT = v0
        t_zero_index = P.T + 1  # 不会达到最小速度
    
    # 生成速度序列
    v_seq = np.zeros(P.T + 1)
    s_seq = np.zeros(P.T + 1)
    v_seq[0] = v0
    for i in range(1, P.T + 1):
        if i < t_zero_index:
            # 正常减速阶段
            t = P.dt * i
            v_seq[i] = v0 + acc * t
            s_seq[i] = v0 * t + 0.5 * acc * t * t
        else:
            # 达到最小速度后保持
            v_seq[i] = P.speed_min
            t_zero = P.dt * t_zero_index
            s_at_min_speed = v0 * t_zero + 0.5 * acc * t_zero * t_zero  # 达到最小速度时的位移
            additional_t = P.dt * (i - t_zero_index)  # 最小速度行驶的时间
            s_seq[i] = s_at_min_speed + P.speed_min * additional_t
        
        v_seq[i] = max(min(v_seq[i], P.speed_max), P.speed_min)
    
    # 计算累积位移序列
    
    for i in range(1, P.T + 1):
        # s = v0*t + 1/2*a*t^2
        t = P.dt * i
        s_seq[i] = v0 * t + 0.5 * acc * t * t
    
    # 找到参考路径上的对应点
    ind, _ = ref_path.nearest_index(node)
    
    # 设置初始点
    z_ref[0, 0] = ref_path.cx[ind]
    z_ref[1, 0] = ref_path.cy[ind]
    z_ref[2, 0] = v_seq[0]
    z_ref[3, 0] = ref_path.cyaw[ind]

    # 根据累积位移找到对应的参考点
    for i in range(1, P.T + 1):
        # 计算索引增量
        ind_move = int(round(s_seq[i] / P.d_dist))
        index = min(ind + ind_move, length - 1)
        
        # 设置参考点
        z_ref[0, i] = ref_path.cx[index]
        z_ref[1, i] = ref_path.cy[index]
        z_ref[2, i] = v_seq[i]  # 使用计算得到的速度序列
        z_ref[3, i] = ref_path.cyaw[index]
    
    return z_ref, ind


def linear_mpc_control(z_ref, z0, a_old, delta_old, consider_obstacles=True, use_linear_constraints=None):
    """
    linear mpc controller
    :param z_ref: reference trajectory in T steps
    :param z0: initial state vector
    :param a_old: acceleration of T steps from last time
    :param delta_old: delta of T steps from last time
    :param consider_obstacles: whether to consider obstacles
    :param use_linear_constraints: whether to use linear obstacle constraints (None=use P.use_linear_obstacle_constraints)
    :return: acceleration and delta strategy based on current information
    """

    if a_old is None or delta_old is None:
        a_old = [0.0] * P.T
        delta_old = [0.0] * P.T

    # 如果没有指定约束类型，使用全局配置
    if use_linear_constraints is None:
        use_linear_constraints = P.use_linear_obstacle_constraints
    
    # 临时保存全局配置并设置新值
    original_setting = P.use_linear_obstacle_constraints
    P.use_linear_obstacle_constraints = use_linear_constraints

    x, y, yaw, v = None, None, None, None

    # for k in range(P.iter_max):
        
    #     # Predict states in T steps
    #     z_bar = predict_states_in_T_step(z0, a_old, delta_old, z_ref)
    #     a_rec, delta_rec = a_old[:], delta_old[:]
        
    #     # Create and solve MPC problem
    #     mpc_problem = MPCProblem(z_ref, z_bar, z0, delta_old, consider_obstacles)

  
    #     a_old, delta_old, x, y, yaw, v = mpc_problem.solve()

        
    #     if a_old is None:
    #         # If optimization fails, use previous control inputs
    #         logger.warning("Optimization failed, using previous control inputs")
    #         return None, None, None, None, None, None
        
    #     # Calculate maximum control change
    #     du_a_max = max([abs(ia - iao) for ia, iao in zip(a_old, a_rec)])
    #     du_d_max = max([abs(ide - ido) for ide, ido in zip(delta_old, delta_rec)])
        
    #     # If control change is less than threshold, stop iteration
    #     if max(du_a_max, du_d_max) < P.du_res:
    #         logger.info(f"MPC iteration converged, iterations: {k+1}")
    #         break
    z_bar = predict_states_in_T_step(z0, a_old, delta_old, z_ref)
    mpc_problem = MPCProblem(z_ref, z_bar, z0, a_old, delta_old, consider_obstacles)
    a_old, delta_old, x, y, yaw, v = mpc_problem.solve()
    # 恢复全局配置
    P.use_linear_obstacle_constraints = original_setting
    
    return a_old, delta_old, x, y, yaw, v


def predict_states_in_T_step(z0, a, delta, z_ref):
    """
    given the current state, using the acceleration and delta strategy of last time,
    predict the states of vehicle in T steps.
    :param z0: initial state
    :param a: acceleration strategy of last time
    :param delta: delta strategy of last time
    :param z_ref: reference trajectory
    :return: predict states in T steps (z_bar, used for calc linear motion model)
    """

    z_bar = z_ref * 0.0

    for i in range(P.NX):
        z_bar[i, 0] = z0[i]

    node = Node(x=z0[0], y=z0[1], v=z0[2], yaw=z0[3])

    for ai, di, i in zip(a, delta, range(1, P.T + 1)):
        node.update(ai, di, 1.0)
        z_bar[0, i] = node.x
        z_bar[1, i] = node.y
        z_bar[2, i] = node.v
        z_bar[3, i] = node.yaw

    return z_bar


def calc_linear_discrete_model(v, phi, delta):
    """
    calc linear and discrete time dynamic model.
    :param v: speed: v_bar
    :param phi: angle of vehicle: phi_bar
    :param delta: steering angle: delta_bar
    :return: A, B, C
    """

    A = np.array([[1.0, 0.0, P.dt * math.cos(phi), - P.dt * v * math.sin(phi)],
                  [0.0, 1.0, P.dt * math.sin(phi), P.dt * v * math.cos(phi)],
                  [0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, P.dt * math.tan(delta) / P.WB, 1.0]])

    B = np.array([[0.0, 0.0],
                  [0.0, 0.0],
                  [P.dt, 0.0],
                  [0.0, P.dt * v / (P.WB * math.cos(delta) ** 2)]])

    C = np.array([P.dt * v * math.sin(phi) * phi,
                  -P.dt * v * math.cos(phi) * phi,
                  0.0,
                  -P.dt * v * delta / (P.WB * math.cos(delta) ** 2)])

    return A, B, C

# Setup logger
def setup_simple_logger():
    # Check if logger is already configured
    if len(logging.root.handlers) > 0:
        # Logger already configured, return the existing logger
        return logging.getLogger("mpc_logger")
        
    # Create new logger configuration
    log_dir = "mpc_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Use a single timestamp for the entire program run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/mpc_constraints_{timestamp}.log"
    
    # Configure logger with a FileHandler for the log file
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    
    # Set formatter for both handlers
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Create logger
    logger = logging.getLogger("mpc_logger")  # Use a specific name
    logger.setLevel(logging.INFO)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    logger.info(f"MPC optimization log started at {timestamp}")
    logger.info("-" * 40)
    
    return logger

# Create logger instance - ensure it's only created once
logger = setup_simple_logger()

class MPCProblem:
    """
    Linear MPC problem construction and solution
    Encapsulates the MPC problem to make the building and solving process clearer
    """
    def __init__(self, z_ref, z_bar, z0, a_old, delta_old, consider_obstacles=True):
        """
        Initialize MPC problem
        :param z_ref: reference trajectory [x, y, v, yaw]
        :param z_bar: predicted states
        :param z0: initial state
        :param a_old: acceleration from previous time step
        :param delta_old: steering angle from previous time step
        :param consider_obstacles: whether to consider obstacles
        """
        self.z_ref = z_ref
        self.z_bar = z_bar
        self.z0 = z0
        self.a_old = a_old
        self.delta_old = delta_old
        self.consider_obstacles = consider_obstacles
        
        # Create CasADi optimizer
        self.opti = ca.Opti()
        
        # Define variables
        self.z = self.opti.variable(P.NX, P.T + 1)  # state variables
        self.u = self.opti.variable(P.NU, P.T)      # control variables
        
        # Set warm start initial values for control variables
        self.set_warm_start_values()
        
        # Construct problem
        self.construct_problem()
    
    def set_warm_start_values(self):
        """Set warm start initial values for optimization variables"""
        # Set initial guess for control variables using previous solution
        for t in range(P.T):
            if t < len(self.a_old):
                self.opti.set_initial(self.u[0, t], self.a_old[t])  # acceleration
            else:
                self.opti.set_initial(self.u[0, t], 0.0)  # default acceleration
                
            if t < len(self.delta_old):
                self.opti.set_initial(self.u[1, t], self.delta_old[t])  # steering
            else:
                self.opti.set_initial(self.u[1, t], 0.0)  # default steering
        
        # Set initial guess for state variables using predicted trajectory
        for t in range(P.T + 1):
            for i in range(P.NX):
                self.opti.set_initial(self.z[i, t], self.z_bar[i, t])
    
    def construct_problem(self):
        """Construct MPC problem"""
        self.define_objective()
        self.add_initial_state_constraint()
        self.add_dynamics_constraints()
        self.add_state_constraints()
        self.add_control_constraints()

        if self.consider_obstacles and P.obstacles is not None:
            if P.use_linear_obstacle_constraints:
                self.add_linear_obstacle_constraints()
                logger.info("使用线性化障碍物约束")
            else:
                self.add_obstacle_constraints()
                logger.info("使用非线性障碍物约束")
    
    def define_objective(self):
        """Define objective function - optimized vectorized version"""
        
        # Vectorized state error cost - replaces T individual calculations
        state_errors = self.z[:, :P.T] - self.z_ref[:, :P.T]  # Shape: (4, T)
        Q_ca = ca.DM(P.Q)
        state_cost = ca.trace(ca.mtimes(ca.mtimes(state_errors.T, Q_ca), state_errors))
        
        # Vectorized control cost - replaces T individual calculations  
        R_ca = ca.DM(P.R)
        control_cost = ca.trace(ca.mtimes(ca.mtimes(self.u[:, :P.T].T, R_ca), self.u[:, :P.T]))
        
        # Vectorized control rate cost - replaces (T-1) individual calculations
        control_rate_cost = 0.0
        if P.T > 1:
            du = self.u[:, 1:P.T] - self.u[:, 0:P.T-1]  # Shape: (2, T-1)
            Rd_ca = ca.DM(P.Rd)
            control_rate_cost = ca.trace(ca.mtimes(ca.mtimes(du.T, Rd_ca), du))
        
        # Terminal cost
        final_state_error = self.z[:, P.T] - self.z_ref[:, P.T]
        Qf_ca = ca.DM(P.Qf)
        terminal_cost = ca.mtimes(ca.mtimes(final_state_error.T, Qf_ca), final_state_error)
        
        # Total cost
        total_cost = state_cost + control_cost + control_rate_cost + terminal_cost
        
        # Set objective
        self.opti.minimize(total_cost)
    
    def add_initial_state_constraint(self):
        """Add initial state constraint"""
        self.opti.subject_to(self.z[:, 0] == ca.DM(self.z0))
    
    def add_dynamics_constraints(self):
        """Add dynamics constraints - optimized batch version"""
        # Pre-compute all linear models in batch to reduce function call overhead
        A_matrices = []
        B_matrices = []
        C_vectors = []
        
        # Batch computation of linear discrete models
        for t in range(P.T):
            A, B, C = calc_linear_discrete_model(self.z_bar[2, t], self.z_bar[3, t], self.delta_old[t])
            A_matrices.append(ca.DM(A))
            B_matrices.append(ca.DM(B))
            C_vectors.append(ca.DM(C))
        
        # Vectorized constraint addition - more efficient than individual constraints
        for t in range(P.T):
            # Add dynamics constraint using pre-computed matrices
            self.opti.subject_to(self.z[:, t+1] == 
                               ca.mtimes(A_matrices[t], self.z[:, t]) + 
                               ca.mtimes(B_matrices[t], self.u[:, t]) + 
                               C_vectors[t])
    
    def add_state_constraints(self):
        """Add state constraints"""
        # Speed constraints (compatible with older CasADi versions)
        self.opti.subject_to(self.z[2, :] >= P.speed_min)
        self.opti.subject_to(self.z[2, :] <= P.speed_max)
    
    def add_control_constraints(self):
        """Add control constraints - optimized vectorized version (CasADi compatible)"""
        # Vectorized acceleration constraints - replaces 2*T individual constraints
        self.opti.subject_to(self.u[0, :] >= -P.acceleration_max)
        self.opti.subject_to(self.u[0, :] <= P.acceleration_max)
        
        # Vectorized steering constraints - replaces 2*T individual constraints  
        self.opti.subject_to(self.u[1, :] >= -P.steer_max)
        self.opti.subject_to(self.u[1, :] <= P.steer_max)
        
        # Vectorized steering rate constraints - replaces 2*(T-1) individual constraints
        if P.T > 1:
            du_steer = self.u[1, 1:] - self.u[1, :-1]  # Calculate steering differences
            max_steer_change = P.steer_change_max * P.dt
            self.opti.subject_to(du_steer >= -max_steer_change)
            self.opti.subject_to(du_steer <= max_steer_change)
    
    def add_obstacle_constraints(self):
        """Add adaptive obstacle constraints: follow dynamic obstacles, avoid static obstacles"""
        treat_as_static = P.treat_obstacles_as_static
        for t in range(min(P.T, P.obstacle_horizon)):
            robot_pos = self.z_bar[:2, t]
            robot_yaw = self.z_bar[3, t]  # Vehicle heading angle
            
            # Calculate threat score for all obstacles
            threat_obstacles = []
            for obs_id, obstacle in P.obstacles.items():
                # Get obstacle initial position and velocity
                obs_pos_t0 = obstacle.positions[0][0]
                
                # Determine if obstacle is dynamic or static
                # Use the obstacle's built-in dynamic/static classification
                is_dynamic = getattr(obstacle, 'is_dynamic', False)
                v_magnitude = getattr(obstacle, 'v', 0.0)
                
                # Calculate predicted obstacle position at time t
                if treat_as_static or not is_dynamic:
                    # Static obstacle: position doesn't change
                    obs_pos = obs_pos_t0
                else:
                    # Dynamic obstacle: predict position using constant velocity model
                    if hasattr(obstacle, 'vx') and hasattr(obstacle, 'vy'):
                        # Use actual velocity components if available
                        vx = obstacle.vx
                        vy = obstacle.vy
                    else:
                        # Fallback: assume stationary obstacle
                        vx, vy = 0.0, 0.0
                    
                    # Predict position at time t using constant velocity model
                    dt = P.dt * t  # Time from current moment
                    obs_pos = obs_pos_t0 + np.array([vx * dt, vy * dt])
                
                # Calculate relative position vector
                relative_pos = obs_pos - robot_pos
                
                # Vehicle forward direction unit vector
                forward_vec = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
                lateral_vec = np.array([-np.sin(robot_yaw), np.cos(robot_yaw)])
                
                # Project relative position onto vehicle coordinate system
                longitudinal_dist = np.dot(relative_pos, forward_vec)  # Forward distance
                lateral_dist = abs(np.dot(relative_pos, lateral_vec))   # Lateral distance
                
                # Only consider obstacles in front of the vehicle
                if longitudinal_dist <= 0:
                    continue
                
                # Classify obstacle type based on lateral distance and motion state
                if lateral_dist <= 2.5:  # Same lane
                    if is_dynamic:
                        # Dynamic obstacle in same lane - use car following logic
                        if longitudinal_dist > 50:  # Too far ahead to matter
                            continue
                        threat_score = 100.0 - longitudinal_dist  # Higher priority for closer vehicles
                        vehicle_type = "same_lane_dynamic"
                    else:
                        # Static obstacle in same lane - use avoidance logic
                        if longitudinal_dist > 30:  # Shorter range for static obstacles
                            continue
                        threat_score = 50.0 - longitudinal_dist  # High priority for avoidance
                        vehicle_type = "same_lane_static"
                elif lateral_dist <= 4.5:  # Adjacent lane
                    if longitudinal_dist > 35:
                        continue
                    threat_score = 150.0 + longitudinal_dist + lateral_dist  # Lower priority
                    vehicle_type = "adjacent_lane"
                else:  # Far lateral distance
                    if longitudinal_dist > 25:
                        continue  
                    threat_score = 200.0 + longitudinal_dist + lateral_dist  # Lowest priority
                    vehicle_type = "distant"
                
                threat_obstacles.append((threat_score, obs_id, obs_pos, longitudinal_dist, lateral_dist, vehicle_type, is_dynamic))
            
            # Sort by threat score (lower score = higher threat)
            threat_obstacles.sort(key=lambda x: x[0])
            
            # Consider obstacles based on type
            max_obstacles = min(4, len(threat_obstacles))
            closest_obstacles = threat_obstacles[:max_obstacles]
            
            # Add constraints for threatening obstacles
            for threat_score, obs_id, obs_pos, long_dist, lat_dist, vehicle_type, is_dynamic in closest_obstacles:
                # Calculate avoidance direction
                diff = robot_pos - obs_pos
                euclidean_dist = np.linalg.norm(diff)
                
                # Avoid numerical issues
                if euclidean_dist >= 0.001:
                    if vehicle_type == "same_lane_dynamic":
                        # Car following constraint: maintain safe following distance
                        # Use longitudinal constraint only (allow lateral freedom for lane changing)
                        forward_vec_ca = ca.DM([np.cos(robot_yaw), np.sin(robot_yaw)])
                        
                        # Calculate longitudinal separation
                        relative_pos_ca = ca.DM([self.z[0, t] - obs_pos[0], self.z[1, t] - obs_pos[1]])
                        longitudinal_separation = forward_vec_ca.T @ relative_pos_ca
                        
                        # Safe following distance based on speed and time gap
                        robot_speed = self.z_bar[2, t]  # Current robot speed
                        time_gap = 1.5  # seconds
                        min_distance = 3.0  # minimum distance in meters
                        safe_following_distance = max(min_distance, robot_speed * time_gap)
                        
                        # Add following constraint (maintain distance behind obstacle)
                        self.opti.subject_to(-longitudinal_separation >= safe_following_distance)
                        
                        # logger.info(f"跟驰约束 t={t}, obs_id={obs_id}, long_dist={long_dist:.1f}m, safe_dist={safe_following_distance:.1f}m")
                        
                    elif vehicle_type == "same_lane_static":
                        # Static obstacle avoidance: strong lateral avoidance constraint
                        a = diff / euclidean_dist  # unit vector from obstacle to robot
                        
                        # Larger safe distance for static obstacles to encourage avoidance
                        base_safe_distance = 6.0  # Increased for static obstacles
                        distance_factor = max(1.2, (25.0 - long_dist) / 20.0)
                        safe_distance = base_safe_distance * distance_factor
                        
                        # Convert to CasADi DM
                        a_ca = ca.DM(a)
                        obs_pos_ca = ca.DM(obs_pos)
                        
                        # Add strong avoidance constraint
                        constraint_expr = a_ca[0] * (self.z[0, t] - obs_pos_ca[0]) + a_ca[1] * (self.z[1, t] - obs_pos_ca[1])
                        self.opti.subject_to(constraint_expr >= safe_distance)
                        
                        # logger.info(f"静态避障约束 t={t}, obs_id={obs_id}, long_dist={long_dist:.1f}m, safe_dist={safe_distance:.1f}m")
                        
                    else:
                        # Other obstacles: standard avoidance
                        a = diff / euclidean_dist  # unit vector from obstacle to robot
                        
                        # Standard safe distance
                        if vehicle_type == "adjacent_lane":
                            base_safe_distance = 3.0
                            distance_factor = max(0.7, (15.0 - long_dist) / 15.0)
                            safe_distance = base_safe_distance * distance_factor
                        else:  # distant
                            safe_distance = 2.0
                        
                        # Convert to CasADi DM
                        a_ca = ca.DM(a)
                        obs_pos_ca = ca.DM(obs_pos)
                        
                        # Add avoidance constraint
                        constraint_expr = a_ca[0] * (self.z[0, t] - obs_pos_ca[0]) + a_ca[1] * (self.z[1, t] - obs_pos_ca[1])
                        self.opti.subject_to(constraint_expr >= safe_distance)
                        
                        # logger.info(f"标准避障约束 t={t}, obs_id={obs_id}, type={vehicle_type}, long_dist={long_dist:.1f}m, safe_dist={safe_distance:.1f}m")
    
    def add_linear_obstacle_constraints(self):
        """Add linearized adaptive obstacle constraints: follow dynamic obstacles, avoid static obstacles"""
        treat_as_static = P.treat_obstacles_as_static
        
        for t in range(min(P.T, P.obstacle_horizon)):
            robot_pos = self.z_bar[:2, t]
            robot_yaw = self.z_bar[3, t]  # Vehicle heading angle
            
            # Calculate threat score for all obstacles using predicted trajectory
            threat_obstacles = []
            for obs_id, obstacle in P.obstacles.items():
                # Get obstacle initial position
                obs_pos_t0 = obstacle.positions[0][0]
                
                # Determine if obstacle is dynamic or static
                # Use the obstacle's built-in dynamic/static classification
                is_dynamic = getattr(obstacle, 'is_dynamic', False)
                v_magnitude = getattr(obstacle, 'v', 0.0)
                
                # Calculate predicted obstacle position at time t
                if treat_as_static or not is_dynamic:
                    # Static obstacle: position doesn't change
                    obs_pos = obs_pos_t0
                else:
                    # Dynamic obstacle: predict position using constant velocity model
                    if hasattr(obstacle, 'vx') and hasattr(obstacle, 'vy'):
                        # Use actual velocity components if available
                        vx = obstacle.vx
                        vy = obstacle.vy
                    else:
                        # Fallback: assume stationary obstacle
                        vx, vy = 0.0, 0.0
                    
                    # Predict position at time t
                    dt = P.dt * t
                    obs_pos = obs_pos_t0 + np.array([vx * dt, vy * dt])
                
                # Calculate relative position vector
                relative_pos = obs_pos - robot_pos
                
                # Vehicle forward direction unit vector
                forward_vec = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
                lateral_vec = np.array([-np.sin(robot_yaw), np.cos(robot_yaw)])
                
                # Project relative position onto vehicle coordinate system
                longitudinal_dist = np.dot(relative_pos, forward_vec)  # Forward distance
                lateral_dist = abs(np.dot(relative_pos, lateral_vec))   # Lateral distance
                
                # Only consider obstacles in front of the vehicle
                if longitudinal_dist <= 0:
                    continue
                    
                # Calculate threat score based on obstacle type and motion state
                if lateral_dist <= 2.5:  # Same lane
                    if is_dynamic:
                        # Dynamic obstacle - lower priority for linear constraints (handled by car following)
                        threat_score = 200.0 + longitudinal_dist
                        vehicle_type = "same_lane_dynamic"
                    else:
                        # Static obstacle - high priority for avoidance
                        threat_score = longitudinal_dist + lateral_dist
                        vehicle_type = "same_lane_static"
                else:
                    # Other lanes
                    threat_score = longitudinal_dist + 2.0 * lateral_dist
                    vehicle_type = "other_lane"
                
                # Filter based on threat criteria
                if (longitudinal_dist > 35 or           
                    lateral_dist > 4.5 or               
                    threat_score > 45):                 
                    continue
                
                threat_obstacles.append((threat_score, obs_id, obs_pos, longitudinal_dist, lateral_dist, vehicle_type, is_dynamic))
            
            # Sort by threat score and get most threatening obstacles
            threat_obstacles.sort(key=lambda x: x[0])
            # Prioritize static obstacles for linear constraints
            static_obstacles = [obs for obs in threat_obstacles if not obs[6]]  # is_dynamic = False
            dynamic_obstacles = [obs for obs in threat_obstacles if obs[6]]     # is_dynamic = True
            
            # Consider static obstacles first, then dynamic if no static obstacles
            if static_obstacles:
                closest_obstacles = static_obstacles[:1]  # Focus on closest static obstacle
            elif dynamic_obstacles:
                closest_obstacles = dynamic_obstacles[:1]  # Fallback to dynamic obstacle
            else:
                closest_obstacles = []
            
            # Add linear constraints for threatening obstacles
            for threat_score, obs_id, obs_pos, long_dist, lat_dist, vehicle_type, is_dynamic in closest_obstacles:
                # Calculate avoidance direction based on predicted trajectory
                diff = robot_pos - obs_pos
                euclidean_dist = np.linalg.norm(diff)
                
                # Avoid numerical issues
                if euclidean_dist >= 0.001:
                    if vehicle_type == "same_lane_dynamic":
                        # For dynamic obstacles, use lighter constraint (mainly handled by nonlinear constraints)
                        forward_vec_ca = ca.DM([np.cos(robot_yaw), np.sin(robot_yaw)])
                        relative_pos_ca = ca.DM([self.z[0, t] - obs_pos[0], self.z[1, t] - obs_pos[1]])
                        longitudinal_separation = forward_vec_ca.T @ relative_pos_ca
                        
                        # Lighter following constraint for linear approximation
                        safe_following_distance = 2.0  # Reduced for linear constraints
                        self.opti.subject_to(-longitudinal_separation >= safe_following_distance)
                        
                        # logger.info(f"线性跟驰约束 t={t}, obs_id={obs_id}, long_dist={long_dist:.2f}m")
                        
                    else:
                        # For static obstacles, use strong avoidance constraint
                        a_fixed = diff / euclidean_dist  # unit vector from obstacle to robot
                        
                        # Adaptive safe distance for linear constraints
                        if vehicle_type == "same_lane_static":
                            base_safe_distance = 2.5  # Larger for static obstacles
                            lateral_factor = max(0.8, (4.5 - lat_dist) / 4.5)
                        else:
                            base_safe_distance = 2.5
                            lateral_factor = max(0.6, (4.5 - lat_dist) / 4.5)
                        
                        safe_distance = base_safe_distance * lateral_factor
                        
                        # Convert to CasADi DM
                        a_ca = ca.DM(a_fixed)
                        obs_pos_ca = ca.DM(obs_pos)
                        
                        # Add linearized constraint
                        constraint_expr = a_ca[0] * (self.z[0, t] - obs_pos_ca[0]) + a_ca[1] * (self.z[1, t] - obs_pos_ca[1])
                        self.opti.subject_to(constraint_expr >= safe_distance)
                        
                        # logger.info(f"线性避障约束 t={t}, obs_id={obs_id}, type={vehicle_type}, long_dist={long_dist:.2f}m, safe_dist={safe_distance:.1f}m")

    
    def setup_solver(self):
        """Setup solver options"""
        p_opts = {
            "expand": True,
            "print_time": False,
            "verbose": False
        }
        s_opts = {
            "max_iter": 100,
            "tol": 1e-5,
            "print_level": 0,
            "warm_start_init_point": "yes"  # Enable warm start
        }
        # s_opts = {
        #             "max_iter": 500,
        #             "tol": 1e-3,
        #             "print_level": 0,
        #             "warm_start_init_point": "yes",  # Enable warm start
        #             # config for faster convergence
        #             "acceptable_tol": 1e-4,
        #             "acceptable_iter": 10,
        #             "acceptable_obj_change_tol": 1e-4,
        #             "max_cpu_time": 10.0,  # Set a reasonable timeout
        #             "max_iter": 1000,  # Increase max iterations for robustness
        #             "hessian_approximation": "limited-memory",  # Use limited-memory BFGS for better performance
        #         }    
        self.opti.solver("ipopt", p_opts, s_opts)
    
    def solve(self):
        """Solve MPC problem"""
        self.setup_solver()
        
        try:
            sol = self.opti.solve()
            
            # Extract solution
            z_opt = sol.value(self.z)
            u_opt = sol.value(self.u)
            
            # Extract individual components
            x = z_opt[0, :]
            y = z_opt[1, :]
            v = z_opt[2, :]
            yaw = z_opt[3, :]
            a = u_opt[0, :]
            delta = u_opt[1, :]
            
            # Verify initial state constraint
            initial_error = np.linalg.norm(z_opt[:, 0] - self.z0)
            # logger.info(f"Initial state error: {initial_error}")
            
            # logger.info("Optimization successful")
            return a, delta, x, y, yaw, v
            
        except Exception as e:
            # 详细记录优化失败的原因
            err_msg = str(e)
            if "infeasible" in err_msg.lower():
                logger.error(f"MPC无解：约束不可满足 - {err_msg}")
                logger.error("可能存在避障约束无法满足，建议使用紧急制动")
            elif "convergence" in err_msg.lower():
                logger.error(f"MPC未收敛：优化求解器未收敛 - {err_msg}")
                logger.error("优化器在最大迭代次数内未收敛，建议使用紧急制动")
            elif "timeout" in err_msg.lower():
                logger.error(f"MPC超时：计算时间过长 - {err_msg}")
            else:
                logger.error(f"MPC优化错误：{err_msg}")
            
            # 记录当前车辆状态和障碍物信息
            if P.obstacles:
                logger.error(f"当前有 {len(P.obstacles)} 个障碍物")
                for obs_id, obstacle in P.obstacles.items():
                    if hasattr(obstacle, 'positions') and len(obstacle.positions) > 0 and len(obstacle.positions[0]) > 0:
                        obs_pos = obstacle.positions[0][0]
                        robot_pos = self.z0[:2]
                        dist = np.linalg.norm(robot_pos - obs_pos)
                        logger.error(f"障碍物 {obs_id}: 位置=({obs_pos[0]:.2f}, {obs_pos[1]:.2f}), 距离={dist:.2f}m")
            
            # 记录详细的调试信息
            self.log_debug_info()
            return None, None, None, None, None, None
    
    def log_debug_info(self):
        """Log debug information"""
        logger.info("Logging constraints at failure:")
        
        try:
            # Get current iteration variable values
            z_val = self.opti.debug.value(self.z)
            u_val = self.opti.debug.value(self.u)
            
            # Log optimization variables
            logger.info(f"State variables (z):")
            logger.info(f"  Shape: {z_val.shape}")
            logger.info(f"  Initial state: {z_val[:, 0]}")
            logger.info(f"  Final state: {z_val[:, -1]}")
            
            logger.info(f"Control variables (u):")
            logger.info(f"  Shape: {u_val.shape}")
            logger.info(f"  Initial control: {u_val[:, 0]}")
            logger.info(f"  Final control: {u_val[:, -1]}")
            
            # Check constraint violations
            all_constraints = self.opti.g
            num_constraints = all_constraints.size1()
            
            for j in range(min(num_constraints, 10)):  # Only show first 10 constraints
                try:
                    con = all_constraints[j]
                    lhs_value = self.opti.debug.value(con.dep(0))
                    rhs_value = self.opti.debug.value(con.dep(1))
                    violation = abs(lhs_value - rhs_value)
                    
                    if violation > 1e-4:
                        logger.warning(f"Constraint {j} violated: |{lhs_value} - {rhs_value}| = {violation}")
                except Exception as e:
                    logger.error(f"Error evaluating constraint violation {j}: {e}")
            
            # Try to extract last iteration results
            return z_val[0, :], z_val[1, :], z_val[2, :], z_val[3, :], u_val[0, :], u_val[1, :]
            
        except Exception as e2:
            logger.error(f"Could not extract last iterate: {str(e2)}")
            return None, None, None, None, None, None


def calc_speed_profile(cx, cy, cyaw, target_speed):
    """
    design appropriate speed strategy
    :param cx: x of reference path [m]
    :param cy: y of reference path [m]
    :param cyaw: yaw of reference path [m]
    :param target_speed: target speed [m/s]
    :return: speed profile
    """

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = - target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile


def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi

    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle


class MPCController:
    def __init__(self, target_speed, initial_state, treat_obstacles_as_static=True):
        """
        Initialize MPC controller
        :param target_speed: target speed
        :param initial_state: initial state [x, y, yaw, v]
        """
        self.ref_path = None
        self.target_speed = target_speed
        self.node = Node(
            x=initial_state[0], y=initial_state[1], yaw=initial_state[2], v=initial_state[3])
        # self.obstacles = obstacles if obstacles is not None else []
        self.x, self.y, self.yaw, self.v, self.t, self.d, self.a = ([self.node.x],
                                                                    [self.node.y],
                                                                    [self.node.yaw],
                                                                    [self.node.v],
                                                                    [0.0],
                                                                    [0.0],
                                                                    [0.0])
        self.delta_opt, self.a_opt = None, None
        self.a_exc, self.delta_exc = 0.0, 0.0

        P.treat_obstacles_as_static = treat_obstacles_as_static
        
        # Debug information
        self.debug_info = {
            'iterations': [],
            'cost_values': [],
            'constraint_violations': []
        }
        
        # 多帧规划相关属性
        self.frame_counter = 0            # 帧计数器
        self.frame_update_rate = 1        # 每隔多少帧重新规划
        self.control_index = 0            # 当前执行的控制量索引
        self.stored_controls = None       # 存储规划得到的控制量序列 [a_opt, delta_opt]

    def update(self, ref_path, initial_state=None, consider_obstacles=True, acc_mode='accelerate', show_plot=False, use_linear_constraints=None):
        """
        Update MPC controller
        :param ref_path: reference path
        :param initial_state: initial state, if None use current state
        :param consider_obstacles: whether to consider obstacles
        :param acc_mode: acceleration mode ('accelerate', 'decelerate', 'maintain')
        :param show_plot: whether to display plots
        :param use_linear_constraints: whether to use linear obstacle constraints (None=use P.use_linear_obstacle_constraints)
        :return: target index and optimized trajectory
        """
        self.ref_path = ref_path
        if initial_state is not None:
            self.node = Node(
                x=initial_state[0], y=initial_state[1], yaw=initial_state[2], v=initial_state[3])
        
        # 获取当前状态
        z0 = [self.node.x, self.node.y, self.node.v, self.node.yaw]
        
        # 增加帧计数器
        self.frame_counter += 1
        
        # 是否需要重新规划
        need_replan = (self.frame_counter >= self.frame_update_rate) or (self.stored_controls is None)
        
        if need_replan:
            # 重置计数器和控制索引
            self.frame_counter = 0
            self.control_index = 0

          
            # 计算参考轨迹
            speed_profile = calc_speed_profile(self.ref_path.cx, self.ref_path.cy, 
                                             self.ref_path.cyaw, self.target_speed)
            z_ref, target_ind = calc_ref_trajectory_in_T_step(self.node, self.ref_path,
                                                            speed_profile, acc_mode=acc_mode)
                      
            # 执行MPC优化，得到控制序列
            self.a_opt, self.delta_opt, x_opt, y_opt, yaw_opt, v_opt = linear_mpc_control(
                z_ref, z0, self.a_opt, self.delta_opt, 
                consider_obstacles=consider_obstacles, use_linear_constraints=use_linear_constraints
            )

            # 存储优化结果供后续帧使用
            if self.a_opt is not None and self.delta_opt is not None:
                self.stored_controls = {
                    'a': self.a_opt,
                    'delta': self.delta_opt,
                    'z_ref': z_ref,
                    'target_ind': target_ind,
                    'x_init': x_opt[0] if x_opt is not None else self.node.x,
                    'y_init': y_opt[0] if y_opt is not None else self.node.y,
                    'yaw_init': yaw_opt[0] if yaw_opt is not None else self.node.yaw,
                    'v_init': v_opt[0] if v_opt is not None else self.node.v,
                    'full_states': {
                        'x': x_opt,
                        'y': y_opt,
                        'yaw': yaw_opt,
                        'v': v_opt
                    }
                }
                
                logger.info(f"MPC replanned, stored {len(self.a_opt)} control points")
            else:
                logger.warning("MPC optimization failed, using emergency deceleration (-2.0 m/s²)")
                # 设置减速度为-2.0，方向盘保持不变
                self.a_exc = -2.0  # 紧急减速度
                self.delta_exc = 0.0  # 保持方向盘不变
                
                # 计算紧急减速轨迹，用于显示
                emergency_node = Node(
                    x=self.node.x, y=self.node.y, 
                    yaw=self.node.yaw, v=self.node.v
                )
                
                # 预测时间步数
                pred_steps = 10
                
                # 初始化预测轨迹数组
                x_pred = [emergency_node.x]
                y_pred = [emergency_node.y]
                yaw_pred = [emergency_node.yaw]
                v_pred = [emergency_node.v]
                
                # 使用紧急减速进行预测
                for i in range(1, pred_steps):
                    # 更新车辆状态，使用紧急减速度
                    emergency_node.update(self.a_exc, self.delta_exc, 1.0)
                    
                    # 记录状态
                    x_pred.append(emergency_node.x)
                    y_pred.append(emergency_node.y)
                    yaw_pred.append(emergency_node.yaw)
                    v_pred.append(emergency_node.v)
                    
                    # 如果速度接近零，则停止预测
                    if emergency_node.v <= 0.1:
                        break
                
                # 使用预测的轨迹作为输出
                x_opt = x_pred
                y_opt = y_pred
                yaw_opt = yaw_pred
                v_opt = v_pred
                
                # 设置参考轨迹和目标索引，用于显示
                speed_profile = calc_speed_profile(self.ref_path.cx, self.ref_path.cy, 
                                                 self.ref_path.cyaw, self.target_speed)
                z_ref, target_ind = calc_ref_trajectory_in_T_step(self.node, self.ref_path,
                                                               speed_profile, acc_mode='decelerate')
                
                logger.warning(f"Emergency brake trajectory generated with {len(x_opt)} points, starting speed: {v_opt[0]:.2f} m/s")
        else:
            # 不需要重新规划，使用上一次规划的控制序列
            if self.stored_controls is not None and self.control_index < len(self.stored_controls['a']):
                # 从存储的控制序列中获取当前应执行的控制量
                self.a_exc = self.stored_controls['a'][self.control_index]
                self.delta_exc = self.stored_controls['delta'][self.control_index]
                
                logger.info(f"Using stored control at index {self.control_index}, a={self.a_exc:.2f}, delta={self.delta_exc:.2f}")
                
                # 获取参考轨迹和目标索引
                z_ref = self.stored_controls['z_ref']
                target_ind = self.stored_controls['target_ind']
                
                # 计算剩余控制量序列
                remaining_indices = list(range(self.control_index, len(self.stored_controls['a'])))
                remaining_a = [self.stored_controls['a'][i] for i in remaining_indices]
                remaining_delta = [self.stored_controls['delta'][i] for i in remaining_indices]
                
                # 使用当前状态和剩余控制量序列预测未来状态
                # 创建模拟节点，从当前状态开始
                sim_node = Node(
                    x=self.node.x, y=self.node.y, 
                    yaw=self.node.yaw, v=self.node.v
                )
                
                # 初始化预测轨迹数组
                x_pred = [sim_node.x]
                y_pred = [sim_node.y]
                yaw_pred = [sim_node.yaw]
                v_pred = [sim_node.v]
                
                # 使用剩余的控制量序列向前模拟
                for i in range(len(remaining_a)):
                    # 使用剩余控制量更新模拟节点
                    sim_node.update(remaining_a[i], remaining_delta[i], 1.0)
                    
                    # 记录状态
                    x_pred.append(sim_node.x)
                    y_pred.append(sim_node.y)
                    yaw_pred.append(sim_node.yaw)
                    v_pred.append(sim_node.v)
                
                # 如果预测的轨迹点太少，则向前多预测几步
                last_a = remaining_a[-1] if remaining_a else 0.0
                last_delta = remaining_delta[-1] if remaining_delta else 0.0
                
                # 确保至少有10个预测点
                min_pred_points = 10
                while len(x_pred) < min_pred_points:
                    sim_node.update(last_a, last_delta, 1.0)
                    x_pred.append(sim_node.x)
                    y_pred.append(sim_node.y)
                    yaw_pred.append(sim_node.yaw)
                    v_pred.append(sim_node.v)
                
                # 使用预测的轨迹作为输出
                x_opt = x_pred
                y_opt = y_pred
                yaw_opt = yaw_pred
                v_opt = v_pred
                
                # 更新控制索引，为下一帧准备
                self.control_index += 1
            else:
                # 如果控制序列已用完但还未到重规划时间，保持最后一个控制量
                logger.warning("Control sequence exhausted before replan, using last control")
                
                # 获取最后一个控制量
                if self.stored_controls is not None and len(self.stored_controls['a']) > 0:
                    self.a_exc = self.stored_controls['a'][-1]
                    self.delta_exc = self.stored_controls['delta'][-1]
                # 否则保持不变
                
                # 虽然不重新规划，但仍需计算参考轨迹用于显示
                speed_profile = calc_speed_profile(self.ref_path.cx, self.ref_path.cy, 
                                                 self.ref_path.cyaw, self.target_speed)
                z_ref, target_ind = calc_ref_trajectory_in_T_step(self.node, self.ref_path,
                                                               speed_profile, acc_mode=acc_mode)
                
                # 使用当前状态和最后的控制量向前预测
                sim_node = Node(
                    x=self.node.x, y=self.node.y, 
                    yaw=self.node.yaw, v=self.node.v
                )
                
                # 预测时间步数
                pred_steps = 10
                
                # 初始化预测轨迹数组
                x_pred = [sim_node.x]
                y_pred = [sim_node.y]
                yaw_pred = [sim_node.yaw]
                v_pred = [sim_node.v]
                
                # 使用最后的控制量向前预测
                for i in range(1, pred_steps):
                    sim_node.update(self.a_exc, self.delta_exc, 1.0)
                    x_pred.append(sim_node.x)
                    y_pred.append(sim_node.y)
                    yaw_pred.append(sim_node.yaw)
                    v_pred.append(sim_node.v)
                
                # 使用预测的轨迹作为输出
                x_opt = x_pred
                y_opt = y_pred
                yaw_opt = yaw_pred
                v_opt = v_pred

        
        # Update vehicle state

        self.node.update(self.a_exc, self.delta_exc, 1.0)

       
        # Update trajectory records
        self.x.append(self.node.x)
        self.y.append(self.node.y)
        self.yaw.append(self.node.yaw)
        self.v.append(self.node.v)
        self.v.append(self.node.v)
        
        self.v.append(self.node.v)          
        
        # 绘图部分，根据show_plot参数决定是否显示

        if show_plot:
            self.plot_current_state(z_ref, x_opt, y_opt)

     
        # 确保返回的轨迹非空
        if x_opt is None or y_opt is None or len(x_opt) == 0 or len(y_opt) == 0:
            x_opt = [self.node.x]
            y_opt = [self.node.y]
            yaw_opt = [self.node.yaw]
            v_opt = [self.node.v]
        return target_ind, x_opt, y_opt, yaw_opt, v_opt
    
    def plot_current_state(self, z_ref, x_opt=None, y_opt=None):
        """绘制当前状态和轨迹"""
        plt.ion()  # 开启交互模式
        plt.figure(1, figsize=(10, 8))
        plt.clf()  # 清除当前图形
        
        # 绘制参考线
        plt.plot(self.ref_path.cx, self.ref_path.cy, 'g-', linewidth=1, label='Reference Path')
        
        # 绘制参考轨迹
        if z_ref is not None:
            plt.plot(z_ref[0, :], z_ref[1, :], 'r-', linewidth=2, label='ref_plan')
        
        # 绘制优化轨迹（如果有）
        if x_opt is not None and y_opt is not None:
            plt.plot(x_opt, y_opt, 'b-', linewidth=2, label='MPC Trajectory')
        
        # 绘制当前位置
        plt.scatter(self.node.x, self.node.y, color='black', s=100, marker='*', 
                   label=f'Current Position (v={self.node.v:.2f} m/s)')
        
        # 绘制障碍物
        if P.obstacles:
            for obs_id, obstacle in P.obstacles.items():
                # 获取障碍物位置
                if hasattr(obstacle, 'positions') and len(obstacle.positions) > 0 and len(obstacle.positions[0]) > 0:
                    obs_pos = obstacle.positions[0][0]
                    plt.scatter(obs_pos[0], obs_pos[1], color='red', s=100, marker='x', 
                               label=f'Obstacle {obs_id}' if obs_id == list(P.obstacles.keys())[0] else "")
                    print(f"障碍物 {obs_id} 位置: {obs_pos}")
                    
                    # 绘制障碍物范围（可选）
                    if hasattr(obstacle, 'width') and hasattr(obstacle, 'length'):
                        length = obstacle.width if obstacle.width > 0 else 2.0
                        width = obstacle.length if obstacle.length > 0 else 5.0
                        # 创建矩形表示障碍物
                        rect = plt.Rectangle(
                            (obs_pos[0] - length/2, obs_pos[1] - width/2),
                            length, width, fill=False, color='red', linestyle='--'
                        )
                        plt.gca().add_patch(rect)
        
        # 绘制历史轨迹
        # plt.plot(self.x, self.y, 'b--', linewidth=1, alpha=0.5, label='Actual Path')
        
        plt.legend()
        plt.axis('equal')
        plt.title(f"MPC Planning - Frame {self.frame_counter}/{self.frame_update_rate}")
        plt.grid(True)
        plt.savefig("mpc_trajectory.png")
        plt.draw()
        plt.show(block=False)
        plt.pause(0.0001)
