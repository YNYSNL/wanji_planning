"""
Linear MPC controller (X-Y frame)
author: huiming zhou
"""

import os
import sys
import math
import cvxpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
import casadi as ca
import logging
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import Control.draw as draw
import CurvesGenerator.reeds_shepp as rs
import CurvesGenerator.cubic_spline as cs

class P:

    # System config
    NX = 4  # state vector: z = [x, y, v, phi]
    NU = 2  # input vector: u = [acceleration, steer]
    T = 20  # finite time horizon length that is the same as PLAN_HORIZON

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
    dt = 0.1  # time step
    d_dist = 1.0  # dist step
    du_res = 0.1  # threshold for stopping iteration

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
    acceleration_max = 2.0  # maximum acceleration [m/s2]

    num_obstacles = 1  # number of obstacles
    obstacles = dict()  # obstacles: {id: [x, y, w, h]}
    obstacle_horizon = 20  # horizon length for obstacle avoidance
    num_modes = 1  # number of modes for Reeds-Shepp path

    @classmethod
    def init(cls, num_obstacles=1, obstacle_horizon=20, num_modes=1):
        cls.num_obstacles = num_obstacles
        cls.obstacles = dict()
        cls.obstacle_horizon = obstacle_horizon
        cls.num_modes = num_modes

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
    comfort_acc = 2.0  # m/s^2
    comfort_dec = -2.0  # m/s^2
    
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


def linear_mpc_control(z_ref, z0, a_old, delta_old, consider_obstacles=True):
    """
    linear mpc controller
    :param z_ref: reference trajectory in T steps
    :param z0: initial state vector
    :param a_old: acceleration of T steps from last time
    :param delta_old: delta of T steps from last time
    :param consider_obstacles: whether to consider obstacles
    :return: acceleration and delta strategy based on current information
    """
    if a_old is None or delta_old is None:
        a_old = [0.0] * P.T
        delta_old = [0.0] * P.T

    x, y, yaw, v = None, None, None, None

    for k in range(P.iter_max):
        # Predict states in T steps
        z_bar = predict_states_in_T_step(z0, a_old, delta_old, z_ref)
        a_rec, delta_rec = a_old[:], delta_old[:]
        
        # Create and solve MPC problem
        mpc_problem = MPCProblem(z_ref, z_bar, z0, delta_old, consider_obstacles)
        a_old, delta_old, x, y, yaw, v = mpc_problem.solve()
        
        if a_old is None:
            # If optimization fails, use previous control inputs
            logger.warning("Optimization failed, using previous control inputs")
            return a_rec, delta_rec, None, None, None, None
        
        # Calculate maximum control change
        du_a_max = max([abs(ia - iao) for ia, iao in zip(a_old, a_rec)])
        du_d_max = max([abs(ide - ido) for ide, ido in zip(delta_old, delta_rec)])
        
        # If control change is less than threshold, stop iteration
        if max(du_a_max, du_d_max) < P.du_res:
            logger.info(f"MPC iteration converged, iterations: {k+1}")
            break

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
    def __init__(self, z_ref, z_bar, z0, d_bar, consider_obstacles=True):
        """
        Initialize MPC problem
        :param z_ref: reference trajectory [x, y, v, yaw]
        :param z_bar: predicted states
        :param z0: initial state
        :param d_bar: steering angle from previous time step
        :param consider_obstacles: whether to consider obstacles
        """
        self.z_ref = z_ref
        self.z_bar = z_bar
        self.z0 = z0
        self.d_bar = d_bar
        self.consider_obstacles = consider_obstacles
        
        # Create CasADi optimizer
        self.opti = ca.Opti()
        
        # Define variables
        self.z = self.opti.variable(P.NX, P.T + 1)  # state variables
        self.u = self.opti.variable(P.NU, P.T)      # control variables
        
        # Construct problem
        self.construct_problem()
    
    def construct_problem(self):
        """Construct MPC problem"""
        self.define_objective()
        self.add_initial_state_constraint()
        self.add_dynamics_constraints()
        self.add_state_constraints()
        self.add_control_constraints()
        
        if self.consider_obstacles and P.obstacles is not None:
            self.add_obstacle_constraints()
    
    def define_objective(self):
        """Define objective function"""
        cost = 0.0
        
        # State and control costs
        for t in range(P.T):
            # State error cost
            state_error = self.z[:, t] - self.z_ref[:, t]
            cost += ca.mtimes(ca.mtimes(state_error.T, ca.DM(P.Q)), state_error)
            
            # Control cost
            cost += ca.mtimes(ca.mtimes(self.u[:, t].T, ca.DM(P.R)), self.u[:, t])
            
            # Control rate cost
            if t < P.T - 1:
                du = self.u[:, t+1] - self.u[:, t]
                cost += ca.mtimes(ca.mtimes(du.T, ca.DM(P.Rd)), du)
        
        # Terminal cost
        final_state_error = self.z[:, P.T] - self.z_ref[:, P.T]
        cost += ca.mtimes(ca.mtimes(final_state_error.T, ca.DM(P.Qf)), final_state_error)
        
        # Set objective
        self.opti.minimize(cost)
    
    def add_initial_state_constraint(self):
        """Add initial state constraint"""
        self.opti.subject_to(self.z[:, 0] == ca.DM(self.z0))
    
    def add_dynamics_constraints(self):
        """Add dynamics constraints"""
        for t in range(P.T):
            A, B, C = calc_linear_discrete_model(self.z_bar[2, t], self.z_bar[3, t], self.d_bar[t])
            # Convert to CasADi DM
            A_ca = ca.DM(A)
            B_ca = ca.DM(B)
            C_ca = ca.DM(C)
            
            # Add dynamics constraint
            self.opti.subject_to(self.z[:, t+1] == ca.mtimes(A_ca, self.z[:, t]) + ca.mtimes(B_ca, self.u[:, t]) + C_ca)
    
    def add_state_constraints(self):
        """Add state constraints"""
        for t in range(P.T + 1):
            # Speed constraints
            self.opti.subject_to(self.z[2, t] <= P.speed_max)
            self.opti.subject_to(self.z[2, t] >= P.speed_min)
    
    def add_control_constraints(self):
        """Add control constraints"""
        for t in range(P.T):
            # Acceleration constraints
            self.opti.subject_to(self.u[0, t] <= P.acceleration_max)
            self.opti.subject_to(self.u[0, t] >= -P.acceleration_max)
            
            # Steering constraints
            self.opti.subject_to(self.u[1, t] <= P.steer_max)
            self.opti.subject_to(self.u[1, t] >= -P.steer_max)
            
            # Steering rate constraints
            if t < P.T - 1:
                self.opti.subject_to(self.u[1, t+1] - self.u[1, t] <= P.steer_change_max * P.dt)
                self.opti.subject_to(self.u[1, t+1] - self.u[1, t] >= -P.steer_change_max * P.dt)
    
    def add_obstacle_constraints(self):
        """Add obstacle constraints"""
        for t in range(min(P.T, P.obstacle_horizon)):
            # Calculate distances to all obstacles
            distances = []
            for obs_id, obstacle in P.obstacles.items():
                obs_pos = obstacle.positions[0][t]
                robot_current_pos = self.z_bar[:2, t]  # Use current position for distance calculation
                dist = np.linalg.norm(robot_current_pos - obs_pos)
                distances.append((dist, obs_id, obs_pos))
            
            # Sort by distance and get closest obstacle
            distances.sort(key=lambda x: x[0])
            closest_obstacles = distances[:1]
            
            # Add constraints for closest obstacle
            for dist, obs_id, obs_pos in closest_obstacles:
                # Calculate unit vector from obstacle to robot
                diff = self.z_bar[:2, t] - obs_pos
                
                # Avoid numerical issues
                if dist >= 0.001:
                    a = diff / dist  # unit vector
                    safe_distance = 3
                    
                    # Convert to CasADi DM
                    a_ca = ca.DM(a)
                    obs_pos_ca = ca.DM(obs_pos)
                    
                    # Add obstacle avoidance constraint
                    constraint_expr = a_ca[0] * (self.z[0, t] - obs_pos_ca[0]) + a_ca[1] * (self.z[1, t] - obs_pos_ca[1])
                    self.opti.subject_to(constraint_expr >= safe_distance)
                break
    
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
            logger.error(f"Optimization error: {str(e)}")
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
    def __init__(self, target_speed, initial_state):
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
        
        # Debug information
        self.debug_info = {
            'iterations': [],
            'cost_values': [],
            'constraint_violations': []
        }

    def update(self, ref_path, initial_state=None, consider_obstacles=True, acc_mode='accelerate', show_plot=False):
        """
        Update MPC controller
        :param ref_path: reference path
        :param initial_state: initial state, if None use current state
        :param consider_obstacles: whether to consider obstacles
        :param acc_mode: acceleration mode ('accelerate', 'decelerate', 'maintain')
        :param show_plot: whether to display plots
        :return: target index and optimized trajectory
        """
        self.ref_path = ref_path
        if initial_state is not None:
            self.node = Node(
                x=initial_state[0], y=initial_state[1], yaw=initial_state[2], v=initial_state[3])
        
        # Calculate reference trajectory
        speed_profile = calc_speed_profile(self.ref_path.cx, self.ref_path.cy, 
                                          self.ref_path.cyaw, self.target_speed)
        z_ref, target_ind = calc_ref_trajectory_in_T_step(self.node, self.ref_path,
                                                         speed_profile, acc_mode=acc_mode)
        
        # Current state
        z0 = [self.node.x, self.node.y, self.node.v, self.node.yaw]
        
        # Solve MPC problem
        self.a_opt, self.delta_opt, x_opt, y_opt, yaw_opt, v_opt = linear_mpc_control(
            z_ref, z0, self.a_opt, self.delta_opt, 
            consider_obstacles=consider_obstacles
        )
        
        # 绘图部分，根据show_plot参数决定是否显示
        if show_plot and x_opt is not None and y_opt is not None:
            # 使用连续刷新的绘图方式
            plt.ion()  # 开启交互模式
            plt.figure(1, figsize=(10, 8))
            plt.clf()  # 清除当前图形
            plt.plot(z_ref[0, :], z_ref[1, :], 'r-', linewidth=3, label='Reference Trajectory')
            plt.plot(x_opt, y_opt, 'b-', linewidth=2, label='MPC Trajectory')
            plt.plot(self.ref_path.cx, self.ref_path.cy, 'g-', linewidth=1, label='Reference Path')
            plt.scatter(self.node.x, self.node.y, color='black', s=100, marker='*', label=f'Current Position (v={self.node.v:.2f} m/s)')
            plt.legend()
            plt.axis('equal')
            plt.title("MPC Trajectory Planning")
            plt.grid(True)
            plt.savefig("mpc_trajectory.png")
            plt.draw()  # 更新图形
            plt.show(block=False)  # 非阻塞显示
            
            # 强制处理图形事件
            plt.pause(0.0001)  # 短暂暂停以使图形更新
        
        # Verify optimization results
        if x_opt is not None and y_opt is not None:
            logger.info(f"Optimization result initial point: ({x_opt[0]}, {y_opt[0]}), Current state: ({z0[0]}, {z0[1]})")
            logger.info(f"Distance between optimization result and current state: {np.hypot(x_opt[0] - z0[0], y_opt[0] - z0[1])}")
        
        # Execute control
        if self.delta_opt is not None:
            self.delta_exc, self.a_exc = self.delta_opt[0], self.a_opt[0]
        else:
            # If optimization fails, use previous control inputs or safe defaults
            logger.warning("Using fallback control")
            self.delta_exc = self.delta_exc if hasattr(self, 'delta_exc') else 0.0
            self.a_exc = self.a_exc if hasattr(self, 'a_exc') else 0.0
        
        # Update vehicle state
        self.node.update(self.a_exc, self.delta_exc, 1.0)
        
        # Update trajectory records
        self.x.append(self.node.x)
        self.y.append(self.node.y)
        self.yaw.append(self.node.yaw)
        self.v.append(self.node.v)
        
        return target_ind, x_opt, y_opt, yaw_opt, v_opt
    
    def plot_trajectories(self, z_ref, x_opt, y_opt):
        """
        Plot reference and optimized trajectories
        :param z_ref: reference trajectory
        :param x_opt: optimized x coordinates
        :param y_opt: optimized y coordinates
        """
        plt.figure(figsize=(10, 8))
        plt.plot(z_ref[0, :], z_ref[1, :], 'r-', linewidth=3, label='Reference Trajectory')
        plt.plot(x_opt, y_opt, 'b-', linewidth=2, label='MPC Trajectory')
        plt.plot(self.ref_path.cx, self.ref_path.cy, 'g-', linewidth=1, label='Reference Path')
        plt.scatter(self.node.x, self.node.y, color='black', s=100, marker='*', label='Current Position')
        plt.axis("equal")
        plt.legend()
        plt.title("MPC Trajectory Planning")
        plt.grid(True)
        
        # Save image or display
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"mpc_trajectory_{timestamp}.png")
        plt.close()
