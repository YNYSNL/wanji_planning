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

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../../MotionPlanning/")

import Control.draw as draw
import CurvesGenerator.reeds_shepp as rs
import CurvesGenerator.cubic_spline as cs

class P:

    # System config
    NX = 4  # state vector: z = [x, y, v, phi]
    NU = 2  # input vector: u = [acceleration, steer]
    T = 6  # finite time horizon length

    # MPC config
    Q = np.diag([1.0, 1.0, 1.0, 1.0])  # penalty for states
    Qf = np.diag([1.0, 1.0, 1.0, 1.0])  # penalty for end state
    R = np.diag([0.01, 0.1])  # penalty for inputs
    Rd = np.diag([0.01, 0.1])  # penalty for change of inputs

    dist_stop = 1.5  # stop permitted when dist to goal < dist_stop
    speed_stop = 0.5 / 3.6  # stop permitted when speed < speed_stop
    time_max = 500.0  # max simulation time
    iter_max = 5  # max iteration
    target_speed = 20.0 / 3.6 * 100  # target speed
    N_IND = 10  # search index number
    dt = 0.1  # time step
    d_dist = 1.0 * 100 # dist step
    du_res = 0.1  # threshold for stopping iteration

    # vehicle config
    RF = 3.3  # [m] distance from rear to vehicle front end of vehicle
    RB = 0.8  # [m] distance from rear to vehicle back end of vehicle
    W = 2.4  # [m] width of vehicle
    WD = 0.7 * W  # [m] distance between left-right wheels
    WB = 2.5 * 100  # [m] Wheel base
    TR = 0.44  # [m] Tyre radius
    TW = 0.7  # [m] Tyre width

    steer_max = np.deg2rad(45.0)  # max steering angle [rad]
    steer_change_max = np.deg2rad(30.0)  # maximum steering speed [rad/s]
    speed_max = 55.0 / 3.6 * 100  # maximum speed [m/s]
    speed_min = -20.0 / 3.6 * 100  # minimum speed [m/s]
    acceleration_max = 4.0 * 100  # maximum acceleration [m/s2]

    num_obstacles = 1  # number of obstacles
    obstacles = dict()  # obstacles: {id: [x, y, w, h]}
    obstacle_horizon = 20  # horizon length for obstacle avoidance
    num_modes = 1  # number of modes for Reeds-Shepp path
    
    @classmethod
    def init(cls, num_obstacles=1, obstacle_horizon = 20, num_modes=1):
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

    def nearest_index(self, node):
        """
        calc index of the nearest node in N steps
        :param node: current information
        :return: nearest index, lateral distance to ref point
        """

        dx = [node.x - x for x in self.cx[self.ind_old:]]
        dy = [node.y - y for y in self.cy[self.ind_old:]]
        dx_array = np.array(dx)
        dy_array = np.array(dy)
        dist = np.hypot(dx_array, dy_array)

        ind_in_N = int(np.argmin(dist))
        ind = self.ind_old + ind_in_N
        self.ind_old = ind

        rear_axle_vec_rot_90 = np.array([[math.cos(node.yaw + math.pi / 2.0)],
                                         [math.sin(node.yaw + math.pi / 2.0)]])

        vec_target_2_rear = np.array([[dx_array[ind_in_N]],
                                      [dy_array[ind_in_N]]])

        er = np.dot(vec_target_2_rear.T, rear_axle_vec_rot_90)
        er = er[0][0]

        return ind, er


def calc_ref_trajectory_in_T_step(node, ref_path, sp):
    """
    使用匀加速运动生成参考轨迹
    """
    z_ref = np.zeros((P.NX, P.T + 1))
    length = ref_path.length
    
    # 设置舒适加速度（可调整）
    comfort_acc = 1.0  # m/s^2
    
    # 计算匀加速运动末端速度
    v0 = abs(node.v)
    vT = v0 + comfort_acc * P.dt * P.T
    
    # 判断是否需要调整加速度
    if vT > P.target_speed:
        # 如果末端速度超过目标速度，反推所需加速度
        vT = P.target_speed
        acc = (vT - v0) / (P.dt * P.T)
    else:
        # 使用舒适加速度
        acc = comfort_acc
    
    # 生成速度序列
    v_seq = np.zeros(P.T + 1)
    v_seq[0] = v0
    for i in range(1, P.T + 1):
        v_seq[i] = v0 + acc * P.dt * i
    
    # 计算累积位移序列
    s_seq = np.zeros(P.T + 1)
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


def linear_mpc_control(z_ref, z0, a_old, delta_old):
    """
    linear mpc controller
    :param z_ref: reference trajectory in T steps
    :param z0: initial state vector
    :param a_old: acceleration of T steps of last time
    :param delta_old: delta of T steps of last time
    :return: acceleration and delta strategy based on current information
    """

    if a_old is None or delta_old is None:
        a_old = [0.0] * P.T
        delta_old = [0.0] * P.T

    x, y, yaw, v = None, None, None, None

    for k in range(P.iter_max):
        z_bar = predict_states_in_T_step(z0, a_old, delta_old, z_ref)
        a_rec, delta_rec = a_old[:], delta_old[:]
        a_old, delta_old, x, y, yaw, v = solve_linear_mpc(z_ref, z_bar, z0, delta_old)

        du_a_max = max([abs(ia - iao) for ia, iao in zip(a_old, a_rec)])
        du_d_max = max([abs(ide - ido) for ide, ido in zip(delta_old, delta_rec)])

        if max(du_a_max, du_d_max) < P.du_res:
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


def solve_linear_mpc(z_ref, z_bar, z0, d_bar):
    """
    solve the quadratic optimization problem using cvxpy, solver: OSQP
    :param z_ref: reference trajectory (desired trajectory: [x, y, v, yaw])
    :param z_bar: predicted states in T steps
    :param z0: initial state
    :param d_bar: delta_bar
    :return: optimal acceleration and steering strategy
    """

    z = cvxpy.Variable((P.NX, P.T + 1))
    u = cvxpy.Variable((P.NU, P.T))

    cost = 0.0
    constrains = []

    for t in range(P.T):
        cost += cvxpy.quad_form(u[:, t], P.R)
        cost += cvxpy.quad_form(z_ref[:, t] - z[:, t], P.Q)

        A, B, C = calc_linear_discrete_model(z_bar[2, t], z_bar[3, t], d_bar[t])

        constrains += [z[:, t + 1] == A @ z[:, t] + B @ u[:, t] + C]

        if t < P.T - 1:
            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], P.Rd)
            constrains += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= P.steer_change_max * P.dt]

    cost += cvxpy.quad_form(z_ref[:, P.T] - z[:, P.T], P.Qf)

    constrains += [z[:, 0] == z0]
    constrains += [z[2, :] <= P.speed_max]
    constrains += [z[2, :] >= P.speed_min]
    constrains += [cvxpy.abs(u[0, :]) <= P.acceleration_max]
    constrains += [cvxpy.abs(u[1, :]) <= P.steer_max]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constrains)
    solver_opts = {
        'verbose': False,
        'eps_abs': 1e-3,        # 放宽误差容限
        'eps_rel': 1e-3,
        'max_iter': 10000,      
        'adaptive_rho': True,   
        'polish': True,         
        'warm_start': True,
        'rho': 1.0,            # 添加初始步长参数
        'sigma': 1e-6,         # 添加正则化参数
        'alpha': 1.6,          # 添加过松弛参数
    }
    
    try:
        result = prob.solve(solver=cvxpy.OSQP, **solver_opts)
    except:
        print("OSQP failed, trying with modified parameters")
        # 如果失败，尝试使用更保守的参数
        solver_opts.update({
            'eps_abs': 1e-2,
            'eps_rel': 1e-2,
            'rho': 0.1,
            'sigma': 1e-4,
        })
        try:
            result = prob.solve(solver=cvxpy.OSQP, **solver_opts)
        except:
            print("OSQP failed again, trying ECOS")
            try:
                result = prob.solve(solver=cvxpy.ECOS,
                                  max_iters=100,
                                  abstol=1e-2,
                                  reltol=1e-2,
                                  verbose=False)
            except:
                print("All optimization attempts failed")
                return None, None, None, None, None, None

    if prob.status in [cvxpy.OPTIMAL, cvxpy.OPTIMAL_INACCURATE]:
        x = z.value[0, :]
        y = z.value[1, :]
        v = z.value[2, :]
        yaw = z.value[3, :]
        a = u.value[0, :]
        delta = u.value[1, :]
        return a, delta, x, y, yaw, v
    else:
        print(f"Optimization failed with status: {prob.status}")
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

    # speed_profile[-1] = 0.0

    return speed_profile


def pi_2_pi(angle):
    if angle > math.pi:
        return angle - 2.0 * math.pi

    if angle < -math.pi:
        return angle + 2.0 * math.pi

    return angle


class MPCController:
    def __init__(self, target_speed, initial_state):
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

    def update(self, ref_path, initial_state=None):
        if ref_path is None or len(ref_path.cx) < 2:
            print("Invalid reference path")
            return None, None, None, None, None
            
        # 检查参考路径的连续性
        path_dists = np.sqrt(np.diff(ref_path.cx)**2 + np.diff(ref_path.cy)**2)
        if np.any(path_dists < 1e-6):
            print("Warning: Reference path contains very close points")
            
        self.ref_path = ref_path
        if initial_state is not None:
            self.node = Node(
                x=initial_state[0], y=initial_state[1], yaw=initial_state[2], v=initial_state[3])
        
        try:
            z_ref, target_ind = calc_ref_trajectory_in_T_step(self.node, self.ref_path,
                                                            calc_speed_profile(self.ref_path.cx,
                                                                            self.ref_path.cy,
                                                                            self.ref_path.cyaw,
                                                                            self.target_speed))

            # 检查参考轨迹的有效性
            if np.any(np.isnan(z_ref)):
                raise ValueError("Reference trajectory contains NaN values")
                
            z0 = [self.node.x, self.node.y, self.node.v, self.node.yaw]
        
        except Exception as e:
            print(f"MPC update failed: {str(e)}")
            return None, None, None, None, None
        
        try:
            self.a_opt, self.delta_opt, x_opt, y_opt, yaw_opt, v_opt = linear_mpc_control(z_ref, z0,
                                                                                    self.a_opt,
                                                                                    self.delta_opt)
            
            if self.delta_opt is not None:
                self.delta_exc, self.a_exc = self.delta_opt[0], self.a_opt[0]
            else:
                # 如果优化失败，使用上一步的控制输入或安全的默认值
                print("Using fallback control")
                self.delta_exc = self.delta_exc if hasattr(self, 'delta_exc') else 0.0
                self.a_exc = self.a_exc if hasattr(self, 'a_exc') else 0.0
                
            # 限制控制输入的变化率
            self.a_exc = np.clip(self.a_exc, -P.acceleration_max, P.acceleration_max)
            self.delta_exc = np.clip(self.delta_exc, -P.steer_max, P.steer_max)

            self.node.update(self.a_exc, self.delta_exc, 1.0)
            
            # 更新记录
            self.x.append(self.node.x)
            self.y.append(self.node.y)
            self.yaw.append(self.node.yaw)
            self.v.append(self.node.v)
            
            return target_ind, x_opt, y_opt, yaw_opt, v_opt
            
        except Exception as e:
            print(f"MPC update failed: {str(e)}")
            # 使用简单的故障安全控制
            self.delta_exc = 0.0
            self.a_exc = -0.1  # 轻微减速
            self.node.update(self.a_exc, self.delta_exc, 1.0)
            
            # 仍然更新记录
            self.x.append(self.node.x)
            self.y.append(self.node.y)
            self.yaw.append(self.node.yaw)
            self.v.append(self.node.v)
            
            return target_ind, None, None, None, None


def main():
    ax = [0.0, 15.0, 30.0, 50.0, 60.0]
    ay = [0.0, 40.0, 15.0, 30.0, 0.0]
    cx, cy, cyaw, ck, s = cs.calc_spline_course(ax, ay, ds=P.d_dist)

    ref_path = PATH(cx, cy, cyaw, ck)
    # 在参考线附近设置一个障碍物位置
    # obstacle_x = cx[10] + 2.0  # 假设障碍物在参考线的第11个点右侧2个单位
    # obstacle_y = cy[10] + 2.0  # 把障碍物设定在参考线的第11个点上方2个单位
    # obstacles = [(obstacle_x, obstacle_y)] 
    mpc_controller = MPCController(ref_path, P.target_speed)

    time = 0.0  # 在 main 函数中控制时间
    while time < P.time_max:
        print("Time: ", time)
        target_ind, x_opt, y_opt, _, _ = mpc_controller.update()


        dist = math.hypot(mpc_controller.node.x - cx[-1], mpc_controller.node.y - cy[-1])

        if dist < P.dist_stop and abs(mpc_controller.node.v) < P.speed_stop:
            break

        # 绘图代码
        plt.cla()
        if len(mpc_controller.yaw) > 1:  # 确保有足够的数据
            steer = rs.pi_2_pi(-math.atan(P.WB * ((mpc_controller.node.yaw - mpc_controller.yaw[-2]) / 
                                                    (mpc_controller.node.v * P.dt))))
        else:
            steer = 0.0  # 如果没有足够的数据，则设定为0

        draw.draw_car(mpc_controller.node.x, mpc_controller.node.y, mpc_controller.node.yaw, steer, P)
        plt.plot(cx, cy, color='gray')
        plt.plot(mpc_controller.x, mpc_controller.y, '-b')
        if x_opt is not None:
            plt.plot(x_opt, y_opt, color='darkviolet', marker='*')
        plt.plot(cx[target_ind], cy[target_ind])
        plt.axis("equal")
        plt.title("Linear MPC, " + "v = " + str(round(mpc_controller.node.v * 3.6, 2)))
        plt.pause(0.001)

        time += P.dt  # 在 main 函数中增量时间

    plt.show()


if __name__ == '__main__':
    main()