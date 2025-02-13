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

def xy_to_latlon(lon_origin, lat_origin, x, y):

    proj_string = "+proj=tmerc +lon_0=" + str(lon_origin) + " +lat_0=" + str(lat_origin) + " +ellps=WGS84"

    proj = Proj(proj_string)

    transformer_inv = Transformer.from_crs(proj_string, "epsg:4326", always_xy=True)

    lon_target, lat_target = transformer_inv.transform(x + proj(lon_origin, lat_origin)[0], 
                                                       y + proj(lon_origin, lat_origin)[1])

    return lon_target, lat_target

def CACS_plan(state_data, reference_data, ego_plan, ego_decision):
    DataFromEgo = copy.deepcopy(state_data)
    # print('DataFromEgo', DataFromEgo)

    # Get the current sensor data
    # sensorgps_data = frame_data.get("sensorgps")

    # v0 = sensorgps_data.velocity
    # heading = sensorgps_data.heading


    ego_state = Node(
        x=DataFromEgo['0'].get("x", 0.0) * 100, 
        y=DataFromEgo['0'].get("y", 0.0) * 100, 
        yaw=DataFromEgo['0'].get("heading", 0.0), 
        v=DataFromEgo['0'].get("v", 0.0) * 100, 
        gx=DataFromEgo['0'].get("gx", 0.0) * 100, 
        gy=DataFromEgo['0'].get("gy", 0.0) * 100, 
        direct=1.0
    )
    # Acceleration and time steps for the motion plan
    a = 0  # Constant acceleration
    # t_max = 3  # Total time for the trajectory
    time_steps = 964  # Number of time steps to break down the trajectory

    # Generate the trajectory using the Node's kinematic update
    trajectory_points = []
    trajectory_points_x = []
    trajectory_points_y = []
    for t in range(time_steps):
        # Update the vehicle's state using the Node's update method
        # Create a roadpoint from the updated state
        point = roadpoint()
        point.x = ego_state.x  # Updated x-coordinate
        point.y = ego_state.y  # Updated y-coordinate    relativetime: 0.0
        point.gx, point.gy = xy_to_latlon(ego_state.gx,ego_state.gy,point.x,point.y)

        point.speed = ego_state.v / 100  # Updated velocity
        point.heading = 0  # Updated heading
        # print(f'heading is {point.heading}')
        point.a = a  
        point.jerk = 0  # Assuming no jerk (smooth motion)
        point.lanewidth = 0  # Default lane width, adjust if needed
        ego_state.update(200, delta=0.0, direct=1.0)  # Assume no steering (delta=0) for simplicity

        point.s = t * 0.1 * ego_state.v # Updated s-coordinate
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
    ego_plan.guidespeed = 3  # Final velocity
    ego_plan.guideangle = 0 # Final heading
    ego_plan.timestamp = int(rospy.Time.now().to_sec())

    # Decision-making (example behavior)
    ego_decision.drivebehavior = 2  # Drive behavior, this could be adjusted

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
    ego_plan.timestamp = int(rospy.Time.now().to_sec())

    ego_decision.drivebehavior = 2  # Drive behavior, this could be adjusted

    return ego_plan, ego_decision