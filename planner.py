import copy
import numpy as np
import rospy

def CACS_plan(data):
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


def bag_plan(bag_data, t, ego_plan, ego_decision):

    # print(bag_data['guidespeed'][t])
    # Update the ego_plan with the generated trajectory points
    ego_plan.points = bag_data['roadpoints'][t]
    # print('ego_plan.points', ego_plan.points.s)

    # plt.plot(ego_plan.points.x, ego_plan.points.y, marker='o', label='Trajectory Points')
    # plt.savefig('trajectory_plot.png')
    ego_plan.guidespeed = bag_data['guidespeed'][t]  # Final velocity
    ego_plan.guideangle = bag_data['guideangle'][t]  # Final heading
    ego_plan.timestamp = int(rospy.Time.now().to_sec())

    ego_decision.drivebehavior = 2  # Drive behavior, this could be adjusted

    return ego_plan, ego_decision