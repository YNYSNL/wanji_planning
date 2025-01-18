import rospy
import rosbag
from common_msgs.msg import actuator
from common_msgs.msg import hdmap
from common_msgs.msg import hdroutetoglobal
from common_msgs.msg import roadpoint
from common_msgs.msg import sensorgps
from common_msgs.msg import sensorobject
from common_msgs.msg import sensorobjects
from common_msgs.msg import planningmotion
from common_msgs.msg import decisionbehavior
import time

def read_planningmotion_from_bag(file_path):
    """从 rosbag 文件中读取 /planningmotion 话题的数据"""
    planning_points = []
    guidespeed = 0.0
    guideangle = 0.0
    timestamp = 0

    try:
        # 打开bag文件
        bag = rosbag.Bag(file_path, 'r')

        # 遍历bag文件中的每条消息
        for topic, msg, t in bag.read_messages(topics='/planningmotion'):
            # 读取 /planningmotion 话题中的数据
            guidespeed = msg.guidespeed
            guideangle = msg.guideangle
            timestamp = msg.timestamp

            # 解析 /planningmotion 中的每个 roadpoint
            for point in msg.points:
                p = roadpoint()
                p.x = point.x
                p.y = point.y
                p.speed = point.speed
                p.heading = point.heading
                p.jerk = point.jerk
                p.lanewidth = point.lanewidth
                p.s = point.s
                p.gx = point.gx
                p.gy = point.gy
                p.roadtype = point.roadtype
                p.a = point.a
                p.lanetype = point.lanetype
                p.turnlight = point.turnlight
                p.mergelanetype = point.mergelanetype
                p.sensorlanetype = point.sensorlanetype
                p.curvature = point.curvature
                p.dkappa = point.dkappa
                p.ddkappa = point.ddkappa
                p.sideroadwidth = point.sideroadwidth
                p.leftlanewidth = point.leftlanewidth
                p.rightlanewidth = point.rightlanewidth
                p.laneswitch = point.laneswitch
                p.lanenum = point.lanenum
                p.lanesite = point.lanesite
                planning_points.append(p)

        bag.close()

    except Exception as e:
        rospy.logerr("Error reading the rosbag file: %s", e)

    return planning_points, guidespeed, guideangle, timestamp

def read_behaviordecision_from_bag(file_path):
    """从 rosbag 文件中读取 /behaviordecision 话题的数据"""
    behavior_decision = decisionbehavior()
    
    try:
        # 打开bag文件
        bag = rosbag.Bag(file_path, 'r')

        # 遍历bag文件中的每条消息
        for topic, msg, t in bag.read_messages(topics='/behaviordecision'):
            # 读取 /behaviordecision 话题中的数据
            behavior_decision.drivebehavior = msg.drivebehavior
            behavior_decision.timestamp = msg.timestamp

        bag.close()

    except Exception as e:
        rospy.logerr("Error reading the rosbag file: %s", e)

    return behavior_decision

def create_planningmotion_message(planning_points, guidespeed, guideangle, timestamp):
    """创建并返回一个 planningmotion 消息"""
    ego_plan = planningmotion()

    # 将读取的数据添加到 ego_plan 中
    ego_plan.points = planning_points
    ego_plan.guidespeed = guidespeed
    ego_plan.guideangle = guideangle
    ego_plan.timestamp = timestamp
    ego_plan.isvalid = True

    return ego_plan

def publish_planningmotion_and_decision(file_path):
    """读取 rosbag 文件中的 /planningmotion 和 /behaviordecision 数据并发布"""
    # 读取 /planningmotion 数据
    label = f"label_{int(time.time())}"
    print('label:', label)
    planning_points, guidespeed, guideangle, timestamp = read_planningmotion_from_bag(file_path)

    # 读取 /behaviordecision 数据
    behavior_decision = read_behaviordecision_from_bag(file_path)

    if planning_points:
        # 创建 planningmotion 消息并发布
        ego_plan = create_planningmotion_message(planning_points, guidespeed, guideangle, timestamp)
        pub_ego_plan = rospy.Publisher("/planningmotion", planningmotion, queue_size=5)
        pub_ego_plan.publish(ego_plan)
        rospy.loginfo("Planning motion result published!")
    else:
        rospy.logwarn("No valid /planningmotion data to publish!")

    if behavior_decision:
        # 发布 behavior_decision 消息
        pub_ego_decision = rospy.Publisher("/behaviordecision", decisionbehavior, queue_size=5)
        pub_ego_decision.publish(behavior_decision)
        rospy.loginfo("Behavior decision result published!")
    else:
        rospy.logwarn("No valid /behaviordecision data to publish!")


def main():

    rospy.init_node("planningmotion_bag_publisher", anonymous=True)

    file_path = '/home/wanji/tongji_vehicle/2025-01-18-17-30-22.bag'

    publish_planningmotion_and_decision(file_path)

    rate = rospy.Rate(100)
    rospy.spin()

if __name__ == "__main__":
    main()