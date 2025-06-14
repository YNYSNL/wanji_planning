#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rosbag
import csv
import numpy as np
from common_msgs.msg import sensorgps
import rospy

def extract_reference_trajectory_from_bag(bag_file_path, output_csv_path):
    """
    从bag文件中提取sensorgps话题的经纬度数据，保存为CSV文件
    
    Args:
        bag_file_path: bag文件路径
        output_csv_path: 输出CSV文件路径
    """
    
    trajectory_data = []
    
    try:
        # 打开bag文件
        print(f"正在读取bag文件: {bag_file_path}")
        bag = rosbag.Bag(bag_file_path, 'r')
        
        # 获取话题信息
        topics = bag.get_type_and_topic_info()[1]
        if '/sensorgps' not in topics:
            print("错误：在bag文件中找不到/sensorgps话题")
            available_topics = list(topics.keys())
            print(f"可用话题: {available_topics}")
            return False
        
        print(f"找到/sensorgps话题，包含 {topics['/sensorgps'].message_count} 条消息")
        
        # 读取sensorgps话题的所有消息
        message_count = 0
        for topic, msg, t in bag.read_messages(topics=['/sensorgps']):
            # 提取经纬度数据
            lon = msg.lon
            lat = msg.lat
            timestamp = t.to_sec()
            
            # 存储数据
            trajectory_data.append({
                'timestamp': timestamp,
                'lon': lon,
                'lat': lat
            })
            
            message_count += 1
            if message_count % 100 == 0:
                print(f"已处理 {message_count} 条消息...")
        
        bag.close()
        print(f"总共提取了 {len(trajectory_data)} 个轨迹点")
        
        if len(trajectory_data) == 0:
            print("错误：没有提取到任何轨迹数据")
            return False
        
        # 按时间戳排序
        trajectory_data.sort(key=lambda x: x['timestamp'])
        
        # 数据预处理：去除重复点和异常点
        filtered_data = filter_trajectory_data(trajectory_data)
        print(f"过滤后剩余 {len(filtered_data)} 个轨迹点")
        
        # 保存为CSV文件
        save_trajectory_to_csv(filtered_data, output_csv_path)
        print(f"参考轨迹已保存到: {output_csv_path}")
        
        # 打印统计信息
        print_trajectory_statistics(filtered_data)
        
        return True
        
    except Exception as e:
        print(f"处理bag文件时出错: {e}")
        return False

def filter_trajectory_data(trajectory_data, min_distance=0.2):
    """
    过滤轨迹数据，去除距离过近的点
    
    Args:
        trajectory_data: 原始轨迹数据列表
        min_distance: 最小距离阈值（米）
    
    Returns:
        filtered_data: 过滤后的轨迹数据
    """
    if len(trajectory_data) == 0:
        return []
    
    filtered_data = [trajectory_data[0]]  # 保留第一个点
    
    for i in range(1, len(trajectory_data)):
        current_point = trajectory_data[i]
        last_point = filtered_data[-1]
        
        # 计算距离（简化的球面距离计算）
        distance = calc_distance(
            last_point['lon'], last_point['lat'],
            current_point['lon'], current_point['lat']
        )
        
        # 如果距离大于阈值，保留该点
        if distance > min_distance:
            filtered_data.append(current_point)
    
    return filtered_data

def calc_distance(lon1, lat1, lon2, lat2):
    """计算两点间大地距离（米）"""
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

def save_trajectory_to_csv(trajectory_data, output_path):
    """将轨迹数据保存为CSV文件"""
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['timestamp', 'lon', 'lat']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # 写入表头
            writer.writeheader()
            
            # 写入数据
            for point in trajectory_data:
                writer.writerow(point)
        
        print(f"CSV文件保存成功: {output_path}")
        
    except Exception as e:
        print(f"保存CSV文件时出错: {e}")

def print_trajectory_statistics(trajectory_data):
    """打印轨迹统计信息"""
    if len(trajectory_data) < 2:
        return
    
    # 计算轨迹总长度
    total_distance = 0
    for i in range(1, len(trajectory_data)):
        distance = calc_distance(
            trajectory_data[i-1]['lon'], trajectory_data[i-1]['lat'],
            trajectory_data[i]['lon'], trajectory_data[i]['lat']
        )
        total_distance += distance
    
    # 计算时间跨度
    time_span = trajectory_data[-1]['timestamp'] - trajectory_data[0]['timestamp']
    
    # 计算经纬度范围
    lons = [point['lon'] for point in trajectory_data]
    lats = [point['lat'] for point in trajectory_data]
    
    print("=== 轨迹统计信息 ===")
    print(f"轨迹点数: {len(trajectory_data)}")
    print(f"轨迹总长度: {total_distance:.2f} 米")
    print(f"时间跨度: {time_span:.2f} 秒")
    print(f"经度范围: {min(lons):.6f} ~ {max(lons):.6f}")
    print(f"纬度范围: {min(lats):.6f} ~ {max(lats):.6f}")
    print("==================")

def load_reference_trajectory_from_csv(csv_path):
    """
    从CSV文件加载参考轨迹
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        ref_lons: 经度列表
        ref_lats: 纬度列表
    """
    ref_lons = []
    ref_lats = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                ref_lons.append(float(row['lon']))
                ref_lats.append(float(row['lat']))
        
        print(f"从CSV文件加载了 {len(ref_lons)} 个参考点")
        return ref_lons, ref_lats
        
    except Exception as e:
        print(f"加载CSV文件时出错: {e}")
        return [], []

if __name__ == "__main__":
    # 设置文件路径
    bag_file_path = "/home/admin/Downloads/2025-06-13-22-01-43.bag"
    output_csv_path = "reference_trajectory.csv"
    
    print("开始提取优秀驾驶人参考轨迹...")
    
    # 提取轨迹数据
    success = extract_reference_trajectory_from_bag(bag_file_path, output_csv_path)
    
    if success:
        print("参考轨迹提取完成！")
        
        # 测试加载功能
        print("\n测试加载功能...")
        ref_lons, ref_lats = load_reference_trajectory_from_csv(output_csv_path)
        if ref_lons and ref_lats:
            print("CSV文件加载测试成功！")
        else:
            print("CSV文件加载测试失败！")
    else:
        print("参考轨迹提取失败！") 