import numpy as np
from pyproj import Transformer, Proj
import matplotlib.pyplot as plt
import matplotlib
from typing import Dict, List, Tuple
import pandas as pd
import rospy


# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False     # 解决保存图像时负号'-'显示为方块的问题

map_transformer = Transformer.from_crs("epsg:4326", "epsg:3857")

'for tongji test ground'
proj_string = "+proj=tmerc +lon_0=121.20585769414902 +lat_0=31.290823210868965 +ellps=WGS84"
# proj_string = "+proj=tmerc +lon_0=121.2092870660126 +lat_0=31.292829882838856 +ellps=WGS84"
proj = Proj(proj_string)
# lon, lat = 121.2032277000,	31.2924949000
#
# # 经纬度转换为 x, y 坐标
# x, y = proj(lon, lat)
# print(f"经纬度转换为 X, Y 坐标: ({x}, {y})")
class lon_lat_to_xy:
    def __init__(self, data_list, flag1=False):
        self.data_list = data_list
        self.veh_num = len(self.data_list)
        self.flag1 = flag1
        print('veh_num', self.veh_num)

    def get_pos(self):
        state_dict = {}
        for veh in range(self.veh_num):
            data = self.data_list[veh]
            # 经纬度转为笛卡尔坐标
            x, y = proj(data.lon, data.lat)
            state_dict[str(veh)] = {
                'x': 0,
                'y': 0,
                'gx': data.lon,
                'gy':data.lat,
                'v': float(data.velocity),  # 车辆速度
                'heading': float(data.heading),  # 航向角
                'a': 0  
            }
        return state_dict

class lon_lat_to_xy_map:
    def __init__(self, data_list, flag1=False):
        self.data_list = data_list
        self.veh_num = len(self.data_list)
        self.flag1 = flag1


    def get_pos(self):
        state_dict = {}

        data = self.data_list.map.point
        # 经纬度转为笛卡尔坐标
        x, y = proj(data.lon, data.lat)
        state_dict[str(veh)] = {
            'x': x,
            'y': y,
            'v': float(data.velocity),  # 车辆速度
            'heading': float(data.heading),  # 航向角
        }
        return state_dict

class xy_to_lon_lat:
    def __init__(self, state_list, data_list):
        self.state_list = state_list
        self.data_list = data_list
        self.veh_num = len(self.state_list)

    def transform(self):
        data_from_simulator = []
        for veh in range(self.veh_num):
            data = self.data_list[veh]
            # 检查 state_list 是否包含 'x' 和 'y'
            if 'x' not in self.state_list[veh] or 'y' not in self.state_list[veh]:
                rospy.logerr(f"Missing 'x' or 'y' for vehicle {veh}")
                continue  # 如果缺少 'x' 或 'y'，跳过该车辆

            # 获取笛卡尔坐标并转换为经纬度
            lon, lat = proj(self.state_list[veh]['x'], self.state_list[veh]['y'], inverse=True)
            # 更新数据
            data.lon = lon
            data.lat = lat
            data_from_simulator.append(data)
        return data_from_simulator


# # 接受到data输入信息 将他转化为协同算法需要的格式
# class lon_lat_to_xy:
#     def __init__(self, data_list, flag1=False):
#         self.data_list = data_list
#         self.veh_num = len(self.data_list)
#         self.platform = platform  # 添加平台参数
#         self.flag1 = flag1

#     def get_pos(self):
#         state_dict = {}
#         for veh in range(self.veh_num):
#             # 打印车辆id
#             print('车辆id', veh)
#             data = self.data_list[veh]
#             x, y = proj(data.longitude, data.latitude)
#             # 打印原始坐标
#             # print('原始坐标', data.longitude, data.latitude)
#             # 打印原始坐标数据类型
#             # print('原始坐标数据类型', type(data.longitude), type(data.latitude))
#             # 打印原始坐标数据类型
#             print('坐标转换结果', x, y)
#             # # 打印坐标转换结果数据类型
#             # print('坐标转换结果数据类型', type(x), type(y))

            
#             # 根据平台处理速度单位
#             speed = float(data.speed)
#             if self.platform == "dc" and veh == 0:  # 域控中速度单位是m/s
#                 speed = speed * 3.6    # m/s 转换为 km/h
#             # 实车平台保持原样（已经是km/h）
#             if self.platform == "dc" and veh == 1 and self.flag1                                                                                                                                                        :
#                 speed = speed * 3.6
            
#             # 打印速度
#             print('速度', round(speed, 2), 'km/h')

#             state_dict[str(veh)] = {
#                 'x': x,
#                 'y': y,
#                 'v': speed,
#                 'heading': float(data.courseAngle),
#                 'a': 0
#             }
#         return state_dict


class planList_form:
    def __init__(self, header, id, timestamp, refpoints):
        self.header = header
        self.Id = id
        self.timestamp = timestamp
        self.refpoints = refpoints


# class xy_to_lon_lat:
#     def __init__(self, state_list, direction_list, aggressiveness_list, data_list):
#         self.state_list = state_list
#         self.direction_list = direction_list
#         self.aggressiveness_list = aggressiveness_list
#         self.data_list = data_list
#         self.veh_num = len(self.state_list)

#     def transform(self):
#         data_from_simulator = []
#         for veh in range(self.veh_num):
#             data = self.data_list[veh]
#             # print('now its veh:', veh, '/original speed', data.refpoints[0].speed, '/PG planning speed', self.state_list[veh][2])
#             data.refpoints[0].speed = str(self.state_list[veh][2])  # 只改第一个参考点的速度信息 改为规划得到的速度
#             # print('after', self.state_list[veh][2], self.state_list[veh])
#             data.Id = str(veh)
#             # if veh == 0:
#             #     data.refpoints[0].speed = str(5)
#                 # action = planList_form(None, veh, data.timestamp, data.refpoints)
#             # action = planList_form(veh, data.timestamp + 1, data.refpoints)
#             data_from_simulator.append(data)
#         return data_from_simulator

def visualize_reference_lines(reference_lines: Dict[str, Dict[str, np.ndarray]], 
                            resampled_lines: Dict[str, Dict[str, np.ndarray]] = None):
    """可视化参考线"""
    plt.figure(figsize=(12, 8))
    
    # 设置颜色映射
    colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']
    
    # 绘制原始参考线
    for i, (name, data) in enumerate(reference_lines.items()):
        color = colors[i % len(colors)]
        plt.plot(data['x'], data['y'], f'{color}.', label=f'{name} (原始)', markersize=2, alpha=0.5)
        
        # 绘制起点和终点
        plt.plot(data['x'][0], data['y'][0], f'{color}o', markersize=10, label=f'{name} 起点')
        plt.plot(data['x'][-1], data['y'][-1], f'{color}s', markersize=10, label=f'{name} 终点')
    
    # 如果有重采样数据，也绘制出来
    if resampled_lines:
        for i, (name, data) in enumerate(resampled_lines.items()):
            color = colors[i % len(colors)]
            plt.plot(data['x'], data['y'], f'{color}-', label=f'{name} (重采样)', alpha=0.7)
    
    plt.grid(True)
    plt.xlabel('X (米)', fontsize=12)
    plt.ylabel('Y (米)', fontsize=12)
    plt.title('参考线可视化', fontsize=14)
    plt.legend(loc='best', fontsize=10)
    
    # 保持横纵比例一致
    plt.axis('equal')
    
    plt.show()

class ReferenceLineParser:
    """参考线解析类"""
    def __init__(self):
        self.transformer = Transformer.from_crs("epsg:4326", "epsg:3857")
    
    def read_reference_lines(self, csv_path: str) -> Dict[str, Dict[str, np.ndarray]]:
        """读取CSV文件中的参考线数据
        
        返回数据结构:
        {
            "轨迹名称1": {
                "x": np.array([x1, x2, ...]),  # X坐标数组
                "y": np.array([y1, y2, ...]),  # Y坐标数组
                "time": np.array([t1, t2, ...])  # 时间戳数组
            },
            "轨迹名称2": {
                "x": np.array([...]),
                "y": np.array([...]),
                "time": np.array([...])
            },
            ...
        }
        """
        df = pd.read_csv(csv_path)
        grouped = df.groupby('TrajectoryName')
        
        reference_lines = {}
        for name, group in grouped:
            # print(f"\n处理参考线 {name}:")
            
            # 直接使用X、Y坐标
            x_values = group['X'].values
            y_values = group['Y'].values
            
            # 打印第一个座标点
            # print('第一个座标点', x_values[0], y_values[0])

            # 打印坐标范围
            # print(f"X坐标范围: {x_values.min():.6f} 到 {x_values.max():.6f}")
            # print(f"Y坐标范围: {y_values.min():.6f} 到 {y_values.max():.6f}")
            
            # 将坐标原点平移到数据范围的中心
            # x_center = (x_values.max() + x_values.min()) / 2
            # y_center = (y_values.max() + y_values.min()) / 2
            
            # x_meters = x_values - x_center
            # y_meters = y_values - y_center
            
            reference_lines[name] = {
                'x': x_values,  # 将x坐标作为y坐标,与轨迹点保持一致
                'y': y_values,
                'time': group['Time'].values
            }
            
            # print(f"处理后坐标范围:")
            # print(f"X: {x_meters.min():.2f} 到 {x_meters.max():.2f} 米")
            # print(f"Y: {y_meters.min():.2f} 到 {y_meters.max():.2f} 米")
            # print(f"成功处理 {name}: {len(x_meters)} 个点")
        
        return reference_lines

    def get_reference_line_segments(self, 
                                  reference_lines: Dict[str, Dict[str, np.ndarray]], 
                                  min_distance: float = 1.0) -> Dict[str, Dict[str, np.ndarray]]:
        """对参考线进行重采样，确保相邻点之间的距离不小于min_distance
        
        参数:
            reference_lines: 原始参考线数据，格式同read_reference_lines的返回值
            min_distance: 重采样后相邻点之间的最小距离（米）
        
        返回:
            重采样后的参考线数据，格式与输入相同:
            {
                "轨迹名称1": {
                    "x": np.array([x1', x2', ...]),  # 重采样后的X坐标
                    "y": np.array([y1', y2', ...]),  # 重采样后的Y坐标
                    "time": np.array([t1', t2', ...])  # 对应的插值时间戳
                },
                ...
            }
        """
        resampled_lines = {}
        
        for name, data in reference_lines.items():
            x, y = data['x'], data['y']
            
            # 计算累积距离
            dx = np.diff(x)
            dy = np.diff(y)
            distances = np.sqrt(dx**2 + dy**2)
            
            # 检查距离是否有效
            if np.any(np.isnan(distances)) or np.any(np.isinf(distances)):
                print(f"警告: {name} 包含无效距离")
                continue
            
            cumulative_distance = np.concatenate(([0], np.cumsum(distances)))
            
            if cumulative_distance[-1] <= 0:
                print(f"警告: {name} 总距离无效")
                continue
            
            # 创建新的采样点
            try:
                target_distances = np.arange(0, cumulative_distance[-1], min_distance)
                
                # 对x和y进行插值
                x_interp = np.interp(target_distances, cumulative_distance, x)
                y_interp = np.interp(target_distances, cumulative_distance, y)
                t_interp = np.interp(target_distances, cumulative_distance, data['time'])
                
                resampled_lines[name] = {
                    'x': x_interp,
                    'y': y_interp,
                    'time': t_interp
                }
                
                # print(f"重采样 {name}:")
                # print(f"原始点数: {len(x)}, 重采样后点数: {len(x_interp)}")
                # print(f"总路径长度: {cumulative_distance[-1]:.2f} 米")
                
            except Exception as e:
                # print(f"重采样错误 {name}: {e}")
                continue
        
        return resampled_lines

# def main():
#     """测试函数"""
#     parser = ReferenceLineParser()
    
#     # 读取参考线
#     reference_lines = parser.read_reference_lines('pre_map.csv')
    
#     # 重采样参考线
#     resampled_lines = parser.get_reference_line_segments(reference_lines, min_distance=1.0)
    
#     # 打印一些基本信息
#     for name, data in resampled_lines.items():
#         print(f"\n参考线 {name}:")
#         print(f"点数量: {len(data['x'])}")
#         print(f"总长度: {np.sum(np.sqrt(np.diff(data['x'])**2 + np.diff(data['y'])**2)):.2f} 米")
# , flag1=False):
#         self.data_list = data_listerence_lines(resampled_lines)

# if __name__ == "__main__":
#     main()
