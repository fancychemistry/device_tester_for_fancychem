import requests
import time
import math
import keyboard
import threading


class PrinterControl:
    def __init__(self, ip="192.168.51.168", port=7125, move_speed=150,
                 general_min_pos=(0, 0, 75), general_max_pos=(215, 190, 200),
                 grid_min_pos=(6, 100, 75), grid_max_pos=(174, 173, 75),
                 min_pos=None, max_pos=None):
        """初始化打印机控制对象。

        参数:
            ip (str): 打印机 IP 地址，默认值为 "192.168.51.168"
            port (int): 打印机端口号，默认值为 7125
            move_speed (float): 移动速度 (mm/s)，默认值为 150
            general_min_pos (tuple): 一般移动安全范围最小坐标 (x, y, z)，默认值为 (0, 0, 75)
            general_max_pos (tuple): 一般移动安全范围最大坐标 (x, y, z)，默认值为 (215, 190, 200)
            grid_min_pos (tuple): 网格移动安全范围最小坐标 (x, y, z)，默认值为 (4, 105, 75)
            grid_max_pos (tuple): 网格移动安全范围最大坐标 (x, y, z)，默认值为 (174, 177, 75)
            min_pos (tuple): 自定义安全范围最小坐标 (x, y, z)，如果提供则覆盖grid_min_pos
            max_pos (tuple): 自定义安全范围最大坐标 (x, y, z)，如果提供则覆盖grid_max_pos
        """
        self.ip = ip
        self.port = port
        self.move_speed = move_speed
        self.general_min_pos = general_min_pos
        self.general_max_pos = general_max_pos
        self.emergency_stop_flag = False

        # 如果提供了min_pos和max_pos，则使用它们来覆盖grid_min_pos和grid_max_pos
        if min_pos is not None:
            self.grid_min_pos = min_pos
        else:
            self.grid_min_pos = grid_min_pos

        if max_pos is not None:
            self.grid_max_pos = max_pos
        else:
            self.grid_max_pos = grid_max_pos

        # 设置紧急停止键监听
        self._setup_emergency_stop()

    def _setup_emergency_stop(self):
        """设置紧急停止键监听"""
        keyboard.on_press_key("esc", self._emergency_stop_callback)
        print("紧急停止功能已启用，按下ESC键可停止移动")

    def _emergency_stop_callback(self, e):
        """ESC键回调函数"""
        self.emergency_stop()

    def emergency_stop(self):
        """紧急停止所有移动"""
        self.emergency_stop_flag = True
        print("\n紧急停止被触发！停止所有移动...")
        # 立即发送停止命令
        self.send_gcode_command("M112")  # 紧急停止
        print("已发送紧急停止命令")

    def reset_emergency_stop(self):
        """重置紧急停止标志"""
        self.emergency_stop_flag = False
        print("紧急停止状态已重置")

    def send_gcode_command(self, command):
        """发送 G-code 命令到打印机。

        参数:
            command (str): 要发送的 G-code 命令
        """
        url = f"http://{self.ip}:{self.port}/printer/gcode/script"
        payload = {"script": command}
        try:
            response = requests.post(url, json=payload)
            if response.status_code == 200:
                print("命令发送成功")
                return True
            else:
                print(f"命令发送失败，状态码: {response.status_code}")
                # 尝试解析错误信息
                try:
                    error_info = response.json()
                    print(f"错误详情: {error_info}")
                except:
                    pass
                
                # 如果是400错误，可能是命令格式有问题
                if response.status_code == 400:
                    print(f"400错误通常表示命令格式错误或参数不正确: {command}")
                    
                return False
        except Exception as e:
            print(f"发送命令时出错: {e}")
            return False

    def get_current_position(self):
        """获取打印头的当前坐标。

        返回:
            tuple: (x, y, z) 当前坐标，若失败则返回 None
        """
        url = f"http://{self.ip}:{self.port}/printer/objects/query?toolhead=position"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                position = data["result"]["status"]["toolhead"]["position"]
                return position[0], position[1], position[2]
            else:
                print(f"获取坐标失败，状态码: {response.status_code}")
                return None
        except Exception as e:
            print(f"获取坐标出错: {e}")
            return None

    def calculate_move_time(self, current_pos, target_x, target_y, target_z):
        """估算移动时间。

        参数:
            current_pos (tuple): 当前坐标 (x, y, z)
            target_x (float): 目标 X 坐标
            target_y (float): 目标 Y 坐标
            target_z (float): 目标 Z 坐标

        返回:
            float: 移动时间 (秒)
        """
        if current_pos is None:
            return 0
        dx = target_x - current_pos[0]
        dy = target_y - current_pos[1]
        dz = target_z - current_pos[2]
        distance = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return distance / self.move_speed

    def is_position_safe(self, x, y, z, min_pos=None, max_pos=None):
        """检查目标位置是否在指定安全范围内。

        参数:
            x (float): 目标 X 坐标
            y (float): 目标 Y 坐标
            z (float): 目标 Z 坐标
            min_pos (tuple): 安全范围最小坐标，默认使用grid_min_pos
            max_pos (tuple): 安全范围最大坐标，默认使用grid_max_pos

        返回:
            bool: 如果位置安全返回 True，否则返回 False
        """
        # 如果没有提供min_pos和max_pos，则使用grid的安全范围
        if min_pos is None:
            min_pos = self.grid_min_pos
        if max_pos is None:
            max_pos = self.grid_max_pos

        return (min_pos[0] <= x <= max_pos[0] and
                min_pos[1] <= y <= max_pos[1] and
                min_pos[2] <= z <= max_pos[2])

    def wait_for_move_completion(self, expected_time):
        """等待移动完成，支持紧急停止。

        参数:
            expected_time (float): 预计移动时间 (秒)

        返回:
            bool: 如果正常完成返回 True，如果被紧急停止返回 False
        """
        start_time = time.time()
        print(f"预计移动时间: {expected_time:.2f} 秒，等待中...")

        # 分段等待，以便能响应紧急停止
        interval = 0.1  # 检查间隔（秒）
        elapsed = 0

        while elapsed < expected_time:
            if self.emergency_stop_flag:
                print("移动被紧急停止！")
                return False

            time.sleep(interval)
            elapsed = time.time() - start_time

            # 打印进度（每秒更新一次）
            if int(elapsed) > int(elapsed - interval):
                print(f"移动进度: {min(100, int(elapsed / expected_time * 100))}%", end="\r")

        print("移动完成                ")  # 额外的空格用于覆盖进度百分比
        return True

    def move_to(self, x, y, z, use_general_safety=True):
        """移动打印头到指定位置，并等待移动完成。

        参数:
            x (float): 目标 X 坐标
            y (float): 目标 Y 坐标
            z (float): 目标 Z 坐标
            use_general_safety (bool): 如果为True，使用一般安全范围检查；如果为False，使用网格安全范围

        返回:
            bool: 如果移动成功完成返回 True，如果被紧急停止返回 False
        """
        # 重置紧急停止标志
        self.reset_emergency_stop()

        # 根据参数选择使用哪个安全范围进行检查
        if use_general_safety:
            if not self.is_position_safe(x, y, z, self.general_min_pos, self.general_max_pos):
                print(f"错误：目标位置 ({x:.2f}, {y:.2f}, {z:.2f}) 超出一般安全范围！")
                return False
        else:
            if not self.is_position_safe(x, y, z):
                print(f"错误：目标位置 ({x:.2f}, {y:.2f}, {z:.2f}) 超出网格安全范围！")
                return False

        current_pos = self.get_current_position()
        if current_pos is None:
            print("无法获取当前位置，移动取消")
            return False

        formatted_x = f"{x:.2f}"
        formatted_y = f"{y:.2f}"
        formatted_z = f"{z:.2f}"

        # 发送移动命令前设置速度
        self.send_gcode_command(f"G1 F{self.move_speed * 60}")  # 转换为mm/min
        self.send_gcode_command(f"G1 X{formatted_x} Y{formatted_y} Z{formatted_z}")
        print(f"移动到: ({formatted_x}, {formatted_y}, {formatted_z})")

        move_time = self.calculate_move_time(current_pos, x, y, z)
        return self.wait_for_move_completion(move_time)

    def move_to_grid_position(self, grid_number):
        """移动打印头到指定的网格位置（1-50），使用安全移动逻辑。

        参数:
            grid_number (int): 网格位置编号（1-50）

        返回:
            bool: 如果移动成功完成返回 True，如果被紧急停止返回 False
        """
        if not 1 <= grid_number <= 50:
            print("错误：网格编号必须在1到50之间！")
            return False

        # 计算行和列
        row = (grid_number - 1) // 10 + 1  # 1-5
        col = (grid_number - 1) % 10 + 1  # 1-10

        # 计算坐标 - 新的网格定义
        # 右上角(1号位置): (174,173,75)
        # 左上角(10号位置): (6,173,75)
        # 右下角(41号位置): (174,100,75)
        # 左下角(50号位置): (6,100,75)
        x_min, x_max = 6, 174
        y_min, y_max = 100, 173
        z_height = 75

        # 网格从右到左，从上到下排列
        col_offset = col - 1  # 列偏移(0-9)
        x = x_max - col_offset * (x_max - x_min) / 9  # 从右到左
        y = y_max - (row - 1) * (y_max - y_min) / 4  # 从上到下

        # 显示计算出的坐标
        print(f"网格位置 {grid_number} 的计算坐标: ({x:.2f}, {y:.2f}, {z_height:.2f})")

        # 开始安全移动过程
        current_pos = self.get_current_position()
        if current_pos is None:
            print("无法获取当前位置，移动取消")
            return False

        # 1. 首先Z轴上升到固定安全高度（85mm），而不是累加高度
        safe_z = 85  # 固定安全高度
        print(f"第1步：Z轴上升到安全高度 {safe_z:.2f}")
        if not self.move_to(current_pos[0], current_pos[1], safe_z, use_general_safety=True):
            return False

        # 2. 移动到目标XY位置，保持Z轴在安全高度
        print(f"第2步：在安全高度移动到目标XY位置 ({x:.2f}, {y:.2f}, {safe_z:.2f})")
        if not self.move_to(x, y, safe_z, use_general_safety=True):
            return False

        # 3. 最后降低Z轴到目标高度
        print(f"第3步：Z轴下降到目标高度 {z_height:.2f}")
        if not self.move_to(x, y, z_height, use_general_safety=False):
            return False

        print(f"成功移动到网格位置 {grid_number}: ({x:.2f}, {y:.2f}, {z_height:.2f})")
        return True

    def home(self, wait_time=10):
        """执行归位操作，并等待指定时间。

        参数:
            wait_time (int): 等待时间（秒）

        返回:
            bool: 如果归位成功完成返回 True，如果被紧急停止返回 False
        """
        # 重置紧急停止标志
        self.reset_emergency_stop()

        self.send_gcode_command("G28")  # 归位
        print(f"执行归位，等待 {wait_time} 秒...")

        # 分段等待，以便能响应紧急停止
        interval = 0.1  # 检查间隔（秒）
        elapsed = 0

        while elapsed < wait_time:
            if self.emergency_stop_flag:
                print("归位操作被紧急停止！")
                return False

            time.sleep(interval)
            elapsed += interval

        print("归位完成")
        return True


# 使用示例
if __name__ == "__main__":
    try:
        printer = PrinterControl()
        print("打印机控制系统已启动")
        print("紧急停止: 按下ESC键")
        print("移动速度: 150mm/s")
        print("网格位置1: (4, 177, 75), 网格位置50: (174, 105, 75)")

        # 示例操作
        printer.home()  # 归位
        printer.move_to(50, 50, 85)  # 移动到 (50, 50, 85)，使用一般安全范围
        printer.move_to_grid_position(1)  # 移动到网格位置 1

        print("测试完成。按Ctrl+C退出程序。")

        # 保持程序运行，以便键盘监听继续工作
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n程序已退出")
    finally:
        # 清理键盘监听
        keyboard.unhook_all()