import asyncio
import logging
import time
from typing import Dict, Any, Optional, Tuple, List

from backend.pubsub import Broadcaster
from .base_adapter import BaseAdapter

# 添加项目根目录到系统路径，以便导入device_control
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))) # REMOVED
from device_control.control_printer import PrinterControl

logger = logging.getLogger(__name__)

class PrinterAdapter(BaseAdapter):
    """打印机适配器，控制打印头移动并监控位置"""
    
    def __init__(self, moonraker_addr: str, broadcaster: Broadcaster, polling_interval: float = 1.0):
        """初始化打印机适配器
        
        Args:
            moonraker_addr: Moonraker API地址，格式为 "http://ip:port" 或 "ip:port"
            broadcaster: WebSocket广播器
            polling_interval: 位置状态轮询间隔（秒）
        """
        super().__init__("3D打印机")
        self.broadcaster = broadcaster
        self.polling_interval = polling_interval
        
        # 解析Moonraker地址，确保有完整URL格式
        if moonraker_addr.startswith("http://") or moonraker_addr.startswith("https://"):
            self.moonraker_addr = moonraker_addr
        else:
            self.moonraker_addr = f"http://{moonraker_addr}"
            
        # 获取主机部分，用于创建PrinterControl实例
        parts = self.moonraker_addr.split("://")
        if len(parts) > 1:
            host_port = parts[1].split(":")
            self.printer_ip = host_port[0]
            self.printer_port = int(host_port[1]) if len(host_port) > 1 else 7125
        else:
            host_port = parts[0].split(":")
            self.printer_ip = host_port[0]
            self.printer_port = int(host_port[1]) if len(host_port) > 1 else 7125
            
        self.printer = None
        self.topic = "hardware_status:printer"
        
        # 最后一次已知位置
        self.last_position = (0, 0, 0)
        
        # 移动操作的状态
        self.is_moving = False
        self.target_position = None
        self.move_start_time = None
    
    async def initialize(self) -> bool:
        """初始化打印机连接
        
        Returns:
            连接是否成功
        """
        try:
            # 创建PrinterControl实例
            self.printer = PrinterControl(
                ip=self.printer_ip,
                port=self.printer_port,
                move_speed=150  # 默认移动速度
            )
            
            # 尝试获取当前位置，验证连接
            position = self.printer.get_current_position()
            if position:
                logger.info(f"打印机连接成功，当前位置: {position}")
                
                # 更新位置状态
                self.last_position = position
                await self.update_status({
                    "position": {
                        "x": position[0],
                        "y": position[1],
                        "z": position[2]
                    },
                    "is_moving": False
                })
                
                return True
            else:
                logger.error("无法获取打印机位置")
                return False
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"打印机初始化失败: {e}", exc_info=True)
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """获取当前打印机状态
        
        Returns:
            包含位置的字典
        """
        # 先更新一下当前位置
        if self.printer:
            try:
                position = self.printer.get_current_position()
                if position:
                    self.last_position = position
                    self._status.update({
                        "position": {
                            "x": position[0],
                            "y": position[1],
                            "z": position[2]
                        }
                    })
            except:
                pass  # 忽略查询错误，使用上次位置
                
        return self._status.copy()
    
    async def move_to(self, x: float, y: float, z: float) -> bool:
        """移动打印头到指定位置
        
        Args:
            x: X坐标
            y: Y坐标
            z: Z坐标
            
        Returns:
            移动操作是否成功开始
        """
        if not self.printer:
            logger.error("打印机未初始化")
            return False
            
        try:
            # 更新状态为移动中
            self.is_moving = True
            self.target_position = (x, y, z)
            self.move_start_time = time.time()
            
            await self.update_status({
                "is_moving": True,
                "target_position": {
                    "x": x,
                    "y": y,
                    "z": z
                }
            })
            
            # 创建一个后台任务执行移动
            # 注意：这里使用asyncio.create_task创建真正的后台任务，
            # 而不是直接在当前协程中同步等待移动完成
            move_task = asyncio.create_task(self._execute_move(x, y, z))
            
            # 我们不等待任务完成
            return True
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"启动移动操作失败: {e}", exc_info=True)
            
            # 重置移动状态
            self.is_moving = False
            self.target_position = None
            self.move_start_time = None
            
            await self.update_status({
                "is_moving": False,
                "error": str(e)
            })
            
            return False
    
    async def move_to_grid_position(self, grid_number: int) -> bool:
        """移动到网格位置
        
        Args:
            grid_number: 网格位置编号(1-50)
            
        Returns:
            是否成功启动移动
        """
        if not self.printer:
            logger.error("打印机未初始化")
            return False
            
        try:
            # 使用PrinterControl的move_to_grid_position方法
            # 由于这是一个阻塞操作，需要在后台任务中执行
            move_task = asyncio.create_task(self._execute_grid_move(grid_number))
            
            return True
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"启动网格移动操作失败: {e}", exc_info=True)
            return False
    
    async def home(self) -> bool:
        """执行归位操作
        
        Returns:
            是否成功启动归位
        """
        if not self.printer:
            logger.error("打印机未初始化")
            return False
            
        try:
            # 使用PrinterControl的home方法
            # 由于这是一个阻塞操作，需要在后台任务中执行
            home_task = asyncio.create_task(self._execute_home())
            
            return True
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"启动归位操作失败: {e}", exc_info=True)
            return False
    
    async def _execute_move(self, x: float, y: float, z: float):
        """执行移动的后台任务
        
        Args:
            x: X坐标
            y: Y坐标
            z: Z坐标
        """
        try:
            # 调用PrinterControl的move_to方法
            # 注意：该方法可能是阻塞的，所以在单独的任务中运行
            result = self.printer.move_to(x, y, z)
            
            # 更新完成状态
            self.is_moving = False
            
            if result:
                # 移动成功完成
                logger.info(f"移动到 ({x}, {y}, {z}) 完成")
                
                # 获取最终位置
                position = self.printer.get_current_position()
                if position:
                    self.last_position = position
                    x_final, y_final, z_final = position
                else:
                    # 如果无法获取，使用目标位置
                    x_final, y_final, z_final = x, y, z
                    
                await self.update_status({
                    "is_moving": False,
                    "position": {
                        "x": x_final,
                        "y": y_final,
                        "z": z_final
                    },
                    "completed": True
                })
            else:
                # 移动被中止
                logger.warning(f"移动到 ({x}, {y}, {z}) 被中止")
                await self.update_status({
                    "is_moving": False,
                    "error": "移动被中止",
                    "completed": False
                })
        except Exception as e:
            logger.error(f"移动执行错误: {e}", exc_info=True)
            await self.update_status({
                "is_moving": False,
                "error": str(e),
                "completed": False
            })
    
    async def _execute_grid_move(self, grid_number: int):
        """执行网格移动的后台任务
        
        Args:
            grid_number: 网格位置编号
        """
        try:
            # 更新状态为移动中
            self.is_moving = True
            await self.update_status({
                "is_moving": True,
                "target_grid": grid_number
            })
            
            # 调用PrinterControl的move_to_grid_position方法
            result = self.printer.move_to_grid_position(grid_number)
            
            # 更新完成状态
            self.is_moving = False
            
            if result:
                # 移动成功完成
                logger.info(f"移动到网格位置 {grid_number} 完成")
                
                # 获取最终位置
                position = self.printer.get_current_position()
                if position:
                    self.last_position = position
                    x_final, y_final, z_final = position
                    
                    await self.update_status({
                        "is_moving": False,
                        "position": {
                            "x": x_final,
                            "y": y_final,
                            "z": z_final
                        },
                        "current_grid": grid_number,
                        "completed": True
                    })
            else:
                # 移动被中止
                logger.warning(f"移动到网格位置 {grid_number} 被中止")
                await self.update_status({
                    "is_moving": False,
                    "error": "移动被中止",
                    "completed": False
                })
        except Exception as e:
            logger.error(f"网格移动执行错误: {e}", exc_info=True)
            await self.update_status({
                "is_moving": False,
                "error": str(e),
                "completed": False
            })
    
    async def _execute_home(self):
        """执行归位的后台任务"""
        try:
            # 更新状态为归位中
            self.is_moving = True
            await self.update_status({
                "is_moving": True,
                "homing": True
            })
            
            # 调用PrinterControl的home方法
            result = self.printer.home()
            
            # 更新完成状态
            self.is_moving = False
            
            if result:
                # 归位成功完成
                logger.info("归位完成")
                
                # 获取最终位置
                position = self.printer.get_current_position()
                if position:
                    self.last_position = position
                    x_final, y_final, z_final = position
                    
                    await self.update_status({
                        "is_moving": False,
                        "homing": False,
                        "position": {
                            "x": x_final,
                            "y": y_final,
                            "z": z_final
                        },
                        "completed": True
                    })
            else:
                # 归位被中止
                logger.warning("归位被中止")
                await self.update_status({
                    "is_moving": False,
                    "homing": False,
                    "error": "归位被中止",
                    "completed": False
                })
        except Exception as e:
            logger.error(f"归位执行错误: {e}", exc_info=True)
            await self.update_status({
                "is_moving": False,
                "homing": False,
                "error": str(e),
                "completed": False
            })
    
    async def _monitor_loop(self):
        """打印机状态监控循环，定期获取并广播当前位置"""
        logger.info("打印机状态监控开始")
        
        while self.monitoring:
            try:
                if self.printer:
                    # 获取当前位置
                    position = self.printer.get_current_position()
                    if position:
                        # 更新内部状态
                        self.last_position = position
                        
                        # 计算已完成的移动进度
                        progress = None
                        if self.is_moving and self.target_position and self.move_start_time:
                            elapsed = time.time() - self.move_start_time
                            
                            # 这里可以实现一个简单的进度模拟
                            # 假设典型移动需要3秒完成
                            progress = min(1.0, elapsed / 3.0)
                        
                        # 更新状态
                        status_update = {
                            "position": {
                                "x": position[0],
                                "y": position[1],
                                "z": position[2]
                            },
                            "is_moving": self.is_moving
                        }
                        
                        if progress is not None:
                            status_update["progress"] = progress
                            
                        # 广播状态
                        await self.update_status(status_update)
                        
            except Exception as e:
                logger.error(f"打印机状态监控异常: {e}", exc_info=True)
                
            # 等待下一个轮询周期
            await asyncio.sleep(self.polling_interval)
            
        logger.info("打印机状态监控停止")
    
    async def update_status(self, status_data: Dict[str, Any], topic: Optional[str] = None):
        """更新状态并广播
        
        Args:
            status_data: 状态数据
            topic: 自定义主题，默认使用printer主题
        """
        # 更新内部状态
        self._status.update(status_data)
        
        # 广播到WebSocket
        await self.broadcaster.publish(topic or self.topic, status_data) 