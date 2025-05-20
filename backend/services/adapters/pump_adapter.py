import asyncio
import sys
import os
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple, List, Union, Dict

from backend.pubsub import Broadcaster
from .base_adapter import BaseAdapter

# 添加项目根目录到系统路径，以便导入core_api
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))) # REMOVED
from core_api.pump_proxy import PumpProxy, MoonrakerError

logger = logging.getLogger(__name__)

class PumpStatus:
    """泵状态常量定义"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"

class PumpAdapter(BaseAdapter):
    """泵适配器，监控泵状态并通过WebSocket广播"""
    
    def __init__(self, moonraker_addr: str, broadcaster: Broadcaster, polling_interval: float = 0.5, ws_listener = None):
        """初始化泵适配器
        
        Args:
            moonraker_addr: Moonraker API地址，格式为 "http://ip:port" 或 "ip:port"
            broadcaster: WebSocket广播器
            polling_interval: 状态轮询间隔（秒）
            ws_listener: MoonrakerWebsocketListener实例，用于监听WebSocket消息
        """
        super().__init__("泵")
        self.broadcaster = broadcaster
        self.polling_interval = polling_interval
        self.ws_listener = ws_listener  # 保存WebSocket监听器引用
        
        # 解析Moonraker地址
        if not moonraker_addr.startswith("http://") and not moonraker_addr.startswith("https://"):
            moonraker_addr = f"http://{moonraker_addr}"
            
        self.moonraker_addr = moonraker_addr
        self.pump_proxy = None
        self.topic = "hardware_status:pump"
        self.base_topic = "hardware_status:pump"
        
        # 泵操作追踪
        self.current_operation = None  # 当前操作类型: "dispense_auto", "dispense_timed"
        self.start_time = None         # 操作开始时间
        self.end_time = None           # 预计结束时间
        self.total_duration = None     # 总操作时长（秒）
        self.target_volume = None      # 目标体积（毫升）
        self.current_port = None       # 当前泵端口
        self.current_unit = None       # 当前泵单元
        self.flow_rate = None          # 流速（毫升/秒）
        
        # 加载泵校准数据
        self.calibration_data = {}
        self.pump_config = {}
        self.speed_rules = {
            0.5: 20.0,   # 默认值，小于0.5ml使用20 RPM
            1.0: 45.0,   # 默认值，小于等于1.0ml使用45 RPM
            3.0: 60.0,   # 默认值，1.0-3.0ml使用60 RPM
            10.0: 90.0,  # 默认值，3.0-10.0ml使用90 RPM
            float("inf"): 120.0  # 默认值，大于10.0ml使用120 RPM
        }
        self._load_calibration_data()
    
    def _load_calibration_data(self):
        """加载泵校准数据和配置"""
        try:
            # 尝试加载pump_config.json
            config_path = "calibration_cache/pump_config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.pump_config = json.load(f)
                logger.info(f"已加载泵配置: {config_path}")
                
                # 提取转速规则
                if "speed_rules" in self.pump_config:
                    rules = {}
                    for rule in self.pump_config["speed_rules"]:
                        volume = float(rule["volume"]) if rule["volume"] != "inf" else float("inf")
                        rules[volume] = float(rule["speed"])
                    if rules:
                        self.speed_rules = rules
                        logger.info(f"已加载转速规则: {self.speed_rules}")
            
            # 尝试加载校准文件
            calibration_files = [f for f in os.listdir("calibration_cache") if f.startswith("calibration_") and f.endswith(".json")]
            for cal_file in calibration_files:
                try:
                    with open(f"calibration_cache/{cal_file}", 'r') as f:
                        cal_data = json.load(f)
                    
                    # 提取端口和单元号
                    parts = cal_file.replace("calibration_", "").replace(".json", "").split("_")
                    if len(parts) >= 1:
                        unit = int(parts[0])
                        self.calibration_data[unit] = cal_data
                        logger.info(f"已加载校准数据: unit={unit}, file={cal_file}")
                except Exception as e:
                    logger.warning(f"加载校准文件失败 {cal_file}: {e}")
        except Exception as e:
            logger.warning(f"加载泵配置或校准数据失败: {e}")
    
    def _get_speed_for_volume(self, volume_ml: float) -> float:
        """根据体积选择合适的转速
        
        Args:
            volume_ml: 目标体积（毫升）
            
        Returns:
            推荐的转速（RPM）
        """
        # 按照control_pump.py中的逻辑选择转速
        if volume_ml < 0.5 and 0.5 in self.speed_rules:
            return self.speed_rules[0.5]
        if volume_ml <= 1.0 and 1.0 in self.speed_rules:
            return self.speed_rules[1.0]
        if 1.0 < volume_ml <= 3.0 and 3.0 in self.speed_rules:
            return self.speed_rules[3.0]
        if 3.0 < volume_ml <= 10.0 and 10.0 in self.speed_rules:
            return self.speed_rules[10.0]
        if volume_ml > 10.0 and float("inf") in self.speed_rules:
            return self.speed_rules[float("inf")]
        
        # 如果没有匹配规则，使用默认最低转速
        return min(self.speed_rules.values())
    
    def _estimate_dispense_time(self, volume_ml: float, rate_mlps: float = None, rpm: float = None) -> float:
        """估算泵送时间
        
        Args:
            volume_ml: 目标体积（毫升）
            rate_mlps: 泵送速率（毫升/秒），如果提供则优先使用
            rpm: 转速（RPM），如果没有提供rate_mlps则根据转速和校准数据计算
            
        Returns:
            估计的泵送时间（秒）
        """
        if rate_mlps is not None and rate_mlps > 0:
            # 直接使用提供的流速计算
            return volume_ml / rate_mlps
        
        if rpm is None:
            # 如果没有提供转速，则根据体积选择转速
            rpm = self._get_speed_for_volume(volume_ml)
        
        # 使用校准数据计算流速
        # 默认流速 - 如果没有校准数据，使用简单估算
        # 假设100RPM约等于5ml/s（这是一个非常粗略的估计）
        estimated_rate = rpm * 0.05  # ml/s
        
        # 尝试从校准数据中获取更准确的估算
        if self.current_unit in self.calibration_data:
            cal_data = self.calibration_data[self.current_unit]
            # 校准数据通常是RPM到ml/min的映射
            # 找到最接近的RPM值
            rpm_values = sorted([float(r) for r in cal_data.keys()])
            
            if rpm_values:
                # 找到小于等于目标RPM的最大值
                closest_rpm = None
                for val in rpm_values:
                    if float(val) <= rpm and (closest_rpm is None or float(val) > closest_rpm):
                        closest_rpm = float(val)
                
                # 如果没有找到小于等于的值，取最小值
                if closest_rpm is None and rpm_values:
                    closest_rpm = min(rpm_values)
                
                if closest_rpm is not None:
                    # 获取流速 (ml/min) 并转换为 ml/s
                    flow_mlpm = float(cal_data[str(closest_rpm)])
                    estimated_rate = flow_mlpm / 60.0  # 转换为ml/s
        
        # 计算时间
        if estimated_rate > 0:
            return volume_ml / estimated_rate
        else:
            # 防止除零错误
            return 60.0  # 默认假设需要1分钟
    
    async def initialize(self) -> bool:
        """初始化泵连接
        
        Returns:
            连接是否成功
        """
        try:
            # 创建PumpProxy实例，传入WebSocket监听器
            self.pump_proxy = PumpProxy(self.moonraker_addr, listener=self.ws_listener)
            
            # 尝试发送一个简单的命令测试连接
            try:
                # 发送一个空命令或者获取状态命令来测试连接
                # 由于没有QUERY_PUMP_STATUS宏，我们不能直接获取泵状态
                # 所以这里只是测试与Moonraker的连接
                # 使用异步方法发送测试命令
                _ = await self.pump_proxy._send_async("M115")  # 发送版本信息请求作为测试
                logger.info(f"泵连接成功: {self.moonraker_addr}")
            except Exception as e:
                logger.error(f"测试泵连接失败: {e}")
                return False
            
            # 初始化成功
            await self.update_status({
                "status": PumpStatus.IDLE,
                "message": "泵已准备就绪"
            })
            
            return True
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"泵初始化失败: {e}", exc_info=True)
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """获取当前泵状态
        
        Returns:
            包含泵状态的字典
        """
        # 计算当前进度（如果有进行中的操作）
        if self.current_operation and self.start_time:
            now = time.time()
            if self.total_duration and self.total_duration > 0:
                elapsed = now - self.start_time
                progress = min(1.0, elapsed / self.total_duration)
                remaining = max(0, self.total_duration - elapsed)
                
                # 更新内部状态
                self._status.update({
                    "progress": progress,
                    "elapsed_seconds": elapsed,
                    "remaining_seconds": remaining
                })
                
                # 检查是否已完成
                if elapsed >= self.total_duration:
                    # 操作可能已完成，但我们没有收到通知
                    # 在下一次轮询中会更新状态为完成
                    pass
        
        # 返回当前状态
        return self._status.copy()
    
    async def dispense(self, volume: float) -> bool:
        """分配指定体积的液体
        
        Args:
            volume: 分配的体积（微升）
            
        Returns:
            操作是否成功
        """
        if not self.pump_proxy:
            logger.error("泵未初始化")
            return False
            
        try:
            # 设置追踪参数
            self.current_operation = "dispense_auto"
            self.start_time = time.time()
            self.target_volume = volume / 1000.0  # 转换为毫升
            
            # 根据体积计算合适的转速
            rpm = self._get_speed_for_volume(self.target_volume)
            
            # 估算持续时间
            self.total_duration = self._estimate_dispense_time(self.target_volume, rpm=rpm)
            self.end_time = self.start_time + self.total_duration
            
            # 默认使用单元0，端口0
            self.current_unit = 0
            self.current_port = 0
            
            # 更新状态为运行中
            await self.update_status({
                "status": PumpStatus.RUNNING,
                "operation": "dispense_auto",
                "target_volume_ml": self.target_volume,
                "estimated_time_seconds": self.total_duration,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "expected_end_time": datetime.fromtimestamp(self.end_time).isoformat(),
                "port": self.current_port,
                "unit": self.current_unit,
                "rpm": rpm,
                "progress": 0.0
            })
            
            # 执行分配操作
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.pump_proxy.dispense_auto(self.target_volume)
            )
            
            if not result:
                # 分配失败
                logger.error(f"分配操作失败: {volume}μL")
                await self.update_status({
                    "status": PumpStatus.ERROR,
                    "error": "分配操作失败"
                })
                
                # 重置操作状态
                self._reset_operation_state()
                return False
            
            # 我们不能直接知道操作已完成（因为没有状态查询机制）
            # 由_monitor_loop根据计时决定何时将状态更新为完成
            
            return True
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"分配操作异常: {e}", exc_info=True)
            
            # 更新为错误状态
            await self.update_status({
                "status": PumpStatus.ERROR,
                "error": str(e)
            })
            
            # 重置操作状态
            self._reset_operation_state()
            
            return False
    
    async def dispense_timed(self, duration_ms: int, rate: float) -> bool:
        """按时间分配液体
        
        Args:
            duration_ms: 持续时间（毫秒）
            rate: 分配速率（微升/秒）
            
        Returns:
            操作是否成功
        """
        if not self.pump_proxy:
            logger.error("泵未初始化")
            return False
            
        try:
            # 设置追踪参数
            self.current_operation = "dispense_timed"
            self.start_time = time.time()
            self.total_duration = duration_ms / 1000.0  # 转换为秒
            self.end_time = self.start_time + self.total_duration
            
            # 计算体积和流速
            rate_mlps = rate / 1000.0  # 微升/秒 转换为 毫升/秒
            self.target_volume = rate_mlps * self.total_duration  # 毫升
            self.flow_rate = rate_mlps
            
            # 根据体积计算合适的转速
            rpm = self._get_speed_for_volume(self.target_volume)
            
            # 默认使用单元0，端口0
            self.current_unit = 0
            self.current_port = 0
            
            # 更新状态为运行中
            await self.update_status({
                "status": PumpStatus.RUNNING,
                "operation": "dispense_timed",
                "duration_ms": duration_ms,
                "rate_ulps": rate,
                "target_volume_ml": self.target_volume,
                "estimated_time_seconds": self.total_duration,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "expected_end_time": datetime.fromtimestamp(self.end_time).isoformat(),
                "port": self.current_port,
                "unit": self.current_unit,
                "rpm": rpm,
                "progress": 0.0
            })
            
            # 为了最大程度兼容pump_proxy接口，我们使用PumpProxy的方法
            # 计算需要的量
            volume_ml = self.target_volume  # 总体积，毫升
            speed_rpm = rpm                # 转速，RPM
            
            # 执行分配操作
            result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.pump_proxy.dispense_speed(volume_ml, speed_rpm)
            )
            
            if not result:
                # 分配失败
                logger.error(f"按时间分配操作失败: {duration_ms}ms, {rate}μL/s")
                await self.update_status({
                    "status": PumpStatus.ERROR,
                    "error": "按时间分配操作失败"
                })
                
                # 重置操作状态
                self._reset_operation_state()
                return False
            
            # 由_monitor_loop根据计时决定何时将状态更新为完成
            
            return True
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"按时间分配操作异常: {e}", exc_info=True)
            
            # 更新为错误状态
            await self.update_status({
                "status": PumpStatus.ERROR,
                "error": str(e)
            })
            
            # 重置操作状态
            self._reset_operation_state()
            
            return False
    
    async def stop(self) -> bool:
        """停止当前泵操作
        
        Returns:
            操作是否成功
        """
        if not self.pump_proxy:
            logger.error("泵未初始化")
            return False
            
        try:
            # 执行停止操作 - 直接使用异步方法
            result_dict = await self.pump_proxy.emergency_stop()
            
            # 无论成功与否，都更新为停止状态
            await self.update_status({
                "status": PumpStatus.IDLE,
                "operation": None,
                "stopped": True,
                "stop_time": datetime.now().isoformat()
            })
            
            # 重置操作状态
            self._reset_operation_state()
            
            return result_dict.get("success", False)
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"停止泵操作异常: {e}", exc_info=True)
            
            # 尝试重置状态
            self._reset_operation_state()
            return False
    
    def _reset_operation_state(self):
        """重置操作状态追踪变量"""
        self.current_operation = None
        self.start_time = None
        self.end_time = None
        self.total_duration = None
        self.target_volume = None
        self.flow_rate = None
    
    async def _monitor_loop(self):
        """泵状态监控循环，基于时间模拟泵进度"""
        logger.info("泵状态监控启动")
        
        while self.monitoring:
            try:
                # 检查是否有正在进行的操作
                if self.current_operation and self.start_time and self.total_duration:
                    now = time.time()
                    elapsed = now - self.start_time
                    
                    # 计算进度
                    progress = min(1.0, elapsed / self.total_duration)
                    remaining = max(0, self.total_duration - elapsed)
                    
                    if progress < 1.0:
                        # 操作进行中，更新进度
                        await self.update_status({
                            "progress": progress,
                            "elapsed_seconds": elapsed,
                            "remaining_seconds": remaining
                        })
                    else:
                        # 操作已完成（基于时间判断）
                        logger.info(f"泵操作完成: {self.current_operation}")
                        
                        # 估算已泵送的体积
                        pumped_volume = self.target_volume
                        if self.current_operation == "dispense_timed" and self.flow_rate:
                            # 根据实际运行时间计算体积
                            pumped_volume = self.flow_rate * min(elapsed, self.total_duration)
                        
                        # 更新为完成状态
                        await self.update_status({
                            "status": PumpStatus.COMPLETED,
                            "progress": 1.0,
                            "elapsed_seconds": self.total_duration,
                            "remaining_seconds": 0,
                            "pumped_volume_ml": pumped_volume,
                            "completed": True,
                            "end_time": datetime.now().isoformat()
                        })
                        
                        # 重置操作状态
                        self._reset_operation_state()
                
            except Exception as e:
                logger.error(f"泵状态监控异常: {e}", exc_info=True)
                
            # 等待下一个轮询周期
            await asyncio.sleep(self.polling_interval)
            
        logger.info("泵状态监控停止")
    
    async def update_status(self, status_data: Dict[str, Any], topic: Optional[str] = None):
        """更新状态并广播
        
        Args:
            status_data: 状态数据
            topic: 自定义主题，默认使用为当前泵生成的主题
        """
        # 更新内部状态
        self._status.update(status_data)
        
        # 确定广播主题
        if topic:
            actual_topic = topic
        elif self.current_port is not None and self.current_unit is not None:
            actual_topic = f"{self.base_topic}:{self.current_port}:{self.current_unit}"
        else:
            actual_topic = self.base_topic
        
        # 广播到WebSocket
        await self.broadcaster.publish(actual_topic, status_data) 

    async def dispense_auto(self, pump_index, volume, speed, direction=1):
        if not self.initialized:
            logger.error("PumpAdapter未初始化，无法执行dispense_auto")
            raise ValueError("泵未初始化")
        
        self._stop_event = False # 重置停止标志
        
        # 记录API调用开始时间点
        api_call_start_time = time.time()
        
        # 转换volume从μL到ml
        volume_ml = volume / 1000.0
        
        # 立即使用预估参数创建初始状态，使UI立即显示正在运行
        # 获取预估转速
        rpm = self._get_speed_for_volume(volume_ml)
        
        # 使用校准数据预估时长
        estimated_duration = self._estimate_dispense_time(volume_ml, rpm=rpm)
        
        # 立即更新状态并开始显示进度条
        self.status = {
            "running": True,
            "pump_index": pump_index,
            "volume": volume, # μL
            "progress": 0,
            "direction": direction,
            "elapsed_time_seconds": 0, # 泵送操作的已过时间，初始为0
            "total_duration_seconds": estimated_duration, # 预估总时长
            "rpm": rpm,
            "revolutions": None, # 稍后从proxy获取
            "raw_response": "正在请求泵服务..."
        }
        await self.broadcast_status() # 立即广播，让前端知道操作已开始
        
        # 立即启动进度监控任务，使用预估参数
        progress_monitor_task = asyncio.create_task(
            self._monitor_pump_progress(
                total_duration_seconds_to_monitor=estimated_duration,
                initial_elapsed_for_progress_calc=0,
                total_estimated_for_progress_calc=estimated_duration
            )
        )
        
        # 并行发送泵送命令，但不阻塞UI更新
        try:
            # 异步调用泵代理 (不等待命令完成，只是启动)
            proxy_response_task = asyncio.create_task(
                self.pump_proxy.dispense_auto(
                    volume_ml=volume_ml, 
                    speed=speed, 
                    direction=direction
                )
            )
            
            # 设置超时，避免无限等待
            try:
                proxy_response = await asyncio.wait_for(proxy_response_task, timeout=5.0)
                
                # 计算API调用和参数获取所花费的时间
                command_send_and_param_fetch_duration = time.time() - api_call_start_time
                logger.info(f"从发送命令到获取参数用时: {command_send_and_param_fetch_duration:.2f}秒")
                
                # 检查proxy_response并更新参数
                if proxy_response and proxy_response.get("success", False):
                    # 更新状态中的实际参数，如果可用
                    actual_rpm = proxy_response.get("rpm")
                    actual_revolutions = proxy_response.get("revolutions")
                    actual_duration = proxy_response.get("estimated_duration", 0)
                    
                    if actual_rpm and actual_revolutions and actual_duration > 0:
                        # 获取了更准确的参数，更新状态
                        # 注意：我们不会修改progress_monitor_task，让它继续使用最初的预估
                        logger.info(f"泵 {pump_index} dispense_auto: 数据来源={proxy_response.get('source', 'unknown')}, "
                                    f"RPM={actual_rpm}, 圈数={actual_revolutions}, "
                                    f"实际泵送时长={actual_duration:.2f}s")
                        
                        # 更新状态，但不改变进度监控参数
                        self.status.update({
                            "rpm": actual_rpm,
                            "revolutions": actual_revolutions,
                            # 不更新 total_duration_seconds，避免进度条跳变
                            "raw_response": proxy_response.get("raw_response", "")
                        })
                        await self.broadcast_status()
                else:
                    logger.warning(f"泵 {pump_index} dispense_auto 无法获取准确参数或命令失败")
            
            except asyncio.TimeoutError:
                logger.warning(f"等待泵 {pump_index} 的参数超时，进度条仍将使用预估参数")
                # 超时后，取消任务但不停止泵操作
                proxy_response_task.cancel()
            
            # 无论命令或参数获取成功与否，进度条都已经开始
            return True
        
        except Exception as e:
            logger.error(f"泵 {pump_index} dispense_auto 执行异常: {e}", exc_info=True)
            # 取消进度监控任务
            if progress_monitor_task and not progress_monitor_task.done():
                progress_monitor_task.cancel()
            # 更新状态为错误
            self.status.update({"running": False, "raw_response": f"执行错误: {str(e)}"})
            await self.broadcast_status()
            return False 