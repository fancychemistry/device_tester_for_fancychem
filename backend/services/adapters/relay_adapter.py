import asyncio
import logging
from typing import Dict, Any, Optional, List

from backend.pubsub import Broadcaster
from .base_adapter import BaseAdapter

# 添加项目根目录到系统路径，以便导入core_api
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))) # REMOVED
from core_api.relay_proxy import RelayProxy, MoonrakerError

logger = logging.getLogger(__name__)

class RelayAdapter(BaseAdapter):
    """继电器适配器，控制继电器状态并通过WebSocket广播"""
    
    def __init__(self, moonraker_addr: str, broadcaster: Broadcaster, relay_count: int = 4):
        """初始化继电器适配器
        
        Args:
            moonraker_addr: Moonraker API地址，格式为 "http://ip:port" 或 "ip:port"
            broadcaster: WebSocket广播器
            relay_count: 继电器数量，默认为4
        """
        super().__init__("继电器/阀门")
        self.broadcaster = broadcaster
        
        # 解析Moonraker地址，确保有完整URL格式
        if moonraker_addr.startswith("http://") or moonraker_addr.startswith("https://"):
            self.moonraker_addr = moonraker_addr
        else:
            self.moonraker_addr = f"http://{moonraker_addr}"
            
        self.relay_proxy = None
        self.relay_count = relay_count
        self.base_topic = "hardware_status:relay"
        
        # 跟踪继电器状态
        self.relay_states = {i: False for i in range(1, relay_count + 1)}
    
    async def initialize(self) -> bool:
        """初始化继电器连接
        
        Returns:
            连接是否成功
        """
        try:
            # 创建RelayProxy实例
            self.relay_proxy = RelayProxy(self.moonraker_addr)
            
            # 测试连接，尝试查询状态（这里没有专门的状态查询方法，可以考虑实际情况）
            # 在这里，我们简单地假设如果创建实例没有异常，则连接成功
            logger.info("继电器连接成功")
            
            # 初始化每个继电器的状态为未知
            for i in range(1, self.relay_count + 1):
                await self.update_relay_status(i, None, "unknown")
                
            return True
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"继电器初始化失败: {e}", exc_info=True)
            return False
    
    async def get_status(self) -> Dict[str, Any]:
        """获取当前继电器状态
        
        Returns:
            包含所有继电器状态的字典
        """
        return {
            "relays": self.relay_states
        }
    
    async def get_relay_state(self, relay_id: int) -> Dict[str, Any]:
        """获取特定继电器的状态
        
        由于relay_proxy.py没有提供状态查询方法，
        该方法返回最后一次操作后记录的本地状态
        
        Args:
            relay_id: 继电器ID
            
        Returns:
            继电器状态信息
        """
        # 从内部状态获取
        if relay_id in self.relay_states:
            return self.relay_states[relay_id]
        else:
            return {"state": None, "status": "unknown"}
    
    async def set_relay_on(self, idx: int) -> bool:
        """打开继电器
        
        Args:
            idx: 继电器索引
            
        Returns:
            操作是否成功
        """
        if not self.relay_proxy:
            logger.error("继电器未初始化")
            return False
            
        try:
            # 执行继电器打开操作
            self.relay_proxy.on(idx)
            
            # 更新状态并广播
            await self.update_relay_status(idx, True, "on")
            
            return True
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"继电器{idx}打开失败: {e}", exc_info=True)
            
            # 更新为错误状态
            await self.update_relay_status(idx, None, "error", str(e))
            
            return False
    
    async def set_relay_off(self, idx: int) -> bool:
        """关闭继电器
        
        Args:
            idx: 继电器索引
            
        Returns:
            操作是否成功
        """
        if not self.relay_proxy:
            logger.error("继电器未初始化")
            return False
            
        try:
            # 执行继电器关闭操作
            self.relay_proxy.off(idx)
            
            # 更新状态并广播
            await self.update_relay_status(idx, False, "off")
            
            return True
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"继电器{idx}关闭失败: {e}", exc_info=True)
            
            # 更新为错误状态
            await self.update_relay_status(idx, None, "error", str(e))
            
            return False
    
    async def toggle_relay(self, relay_id: int, state: Optional[str] = None) -> bool:
        """切换继电器状态
        
        Args:
            relay_id: 继电器索引 (与relay_proxy.py的idx参数对应)
            state: 可选，明确指定状态 ("on" 或 "off")，如果不提供则自动切换
            
        Returns:
            操作是否成功
        """
        if not self.relay_proxy:
            logger.error("继电器未初始化")
            return False
            
        try:
            # 执行继电器切换操作，直接转发参数
            self.relay_proxy.toggle(relay_id, state)
            
            # 根据请求的状态更新本地状态
            new_state = None
            status_text = "toggled"
            
            if state:
                new_state = state.lower() == "on"
                status_text = "on" if new_state else "off"
            else:
                # 如果没有明确指定状态，取反当前状态
                current_state = self.relay_states.get(relay_id, {}).get("state")
                if isinstance(current_state, bool):
                    new_state = not current_state
                    status_text = "on" if new_state else "off"
            
            # 更新状态并广播
            await self.update_relay_status(relay_id, new_state, status_text)
            
            return True
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"继电器{relay_id}切换失败: {e}", exc_info=True)
            
            # 更新为错误状态
            await self.update_relay_status(relay_id, None, "error", str(e))
            
            return False
    
    async def update_relay_status(self, idx: int, state: Optional[bool], status: str, error: str = None):
        """更新继电器状态并广播
        
        Args:
            idx: 继电器索引
            state: 继电器状态（True=开，False=关，None=未知）
            status: 状态文本
            error: 错误信息（如有）
        """
        # 更新内部状态
        relay_status = {
            "state": state,
            "status": status
        }
        
        if error:
            relay_status["error"] = error
            
        self.relay_states[idx] = relay_status
        
        # 广播到特定继电器的主题
        topic = f"{self.base_topic}:{idx}"
        await self.broadcaster.publish(topic, relay_status)
        
        # 同时更新适配器的总状态
        self._status = {
            "relays": self.relay_states
        }
    
    async def _monitor_loop(self):
        """继电器状态监控循环
        
        注意：由于继电器通常没有状态反馈机制，所以这里监控循环
        主要用于定期广播当前已知的继电器状态
        """
        logger.info("继电器状态监控开始")
        
        while self.monitoring:
            try:
                # 广播所有继电器的当前状态
                for idx, state in self.relay_states.items():
                    topic = f"{self.base_topic}:{idx}"
                    await self.broadcaster.publish(topic, state)
                    
                # 同时发送汇总状态
                await self.broadcaster.publish(self.base_topic, {
                    "relays": self.relay_states
                })
            except Exception as e:
                logger.error(f"继电器状态广播异常: {e}", exc_info=True)
                
            # 监控间隔较长，如每30秒广播一次
            await asyncio.sleep(30)
            
        logger.info("继电器状态监控停止") 