import abc
import logging
from typing import Dict, Any, Optional, Callable
import asyncio

logger = logging.getLogger(__name__)

class BaseAdapter(abc.ABC):
    """硬件适配器基类，定义与硬件交互的通用接口"""
    
    def __init__(self, device_name: str):
        """初始化适配器
        
        Args:
            device_name: 设备名称，用于日志和状态标识
        """
        self.device_name = device_name
        self.monitoring = False
        self._monitoring_task = None
        self._status = {"status": "inactive"}
        self._last_error = None
        
    @abc.abstractmethod
    async def initialize(self) -> bool:
        """初始化硬件连接
        
        Returns:
            连接是否成功
        """
        pass
    
    @abc.abstractmethod
    async def get_status(self) -> Dict[str, Any]:
        """获取当前硬件状态
        
        Returns:
            包含硬件状态的字典
        """
        return self._status.copy()
    
    @abc.abstractmethod
    async def _monitor_loop(self):
        """监控循环的具体实现，在子类中实现"""
        pass
    
    async def start_monitoring(self) -> bool:
        """开始监控硬件状态
        
        Returns:
            是否成功启动监控
        """
        if self.monitoring:
            logger.warning(f"{self.device_name} 适配器已经在监控中")
            return True
            
        try:
            # 确保硬件已初始化
            if not await self.initialize():
                logger.error(f"{self.device_name} 初始化失败，无法启动监控")
                return False
                
            self.monitoring = True
            self._monitoring_task = asyncio.create_task(self._monitor_loop())
            logger.info(f"{self.device_name} 监控已启动")
            return True
        except Exception as e:
            self._last_error = str(e)
            logger.error(f"{self.device_name} 启动监控失败: {e}", exc_info=True)
            return False
    
    async def stop_monitoring(self):
        """停止监控硬件状态"""
        if not self.monitoring:
            return
            
        self.monitoring = False
        if self._monitoring_task and not self._monitoring_task.done():
            try:
                # 等待监控任务正常结束
                await asyncio.wait_for(self._monitoring_task, timeout=5.0)
            except asyncio.TimeoutError:
                logger.warning(f"{self.device_name} 监控停止超时，强制取消任务")
                self._monitoring_task.cancel()
                try:
                    await self._monitoring_task
                except asyncio.CancelledError:
                    pass
        
        self._monitoring_task = None
        logger.info(f"{self.device_name} 监控已停止")
    
    async def update_status(self, status_data: Dict[str, Any], topic: Optional[str] = None):
        """更新状态并发布到WebSocket
        
        Args:
            status_data: 状态数据
            topic: WebSocket主题，如不提供则使用默认主题
        """
        self._status.update(status_data)
        # 实际的广播逻辑在子类实现 