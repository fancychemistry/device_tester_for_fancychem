from fastapi_websocket_pubsub import PubSubEndpoint
from fastapi import WebSocket
from typing import List
import json
import logging

logger = logging.getLogger(__name__)

# 创建PubSub终端
pubsub_endpoint = PubSubEndpoint()

# 广播器
class Broadcaster:
    def __init__(self):
        self.subscriptions = {}  # 存储订阅信息
        self.active_connections: List[WebSocket] = []  # 存储活跃的WebSocket连接
        
    async def connect(self, websocket: WebSocket):
        """
        连接新的WebSocket
        """
        self.active_connections.append(websocket)
        logger.debug(f"新的WebSocket连接：当前连接数={len(self.active_connections)}")
        
    async def disconnect(self, websocket: WebSocket):
        """
        断开WebSocket连接
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.debug(f"WebSocket断开连接：当前连接数={len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """
        广播消息给所有连接的客户端
        """
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"向WebSocket发送消息失败: {e}")
                disconnected.append(connection)
        
        # 清理断开连接
        for conn in disconnected:
            await self.disconnect(conn)
        
    async def subscribe(self, topic: str, callback):
        """
        订阅特定主题
        """
        logger.debug(f"订阅主题 {topic}")
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        self.subscriptions[topic].append(callback)
        return len(self.subscriptions[topic]) - 1  # 返回订阅索引作为订阅ID
    
    async def unsubscribe(self, topic: str, callback=None):
        """
        取消订阅特定主题
        """
        if callback and topic in self.subscriptions:
            self.subscriptions[topic].remove(callback)
            logger.debug(f"已取消订阅主题 {topic}")
        elif topic in self.subscriptions:
            self.subscriptions[topic] = []
            logger.debug(f"已清空主题 {topic} 的所有订阅")
    
    async def publish(self, topic: str, message: dict):
        """
        发布消息到特定主题
        """
        logger.debug(f"广播消息到 {topic}: {message}")
        await pubsub_endpoint.publish(topic, message)
        # 同时使用WebSocket广播
        await self.broadcast(message) 