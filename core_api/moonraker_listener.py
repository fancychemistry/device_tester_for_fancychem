"""
Moonraker WebSocket Listener - 用于监听Moonraker WebSocket消息并提取关键信息

这个模块提供了MoonrakerWebsocketListener类，用于：
1. 连接到Moonraker的WebSocket
2. 监听和处理消息
3. 提取关键参数（如泵的RPM、圈数等）
4. 提供API以供其他模块获取这些参数

依赖:
- websockets (pip install websockets)
- asyncio
"""

import asyncio
import json
import logging
import re
import time
import uuid
from typing import Dict, Any, Optional, List, Callable, Awaitable, Union

# 尝试导入websockets，如果无法导入，记录错误但不立即退出
try:
    import websockets
except ImportError:
    logging.error("缺少必要的依赖：websockets。请安装：pip install websockets")
    # 不抛出异常，让实际使用时再报错，便于调试

# 配置日志
log = logging.getLogger(__name__)


class MoonrakerWebsocketListener:
    """Moonraker WebSocket监听器

    监听Moonraker WebSocket消息，解析和存储关键信息（如泵送参数）。
    提供API供其他模块等待和查询这些信息。
    """

    def __init__(self, websocket_url: str):
        """初始化WebSocket监听器

        Args:
            websocket_url: Moonraker WebSocket URL，例如 "ws://192.168.1.100:7125/websocket"
        """
        self.websocket_url = websocket_url
        self.websocket = None
        self.running = False
        self.connected = False
        
        # 用于存储解析出的参数
        self.parsed_data = {}
        self.pending_requests = {}
        
        # 参数缓存：因为RPM和圈数可能在不同消息中
        self._parameter_cache = {}
        
        # 更新正则表达式以匹配更通用的日志格式，并忽略大小写
        # 例如: "选择转速: 5.0 RPM", "PumpService - INFO - ... 选择转速: 5.0 RPM", "已设置转速: 5.0 RPM"
        #       "需要转动 1.224 圈", "PumpService - INFO - ... 需要转动 1.224 圈", "已设置圈数: 1.224 圈"
        self.rpm_regex = re.compile(r"(?:选择转速|转速是|已设置转速|设置转速|使用转速|泵转速|转速设定|转速设置|转速|PumpService.*?转速)[:：\s]\s*([\d\.]+)\s*(?:RPM|rpm)", re.IGNORECASE)
        self.revolutions_regex = re.compile(r"(?:需要转动|转动圈数是|已设置圈数|设置圈数|圈数设定|圈数设置|转动|PumpService.*?[转动|圈数])[:：\s]\s*([\d\.]+)\s*(?:圈|转)", re.IGNORECASE)
        
    async def start(self):
        """启动WebSocket连接并开始监听消息"""
        if self.running:
            log.warning("WebSocket监听器已经在运行")
            return
            
        self.running = True
        
        while self.running:
            try:
                log.info(f"尝试连接到Moonraker WebSocket: {self.websocket_url}")
                async with websockets.connect(
                    self.websocket_url, 
                    ping_interval=20,
                    ping_timeout=60,
                ) as websocket:
                    self.websocket = websocket
                    self.connected = True
                    log.info("已连接到Moonraker WebSocket")
                    
                    await self._subscribe()
                    
                    async for message in websocket:
                        # 首先记录原始消息，便于调试订阅问题
                        # log.debug(f"RAW WS MSG: {message}") 
                        await self._process_message(message)
                        
            except websockets.exceptions.ConnectionClosed as e:
                self.connected = False
                log.warning(f"WebSocket连接已关闭: {e}")
            except asyncio.CancelledError:
                self.connected = False
                self.running = False
                log.info("WebSocket监听器任务被取消")
                break
            except Exception as e:
                self.connected = False
                log.error(f"WebSocket监听器出错: {e}", exc_info=True)
            finally:
                if self.running:
                    self.websocket = None
                    self.connected = False
                    log.info("等待5秒后重新连接...")
                    await asyncio.sleep(5)
    
    async def stop(self):
        """停止WebSocket监听器"""
        self.running = False
        if self.websocket:
            try:
                close_timeout = 3
                await asyncio.wait_for(self.websocket.close(), close_timeout)
                log.info("WebSocket连接已正常关闭")
            except asyncio.TimeoutError:
                log.warning("WebSocket关闭操作超时，可能需要强制断开")
            except Exception as e:
                log.error(f"关闭WebSocket连接时出错: {e}")
            finally:
                self.websocket = None
        self.connected = False
        log.info("WebSocket监听器已停止")
    
    async def _subscribe(self):
        """向Moonraker订阅相关事件"""
        if not self.websocket or not self.connected:
            return
        try:
            # 订阅服务器G-code存储事件，这通常包括G-code命令的响应和可能的日志输出
            subscribe_gcode_store = {
                "jsonrpc": "2.0",
                "method": "server.gcode_store", # 获取G-code命令历史和响应
                "params": {"count": 5}, # 获取最近5条，也许可以调整或不需要count
                "id": int(time.time() * 1000) # 使用时间戳作为ID
            }
            
            # 订阅打印机对象状态，特别是toolhead和gcode_move，可能也包含间接日志或状态
            subscribe_printer_objects = {
                "jsonrpc": "2.0",
                "method": "printer.objects.subscribe",
                "params": {
                    "objects": {
                        "toolhead": None, 
                        "gcode_move": None,
                        "print_stats": None, # 或许print_stats中也可能包含相关信息
                        # "extruder": None, # 如果泵被模拟为挤出机
                    }
                },
                "id": int(time.time() * 1000) + 1
            }
            
            await self.websocket.send(json.dumps(subscribe_gcode_store))
            log.info(f"已发送订阅请求: server.gcode_store")
            await self.websocket.send(json.dumps(subscribe_printer_objects))
            log.info(f"已发送订阅请求: printer.objects.subscribe")
            
        except Exception as e:
            log.error(f"发送WebSocket订阅请求失败: {e}")
    
    async def _process_message(self, message: str):
        """处理从WebSocket接收到的消息"""
        if not message:
            return
        try:
            data = json.loads(message)
            
            # 通用日志记录，查看所有通知类型的方法和参数
            if "method" in data and data["method"].startswith("notify_"):
                log.info(f"WS NOTIFY method: {data['method']}, params: {data.get('params')}")

            # 处理G-code相关的响应或日志
            # server.gcode_store 订阅通常通过 notify_gcode_response 返回G-code的stdout
            if data.get("method") == "notify_gcode_response":
                message_content = data["params"][0]
                if isinstance(message_content, str):
                    log.info(f"G-code响应行: {message_content.strip()}")
                    self._parse_pump_parameters(message_content)
            
            # printer.objects.subscribe 的更新通常通过 notify_status_update
            elif data.get("method") == "notify_status_update":
                # status更新可能包含多项内容，我们需要查找与泵相关的日志或状态
                # 这部分比较复杂，取决于Klipper如何将PumpService的日志或状态通过此通道暴露
                # 暂时只记录，后续根据实际日志内容决定是否从中解析参数
                # log.info(f"Status Update: {data['params'][0]}") 
                # 示例：如果PumpService的日志在gcode_store对象中更新
                if "gcode_store" in data["params"][0]:
                    gcode_store = data["params"][0]["gcode_store"]
                    if isinstance(gcode_store, dict) and "history" in gcode_store:
                        for entry in gcode_store["history"]:
                            if "message" in entry and isinstance(entry["message"], str):
                                # log.info(f"G-code history message: {entry['message'].strip()}")
                                self._parse_pump_parameters(entry["message"])
                # 可以根据需要添加对其他对象的检查，如 toolhead, extruder 等

            # 其他可能的通知方法，例如 'notify_klippy_shutdown', 'notify_error_message' 等
            # 也可以在这里添加特定处理

        except json.JSONDecodeError:
            log.warning(f"无效的JSON消息: {message}")
        except Exception as e:
            log.error(f"处理WebSocket消息时出错: {e}", exc_info=True)
    
    def _parse_pump_parameters(self, message: str):
        """从消息中解析泵参数"""
        if not message or not isinstance(message, str):
            return

        message_to_parse = message.strip()
        if not message_to_parse: # 跳过空消息
            return

        rpm_changed = False
        revolutions_changed = False

        rpm_match = self.rpm_regex.search(message_to_parse)
        if rpm_match:
            try:
                rpm = float(rpm_match.group(1))
                if self._parameter_cache.get("rpm") != rpm:
                    log.info(f"Listener解析到RPM: {rpm} (来自: '{message_to_parse[:100]}...')")
                    self._parameter_cache["rpm"] = rpm
                    rpm_changed = True
            except ValueError:
                log.warning(f"无法将解析的RPM '{rpm_match.group(1)}' 转换为float")
        
        revolutions_match = self.revolutions_regex.search(message_to_parse)
        if revolutions_match:
            try:
                revolutions = float(revolutions_match.group(1))
                if self._parameter_cache.get("revolutions") != revolutions:
                    log.info(f"Listener解析到圈数: {revolutions} (来自: '{message_to_parse[:100]}...')")
                    self._parameter_cache["revolutions"] = revolutions
                    revolutions_changed = True
            except ValueError:
                log.warning(f"无法将解析的圈数 '{revolutions_match.group(1)}' 转换为float")

        # 仅当参数有变化，并且RPM和圈数都存在时，才尝试计算和通知
        if (rpm_changed or revolutions_changed) and "rpm" in self._parameter_cache and "revolutions" in self._parameter_cache:
            current_rpm = self._parameter_cache["rpm"]
            current_revolutions = self._parameter_cache["revolutions"]
            
            params_for_request = {
                "rpm": current_rpm,
                "revolutions": current_revolutions
            }
            
            if current_rpm > 0:
                estimated_duration = (current_revolutions / current_rpm) * 60
                params_for_request["estimated_duration"] = estimated_duration
                log.info(f"Listener计算的估算时长: {estimated_duration:.2f}秒 (RPM: {current_rpm}, Revs: {current_revolutions})")
            else:
                log.warning(f"无法计算估算时长: RPM为零或无效 (RPM: {current_rpm})")
                # 如果没有有效的时长，可能不应该满足请求，或者由请求方决定如何处理
                # params_for_request["estimated_duration"] = 0 # 或者不设置
            
            # 只有在可以计算出预估时长时才满足请求
            if "estimated_duration" in params_for_request:
                # 使用一个唯一的ID（例如当前时间戳）来存储这次完整的解析结果，
                # 但更重要的是通知等待的请求。
                # self.parsed_data[str(int(time.time() * 1000))] = params_for_request 
                self._check_pending_requests(params_for_request)
    
    def _check_pending_requests(self, params: Dict[str, Any]):
        """检查有无等待此参数的请求"""
        # 确保RPM, revolutions, 和 estimated_duration 都存在于提供的params中
        if not all(k in params for k in ("rpm", "revolutions", "estimated_duration")):
            # log.debug(f"参数不完整，不满足pending requests: {params}")
            return
            
        completed_requests = []
        # Iterate over a copy of items in case a future callback modifies the dict
        for request_id, future in list(self.pending_requests.items()): 
            if not future.done():
                log.info(f"Listener满足请求 {request_id} 的参数要求: {params}")
                future.set_result(params.copy()) # 发送参数的副本
                completed_requests.append(request_id)
        
        for request_id in completed_requests:
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]
    
    async def wait_for_parsed_data(self, request_id: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """等待解析出的泵参数"""
        if not self.connected:
            log.warning("WebSocket未连接，无法等待数据")
            return None
            
        # 检查参数缓存中是否已经有可用数据 (RPM 和 Revolutions)
        # 这是为了在请求发起时，如果参数已经被解析过，可以立即返回
        # 但仍然需要计算 estimated_duration
        if "rpm" in self._parameter_cache and "revolutions" in self._parameter_cache:
            rpm = self._parameter_cache["rpm"]
            revolutions = self._parameter_cache["revolutions"]
            if rpm > 0:
                estimated_duration = (revolutions / rpm) * 60
                log.info(f"wait_for_parsed_data: 请求 {request_id} 时，缓存中已有参数，立即返回。")
                return {
                    "rpm": rpm,
                    "revolutions": revolutions,
                    "estimated_duration": estimated_duration
                }

        future = asyncio.get_event_loop().create_future()
        self.pending_requests[request_id] = future
        log.info(f"请求 {request_id} 正在等待泵参数... (超时: {timeout}s)")
        
        try:
            result = await asyncio.wait_for(future, timeout)
            return result
        except asyncio.TimeoutError:
            log.warning(f"等待请求 {request_id} 的泵参数超时。检查参数缓存: RPM={self._parameter_cache.get('rpm')}, Revs={self._parameter_cache.get('revolutions')}")
            # 超时后，如果缓存中有部分或全部参数，也尝试返回它们
            # 这允许调用者至少获得部分信息，而不是完全失败
            if "rpm" in self._parameter_cache and "revolutions" in self._parameter_cache:
                 rpm = self._parameter_cache["rpm"]
                 revolutions = self._parameter_cache["revolutions"]
                 if rpm > 0:
                    estimated_duration = (revolutions / rpm) * 60
                    log.info(f"wait_for_parsed_data: 请求 {request_id} 超时，但从缓存中构建了部分结果。")
                    return {
                        "rpm": rpm,
                        "revolutions": revolutions,
                        "estimated_duration": estimated_duration
                    }
            return None # 最终超时且缓存不足
        finally:
            if request_id in self.pending_requests:
                del self.pending_requests[request_id]


# 如果直接运行此文件，执行简单的连接测试
if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 替换为实际的Moonraker WebSocket URL
    WS_URL = "ws://192.168.51.168:7125/websocket"
    
    # 运行测试
    asyncio.run(MoonrakerWebsocketListener(WS_URL).start()) 