"""pump_proxy.py  –  v3
Moonraker 蠕动泵 HTTP 封装，带结果输出

与MoonrakerWebsocketListener集成，实现从WebSocket获取精确的泵送参数。
"""
import requests
import logging
import re
import json
import uuid
import asyncio
from typing import Dict, Any, Optional

log = logging.getLogger(__name__)

# 如果MoonrakerWebsocketListener类在单独文件中
try:
    from core_api.moonraker_listener import MoonrakerWebsocketListener
except ImportError:
    # 这将使下面的类型提示能够通过，但实际初始化时需要提供正确的实例
    MoonrakerWebsocketListener = Any


class MoonrakerError(RuntimeError):
    pass


class PumpProxy:
    def __init__(self, base_url: str, listener: Optional[MoonrakerWebsocketListener] = None):
        """初始化泵代理

        Args:
            base_url: Moonraker HTTP API基础URL，例如 "http://192.168.1.100:7125"
            listener: MoonrakerWebsocketListener实例，用于接收泵服务的参数。如果为None，将使用传统方式估算参数。
        """
        self.base = base_url.rstrip('/')
        self.listener = listener  # WebSocket监听器
        
        # 泵校准基准值，用于估算时间（仅在无法从WebSocket获取精确值时使用）
        self.fallback_calibration = {
            'slow': {'rpm': 5.0, 'ml_per_rev': 0.08},
            'normal': {'rpm': 20.0, 'ml_per_rev': 0.08},
            'fast': {'rpm': 60.0, 'ml_per_rev': 0.08},
        }
        # 正则表达式用于解析PumpService日志
        self.rpm_regex = re.compile(r"(?:选择|设定|使用)(?:转速|速度)[:：]\s*([\d\.]+)\s*(?:RPM|rpm)")
        self.revolutions_regex = re.compile(r"(?:需要|将|要)(?:转动|旋转)[:：]?\s*([\d\.]+)\s*(?:圈|转)")

    # --- private ---
    async def _send_async(self, script: str):
        """向Moonraker发送G-code脚本命令（异步执行）

        Args:
            script: G-code脚本命令

        Returns:
            Dict: Moonraker的响应
        """
        url = f"{self.base}/printer/gcode/script"
        log.debug("POST (async) %s | %s", url, script)
        try:
            loop = asyncio.get_event_loop()
            # 使用 functools.partial 将参数绑定到同步函数
            # This makes self._blocking_post suitable for run_in_executor
            import functools
            partial_blocking_post = functools.partial(self._blocking_post, url, script)
            
            # 在executor中运行阻塞的requests调用
            raw_response_text = await loop.run_in_executor(None, partial_blocking_post)
            
            data = json.loads(raw_response_text)
            log.info(">> G-code Script (async): %s -> Moonraker Response: %s", script, data)
            return data.get('result', '')
        except requests.exceptions.RequestException as e:
            log.error(f"Moonraker API request failed (async) for script '{script}': {e}")
            raise MoonrakerError(f"Failed to send command to Moonraker (async): {e}")
        except json.JSONDecodeError as e:
            log.error(f"Failed to decode JSON response from Moonraker (async) for script '{script}': {e}. Response text: {raw_response_text if 'raw_response_text' in locals() else 'Unknown'}")
            raise MoonrakerError(f"Invalid JSON response from Moonraker (async): {e}")

    def _blocking_post(self, url: str, script: str) -> str:
        """实际执行阻塞的POST请求的辅助方法"""
        r = requests.post(url, json={"script": script}, timeout=180)
        r.raise_for_status()
        return r.text

    def _parse_pump_service_logs(self, logs: str):
        """从PumpService日志中解析RPM和圈数"""
        if not isinstance(logs, str):
            log.warning(f"无法解析日志：期望字符串，但得到 {type(logs)}")
            return None, None
            
        log.debug(f"尝试解析PumpService日志: {logs[:200]}...") # 打印前200个字符用于调试

        rpm_match = self.rpm_regex.search(logs)
        revolutions_match = self.revolutions_regex.search(logs)
        
        log.debug(f"RPM正则表达式匹配结果: {rpm_match}")
        log.debug(f"圈数正则表达式匹配结果: {revolutions_match}")

        rpm = float(rpm_match.group(1)) if rpm_match else None
        revolutions = float(revolutions_match.group(1)) if revolutions_match else None

        log.info(f"从日志中解析得到: RPM={rpm}, Revolutions={revolutions}")

        if rpm and revolutions:
            log.info(f"从日志中解析成功: RPM={rpm}, Revolutions={revolutions}")
        else:
            # 记录更详细的信息，帮助识别模式不匹配的原因
            log.warning(f"未能从日志中完整解析RPM和Revolutions。")
            # 尝试找出可能包含关键信息的行
            rpm_lines = [line for line in logs.split('\n') if 'rpm' in line.lower() or '转速' in line]
            rev_lines = [line for line in logs.split('\n') if '圈' in line or '转动' in line]
            if rpm_lines:
                log.warning(f"可能包含RPM信息的行: {rpm_lines}")
            if rev_lines:
                log.warning(f"可能包含圈数信息的行: {rev_lines}")
                
        return rpm, revolutions

    def _estimate_parameters_fallback(self, volume_ml: float, speed: str = "normal") -> Dict[str, Any]:
        """估算泵参数（回退方法）

        当无法从WebSocket获取精确参数时，使用此方法进行估算。

        Args:
            volume_ml: 目标体积(ml)
            speed: 速度类型，"slow", "normal", 或 "fast"

        Returns:
            Dict: 包含估算的rpm、revolutions和estimated_duration
        """
        # 确保速度类型有效
        speed_key = speed.lower() if speed.lower() in self.fallback_calibration else "normal"
        
        # 获取对应速度的RPM和ml_per_rev
        rpm = self.fallback_calibration[speed_key]['rpm']
        ml_per_rev = self.fallback_calibration[speed_key]['ml_per_rev']
        
        # 估算圈数
        revolutions = volume_ml / ml_per_rev if ml_per_rev > 0 else 0
        
        # 估算时长（秒）
        estimated_duration = (revolutions / rpm) * 60 if rpm > 0 else 0
        
        result = {
            "rpm": rpm,
            "revolutions": revolutions,
            "estimated_duration": estimated_duration
        }
        
        log.info(f"估算参数: 体积={volume_ml}ml, 速度={speed}, "
                f"估算RPM={rpm}, 估算圈数={revolutions:.2f}, "
                f"估算时长={estimated_duration:.2f}秒 (回退方法)")
        
        return result

    # --- public API ---
    async def dispense_auto(self, volume_ml: float, speed: str = "normal", direction: int = 1) -> Dict[str, Any]:
        """自动泵送指定体积

        发送自动泵送命令，并从WebSocket监听器获取精确参数。
        如果无法获取精确参数，则回退到估算方法。

        Args:
            volume_ml: 目标体积(ml)
            speed: 速度类型，"slow", "normal", 或 "fast"
            direction: 方向，1表示顺时针，0表示逆时针

        Returns:
            Dict: 包含success、rpm、revolutions和estimated_duration的字典
        """
        # 调整参数格式
        parts = [f"V={volume_ml}"]
        for_arg = "N"
        if speed and speed.lower() in ["slow", "normal", "fast"]:
            for_arg = speed[0].upper()
        parts.append(f"FOR={for_arg}")
        
        if direction is not None:
            dir_value = 0 if direction == 0 else 1  # 确保方向值为0或1
            parts.append(f"DIR={dir_value}")
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        log.info(f"创建泵送任务 {task_id}: volume={volume_ml}ml, speed={speed}, direction={direction}")
            
        # 构建G-code命令
        script = "DISPENSE_FLUID_AUTO " + " ".join(parts)
        
        # 提前使用预估参数建立基本信息
        # 即使没有获取到确切参数也可以返回合理估计
        fallback_params = self._estimate_parameters_fallback(volume_ml, speed)
        
        # 发送命令和等待WebSocket参数并行进行
        # 使用asyncio.gather同时启动两个任务
        
        # 命令发送任务 - 设置超时
        async def send_command_with_timeout():
            try:
                # 为_send_async设置超时，避免无限等待
                return await asyncio.wait_for(
                    self._send_async(script),
                    timeout=2.0  # 2秒超时 - 只为获取命令确认，非常快
                )
            except asyncio.TimeoutError:
                log.warning(f"任务 {task_id} G-code命令发送超时，但命令可能已发送。继续执行...")
                return "Command sent, but response timed out"
            
        # WebSocket参数获取任务 - 设置超时
        async def get_ws_params_with_timeout():
            if not self.listener:
                log.info(f"任务 {task_id} 未提供WebSocket监听器，返回None")
                return None
                
            try:
                # 等待WebSocket参数，最多等5秒
                return await asyncio.wait_for(
                    self.listener.wait_for_parsed_data(task_id),
                    timeout=5.0  # 5秒超时，如果5秒内未收到参数，使用回退值
                )
            except asyncio.TimeoutError:
                log.warning(f"任务 {task_id} 等待WebSocket参数超时，将使用回退估算")
                return None
        
        # 并行执行命令发送和参数等待
        send_task = asyncio.create_task(send_command_with_timeout())
        params_task = asyncio.create_task(get_ws_params_with_timeout())
        
        # 等待其中任何一个任务完成即可继续
        done, pending = await asyncio.wait(
            [send_task, params_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # 检查参数任务是否完成
        ws_pump_params = None
        if params_task in done:
            ws_pump_params = params_task.result()
            if ws_pump_params:
                log.info(f"任务 {task_id} 成功从WebSocket获取参数: {ws_pump_params}")
                # 即使命令发送任务未完成，也可以返回参数结果
                if send_task in pending:
                    # 取消命令发送等待但不中断命令本身，命令已经发送
                    send_task.cancel()
                    
                return {
                        "success": True,
                        "rpm": ws_pump_params["rpm"],
                        "revolutions": ws_pump_params["revolutions"],
                        "estimated_duration": ws_pump_params["estimated_duration"],
                        "source": "websocket",
                    "raw_response": "WebSocket parameters received before command completion"
                }
        
        # 如果WebSocket参数未获取或未完成，等待命令发送完成
        # 但设置总超时，避免无限等待
        try:
            raw_response = None
            if send_task in done:
                raw_response = send_task.result()
            else:
                # 等待命令发送完成，但设置较短超时
                try:
                    raw_response = await asyncio.wait_for(send_task, timeout=1.0)
                except asyncio.TimeoutError:
                    log.warning(f"任务 {task_id} 命令发送确认超时，但命令可能已发送")
                    raw_response = "Command sent, confirmation timed out"
                    
            # 继续等待参数任务一小段时间，如果还未完成
            if params_task in pending:
                try:
                    ws_pump_params = await asyncio.wait_for(params_task, timeout=2.0)
                    if ws_pump_params:
                        log.info(f"任务 {task_id} 在命令完成后获取到WebSocket参数")
                except asyncio.TimeoutError:
                    # 如果参数任务超时，取消它
                    params_task.cancel()
                    log.warning(f"任务 {task_id} 参数获取最终超时，将使用回退估算")
        except Exception as e:
            log.error(f"任务 {task_id} 等待命令或参数时出错: {e}", exc_info=True)
            # 确保取消所有未完成的任务
            for task in pending:
                task.cancel()
        
        # 如果已获取WebSocket参数，返回它
        if ws_pump_params:
            return {
                "success": True,
                "rpm": ws_pump_params["rpm"],
                "revolutions": ws_pump_params["revolutions"],
                "estimated_duration": ws_pump_params["estimated_duration"],
                "source": "websocket",
                "raw_response": str(raw_response) if raw_response else "Unknown response"
            }
            
        # 如果没有WebSocket参数，尝试从响应日志解析
        if raw_response:
            rpm_from_response, revolutions_from_response = self._parse_pump_service_logs(str(raw_response))
            if rpm_from_response is not None and revolutions_from_response is not None:
                log.info(f"任务 {task_id} 从响应日志中解析得到: RPM={rpm_from_response}, 圈数={revolutions_from_response}")
                estimated_duration = (revolutions_from_response / rpm_from_response) * 60 if rpm_from_response > 0 else 0
                return {
                    "success": True,
                    "rpm": rpm_from_response,
                    "revolutions": revolutions_from_response,
                    "estimated_duration": estimated_duration,
                    "source": "response_log",
                    "raw_response": str(raw_response)
                }
        
        # 如果都失败了，使用回退估算
        log.warning(f"任务 {task_id} 无法获取精确参数，使用回退估算")
        return {
                "success": True,
                "rpm": fallback_params["rpm"],
                "revolutions": fallback_params["revolutions"],
                "estimated_duration": fallback_params["estimated_duration"],
                "source": "fallback",
            "raw_response": str(raw_response) if raw_response else "Unknown response"
            }

    async def dispense_speed(self, volume_ml: float, speed_rpm: float, direction: int = 1) -> Dict[str, Any]:
        """使用固定速度泵送（定时泵送的底层方法）

        Args:
            volume_ml: 目标体积(ml)，用于估算圈数
            speed_rpm: 转速(RPM)
            direction: 方向，1表示顺时针，0表示逆时针

        Returns:
            Dict: 包含success、rpm、revolutions和estimated_duration的字典
        """
        # 调整参数格式
        parts = [f"V={volume_ml}", f"S={speed_rpm}"]
        
        if direction is not None:
            dir_value = 0 if direction == 0 else 1  # 确保方向值为0或1
            parts.append(f"DIR={dir_value}")
        
        # 生成任务ID
        task_id = str(uuid.uuid4())
        log.info(f"创建定速泵送任务 {task_id}: volume={volume_ml}ml, speed_rpm={speed_rpm}, direction={direction}")
            
        # 构建G-code命令
        script = "DISPENSE_FLUID_SPEED " + " ".join(parts)
        
        # 首先启动命令发送
        send_task = asyncio.create_task(self._send_async(script))
            
        # 同时，开始等待WebSocket的参数 (主要是圈数)
        ws_pump_params = None
        if self.listener:
            log.info(f"任务 {task_id} 开始等待WebSocket泵参数 (与命令发送并行)...")
            ws_pump_params = await self.listener.wait_for_parsed_data(task_id, timeout=7.0) 
            if ws_pump_params and ws_pump_params.get("revolutions"):
                log.info(f"从WebSocket获取到泵参数: {ws_pump_params}")
            else:
                log.warning(f"无法从WebSocket获取任务 {task_id} 的泵参数(圈数)，将依赖后续处理。")
        else:
            log.info("未提供WebSocket监听器，将依赖后续处理。")
        
        # 等待命令发送完成
        try:
            raw_response = await send_task
            log.info(f"任务 {task_id} G-code命令发送完成，Moonraker响应: {str(raw_response)[:200]}...")
        except Exception as e:
            log.error(f"任务 {task_id} G-code命令发送失败: {e}", exc_info=True)
            return {"success": False, "error": str(e), "source": "send_error"}

        # 参数获取和组装逻辑
        try:
            if ws_pump_params and ws_pump_params.get("revolutions"):
                revolutions = ws_pump_params["revolutions"]
                estimated_duration = (revolutions / speed_rpm) * 60 if speed_rpm > 0 else 0
                return {
                    "success": True,
                    "rpm": speed_rpm, 
                    "revolutions": revolutions,
                    "estimated_duration": estimated_duration,
                    "source": "websocket",
                    "raw_response": str(raw_response)
                }
            
            # 如果WebSocket失败，尝试从响应日志解析圈数
            _, revolutions_from_response = self._parse_pump_service_logs(str(raw_response))
            if revolutions_from_response is not None:
                log.info(f"从响应日志中直接解析得到: 圈数={revolutions_from_response}")
                estimated_duration = (revolutions_from_response / speed_rpm) * 60 if speed_rpm > 0 else 0
                return {
                    "success": True,
                    "rpm": speed_rpm,
                    "revolutions": revolutions_from_response,
                    "estimated_duration": estimated_duration,
                    "source": "response_log",
                    "raw_response": str(raw_response)
                }
            
            # 回退估算
            ml_per_rev = 0.08 # 默认值
            revolutions_fallback = volume_ml / ml_per_rev if ml_per_rev > 0 else 0
            estimated_duration_fallback = (revolutions_fallback / speed_rpm) * 60 if speed_rpm > 0 else 0
            log.info(f"估算参数: 体积={volume_ml}ml, RPM={speed_rpm}, "
                    f"估算圈数={revolutions_fallback:.2f}, 估算时长={estimated_duration_fallback:.2f}秒 (基本估算)")
            return {
                "success": True,
                "rpm": speed_rpm,
                "revolutions": revolutions_fallback,
                "estimated_duration": estimated_duration_fallback,
                "source": "estimate",
                "raw_response": str(raw_response)
            }
            
        except Exception as e:
            log.error(f"处理任务 {task_id} 的定速泵送参数时出错: {e}", exc_info=True)
            return {"success": False, "error": str(e), "source": "parameter_error", "raw_response": str(raw_response)}

    async def emergency_stop(self) -> Dict[str, Any]:
        """紧急停止所有泵 (异步)

        Returns:
            Dict: 包含success的字典
        """
        try:
            script = "STOP_PUMP"
            response = await self._send_async(script) # 改为调用异步版本
            return {
                "success": True,
                "raw_response": str(response)
            }
        except Exception as e:
            log.error(f"执行emergency_stop时出错: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
