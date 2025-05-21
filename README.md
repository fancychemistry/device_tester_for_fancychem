# 设备测试器

这是一个用于测试和控制实验设备的集成平台，提供了Web界面来控制以下设备：

- 打印机控制：移动、定位和归位
- 蠕动泵控制：自动泵送、定时泵送和停止
- 继电器控制：切换多个继电器状态
- CHI工作站控制：支持多种电化学测试(CV、CA、EIS、LSV、i-t等)

## 功能特点

- 基于FastAPI的RESTful API接口
- 交互式WebSocket通信
- 实时状态更新和监控
- 支持多种设备并行操作
- 自动化测试流程

## 技术架构

- 后端：Python FastAPI
- 前端：HTML/JavaScript
- 通信：WebSocket、HTTP REST API
- 设备控制：自定义适配器模式

## 使用说明

1. 安装依赖：`pip install -r requirements.txt`
2. 启动服务器：`python device_tester.py`
3. 通过浏览器访问：`http://localhost:8001`
4. 配置设备连接参数
5. 开始使用设备测试功能 