# 设备测试器 (Device Tester) - 技术文档

本系统是一个专为实验室环境设计的集成控制与测试平台。它通过统一的Web界面，使用户能够方便地对多种常见的实验设备进行远程操作和自动化测试。主要支持的设备包括3D打印机（XYZ位移平台）、蠕动泵、多通道继电器以及CHI电化学工作站，从而覆盖了从样品处理、流体控制到电化学分析等多种实验场景。

该平台旨在简化实验流程，提高工作效率，并为实验过程提供实时的监控与数据记录能力。

## 快速开始 (Quick Start) / 使用说明 (Usage Instructions)

1.  **安装依赖 (Install Dependencies):**
    ```bash
    pip install -r requirements.txt
    ```
2.  **启动服务器 (Start Server):**
    ```bash
    python device_tester.py
    ```
3.  **访问应用 (Access Application):**
    通过浏览器访问部署的地址，默认为: `http://localhost:8001`
4.  **配置设备参数 (Configure Device Parameters):**
    首次使用时，请在Web界面的“设置”或相应配置区域，根据您的实际硬件连接情况，配置Moonraker服务地址、CHI工作站路径、实验结果保存目录等关键参数。
5.  **开始使用 (Start Using):**
    配置完成后，即可通过各个设备控制标签页开始进行设备初始化、操作和测试。

## PRD (项目需求文档)

### 1. 应用概述
本系统是一个集成平台，旨在通过Web界面对实验室设备进行测试和控制，主要包括3D打印机（Printer）、蠕动泵（Pump）、继电器（Relay）和CHI电化学工作站（CHI Workstation）。

### 2. 用户流程
*   用户在设置页面配置Moonraker地址、实验结果保存目录以及CHI工作站的路径。
*   用户通过导航栏进入相应的设备控制标签页（如：打印机、蠕动泵、继电器、CHI工作站）。
*   用户初始化所选设备，确保设备连接正常。
*   用户输入具体操作参数，并触发相应动作（例如：移动打印机喷头、控制蠕动泵分配液体、切换继电器状态、运行CHI电化学测试）。
*   系统通过用户界面（UI）更新和日志记录，提供实时的操作反馈。
*   对于CHI工作站，用户可以查看测试曲线（如果前端支持）并下载生成的结果文件。

### 3. 技术栈 & API
*   **后端 (Backend):**
    *   Python
    *   FastAPI: 用于构建主要的RESTful API。
    *   Uvicorn: ASGI服务器，用于运行FastAPI应用。
    *   WebSockets:
        *   用于与Moonraker服务进行实时双向通信。
        *   通过`broadcaster`库实现向客户端浏览器推送实时更新。
*   **前端 (Frontend):**
    *   HTML
    *   JavaScript: 处理用户交互和动态内容更新。
    *   Bootstrap 5: 用于构建响应式的用户界面。
*   **设备控制 (Device Control):**
    *   通过与Moonraker API交互来控制3D打印机、蠕动泵和继电器。
    *   通过`device_control.control_chi`模块（可能包装了CHI工作站的官方软件或命令行接口）控制CHI电化学工作站。
*   **主要API接口 (APIs):**
    *   WebSocket端点: `/ws` (用于与前端的实时通信)
    *   打印机:
        *   `/api/printer/initialize`: 初始化打印机连接。
        *   `/api/printer/move`: 控制打印机移动。
        *   `/api/printer/home`: 控制打印机归位。
        *   `/api/printer/set_relative`: 设置相对定位。
        *   `/api/printer/set_absolute`: 设置绝对定位。
    *   蠕动泵:
        *   `/api/pump/initialize`: 初始化蠕动泵。
        *   `/api/pump/dispense_auto`: 自动模式分配液体。
        *   `/api/pump/dispense_timed`: 定时模式分配液体。
        *   `/api/pump/stop`: 停止蠕动泵。
    *   继电器:
        *   `/api/relay/initialize`: 初始化继电器。
        *   `/api/relay/toggle`: 切换继电器状态。
    *   CHI工作站:
        *   `/api/chi/initialize`: 初始化CHI工作站。
        *   `/api/chi/cv`: 运行循环伏安法 (CV) 测试。
        *   `/api/chi/ca`: 运行计时电流法 (CA) 测试。
        *   `/api/chi/eis`: 运行电化学阻抗谱 (EIS) 测试。
        *   `/api/chi/lsv`: 运行线性扫描伏安法 (LSV) 测试。
        *   `/api/chi/it`: 运行电流-时间曲线 (i-t) 测试。
        *   `/api/chi/ocp`: 运行开路电位 (OCP) 测试。
        *   `/api/chi/dpv`: 运行差分脉冲伏安法 (DPV) 测试。
        *   `/api/chi/scv`: 运行方波伏安法 (SCV) 测试。
        *   `/api/chi/cp`: 运行计时电位法 (CP) 测试。
        *   `/api/chi/acv`: 运行交流伏安法 (ACV) 测试。
        *   `/api/chi/get_results_list`: 获取CHI测试结果文件列表。
        *   `/api/chi/download_result/{filename}`: 下载指定的CHI测试结果文件。

### 4. 核心功能
*   远程控制3D打印机的XYZ轴移动、归位操作。
*   控制蠕动泵进行自动分配（基于设定的流速和体积）和定时分配（基于设定的流速和时间）。
*   控制多通道继电器的开关状态。
*   通过CHI工作站执行和管理多种电化学测试方法，包括CV, CA, EIS, LSV, IT, OCP, DPV, SCV, CP, ACV。
*   提供设备状态的实时更新和操作日志记录。
*   支持对设备连接参数（如Moonraker地址）和文件路径（如结果保存目录、CHI路径）的配置管理。
*   支持查看CHI电化学测试生成的结果文件列表，并提供下载功能。
*   基于FastAPI的RESTful API接口。
*   交互式WebSocket通信，用于实时状态更新和监控。
*   支持多种设备并行操作（部分设备间操作可能存在互斥，例如CHI测试运行时）。
*   自动化测试流程（通过组合API调用实现）。

### 5. 项目范围 (In/Out Scope)
*   **范围内 (In Scope):**
    *   对当前已支持设备（打印机、蠕动泵、继电器、CHI工作站）的远程控制和操作。
    *   基本的设备连接参数和路径配置。
    *   操作日志记录。
    *   CHI测试结果文件的列表展示和下载。
*   **范围外 (Out Scope):**
    *   用户认证和授权系统。
    *   平台内嵌的高级数据分析功能（例如，直接在网页上对电化学数据进行拟合、作图等）。
    *   对列表中未提及的其他类型实验室设备的支持。

## App Flow (应用流程文档)

### 1. 配置页面/区域 (Configuration Page/Section)
*   **功能描述:** 用户在此页面输入和保存系统运行所需的关键配置信息。
*   **用户操作:**
    *   输入Moonraker服务的URL地址 (例如: `ws://192.168.1.100:7125`)。
    *   指定CHI电化学工作站测试结果的保存目录路径。
    *   提供CHI电化学工作站可执行文件（或相关脚本）的路径。
*   **系统交互:**
    *   用户点击“保存配置”按钮。
    *   前端将配置信息通过API调用（例如: `/api/config/save`）发送到后端进行存储。
    *   保存成功或失败的状态会反馈给用户。

### 2. 主界面 (Main Interface)
*   **功能描述:** 应用的核心操作界面，提供设备状态概览和不同设备控制模块的导航。
*   **用户操作:**
    *   查看各个设备（打印机、蠕动泵、继电器、CHI工作站）的状态指示灯或文本，快速了解其连接和工作状态。
    *   点击不同的标签页（Tabs）在打印机、蠕动泵、继电器和CHI工作站的控制面板之间切换。

### 3. 打印机控制标签页 (Printer Tab)
*   **功能描述:** 控制3D打印机（或类似XYZ位移平台）的移动和定位。
*   **用户操作:**
    *   点击“初始化打印机”按钮，通过API (`/api/printer/initialize`) 连接到打印机。
    *   在表单中输入X, Y, Z目标坐标，点击“移动”按钮，通过API (`/api/printer/move`) 控制打印机精确移动。
    *   在表单中输入Grid X, Grid Y, Grid Z坐标，点击“移动到网格点”按钮，通过API (`/api/printer/grid`) 控制打印机移动到预设网格位置。
    *   点击“归位”按钮，通过API (`/api/printer/home`) 使打印机所有轴回归原点。
    *   点击“获取当前位置”按钮，通过API (`/api/printer/position`) 查询并显示打印机当前坐标。
*   **系统交互:**
    *   所有操作通过相应的API调用与后端通信，后端再通过Moonraker接口与打印机硬件交互。
    *   操作结果和打印机状态通过WebSocket或API响应更新到UI和日志面板。

### 4. 蠕动泵控制标签页 (Pump Tab)
*   **功能描述:** 控制蠕动泵进行液体分配。
*   **用户操作:**
    *   点击“初始化蠕动泵”按钮，通过API (`/api/pump/initialize`) 准备蠕动泵。
    *   **自动分配模式:** 输入目标体积 (µL)、流速 (µL/s) 和方向（正/反），点击“开始自动分配”，通过API (`/api/pump/dispense_auto`) 启动。
    *   **定时分配模式:** 输入运行时间 (s)、转速 (RPM) 和方向（正/反），点击“开始定时分配”，通过API (`/api/pump/dispense_timed`) 启动。
    *   点击“停止泵”按钮，通过API (`/api/pump/stop`) 立即停止当前的泵送操作。
    *   点击“获取泵状态”按钮，通过API (`/api/pump/status`) 查询泵的当前状态。
*   **系统交互:**
    *   泵送操作进行时，UI会显示进度条和剩余时间/已分配体积，这些信息通过WebSocket (`pump_status`消息) 实时更新。
    *   操作日志会记录启动、停止和参数等信息。

### 5. 继电器控制标签页 (Relay Tab)
*   **功能描述:** 控制多个继电器的开关状态。
*   **用户操作:**
    *   点击“初始化继电器”按钮，通过API (`/api/relay/initialize`) 准备继电器模块。
    *   界面上会为每个继电器（例如1-4号）显示独立的控制元件（如开关按钮或卡片）。
    *   点击特定继电器的按钮来切换其状态（开/关），此操作通过API (`/api/relay/toggle`，通常会带上继电器编号和目标状态）执行。
    *   点击“刷新状态”按钮，通过API (`/api/relay/status`) 获取所有继电器的当前状态。
*   **系统交互:**
    *   继电器的当前状态（开/关）会实时显示在UI上，状态变更信息通过WebSocket (`relay_status`消息) 推送。
    *   操作日志记录哪个继电器被操作以及操作结果。

### 6. CHI工作站控制标签页 (CHI Workstation Tab)
*   **功能描述:** 控制CHI电化学工作站执行各种电化学测试，并管理测试结果。
*   **用户操作:**
    *   点击“初始化CHI”按钮，通过API (`/api/chi/initialize`) 连接并准备CHI工作站。
    *   为选择的电化学测试方法（如CV, CA, EIS, LSV, IT, OCP, DPV, SCV, CP, ACV）在对应的表单中填写所需的测试参数（例如扫描范围、速率、采样间隔等）。
    *   点击相应测试方法旁边的“开始测试”按钮，通过对应的API (如 `/api/chi/cv`, `/api/chi/ca` 等) 启动测试。
    *   点击“停止当前测试”按钮，通过API (`/api/chi/stop`) 终止正在进行的CHI测试。
    *   **结果管理:**
        *   点击“刷新结果列表”按钮，通过API (`/api/chi/results` 或 `/api/chi/get_results_list`) 获取已完成测试的结果文件列表。
        *   从列表中选择一个文件，点击“下载”按钮 (或文件名链接)，通过API (`/api/chi/download/{filename}`) 下载该结果文件。
*   **系统交互:**
    *   当前测试的类型、状态（运行中、已完成、错误）、已用时间和进度等信息会显示在UI上，这些信息通过WebSocket (`chi_status`消息) 和/或定期的API轮询 (`/api/chi/status`) 进行更新。
    *   操作日志记录测试的启动、停止、参数以及结果文件的生成和下载。

### 7. 日志面板 (Log Panel)
*   **功能描述:** 集中显示系统操作日志，帮助用户追踪设备行为和排查问题。
*   **用户操作:**
    *   用户可随时查看此面板。
*   **系统交互:**
    *   所有前端发起的控制操作（如点击按钮）及其结果（成功/失败）都会生成带时间戳的日志条目。
    *   后端通过WebSocket推送的设备状态更新、错误信息等也会被记录在此。
    *   日志条目清晰标明来源（例如：打印机控制、CHI测试、WebSocket消息）和内容。

## Tech Stack (技术栈文档)

### 1. 后端 (Backend)
*   **编程语言:** Python 3
*   **Web框架:** FastAPI
    *   用于构建高效、易用的RESTful API接口。
*   **Web服务器:** Uvicorn
    *   作为ASGI服务器，用于运行FastAPI应用。
*   **实时通信:** WebSockets
    *   **FastAPI WebSocket 支持:** 用于客户端与服务器之间的双向通信。
    *   **`broadcaster` 库:** 用于实现服务器到客户端的实时消息广播。
    *   **`websockets` 库:** 用于与Moonraker服务进行WebSocket通信。
*   **设备通信:**
    *   **Moonraker API:**
        *   通过HTTP (`aiohttp`库) 和 WebSockets (`websockets`库) 与Moonraker服务交互，以控制打印机、蠕动泵和继电器。
    *   **CHI电化学工作站:**
        *   通过 `device_control.control_chi` 模块进行控制。该模块封装了CHI仪器供应商提供的命令行工具(CLI)或软件开发工具包(SDK)的调用逻辑。
*   **关键Python库:**
    *   `pydantic`: 用于API数据模型的定义和验证。
    *   `python-dotenv`: 用于管理环境变量，例如从 `.env` 文件加载配置。
    *   `psutil`: 用于获取系统和进程相关信息。
    *   `pandas`: 用于数据处理和分析 (尤其可能用于处理CHI数据)。
    *   `numpy`: 用于数值计算 (与`pandas`配合使用)。
    *   `aiohttp`: 用于异步HTTP客户端请求 (例如与Moonraker HTTP API通信)。
    *   `requests`: 用于同步HTTP客户端请求 (备用或用于某些场景)。
    *   (其他在 `requirements.txt` 中列出的相关库，根据实际项目情况添加)

### 2. 前端 (Frontend)
*   **编程语言:** HTML, CSS, JavaScript (ES6+)
*   **框架/库:** Bootstrap 5.3.0
    *   用于构建响应式的用户界面组件和整体样式。
*   **API 通信:**
    *   **Fetch API:** 用于执行标准的HTTP请求 (GET, POST等) 与后端API交互。
    *   **原生WebSocket API:** 用于建立和管理与后端服务器的WebSocket连接，实现实时数据交换。

### 3. 开发工具与环境 (Development Tools & Environment)
*   **包管理:** `pip` (通过 `requirements.txt`)
    *   用于安装和管理Python项目的依赖库。
*   **版本控制:** Git (未明确列出，但通常用于项目管理)

### 4. 配置 (Configuration)
*   **配置文件:** `device_config.json` (或通过环境变量)
    *   用于存储应用层面的配置，例如Moonraker服务地址、CHI路径、文件保存目录等。
    *   通常配合 `python-dotenv` 使用，允许通过 `.env` 文件覆盖默认配置或在部署时设置环境变量。

### 5. API文档链接 (API Documentation Links)
*   **FastAPI:** [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)
    *   FastAPI 官方文档，包含框架使用、教程和API参考。
*   **Moonraker API:** [https://moonraker.readthedocs.io/](https://moonraker.readthedocs.io/)
    *   Moonraker 官方API文档，详细说明了其HTTP和WebSocket接口。
*   **Bootstrap 5.3:** [https://getbootstrap.com/docs/5.3/](https://getbootstrap.com/docs/5.3/)
    *   Bootstrap 官方文档，包含所有组件、表单、布局等的使用方法。
*   **CHI电化学工作站:**
    *   CHI仪器的操作和编程文档通常由设备供应商提供，并且可能是特定于型号的。请参考您所使用CHI设备的官方手册或技术支持资源。

## Frontend Guidelines (前端指南文档)

### 1. 字体 (Font)
*   **默认字体栈:** 遵循Bootstrap 5的默认字体栈，优先使用操作系统UI字体。
    *   `system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans", sans-serif, "Apple Color Emoji", "Segoe UI Emoji", "Segoe UI Symbol", "Noto Color Emoji"`

### 2. 颜色规范 (Color Palette)
*   **主要颜色 (Primary):** Bootstrap 默认蓝色 (`#0d6efd`) - 用于主要按钮、活动指示器等。
*   **次要颜色 (Secondary):** Bootstrap 默认灰色 (`#6c757d`) - 用于辅助文本、非活动元素等。
*   **成功状态 (Success):** Bootstrap 默认绿色 (`#198754`) - 用于成功消息、初始化完成状态等。
*   **危险/错误状态 (Danger):** Bootstrap 默认红色 (`#dc3545`) - 用于错误消息、离线状态、停止按钮等。
*   **信息提示 (Info):** Bootstrap 默认青色 (`#0dcaf0`) - 用于信息性提示或特定组件高亮。
*   **状态指示器:**
    *   **设备初始化/在线状态:** 绿色 (`#198754`)。
    *   **设备离线/关闭状态:** 红色 (`#dc3545`)。
    *   **继电器状态:** 使用Bootstrap徽章 (Badges)
        *   开启: `bg-success` (绿色背景)。
        *   关闭: `bg-danger` (红色背景)。
        *   未知/未初始化: `bg-secondary` (灰色背景)。

### 3. 布局 (Layout)
*   **应用类型:** 单页面应用程序 (SPA) 结构。
*   **栅格系统:** 使用Bootstrap的栅格系统 (`container`, `row`, `col-md-*` 等) 实现响应式布局。
    *   主内容区域使用 `container` 或 `container-fluid`。
    *   设备控制面板和相关信息块使用 `row` 和 `col-*` 进行组织。
*   **导航:**
    *   主要导航采用标签页 (Tabbed navigation) 形式 (`nav-tabs`)，用于在不同的设备控制面板（打印机、蠕动泵、继电器、CHI工作站）之间切换。
*   **内容分组:**
    *   使用Bootstrap卡片 (`card`) 组件来组织和展示相关的控件组、参数输入区域和信息显示区域。
    *   例如，每个设备的控制表单和状态信息可以放置在一个或多个卡片中。

### 4. 组件库 (Component Library)
*   **核心库:** Bootstrap 5.3.0。
*   **常用组件:**
    *   **按钮 (Buttons):** `<button>` 元素配合 `.btn`, `.btn-primary`, `.btn-danger` 等类。
    *   **表单 (Forms):** `<form>`, `<input>`, `<label>`, `<select>`, `.form-control`, `.form-label`, `.form-select` 等。
    *   **卡片 (Cards):** `.card`, `.card-body`, `.card-title`, `.card-header` 等。
    *   **导航 (Navs):** `.nav`, `.nav-tabs`, `.nav-item`, `.nav-link` 用于标签页导航。
    *   **进度条 (Progress bars):** `.progress`, `.progress-bar` 用于显示泵送进度或测试进度。
    *   **徽章 (Badges):** `.badge`, `bg-success`, `bg-danger` 等用于显示状态。
    *   **列表组 (List groups):** `.list-group`, `.list-group-item` 用于显示CHI结果文件列表或日志条目。

### 5. 图标集 (Icon Set)
*   **策略:** 不强制使用特定的第三方图标库。
*   **现有方式:**
    *   依赖Bootstrap组件内置的简单指示（例如关闭按钮的 `×`）。
    *   使用Unicode字符（例如，状态指示的圆点符号）。
    *   如果需要更丰富的图标，未来可以考虑集成如Bootstrap Icons或Font Awesome。

### 6. 日志记录 (Logging)
*   **显示区域:** 页面上存在一个专用的日志面板 (例如，一个ID为 `log-container` 的 `<div>` 元素，通常是一个 `textarea` 或者使用 `list-group` 实现)。
*   **样式区分:**
    *   **错误消息:** 日志条目应具有明显的错误指示样式（例如，红色文本，或使用Bootstrap的 `.text-danger` 类）。
    *   **成功消息:** 日志条目应具有成功指示样式（例如，绿色文本，或使用Bootstrap的 `.text-success` 类）。
    *   **一般信息:** 默认文本颜色。
*   **格式:** 每条日志应包含时间戳和清晰的事件描述。

### 7. 响应式设计 (Responsiveness)
*   **实现基础:** 基于Bootstrap 5框架构建，使其具有固有的响应式能力。
*   **关键点:**
    *   使用Bootstrap的栅格系统 (`col-sm-*`, `col-md-*`, `col-lg-*`) 确保在不同屏幕尺寸（手机、平板、桌面）下布局合理。
    *   表单控件、卡片和导航栏等组件会根据屏幕宽度自动调整其展现方式。
    *   避免使用固定宽度元素，优先使用相对单位（如百分比）或Bootstrap的响应式工具类。

## Backend Architecture (后端结构文档)

### 1. 整体架构 (Overall Architecture)
*   后端采用基于FastAPI的单体架构，提供RESTful API接口和WebSocket通信能力，用于处理前端请求和控制硬件设备。
*   **设备控制适配模式:** 通过自定义适配器模式与各类硬件设备（打印机、蠕动泵、继电器、CHI工作站）进行通信和控制。

### 2. API层 (API Layer)
*   **框架:** 使用FastAPI定义API路由，处理HTTP请求（GET, POST等）。
*   **数据验证:** 利用Pydantic模型对请求体和响应体进行数据类型定义、格式校验和文档自动生成。

### 3. 设备控制层 (Device Control Layer)
*   **设计模式:** 在 `device_tester.py` 中采用适配器模式（Adapter Pattern），通过特定的适配器类或函数封装对不同硬件设备的控制逻辑。
*   **打印机、蠕动泵、继电器控制:**
    *   主要通过与Moonraker API进行交互（发送HTTP请求或G-code指令）来实现控制。
    *   `PumpAdapter` (蠕动泵适配器) 利用 `MoonrakerWebsocketListener` 监听来自Moonraker的实时参数，例如用于精确控制液体分配的挤出机状态。
*   **CHI电化学工作站控制:**
    *   `device_tester.py` 中的CHI处理逻辑通过 `device_control.control_chi` 模块实现。
    *   该模块封装了对CHI仪器制造商提供的软件（例如 `chi760e.exe` 命令行工具）的调用。
    *   功能包括：运行电化学测试、监控测试过程中生成的输出文件（如 `.txt` 数据文件和 `.png` 图像文件）、以及管理测试的整个生命周期（启动、监控、停止）。

### 4. 实时通信 (Real-time Communication)
*   **客户端更新:**
    *   通过一个全局的 `Broadcaster` 实例，将设备状态更新、操作日志等信息通过WebSocket (`/ws` 端点) 实时推送给所有连接的前端客户端。
*   **Moonraker监听器:**
    *   `core_api.moonraker_listener.py` 模块负责与Moonraker服务建立一个持久化的WebSocket连接。
    *   此连接用于接收Moonraker推送的实时事件和状态更新，特别是对于蠕动泵这类需要精确反馈的设备。

### 5. 配置管理 (Configuration Management)
*   **加载机制:** 应用启动时，从 `device_config.json` 文件中加载各项配置参数。
*   **动态更新:** 提供API端点 (`/api/config`) 允许用户通过前端界面修改配置，修改后的配置会保存回 `device_config.json` 文件。

### 6. 数据存储与持久化 (Data Storage & Persistence)
*   **CHI测试结果:**
    *   电化学测试生成的原始数据文件 (`.txt`) 和可能的图像文件 (`.png`) 被存储在文件系统中，具体路径由配置中的 `results_dir` 指定。
*   **应用配置:**
    *   系统的各项配置参数（如Moonraker地址、CHI程序路径、结果保存目录等）以JSON格式存储在 `device_config.json` 文件中。
*   **数据库使用情况:**
    *   尽管 `requirements.txt` 文件中可能列出了异步数据库驱动 (如 `databases[sqlite]`, `aiosqlite`)，但在主要应用逻辑 (`device_tester.py`) 中并未直接体现其用于核心数据的存储。
    *   当前版本主要依赖文件系统进行数据持久化。数据库驱动可能是为未来扩展或特定可选功能预留的。

### 7. 认证与授权 (Authentication & Authorization)
*   **现状:** 当前版本的后端未实现用户认证或授权机制。所有API端点和设备操作均对任何能够访问服务的用户开放。

### 8. 错误处理 (Error Handling)
*   **常规处理:** 在API端点和设备控制逻辑中包含基本的错误处理机制。
*   **反馈:** 当发生错误时，系统会记录错误信息到日志，并向客户端返回结构化的JSON错误响应（通常包含错误详情）。
*   **设备交互错误:** 与硬件设备通信失败（如连接超时、设备忙碌）会被捕获，并尝试向用户提供有意义的错误提示。

### 9. 边界场景处理策略 (Edge Case Handling Examples)
*   **设备不可用/未初始化:**
    *   在执行设备操作前，通常会有初始化检查。
    *   如果尝试操作一个未成功初始化或当前不可用的设备，API会返回错误信息，阻止后续操作。
*   **并发操作控制:**
    *   对于CHI电化学测试这类耗时且独占资源的操作，`device_tester.py` 中使用了锁机制 (`chi_test_lock`) 来确保在任何时候只有一个CHI测试在运行，防止并发冲突。
    *   (注: 此锁机制是在 `device_tester.py` 主程序中实现的。)
*   **文件I/O问题:**
    *   在进行文件读写操作（如保存CHI结果、读写配置文件）时，使用 `try-except` 块来捕获和处理可能发生的IO异常（如权限不足、磁盘空间满）。
*   **外部服务依赖 (Moonraker):**
    *   如果Moonraker服务连接失败或无响应，相关设备的初始化和操作会失败，并通过错误处理机制向上层报告。

## 总结 (Summary)

### 1. 目标 (Objective)
本文档旨在为“设备测试器”(Device Tester)应用提供一个全面的技术概述。它详细说明了项目的需求、应用流程、技术栈、前端设计原则以及后端架构。

### 2. 边界 (Boundaries)
本文档覆盖了从代码库（包括 `device_tester.py`, `device_tester.html`, `requirements.txt`, `device_control/control_chi.py` 及其他相关后端文件）观察到的现有功能。它描述了本系统如何与外部硬件（通过Moonraker控制的打印机、蠕动泵、继电器，以及通过其专有软件控制的CHI电化学工作站）进行交互。本文档不涉及超出其控制接口范围的底层硬件细节，也不对代码中未明确存在或清晰暗示的功能进行推测。重点在于软件的设计和操作。