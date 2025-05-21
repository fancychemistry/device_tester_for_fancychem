# -*- coding: utf-8 -*-
"""
generate_report_from_folder.py
==============================
给定实验文件夹 → 生成 Excel 报告（项目 + 中央）。

更新 V3.1 (优化版 EIS增强)
-----------------
1. 单独解析并绘制 3 类曲线：
   • CV_Main  (文件 *_CV.txt)
   • CV_Cdl   (文件 *_CV_Cdl.txt)
   • LSV_Main (文件 *_LSV.txt)

2. EIS → Nyquist + Bode（文件 *_EIS.txt） - 增强匹配与容错能力。

3. 计时电流 IT 图:
   • 修正了IT文件名解析和构造逻辑。
   • 优化对 IT 文件的解析，确保数据正确读取。

4. 图片 600 dpi，Excel 中显示尺寸约为 520×360 px (Nyquist 为方图，尺寸调整)；
   Anchor 统一在 layout 字典。

5. 减少冗余调试日志，专注于重要运行信息和错误提示。
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import List, Tuple
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm  # 添加进度条支持

# 确定项目根目录，并将 utils 添加到 sys.path
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from utils.excel_reporting import ExcelReporter
except ImportError as e:
    print(f"错误：无法导入 ExcelReporter 模块。请确保 'utils/excel_reporting.py' 文件存在于脚本 '{__file__}' 同级目录下的 'utils' 文件夹中。")
    print(f"详细错误: {e}")
    print(f"Python搜索路径 (sys.path): {sys.path}")
    sys.exit(1)


# ------------------ 日志配置 ------------------
logging.basicConfig(
    level=logging.INFO, # <--- 从DEBUG改为INFO级别，减少过量日志输出
    format="%(asctime)s | %(levelname)-5s | %(filename)-30s:%(lineno)-4d | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("ReportGeneratorStandalone")

# ------------------ 常量定义 ------------------
DEFAULT_DPI = 600
MAIN_PLOT_SIZE_PX = (520, 360)
NYQUIST_PLOT_SIZE_PX = (400, 400)

# ------------------ 工具函数 ------------------

def _read_curve_txt(fp: Path) -> pd.DataFrame:
    if not fp.is_file():
        log.warning(f"文件不存在: {fp}")
        return pd.DataFrame()
    if fp.stat().st_size == 0:
        log.warning(f"文件为空: {fp}")
        return pd.DataFrame()

    rows = []
    data_section_started = False # True 表示表头已找到，下一行开始是数据

    try:
        lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as e:
        log.error(f"读取文件失败 {fp}: {e}")
        return pd.DataFrame()

    # 表头模式：灵活匹配空格/制表符，忽略大小写
    # 注意： CHI导出的列名中，斜杠前后可能有空格，也可能没有
    # 例如 "Potential / V" 或 "Potential/V"
    header_patterns = [
        re.compile(r"Potential\s*/\s*V\s*[,;\s\t]+\s*Current\s*/\s*A", re.IGNORECASE),
        re.compile(r"Time\s*/\s*sec\s*[,;\s\t]+\s*Current\s*/\s*A", re.IGNORECASE), # IT文件头是 Time/sec
        re.compile(r"Time\s*/\s*s\s*[,;\s\t]+\s*Current\s*/\s*A", re.IGNORECASE), # 也可能是 Time/s
        re.compile(r"Potential\s*\(V\)\s*[,;\s\t]+\s*Current\s*\(A\)", re.IGNORECASE),
        re.compile(r"Time\s*\(s\)\s*[,;\s\t]+\s*Current\s*\(A\)", re.IGNORECASE),
    ]

    for line_idx, line_content in enumerate(lines):
        line_stripped = line_content.strip()
        
        if not line_stripped: # 跳过空行
            continue

        if not data_section_started:
            is_header = False
            for pattern_idx, pattern in enumerate(header_patterns):
                if pattern.search(line_stripped): # 使用 search，因为表头可能不是一行的全部内容
                    is_header = True
                    break
            
            if is_header:
                log.info(f"在 {fp.name} 第 {line_idx+1} 行找到表头: '{line_stripped}'。数据将从下一行开始解析。")
                data_section_started = True # 标记表头已找到，下一行开始是数据
            continue # 无论是否找到表头，这一行（元数据或表头本身）都处理完了，继续下一行
        
        # 此处意味着 data_section_started is True，我们正在处理数据行
        segments = re.split(r"[,\s\t]+", line_stripped) 
        if len(segments) >= 2:
            try:
                x_val = float(segments[0])
                y_val = float(segments[1])
                rows.append([x_val, y_val])
            except ValueError:
                log.warning(f"在数据区无法解析数据行: '{line_stripped}' in {fp.name} at line {line_idx+1}。已跳过此行。")
        else:
            # 数据行列数不足，不输出过多日志
            pass
            
    if not rows:
        log.warning(f"未能从文件 {fp.name} 读取到任何有效数据行。文件可能格式不兼容，数据区无法识别，或表头后无数据。")
        return pd.DataFrame()
    
    log.info(f"成功从 {fp.name} 读取 {len(rows)} 行数据。")
    return pd.DataFrame(rows, columns=["x", "y"])


def _simple_plot(df: pd.DataFrame, title: str, xlabel: str, ylabel: str, png_path: Path, figsize_inches: Tuple[float, float] = (6.5, 4.5)):
    if df.empty:
        log.warning(f"输入给 _simple_plot 的DataFrame为空，无法为 '{title}' 生成图像 {png_path.name}。")
        return

    plt.figure(figsize=figsize_inches)
    plt.plot(df["x"], df["y"], linewidth=1.5)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.title(title, fontsize=12, fontweight='bold')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    try:
        png_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(png_path, dpi=DEFAULT_DPI)
        log.info(f"图像已保存: {png_path}")
    except Exception as e:
        log.error(f"保存图像失败 {png_path}: {e}")
    finally:
        plt.close()


def parse_eis_file(file_path):
    """
    解析EIS数据文件，支持多种格式，包括CHI仪器格式
    
    Args:
        file_path: EIS文件路径
        
    Returns:
        包含freq, z_real, z_imag, z_mag, phase列的DataFrame，或None（如果解析失败）
    """
    log.info(f"开始解析EIS文件: {os.path.basename(file_path)}")
    
    # 定义EIS数据的可能表头模式
    eis_headers = [
        # CHI仪器格式特有的表头（非常特定的格式）
        (r'freq/hz,\s*z\'/ohm,\s*z"/ohm,\s*z/ohm,\s*phase/deg', ['freq', 'z_real', 'z_imag', 'z_mag', 'phase']),
        # 原始标准格式(不分大小写)
        (r'freq.*hz.*[,\s]+z.*real.*[,\s]+z.*imag.*[,\s]+z.*mag.*[,\s]+phase', ['freq', 'z_real', 'z_imag', 'z_mag', 'phase']),
        # 新增对CHI格式的支持（频率，实部，虚部，幅值，相位）
        (r'freq.*hz.*[,\s]+z\'.*ohm.*[,\s]+z".*ohm.*[,\s]+z.*ohm.*[,\s]+phase.*deg', ['freq', 'z_real', 'z_imag', 'z_mag', 'phase']),
        # |Z| 变体
        (r'freq.*hz.*[,\s]+z\'.*[,\s]+z\'\'.*[,\s]+\|z\|.*[,\s]+phase', ['freq', 'z_real', 'z_imag', 'z_mag', 'phase']),
        # 括号变体 (Hz) (ohm) 等
        (r'freq.*\(hz\).*[,\s]+z\'.*\(ohm\).*[,\s]+z\'\'.*\(ohm\).*[,\s]+z.*\(ohm\).*[,\s]+phase.*\(deg\)', ['freq', 'z_real', 'z_imag', 'z_mag', 'phase']),
        # 频率, 实部, 虚部
        (r'freq.*hz.*[,\s]+re\(z\).*[,\s]+im\(z\)', ['freq', 'z_real', 'z_imag']),
        # 频率, 幅值, 相位
        (r'freq.*hz.*[,\s]+z.*[,\s]+phase', ['freq', 'z_mag', 'phase']),
        # 频率, 实部, 虚部
        (r'frequency.*[,\s]+z\'.*[,\s]+z\'\'', ['freq', 'z_real', 'z_imag']),
        # 频率, 实部, 虚部 (另一种表示)
        (r'frequency.*[,\s]+z_real.*[,\s]+z_imag', ['freq', 'z_real', 'z_imag']),
        # 松散匹配前三列就足够了，但匹配度要设置低一些
        (r'freq.*[,\s]+z.*[,\s]+z', ['freq', 'z_real', 'z_imag']),
    ]
    
    try:
        # 检查文件是否存在
        if not os.path.exists(file_path):
            log.error(f"EIS文件不存在: {file_path}")
            return None
        
        # 首先尝试检测表头
        header_line = None
        header_index = None
        column_names = None
        found_chi_format = False
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
            # 检查是否是CHI格式（有特征性的仪器信息行）
            for i, line in enumerate(lines):
                if "Instrument Model:  CHI" in line or "CHI" in line and "EIS" in line:
                    found_chi_format = True
                    log.info(f"检测到CHI仪器格式的EIS文件")
                    break
            
            # 寻找数据表头
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                # 尝试匹配所有可能的EIS表头模式
                for pattern, names in eis_headers:
                    if re.search(pattern, line.lower(), re.IGNORECASE):
                        header_line = line
                        header_index = i
                        column_names = names
                        log.info(f"在EIS文件第 {i+1} 行找到表头: '{line}'")
                        break
                
                if header_line:
                    break
        
        # 如果找到表头，则从下一行开始读取数据
        if header_line and header_index is not None:
            try:
                log.info(f"开始解析EIS数据，表头在第 {header_index+1} 行")
                # 读取表头后的数据
                data_lines = lines[header_index+1:]
                data_rows = []
                
                # 处理科学计数法格式 (如 1.234e+3)
                for line in data_lines:
                    line = line.strip()
                    if not line:
                        continue
                        
                    # 根据逗号分隔，处理可能的空格
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:  # 至少需要频率、实部、虚部
                        try:
                            # 尝试转换为浮点数，支持科学计数法
                            row_values = []
                            for part in parts:
                                # 处理科学计数法 (1.234e+3)
                                value = float(part.replace('e+', 'e').replace('e-', 'e-'))
                                row_values.append(value)
                                
                            # 确保有足够的数据列
                            while len(row_values) < len(column_names):
                                row_values.append(None)
                                
                            data_rows.append(row_values[:len(column_names)])
                        except ValueError:
                            # 跳过无法解析的行
                            continue
                
                if not data_rows:
                    log.warning(f"未能从数据表头后解析到任何有效数据行")
                    return None
                    
                # 创建DataFrame
                df = pd.DataFrame(data_rows, columns=column_names)
                
                # 如果缺少某些列，尝试计算它们
                if 'z_real' in df.columns and 'z_imag' in df.columns and 'z_mag' not in df.columns:
                    df['z_mag'] = np.sqrt(df['z_real']**2 + df['z_imag']**2)
                    
                if 'z_real' in df.columns and 'z_imag' in df.columns and 'phase' not in df.columns:
                    df['phase'] = np.arctan2(df['z_imag'], df['z_real']) * 180 / np.pi
                
                log.info(f"已成功从EIS文件 {os.path.basename(file_path)} 解析 {len(df)} 行数据。")
                
                # 根据频率排序
                if 'freq' in df.columns:
                    df = df.sort_values(by='freq').reset_index(drop=True)
                
                return df
                
            except Exception as e:
                log.error(f"解析EIS文件(有表头模式)时出错: {str(e)}")
        
        # 如果找不到表头或者上面的解析失败，尝试直接解析数据行
        try:
            log.warning(f"未在EIS文件 {os.path.basename(file_path)} 中找到任何可识别的EIS表头。尝试直接解析数据...")
            
            data_rows = []
            
            # 跳过文件开头的非数据行
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # 寻找看起来像数据的行 (数值+逗号格式)
                if re.match(r'^[\d\.\-+eE]+\s*,', line):
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:  # 至少需要频率、实部、虚部
                        try:
                            # 尝试将所有部分转换为浮点数
                            row_values = [float(p.replace('e+', 'e').replace('e-', 'e-')) for p in parts if p]
                            if len(row_values) >= 3:
                                data_rows.append(row_values[:5] if len(row_values) > 5 else row_values)
                        except ValueError:
                            # 跳过无法解析的行
                            continue
            
            if not data_rows:
                log.warning(f"直接解析也未能提取任何数据行")
                return None
                
            # 假设列的顺序为：频率、实部、虚部、[幅值、相位]
            # 确定提取的列数
            max_columns = max(len(row) for row in data_rows)
            column_names = []
            
            if max_columns >= 3:
                column_names = ['freq', 'z_real', 'z_imag']
                if max_columns >= 4:
                    column_names.append('z_mag')
                    if max_columns >= 5:
                        column_names.append('phase')
            
            # 创建DataFrame并填充可能缺失的列
            df = pd.DataFrame(data_rows, columns=column_names)
            
            # 填充计算列
            if 'z_real' in df.columns and 'z_imag' in df.columns:
                if 'z_mag' not in df.columns:
                    df['z_mag'] = np.sqrt(df['z_real']**2 + df['z_imag']**2)
                if 'phase' not in df.columns:
                    df['phase'] = np.arctan2(df['z_imag'], df['z_real']) * 180 / np.pi
            
            # 确保所有必需的列都存在
            required_columns = ['freq', 'z_real', 'z_imag', 'z_mag', 'phase']
            for col in required_columns:
                if col not in df.columns:
                    log.warning(f"缺少必需的列 '{col}'，尝试计算")
                    if col == 'z_mag' and 'z_real' in df.columns and 'z_imag' in df.columns:
                        df[col] = np.sqrt(df['z_real']**2 + df['z_imag']**2)
                    elif col == 'phase' and 'z_real' in df.columns and 'z_imag' in df.columns:
                        df[col] = np.arctan2(df['z_imag'], df['z_real']) * 180 / np.pi
            
            log.info(f"在无表头模式下从EIS文件 {os.path.basename(file_path)} 成功解析 {len(df)} 行数据。")
            
            # 根据频率排序
            if 'freq' in df.columns:
                df = df.sort_values(by='freq').reset_index(drop=True)
                
            return df
                
        except Exception as e:
            log.error(f"直接解析EIS文件时出错: {str(e)}")
            return None
    
    except Exception as e:
        log.error(f"EIS文件解析过程中出现未处理的异常: {str(e)}")
        return None
    
    return None


def _plot_eis(df_eis: pd.DataFrame, title_prefix: str,
              nyquist_path: Path, bode_path: Path,
              plot_size_px: Tuple[int, int] = NYQUIST_PLOT_SIZE_PX):
    """绘制EIS图表：奈奎斯特图和波德图"""
    if df_eis is None or df_eis.empty:
        log.warning("EIS DataFrame为空或为None，无法生成Nyquist和Bode图。")
        return
    
    # 确保有必要的列
    required_columns = {
        'Nyquist': ['z_real', 'z_imag'],
        'Bode': ['freq', 'z_mag', 'phase']
    }
    
    # 检查并规范化列名
    column_mapping = {}
    df_columns_lower = [col.lower() for col in df_eis.columns]
    
    # 尝试映射现有列到所需列
    for col in ['freq', 'z_real', 'z_imag', 'z_mag', 'phase']:
        if col in df_eis.columns:
            column_mapping[col] = col
        else:
            # 检查是否有类似的列名
            for existing_col, lower_col in zip(df_eis.columns, df_columns_lower):
                if col == 'freq' and ('freq' in lower_col or 'f' == lower_col):
                    column_mapping[col] = existing_col
                    break
                elif col == 'z_real' and any(x in lower_col for x in ["z'", "real", "re(z)", "z_real"]):
                    column_mapping[col] = existing_col
                    break
                elif col == 'z_imag' and any(x in lower_col for x in ["z\"", "imag", "im(z)", "z_imag", "z''"]):
                    column_mapping[col] = existing_col
                    break
                elif col == 'z_mag' and any(x in lower_col for x in ['|z|', 'abs', 'magnitude', 'z_mag', 'z/ohm']):
                    column_mapping[col] = existing_col
                    break
                elif col == 'phase' and any(x in lower_col for x in ['phase', 'φ', 'angle', 'phase/deg']):
                    column_mapping[col] = existing_col
                    break
    
    # 创建一个工作副本，以便映射列名
    df_work = df_eis.copy()
    
    # 重命名列
    for target_col, source_col in column_mapping.items():
        if source_col in df_eis.columns and target_col != source_col:
            df_work[target_col] = df_eis[source_col]
    
    # 检查是否有足够的列用于绘图
    can_plot_nyquist = all(col in df_work.columns for col in required_columns['Nyquist'])
    can_plot_bode = all(col in df_work.columns for col in required_columns['Bode'])
    
    if not can_plot_nyquist and not can_plot_bode:
        log.warning(f"EIS数据缺少必要的列用于绘图。需要 {required_columns['Nyquist']} 用于Nyquist图，{required_columns['Bode']} 用于Bode图。")
        log.warning(f"当前可用列: {list(df_eis.columns)}")
        return

    # Nyquist Plot
    if can_plot_nyquist:
        try:
            plt.figure(figsize=plot_size_px)
            plt.plot(df_work["z_real"], -df_work["z_imag"], 'o-', markersize=4, linewidth=1.5, color='dodgerblue')
            plt.xlabel("Z' / Ω", fontsize=11)
            plt.ylabel("−Z'' / Ω", fontsize=11)
            plt.title(f"{title_prefix}Nyquist Plot", fontsize=13, fontweight='bold')
            plt.grid(True, linestyle=':', alpha=0.6)
            
            # 确保图形有数据并且比例合适
            if len(df_work) > 1:
                # 计算适当的轴范围
                x_min, x_max = df_work["z_real"].min(), df_work["z_real"].max()
                y_min, y_max = (-df_work["z_imag"]).min(), (-df_work["z_imag"]).max()
                
                # 给轴增加一点余量
                x_pad = max(0.1, (x_max - x_min) * 0.1) if x_max > x_min else 0.1
                y_pad = max(0.1, (y_max - y_min) * 0.1) if y_max > y_min else 0.1
                
                plt.xlim(max(0, x_min - x_pad), x_max + x_pad)
                plt.ylim(max(0, y_min - y_pad), y_max + y_pad)
            
            plt.axis("equal")  # 保持纵横比
            plt.tight_layout(pad=0.5)
            
            nyquist_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(nyquist_path, dpi=600)
            log.info(f"Nyquist图已保存: {nyquist_path}")
        except Exception as e:
            log.error(f"保存Nyquist图失败 {nyquist_path}: {e}")
        finally:
            plt.close()
    else:
        log.warning("无法绘制Nyquist图：缺少必要的列")
    
    # Bode Plot - 改进错误处理和轴范围设置
    if can_plot_bode:
        try:
            fig, ax1 = plt.subplots(figsize=(7.0, 4.8))  # 稍微调整尺寸
            ax1.set_xscale("log")
            ax1.set_xlabel("Frequency / Hz", fontsize=11)
            ax1.set_ylabel("|Z| / Ω", fontsize=11, color='mediumblue')
            
            # 检查频率数据是否有效
            if (df_work["freq"] <= 0).any():
                log.warning("发现频率值 <= 0，这对对数刻度无效。将过滤这些值。")
                df_work = df_work[df_work["freq"] > 0].copy()
                if df_work.empty:
                    log.error("过滤后无有效频率数据，无法绘制Bode图。")
                    plt.close()
                    return
            
            ax1.plot(df_work["freq"], df_work["z_mag"], 'o-', color='mediumblue', markersize=4, linewidth=1.5, label="|Z|")
            ax1.tick_params(axis='y', labelcolor='mediumblue', labelsize=10)
            ax1.tick_params(axis='x', labelsize=10)
            ax1.grid(True, which="both", linestyle=':', alpha=0.4)  # 网格更细
            
            ax2 = ax1.twinx()
            ax2.set_ylabel("Phase / °", fontsize=11, color='crimson')
            
            # 相位值范围计算和边界设置
            min_phase_val = df_work["phase"].min()
            max_phase_val = df_work["phase"].max()
            
            # 更合理的Y轴范围，处理极端情况
            if np.isfinite(min_phase_val) and np.isfinite(max_phase_val):
                y_min_phase = min(-5, np.floor(min_phase_val / 10) * 10 - 5)  # 向下取整到10的倍数再减5
                y_max_phase = max(10, np.ceil(max_phase_val / 10) * 10 + 5 if max_phase_val < 85 else 90)
                ax2.set_ylim(y_min_phase, y_max_phase)
            
            ax2.plot(df_work["freq"], df_work["phase"], 's--', color='crimson', markersize=4, linewidth=1.5, label="Phase")
            ax2.tick_params(axis='y', labelcolor='crimson', labelsize=10)
        
            # 图例
            handles1, labels1 = ax1.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            fig.legend(handles1 + handles2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, 0.01), ncol=2, fontsize=9, frameon=False)
        
            plt.title(f"{title_prefix}Bode Plot", fontsize=13, fontweight='bold', y=1.03)  # 调整标题位置
            fig.tight_layout(rect=[0, 0.05, 1, 0.95])  # 为底部图例留出空间
            
            bode_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(bode_path, dpi=600)
            log.info(f"Bode图已保存: {bode_path}")
        except Exception as e:
            log.error(f"保存Bode图失败 {bode_path}: {e}", exc_info=True)
        finally:
            plt.close()
    else:
        log.warning("无法绘制Bode图：缺少必要的列")


# ------------------ 主流程函数 ------------------
def generate_report(experiment_folder: Path, central_results_dir: Path):
    """
    为给定的实验文件夹生成Excel报告
    
    Args:
        experiment_folder: 实验数据文件夹路径
        central_results_dir: 中央结果目录路径
    """
    project_name = experiment_folder.name
    log.info(f"开始为项目 '{project_name}' 生成报告，数据源: {experiment_folder}")

    voltage_pattern = re.compile(rf"{re.escape(project_name)}_IT_(neg)?(\d+)V\.txt", re.I)
    voltages: List[float] = []
    log.info("开始从IT文件名推断电位序列...")
    for txt_file in experiment_folder.glob("*.txt"):
        match = voltage_pattern.match(txt_file.name)
        if match:
            is_negative_str, number_str = match.groups()
            try:
                raw_num = int(number_str)
                voltage_val = raw_num / 10.0
                if is_negative_str: voltage_val = -voltage_val
                voltages.append(voltage_val)
                log.debug(f"从文件名 '{txt_file.name}' 推断出电位: {voltage_val:.2f} V (raw num: {number_str})")
            except ValueError:
                log.warning(f"无法从文件名 '{txt_file.name}' 中的数字部分 '{number_str}' 解析电位。")
    
    voltages.sort()
    positions = [f"推断位置#{i+1}" for i in range(len(voltages))]
    if voltages: log.info(f"成功推断并排序的电位序列: {voltages}")
    else: log.warning(f"未能从项目 '{project_name}' 的IT文件名中推断出任何电位值。")

    layout_config = {
        "cv_main_plot_anchor": "E2", "cv_cdl_plot_anchor": "M2", "lsv_main_plot_anchor": "U2",
        "eis_nyquist_plot_anchor": "E20", "eis_bode_plot_anchor": "M20",
        "central_cv_main_plot_anchor": "H2", "central_cv_cdl_plot_anchor": "P2",
        "central_lsv_main_plot_anchor": "X2", 
        "central_eis_nyquist_plot_anchor": "H22", "central_eis_bode_plot_anchor": "P22",
        "it_plots_start_anchor": "E40", "central_it_plots_start_anchor": "H42",
    }
    
    project_excel_path = experiment_folder / f"{project_name}_实验结果报告.xlsx"
    central_excel_path = central_results_dir / "ALL_Experiments_Summary_Standalone.xlsx"

    reporter = ExcelReporter(
        project_name=project_name, project_excel_path=project_excel_path,
        central_excel_path=central_excel_path, voltages_list=voltages,
        positions_list=positions, layout_map=layout_config,
    )
    log.info(f"ExcelReporter已为项目 '{project_name}' 初始化。")

    curve_configurations = {
        "CV_Main": {"file_suffix": "CV", "xlabel": "Potential / V", "ylabel": "Current / A", "plot_size_px": MAIN_PLOT_SIZE_PX},
        "CV_Cdl": {"file_suffix": "CV_Cdl", "xlabel": "Potential / V", "ylabel": "Current / A", "plot_size_px": MAIN_PLOT_SIZE_PX},
        "LSV_Main": {"file_suffix": "LSV", "xlabel": "Potential / V", "ylabel": "Current / A", "plot_size_px": MAIN_PLOT_SIZE_PX},
    }

    for tag, config in curve_configurations.items():
        txt_path = experiment_folder / f"{project_name}_{config['file_suffix']}.txt"
        png_path = experiment_folder / f"{project_name}_{tag.replace(' ', '_')}.png"
        log.info(f"处理曲线类型 '{tag}': {txt_path.name}")
        df_curve = pd.DataFrame()
        if txt_path.exists():
            df_curve = _read_curve_txt(txt_path)
            if not df_curve.empty:
                if not png_path.exists():
                    _simple_plot(df_curve, title=f"{project_name} - {tag}", output_path=png_path, curve_config=config, plot_size_px=config['plot_size_px'])
                else: log.info(f"曲线图 {png_path.name} 已存在，跳过生成。")
            else: log.warning(f"无法从 {txt_path.name} 读取数据，跳过 '{tag}' 图像生成。")
        else: log.warning(f"数据文件 {txt_path.name} 不存在，跳过 '{tag}' 处理。")

        # Enhanced logging for image path decision
        effective_png_path = None
        if df_curve is not None and not df_curve.empty:
            if png_path.exists() and png_path.stat().st_size > 0:
                effective_png_path = png_path
                log.info(f"准备为 '{tag}' 记录主图表，有效图像路径: {effective_png_path}")
            else:
                log.warning(f"准备为 '{tag}' 记录主图表，数据有效但图像文件不存在或为空: {png_path}")
        else:
            log.warning(f"准备为 '{tag}' 记录主图表，但数据框为空。图像文件检查被跳过: {png_path.name}")

        reporter.record_main_plot(
            tag_name=tag, png_image_path=effective_png_path,
            project_anchor_key=f"{tag.lower()}_plot_anchor",
            central_anchor_key=f"central_{tag.lower()}_plot_anchor",
            image_size_px=config['plot_size_px']
        )

    # 优化EIS处理流程
    eis_txt_path = experiment_folder / f"{project_name}_EIS.txt"
    nyquist_png_path = experiment_folder / f"{project_name}_EIS_Nyquist_plot.png"
    bode_png_path = experiment_folder / f"{project_name}_EIS_Bode_plot.png"
    log.info(f"处理EIS数据: {eis_txt_path.name}")
    df_eis_data = None
    
    # 1. 检查EIS文件是否存在
    if eis_txt_path.exists():
        # 2. 尝试解析EIS数据 - 使用改进的函数
        df_eis_data = parse_eis_file(eis_txt_path)
        
        # 3. 如果成功解析数据，尝试生成图表
        if df_eis_data is not None and not df_eis_data.empty:
            # 3.1 检查是否需要重新生成图像
            regenerate_images = not nyquist_png_path.exists() or not bode_png_path.exists()
            if regenerate_images: 
                try:
                    log.info(f"开始绘制EIS图表: Nyquist和Bode图")
                    _plot_eis(df_eis_data, title_prefix=f"{project_name} - ",
                              nyquist_path=nyquist_png_path, bode_path=bode_png_path,
                              plot_size_px=config.get('eis_plot_size_px', NYQUIST_PLOT_SIZE_PX))
                except Exception as e:
                    log.error(f"绘制EIS图表时发生错误: {e}", exc_info=True)
            else: 
                log.info(f"EIS图像 {nyquist_png_path.name} 和 {bode_png_path.name} 已存在，跳过生成。")
        else: 
            log.warning(f"无法从 {eis_txt_path.name} 读取EIS数据，跳过图像生成。")
    else: 
        log.warning(f"EIS数据文件 {eis_txt_path.name} 不存在，跳过处理。")
    
    # 4. 记录EIS图表结果
    effective_nyquist_path = None
    if not df_eis_data.empty: # df_eis_data is confirmed not None before this block
        if nyquist_png_path.exists() and nyquist_png_path.stat().st_size > 0:
            effective_nyquist_path = nyquist_png_path
            log.info(f"准备记录 EIS Nyquist 图，有效图像路径: {effective_nyquist_path}")
        else:
            log.warning(f"准备记录 EIS Nyquist 图，数据有效但图像文件不存在或为空: {nyquist_png_path}")
    else: # This case should ideally not be hit if df_eis_data was checked before plotting
        log.warning(f"准备记录 EIS Nyquist 图，但数据框为空。图像文件检查被跳过: {nyquist_png_path.name}")

    reporter.record_main_plot(
        "EIS_Nyquist",
        effective_nyquist_path,
        "eis_nyquist_plot_anchor", 
        "central_eis_nyquist_plot_anchor", 
        image_size_px=NYQUIST_PLOT_SIZE_PX
    )

    effective_bode_path = None
    if not df_eis_data.empty: # df_eis_data is confirmed not None before this block
        if bode_png_path.exists() and bode_png_path.stat().st_size > 0:
            effective_bode_path = bode_png_path
            log.info(f"准备记录 EIS Bode 图，有效图像路径: {effective_bode_path}")
        else:
            log.warning(f"准备记录 EIS Bode 图，数据有效但图像文件不存在或为空: {bode_png_path}")
    else: # This case should ideally not be hit
        log.warning(f"准备记录 EIS Bode 图，但数据框为空。图像文件检查被跳过: {bode_png_path.name}")

    reporter.record_main_plot(
        "EIS_Bode",
        effective_bode_path,
        "eis_bode_plot_anchor", 
        "central_eis_bode_plot_anchor", 
        image_size_px=MAIN_PLOT_SIZE_PX
    )

    log.info("处理IT曲线数据...")
    it_plot_gallery_items = []
    if not voltages: log.warning("未找到任何IT电压进行处理。IT部分将为空。")
    for idx, voltage_val in enumerate(voltages):
        num_abs_scaled = int(round(abs(voltage_val) * 10))
        voltage_tag_part = f"{'neg' if voltage_val < 0 else ''}{num_abs_scaled}"
        it_txt_path = experiment_folder / f"{project_name}_IT_{voltage_tag_part}V.txt"
        it_png_path = experiment_folder / f"{project_name}_IT_{voltage_tag_part}V_plot.png"
        log.debug(f"尝试查找IT文件: {it_txt_path.name} for voltage {voltage_val:.2f}V")
        charge_str = "源文件缺失"
        df_it = pd.DataFrame() # 确保df_it在if/else作用域外可用

        if it_txt_path.exists():
            log.info(f"开始读取IT文件: {it_txt_path.name}")
            df_it = _read_curve_txt(it_txt_path)
            if not df_it.empty:
                if len(df_it['x']) >= 2 and len(df_it['y']) >= 2:
                    try:
                        charge_coulombs = abs(float(np.trapezoid(df_it['y'], df_it['x'])))
                        charge_str = f"{charge_coulombs:.6f}"
                        log.info(f"IT @ {voltage_val:.2f}V ({it_txt_path.name}): 计算电荷量 = {charge_str} C")
                    except Exception as e:
                        log.error(f"为 {it_txt_path.name} 计算电荷量失败: {e}")
                        charge_str = "计算错误"
                else:
                    charge_str = "数据点不足"
                    log.warning(f"{it_txt_path.name} 数据点不足 ({len(df_it['x'])}点)，无法计算电荷量。")

                if not it_png_path.exists(): # 只有在成功读取数据后才尝试绘图
                    _simple_plot(df_it, f"IT Curve @ {voltage_val:.2f} V", "Time / s", "Current / A", it_png_path)
                else: log.info(f"IT曲线图 {it_png_path.name} 已存在，跳过生成。")
            else:
                charge_str = "空数据文件"
                log.warning(f"IT数据文件 {it_txt_path.name} 读取后DataFrame为空。")
        else:
            log.warning(f"IT数据文件 {it_txt_path.name} 不存在。")

        reporter.record_it_data_row(idx, voltage_val, charge_str, positions[idx] if idx < len(positions) else "N/A")
        # 只有当数据文件存在，数据被读取，且电荷计算未出错（或至少不是因文件缺失导致）时，才认为图片可能存在
        can_have_plot = it_txt_path.exists() and not df_it.empty and charge_str not in ["源文件缺失", "空数据文件"]
        it_plot_gallery_items.append({
            "voltage": voltage_val,
            "path": it_png_path if can_have_plot and it_png_path.exists() else None
        })

    reporter.add_it_plots_gallery(it_plot_gallery_items, target_workbook="project", anchor_key="it_plots_start_anchor", image_size_px=MAIN_PLOT_SIZE_PX)
    reporter.add_it_plots_gallery(it_plot_gallery_items, target_workbook="central", anchor_key="central_it_plots_start_anchor", image_size_px=MAIN_PLOT_SIZE_PX)
    log.info("IT曲线处理完成，图库信息已准备。")

    summary_info = {
        "脚本版本": "generate_report_from_folder.py V3.1 (EIS增强优化版)",
        "报告模式": "独立运行模式", 
        "数据文件夹": str(experiment_folder)
    }
    reporter.add_summary_info(additional_info=summary_info)
    
    try:
        reporter.save_all_workbooks()
        log.info(f"√ 项目 '{project_name}' 的Excel报告已成功生成!")
    except Exception as e:
        log.error(f"保存Excel工作簿失败: {e}", exc_info=True)


if __name__ == "__main__":
    log.info("报告生成脚本启动...")
    # 简化命令行参数处理
    if len(sys.argv) < 2:
        print("错误：缺少必要的参数。")
        print("用法：python generate_report_from_folder.py <实验文件夹路径> [中央结果目录路径]")
        sys.exit(1)

    try:
        experiment_data_folder = Path(sys.argv[1]).expanduser().resolve(strict=True)
        if not experiment_data_folder.is_dir():
            log.error(f"错误：指定的路径不是一个文件夹 -> {experiment_data_folder}")
            sys.exit(1)
    except FileNotFoundError:
        log.error(f"错误：指定的实验文件夹不存在 -> {sys.argv[1]}")
        sys.exit(1)
    except Exception as e:
        log.error(f"错误：指定的实验文件夹路径无效 -> {sys.argv[1]}: {e}")
        sys.exit(1)

    # 中央结果目录处理
    central_dir_path_str = "" # For logging in except block
    try:
        if len(sys.argv) > 2:
            central_dir = Path(sys.argv[2]).expanduser().resolve()
            central_dir_path_str = sys.argv[2]
        else:
            central_dir = experiment_data_folder.parent
            central_dir_path_str = str(experiment_data_folder.parent)
            log.info(f"未指定中央结果目录，将使用默认路径: {central_dir}")
        
        central_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"中央结果目录已确认/创建: {central_dir}")

    except Exception as e:
        log.error(f"无法创建或访问中央结果目录 (尝试路径: {central_dir_path_str}): {e}")
        sys.exit(1)
        
    log.info(f"实验文件夹: {experiment_data_folder}")
    log.info(f"中央结果目录: {central_dir}")

    # 检查是否已存在Excel文件，提供警告
    project_summary_file = experiment_data_folder / f"{experiment_data_folder.name}_实验结果报告.xlsx"
    central_summary_file = central_dir / "ALL_Experiments_Summary_Standalone.xlsx"
    
    for f_path in [project_summary_file, central_summary_file]:
        if f_path.exists():
            log.warning(f"文件 {f_path} 已存在，将被更新或覆盖。")

    # 主流程执行与错误处理
    try:
        generate_report(experiment_data_folder, central_dir)
        log.info("✓ 报告生成脚本执行完毕，所有任务已完成。")
    except Exception as e:
        log.error(f"主流程执行时发生未捕获的错误: {e}", exc_info=True)
        sys.exit(1)