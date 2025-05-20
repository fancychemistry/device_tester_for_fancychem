# control_chi.py
import subprocess
import os
import time
import psutil

# 全局变量
folder_save = '.'
model_pstat = 'chi760e'
path_lib = 'C:\\CHI760E\\chi760e\\chi760e.exe'


class Setup:
    """初始化 CHI760E 电位站设置"""

    def __init__(self, model='chi760e', path='C:\\CHI760E\\chi760e\\chi760e.exe', folder='.'):
        global folder_save, model_pstat, path_lib
        if model != 'chi760e':
            raise ValueError("仅支持 'chi760e' 型号")
        folder_save = folder
        model_pstat = model
        path_lib = path
        print(f"型号: {model}, 路径: {path}, 保存文件夹: {folder}")


class Technique:
    """技术基类"""

    def __init__(self, text, fileName, technique):
        self.text = text
        self.fileName = fileName
        self.technique = technique
        self.process = None

    def writeToFile(self):
        """将宏命令写入 .mcr 文件"""
        # 确保保存目录存在
        os.makedirs(folder_save, exist_ok=True)

        with open(f'{folder_save}/{self.fileName}.mcr', 'wb') as file:
            file.write(self.text.encode('ascii'))

    def run(self):
        """执行实验"""
        self.writeToFile()
        print(f"运行 {self.technique}")
        command = f'"{path_lib}" /runmacro:"{folder_save}/{self.fileName}.mcr"'
        self.process = subprocess.Popen(command)
        return self.process

    def stop(self):
        """停止当前实验"""
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                time.sleep(0.5)
                if self.process.poll() is None:
                    self.process.kill()
                print(f"{self.technique} 实验已停止")
            except Exception as e:
                print(f"停止 {self.technique} 实验时出错: {e}")


class CV(Technique):
    """循环伏安法 (CV)"""

    def __init__(self, ei, eh, el, v, si, cl, sens=1e-5, qt=2.0, pn='p', fileName=None, prefix=None, suffix=None, autosens=False):
        # 处理文件名参数
        if prefix is not None and suffix is not None:
            # 使用前缀和后缀构建文件名
            finalFileName = f"{prefix}_{suffix}"
        elif fileName is not None:
            # 使用提供的文件名
            finalFileName = fileName
        else:
            # 默认文件名
            finalFileName = 'CV'

        header = f"CHI760E CV"
        text = f'c\x02\0\0\nfolder: {folder_save}\nfileoverride\nheader: {header}\n\n'
        text += f'tech=cv\nei={ei}\neh={eh}\nel={el}\npn={pn}\ncl={cl}\n'
        text += f'si={si}\nqt={qt}\nv={v}\n'
        
        # 处理灵敏度设置
        if autosens:
            text += 'autosens\n'
        else:
            text += f'sens={sens}\n'
            
        text += 'run\n'
        text += f'save:{finalFileName}\ntsave:{finalFileName}\nforcequit: yesiamsure\n'
        Technique.__init__(self, text, finalFileName, 'CV')


class LSV(Technique):
    """线性扫描伏安法 (LSV)"""

    def __init__(self, ei, ef, v, si, sens, qt=2, fileName=None, prefix=None, suffix=None):
        # 处理文件名参数
        if prefix is not None and suffix is not None:
            finalFileName = f"{prefix}_{suffix}"
        elif fileName is not None:
            finalFileName = fileName
        else:
            finalFileName = 'LSV'

        header = f"CHI760E LSV"
        text = f'C\x02\0\0\nfolder: {folder_save}\nfileoverride\nheader: {header}\n\n'
        text += f'tech=lsv\nei={ei}\nef={ef}\nv={v}\nsi={si}\nqt={qt}\nsens={sens}\n'
        text += 'run\n'
        text += f'save:{finalFileName}\ntsave:{finalFileName}\nforcequit: yesiamsure\n'
        Technique.__init__(self, text, finalFileName, 'LSV')


class CA(Technique):
    """计时安培法 (CA)"""

    def __init__(self, ei, eh, el, cl, pw, si, sens=1e-5, qt=2.0, pn='p', fileName=None, prefix=None, suffix=None, autosens=False):
        # 处理文件名参数
        if prefix is not None and suffix is not None:
            finalFileName = f"{prefix}_{suffix}"
        elif fileName is not None:
            finalFileName = fileName
        else:
            finalFileName = 'CA'

        header = f"CHI760E CA"
        text = f'C\x02\0\0\nfolder: {folder_save}\nfileoverride\nheader: {header}\n\n'
        text += f'tech=ca\nei={ei}\neh={eh}\nel={el}\npn={pn}\ncl={cl}\npw={pw}\nsi={si}\nqt={qt}\n'
        
        # 处理灵敏度设置
        if autosens:
            text += 'autosens\n'
        else:
            text += f'sens={sens}\n'
            
        text += 'run\n'
        text += f'save:{finalFileName}\ntsave:{finalFileName}\nforcequit: yesiamsure\n'
        Technique.__init__(self, text, finalFileName, 'CA')


class IT(Technique):
    """i-t 曲线 (Amperometric i-t Curve)"""

    def __init__(self, ei, si, st, sens, qt=2, fileName=None, prefix=None, suffix=None):
        # 处理文件名参数
        if prefix is not None and suffix is not None:
            finalFileName = f"{prefix}_{suffix}"
        elif fileName is not None:
            finalFileName = fileName
        else:
            finalFileName = 'IT'

        header = f"CHI760E i-t"
        text = f'C\x02\0\0\nfolder: {folder_save}\nfileoverride\nheader: {header}\n\n'
        text += f'tech=i-t\nei={ei}\nsi={si}\nst={st}\nqt={qt}\nsens={sens}\n'
        text += 'run\n'
        text += f'save:{finalFileName}\ntsave:{finalFileName}\nforcequit: yesiamsure\n'
        Technique.__init__(self, text, finalFileName, 'i-t')


class OCP(Technique):
    """开路电位 (OCP)"""

    def __init__(self, st, si, eh=10.0, el=-10.0, fileName='OCP'):
        # fileName 参数现在有默认值 'OCP'，并且 prefix/suffix 已移除
        # finalFileName 直接使用传入的 fileName (其默认值为 'OCP')
        finalFileName = fileName

        header = f"CHI760E OCP"
        text = f'C\x02\0\0\nfolder: {folder_save}\nfileoverride\nheader: {header}\n\n'
        text += f'tech=ocpt\nst={st}\neh={eh}\nel={el}\nsi={si}\n'  # si is included as per existing and TASKCHI.md table
        text += 'run\n'
        text += f'save:{finalFileName}\ntsave:{finalFileName}\nforcequit: yesiamsure\n'
        Technique.__init__(self, text, finalFileName, 'OCP')


class DPV(Technique):
    """差分脉冲伏安法 (DPV)"""

    def __init__(self, ei, ef, incre, amp, pw, sw, prod, sens, qt=2.0, fileName=None, autosens=False):
        # 处理文件名参数
        if fileName is not None:
            finalFileName = fileName
        else:
            finalFileName = 'DPV'

        header = f"CHI760E DPV"
        text = f'C\x02\0\0\nfolder: {folder_save}\nfileoverride\nheader: {header}\n\n'
        text += f'tech=dpv\nei={ei}\nef={ef}\nincre={incre}\namp={amp}\n'
        text += f'pw={pw}\nsw={sw}\nprod={prod}\nqt={qt}\n'
        
        # 处理灵敏度设置
        if autosens:
            text += 'autosens\n'
        else:
            text += f'sens={sens}\n'
            
        text += 'run\n'
        text += f'save:{finalFileName}\ntsave:{finalFileName}\nforcequit: yesiamsure\n'
        Technique.__init__(self, text, finalFileName, 'DPV')


class SCV(Technique):
    """阶梯伏安法 (SCV)"""

    def __init__(self, ei, ef, incre, sw, prod, sens, qt=2.0, fileName=None, autosens=False):
        # 处理文件名参数
        if fileName is not None:
            finalFileName = fileName
        else:
            finalFileName = 'SCV'

        header = f"CHI760E SCV"
        text = f'C\x02\0\0\nfolder: {folder_save}\nfileoverride\nheader: {header}\n\n'
        text += f'tech=scv\nei={ei}\nef={ef}\nincre={incre}\nsw={sw}\nprod={prod}\nqt={qt}\n'
        
        # 处理灵敏度设置
        if autosens:
            text += 'autosens\n'
        else:
            text += f'sens={sens}\n'
            
        text += 'run\n'
        text += f'save:{finalFileName}\ntsave:{finalFileName}\nforcequit: yesiamsure\n'
        Technique.__init__(self, text, finalFileName, 'SCV')


class CP(Technique):
    """计时电位法 (CP)"""

    def __init__(self, ic, ia, tc, ta, eh=10.0, el=-10.0, pn='p', si=0.1, cl=1, priority='time', fileName=None):
        # 处理文件名参数
        if fileName is not None:
            finalFileName = fileName
        else:
            finalFileName = 'CP'

        header = f"CHI760E CP"
        text = f'C\x02\0\0\nfolder: {folder_save}\nfileoverride\nheader: {header}\n\n'
        text += f'tech=cp\nic={ic}\nia={ia}\ntc={tc}\nta={ta}\neh={eh}\nel={el}\npn={pn}\nsi={si}\ncl={cl}\n'
        
        # 根据priority参数添加相应命令
        if priority.lower() == 'time':
            text += 'priot\n'
        else:  # priority == 'potential'
            text += 'prioe\n'
            
        text += 'run\n'
        text += f'save:{finalFileName}\ntsave:{finalFileName}\nforcequit: yesiamsure\n'
        Technique.__init__(self, text, finalFileName, 'CP')


class EIS(Technique):
    """电化学阻抗谱 (EIS)"""

    def __init__(self, ei, fl, fh, amp, sens, qt=2, fileName=None, prefix=None, suffix=None):
        # 处理文件名参数
        if prefix is not None and suffix is not None:
            finalFileName = f"{prefix}_{suffix}"
        elif fileName is not None:
            finalFileName = fileName
        else:
            finalFileName = 'EIS'

        header = f"CHI760E EIS"
        text = f'C\x02\0\0\nfolder: {folder_save}\nfileoverride\nheader: {header}\n\n'
        text += f'tech=imp\nei={ei}\nfl={fl}\nfh={fh}\namp={amp}\nsens={sens}\nqt={qt}\n'
        text += 'run\n'
        text += f'save:{finalFileName}\ntsave:{finalFileName}\nforcequit: yesiamsure\n'
        Technique.__init__(self, text, finalFileName, 'EIS')


class ACV(Technique):
    """交流伏安法 (ACV)"""

    def __init__(self, ei, ef, incre, amp, freq, quiet=2.0, sens=1e-5, fileName=None):
        # 处理文件名参数
        if fileName is not None:
            finalFileName = fileName
        else:
            finalFileName = 'ACV'

        header = f"CHI760E ACV"
        text = f'C\x02\0\0\nfolder: {folder_save}\nfileoverride\nheader: {header}\n\n'
        text += f'tech=acv\nei={ei}\nef={ef}\nincre={incre}\namp={amp}\nfreq={freq}\nqt={quiet}\nsens={sens}\n'
        text += 'run\n'
        text += f'save:{finalFileName}\ntsave:{finalFileName}\nforcequit: yesiamsure\n'
        Technique.__init__(self, text, finalFileName, 'ACV')


# 全局函数
def stop_all():
    """停止所有正在运行的 CHI760E 实验"""
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            if 'chi760e' in proc.info['name'].lower():
                print(f"正在终止 CHI760E 进程 (PID: {proc.info['pid']})")
                psutil.Process(proc.info['pid']).terminate()

        # 确保所有进程都被终止
        time.sleep(1)
        for proc in psutil.process_iter(['pid', 'name']):
            if 'chi760e' in proc.info['name'].lower():
                print(f"强制终止 CHI760E 进程 (PID: {proc.info['pid']})")
                psutil.Process(proc.info['pid']).kill()

        print("已停止所有 CHI760E 实验")
    except Exception as e:
        print(f"停止实验时出错: {e}")


# 运行多个实验的帮助函数
def run_sequence(techniques):
    """按顺序运行多个电化学技术实验

    Args:
        techniques: 包含电化学技术实例的列表
    """
    for i, technique in enumerate(techniques):
        print(f"正在运行实验 {i + 1}/{len(techniques)}: {technique.technique}")
        process = technique.run()
        process.wait()  # 等待当前实验完成
        print(f"实验 {i + 1}/{len(techniques)} 已完成")