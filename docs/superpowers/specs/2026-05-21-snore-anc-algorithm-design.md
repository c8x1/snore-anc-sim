# Snore ANC Algorithm Development — Design Spec

**Date:** 2026-05-21
**Status:** Approved
**Based on:** `snore_anc_pillow_report.html` (技术研究报告)

## 1. Overview

基于已完成的技术研究，开发一套完整的鼾声 ANC（主动噪声消除）算法，包含：
- PC 仿真验证（Python）
- 声学路径建模
- 多通道 ANC 算法（标准 FxLMS → 子带 FxLMS → 2×2 多通道）
- 临时硬件测试方案（STM32F4 + MEMS 麦 + 微型扬声器）

**目标场景：** 妻子侧枕头消除丈夫鼾声，双安静区覆盖双耳。

## 2. Architecture: Layered Modular Framework

三层单向依赖架构：

```
config → core/ (DSP primitives) → simulation/ (acoustic modeling) → evaluation/ (metrics + viz)
```

### 2.1 Project Structure

```
snore-anc-sim/
├── pyproject.toml
├── src/
│   └── snore_anc/
│       ├── __init__.py
│       ├── core/
│       │   ├── fxLMS.py              # 标准 FxLMS 自适应滤波器
│       │   ├── subband.py            # 延迟less子带分解 (M=64, FFT-based)
│       │   ├── secondary_path.py     # 次级通路在线/离线估计
│       │   └── multichannel.py       # 多通道 ANC 矩阵扩展
│       ├── simulation/
│       │   ├── acoustic_path.py      # 主/次级通路 FIR 模型
│       │   ├── snoring_source.py     # 合成鼾声 + 公开数据集加载
│       │   └── pillow_model.py       # 枕头几何、安静区、双枕间距场景
│       ├── evaluation/
│       │   ├── metrics.py            # NR(dB), 收敛速度, PSD 对比
│       │   └── visualization.py      # 频谱图, 时域波形, 收敛曲线
│       └── config.py                 # 统一参数 (fs, 滤波器阶数, 步长)
├── tests/
│   ├── test_fxlms.py
│   ├── test_subband.py
│   ├── test_secondary_path.py
│   └── test_multichannel.py
├── notebooks/                        # 可选 Jupyter 实验
│   ├── 01_basic_fxlms.ipynb
│   ├── 02_subband_comparison.ipynb
│   └── 03_multichannel_dualzone.ipynb
└── docs/
    └── hardware_setup.md             # 临时硬件测试方案
```

### 2.2 Dependencies

- `numpy` — 数组运算
- `scipy.signal` — FIR 滤波器设计、FFT
- `matplotlib` — 可视化
- `soundfile` — 音频文件读写
- `pytest` — 单元测试

## 3. Core DSP Layer

### 3.1 FxLMS (`core/fxLMS.py`)

**算法：** Filtered-x Least Mean Squares

```
信号流：
x(n) → [W(z)] → y(n) → [S(z)] → s(n)*y(n)
                                  ↓
d(n) + s(n)*y(n) = e(n) ← 误差麦克风

权重更新：
W(n+1) = W(n) + μ · e(n) · x'(n)
其中 x'(n) = Ŝ(z) * x(n)  (Filtered-x)
```

**接口：**

```python
class FxLMS:
    def __init__(self, filter_length=256, step_size=0.01, secondary_path_est=None):
        """
        filter_length: 自适应滤波器长度 L (默认 256)
        step_size: 步长 μ (默认 0.01)
        secondary_path_est: 次级通路估计 Ŝ(z) 的 FIR 系数
        """
    def process_sample(self, x_n, d_n, e_n):
        """处理单个采样点，返回 anti-noise y(n)"""
    def process_block(self, x_block, d_block, primary_path, secondary_path):
        """处理一个 block（逐采样调用 process_sample）"""
```

**默认参数（基于报告鼾声特性）：**

| 参数 | 默认值 | 依据 |
|------|-------|------|
| 采样率 fs | 16 kHz | 鼾声能量 < 2kHz，Nyquist 充裕 |
| 滤波器长度 L | 256 | ~23ms @ 16kHz，覆盖鼾声周期性 |
| 步长 μ | 0.01 | 初始保守值，后续参数扫描优化 |

### 3.2 Delayless Subband FxLMS (`core/subband.py`)

**架构：**

```
x(n) → [M点FFT分析滤波器组] → M个子带 x_k(n)
              │
              ├── 子带 0: FxLMS_0 (独立步长 μ_0)
              ├── 子带 1: FxLMS_1 (独立步长 μ_1)
              ├── ...
              └── 子带 M-1: FxLMS_M-1
              │
         [延迟less合成] → y(n)
```

- M = 64 子带（250Hz 带宽/子带 @ 16kHz，覆盖鼾声主频段）
- FFT-based 分析/合成（Morgan & Thi prototype filter）
- 延迟less 结构：子带输出直接到时域，无合成排队延迟
- 每个子带独立步长 μ_k，低频子带可设更大步长（鼾声能量集中低频）

**接口：**

```python
class DelaylessSubbandFxLMS:
    def __init__(self, n_subbands=64, filter_length_per_band=32,
                 step_sizes=None, secondary_path_est=None):
    def process_sample(self, x_n, d_n, e_n):
    def process_block(self, x_block, d_block, primary_path, secondary_path):
```

### 3.3 Secondary Path Estimation (`core/secondary_path.py`)

两种模式：

```python
class SecondaryPathEstimator:
    def offline_estimate(self, probe_signal, measured_response, order=128):
        """离线模式：白噪声 probe → LMS 系统辨识 → Ŝ(z)"""

    def online_update(self, x_n, y_n, e_n):
        """在线模式：运行时用 probe + LMS 持续更新 Ŝ(z)"""
```

- 离线：上电时自动运行（嵌入式场景），或仿真中直接给定
- 在线：枕头位置变化时自适应更新，需附加 probe 噪声

### 3.4 Multi-Channel ANC (`core/multichannel.py`)

1-ref × 2-spk × 2-err 配置（1 参考麦克风 + 2 扬声器 + 2 误差麦克风）：

```
            ┌── W_0 → SPK_L → S_0L → Mic_eL
Ref Mic ──┤               ╲→ S_1L → Mic_eR
            └── W_1 → SPK_R → S_0R → Mic_eL
                            ╲→ S_1R → Mic_eR
```

- 滤波器向量 W = [W_0, W_1]（2 个自适应滤波器，每个扬声器 1 个）
- 次级通路矩阵 S = [[S_eL,SpkL, S_eL,SpkR], [S_eR,SpkL, S_eR,SpkR]]（2×2，4 条耦合路径）
- 更新规则：ΔW_j = μ · Σ_i(e_i · x'_ij)（对所有误差麦克风 i 求和，x'_ij 为参考信号经 Ŝ_ij 滤波后）

```python
class MultiChannelANC:
    def __init__(self, n_references=1, n_speakers=2, n_errors=2,
                 filter_length=256, step_size=0.01):
    def process_sample(self, x_ref, error_signals):
    def process_block(self, x_block, d_block, primary_path_matrix, secondary_path_matrix):
```

## 4. Simulation Layer

### 4.1 Acoustic Path Models (`simulation/acoustic_path.py`)

**Primary Path P(z)** — 打鼾者 → 妻子耳朵

- 建模：延迟线 + 房间脉冲响应 (RIR) 衰减
- 参数：距离 0.5-1.5m，延迟 ≈ 1.5-4.4ms（24-70 samples @ 16kHz）
- RIR 生成：image method（Allen & Berkley 1979）或简化 direct + 3-5 早期反射
- 仿真中 P(z) 输出 = ANC 要抵消的目标信号 d(n)

**Secondary Path S(z)** — 扬声器 → 误差麦克风

- 枕头内距离 5-20cm，延迟 0.15-0.6ms（2-10 samples @ 16kHz）
- 建模：实测 FIR 或简化指数衰减 + 谐振峰

**Feedback Coupling** — 扬声器 → 参考麦克风泄漏

- 建模为 S(z) 的衰减版（-10 到 -20dB）

### 4.2 Snoring Source (`simulation/snoring_source.py`)

**合成鼾声生成器：**

```python
class SnoringSynthesizer:
    def generate(self, duration_s=30, fs=16000, pattern='regular'):
        """
        pattern: 'regular' | 'irregular' | 'apnea'
        组成：
        - 基频 100-200Hz，微弱频率调制 (±5Hz)
        - 谐波 f0, 2*f0, 3*f0... 幅值按 1/n 衰减
        - 带通 50-500Hz 色噪声
        - 周期性 ON/OFF 包络 (吸气 ~0.4s, 呼气 ~0.3s)
        """
```

**公开数据集加载：**

- Munich Passau Snoring Sound Corpus (MPSSC): ~400 条标注鼾声片段
- 统一接口：`load_dataset(name, segment)` → `(signal, fs, annotation)`

### 4.3 Pillow Scenario (`simulation/pillow_model.py`)

```python
class PillowScenario:
    def __init__(self, config):
        self.distance = config.husband_wife_distance   # 双枕间距 0.5-1.5m
        self.pillow_geometry = config.pillow_geometry   # 麦/扬声器位置
        self.room_dims = config.room_dims               # 卧室尺寸

    def build_paths(self):
        """根据几何参数生成 P(z) 和 S(z)"""

    def run_simulation(self, source_signal, anc_algorithm):
        """完整仿真：source → P(z) → + ANC → error → 输出残差"""
```

## 5. Evaluation & Verification

### 5.1 Verification Levels

| Level | 内容 | 通过条件 |
|-------|------|---------|
| L0: 单元测试 | 每个 DSP 原语数学正确性 | 所有 test pass |
| L1: 合成信号 | 合成正弦波 + 合成鼾声抵消 | NR > 20dB (正弦), > 15dB (合成鼾声) |
| L2: 真实数据 | MPSSC + 仿真路径 | NR > 10dB (真实鼾声) |
| L3: 参数扫描 | 网格搜索 μ, L, M | 找到最优参数组合 |

### 5.2 Unit Tests

| 模块 | 测试用例 | 验证内容 |
|------|---------|---------|
| fxLMS | 单频正弦波抵消 | 10s 内 NR > 20dB |
| fxLMS | 已知 FIR + 白噪声 | 系数收敛到 Wiener 解 |
| subband | 分析-合成重建误差 | < -60dB |
| subband | 子带 vs 全带收敛速度 | 子带 ≥ 2x 更快 |
| secondary_path | 离线估计已知 FIR | 误差 < -30dB |
| multichannel | 2×2 独立源抵消 | 双安静区 NR > 15dB |

### 5.3 Metrics (`evaluation/metrics.py`)

```python
class ANCEvaluator:
    def noise_reduction_db(self, d_signal, e_signal):
        """NR = 20*log10(rms(d) / rms(e))"""

    def convergence_time(self, e_signal, threshold_db=-10, fs=16000):
        """从起始到 NR 稳定达到 threshold_db 的时间"""

    def power_spectrum_comparison(self, d_signal, e_signal, fs=16000):
        """抵消前/后功率谱密度对比"""
```

### 5.4 Parameter Sweep

| 参数 | 范围 | 评估指标 |
|------|------|---------|
| 步长 μ | 0.001 - 0.1 (log) | NR vs 收敛时间 |
| 滤波器长度 L | 64, 128, 256, 512 | NR vs 复杂度 |
| 子带数 M | 32, 64, 128 | 收敛速度 vs 复杂度 |
| 双枕间距 | 0.3 - 2.0m | 因果性约束下 NR 退化 |

### 5.5 Visualization (`evaluation/visualization.py`)

每次仿真自动生成：
1. 时域波形：d(n) vs e(n)，标注 NR
2. 功率谱对比：抵消前/后 PSD
3. 收敛曲线：NR(dB) vs 时间
4. 安静区空间 map：NR 等高线

## 6. Hardware Prototype Setup

### 6.1 BOM

| 角色 | 型号 | 理由 | 价格 |
|------|------|------|------|
| DSP | STM32F407 Discovery | Cortex-M4F + FPU, I²S, ¥80-120 | ¥100 |
| 参考麦 | SPH0645LM4H | 65dB SNR, I²S 直连 | ¥20 |
| 误差麦 ×2 | SPH0645LM4H | 双安静区 | ¥40 |
| 扬声器 ×2 | 25mm 全频动圈 | 80Hz-15kHz | ¥16 |
| 电源 | USB 5V 或 3.7V 锂电 + LDO | 便携 | ¥10 |
| **总计** | | | **¥150-300** |

### 6.2 Physical Setup

```
打鼾者(手机播放) → 0.8-1.2m → 妻子枕头
                              ├── Mic_ref (参考麦，枕面朝打鼾者)
                              ├── SPK_L, SPK_R (左/右扬声器，枕内)
                              └── Mic_eL, Mic_eR (左/右误差麦，耳位)
                                    │ I²S
                              STM32F4 Discovery
                                    │ UART
                              PC (实时 NR 监控)
```

### 6.3 Test Phases

**Phase A: 单通道验证** (1 SPK + 1 Error Mic)
- 手机播放预录/合成鼾声
- STM32 运行标准 FxLMS
- 串口输出实时 NR
- 目标：150Hz 单频 > 20dB，真实鼾声 > 10dB

**Phase B: 子带 FxLMS 验证**
- 同硬件，切换子带固件
- 对比标准 vs 子带收敛速度 + 稳态 NR
- 目标：收敛速度 ≥ 2x 提升

**Phase C: 多通道验证** (2 SPK + 2 Error Mic)
- 接入第 2 组扬声器 + 误差麦
- 枕头两侧耳位各一个误差麦
- 目标：双安静区各自 NR > 10dB

### 6.4 Embedded Firmware Structure

```
STM32F4 firmware/
├── audio_passthrough/   # I²S DMA 双缓冲 (16kHz)
├── fxlms_core.c         # C 移植（从 Python 验证版）
├── secondary_path.c     # 次级通路离线标定（上电自动）
├── uart_logger.c        # 串口 NR 值输出
└── main.c               # DMA 半满/全满中断 → ANC 处理
```

Python 仿真验证过的参数（μ, L, M）直接用于嵌入式固件。

## 7. Development Order

1. `config.py` — 参数定义
2. `core/fxLMS.py` + `tests/test_fxlms.py` — FxLMS 核心验证
3. `simulation/acoustic_path.py` — 声学路径建模
4. `simulation/snoring_source.py` — 鼾声源
5. `simulation/pillow_model.py` — 场景集成
6. `evaluation/metrics.py` + `evaluation/visualization.py` — 评估框架
7. `core/secondary_path.py` + tests — 次级通路估计
8. `core/subband.py` + `tests/test_subband.py` — 子带 FxLMS
9. `core/multichannel.py` + `tests/test_multichannel.py` — 多通道
10. 硬件原型搭建 + 固件移植
