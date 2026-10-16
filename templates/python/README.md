# 数据处理模板（Python）

只依赖 `numpy` / `matplotlib`，可直接复制到任何项目里。

## 文件

| 文件 | 作用 |
| --- | --- |
| `radar_io.py` | 读 `signal.bin` + `meta.json`：两种布局、按帧 seek、标准距离/速度/角度轴、距离-多普勒、真值 CSV |
| `cfar.py` | 标准 CA-CFAR、FFT 测角、点云提取 |
| `process_demo.py` | 端到端示例：出 4 张图（原始 IQ / 距离像 / 距离-多普勒 / 点云）+ 检测列表 |
| `compare_gt.py` | 与真值对比：距离/速度/方位 RMS 误差、虚警率、目标可见率 |

## 30 秒上手

```bash
# 1) 生成一份数据（任选其一）
radar-sim-cli --demo --out ../../dataset/demo --frames 20

# 2) 处理并出图
cd templates/python
python process_demo.py   --data ../../dataset/demo --out figures
python compare_gt.py     --data ../../dataset/demo --frames 20 --out .
```

## 在自己的代码里调用

```python
from radar_io import RadarFile
from cfar import extract_pointcloud

with RadarFile("dataset/demo") as rf:
    print(rf.summary())                      # 参数与派生指标

    cube = rf.frame(0)                       # (Nc, Nv, Ns) complex128
    rd, r_axis, v_axis = rf.range_doppler(0) # (Nc, Ns) dB + 轴

    cloud = extract_pointcloud(rf, 0, pfa=1e-3)
    print(cloud["range_m"], cloud["velocity_mps"], cloud["azimuth_deg"])
```

## 必须遵守的约定（`docs/DESIGN.md` 第 10.6 节）

这些不是风格问题，写错了结果是错的：

| 项目 | 正确做法 | 常见错误 |
| --- | --- | --- |
| 帧定位 | `seek(f * frame_bytes)` | 每帧从头读 → O(N²) I/O |
| 距离轴 | `k·c·fs/(2·S·Ns)`，全部 `Ns` 个 bin | 用 `c/(2·S·Tr)`（分母应是**有效带宽** `S·Ns/fs`） |
| 速度轴 | `(i − Nc/2)·λ/(2·Nc·Tc)` | `linspace(-vmax, vmax, Nc)` → 步长错、偶 `Nc` 无零速 bin |
| 角度轴 | `asin((k − Nfft/2)·λ/(Nfft·d))` | 线性近似 `(k−N/2)·180/N` |
| 零频 bin | `fftshift` 后索引 `N/2` | 取 `N/2−1` → 整体偏 1 bin |
| CFAR | `α = Nref·(Pfa^(−1/Nref) − 1)` | `mean + 1.5σ` → 虚警率不可控 |
| 测角快拍 | 取**复数距离-多普勒立方体** | 只对几个 chirp 相加 → 白丢约 20 dB 增益 |
| 窗函数 | 周期窗 `hann(N)` 的 DFT-even 形式 | `np.hanning(N)` 是对称窗 |
| MATLAB | 列主序 `reshape(raw, [2, Ns, Nv, Nc, F])` | 照抄 Python 的 `[2, Nc, Nv, Ns]` |

`tests/test_templates.py` 会自动校验模板公式与仿真内核完全一致，改了任一边都会失败。

## 数据布局

`meta.json` 的 `signal.layout` 决定 `signal.bin` 的排列：

| 布局 | 排列 |
| --- | --- |
| `ti_dca1000`（默认） | 每 chirp 内「阵元在外、采样在内」，每 2 个复数打包成 4 个 int16：`[Re(v0), Re(v1), Im(v0), Im(v1), …]`，与 TI 官方 `readDCA1000.m` 一致 |
| `interleaved_iq` | 每采样直接 `[Re, Im]` |

复数恢复：`(Re + 1j·Im) / iq_scale`。`iq_scale_mode="frame0"` 时整个文件用同一比例，**不要**做逐帧 AGC，否则幅度信息就没了。

## TDM-MIMO 注意

`meta.json` 里 `radar.tdm_interleave = true` 时，同一个虚拟阵元每 `tx_elements` 个 chirp 才被照射一次。做多普勒 FFT 之前必须先解交织，否则速度会错 `tx_elements` 倍：

```python
cube = rf.frame(0)
if rf.tdm:
    cube = cube[0::rf.n_elements // rf.radar["rx_elements"], : rf.radar["rx_elements"], :]
```

默认 `mimo_mode="ideal"`，不存在这个问题。
