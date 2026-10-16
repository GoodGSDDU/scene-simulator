# 打包成 exe

把仿真器做成 Windows 上双击即可运行的 `.exe`。

## 一键打包

```bash
python packaging/build.py
```

产出在 `dist/`：

```
dist/
├─ 雷达仿真器/                    ← 文件夹版，推荐
│  ├─ 雷达仿真器.exe              ← 双击这个
│  ├─ radar-sim-cli.exe           ← 命令行版
│  ├─ _internal/                  ← 运行库（别动）
│  ├─ templates/                  ← Python / MATLAB / Notebook 模板
│  ├─ examples/                   ← 示例场景
│  ├─ docs/                       ← DESIGN.md
│  ├─ 使用说明.txt
│  └─ 运行自检.cmd
└─ 雷达仿真器-单文件版.exe        ← 单文件版，启动慢
```

常用参数：

| 参数 | 作用 |
| --- | --- |
| `--only dir` | 只打文件夹版（快，约 2 分钟） |
| `--only onefile` | 只打单文件版 |
| `--no-clean` | 复用上次的缓存，重打包快很多 |
| `--no-selftest` | 跳过构建后的自检（不推荐） |

**每次构建都会自动跑一遍冻结包的自检**（见下），失败就报告失败——不会把一个坏 exe 交到你手上。

## 文件夹版 vs 单文件版

| | 文件夹版 | 单文件版 |
| --- | --- | --- |
| 启动耗时 | 约 3–5 s | 约 10–20 s（每次解压到临时目录） |
| 分发 | 整个文件夹打包成 zip | 直接发那一个 exe |
| 体积 | 约 200–250 MiB | 约 100 MiB（压缩过） |
| 模板可编辑 | ✅ 就是普通文件 | ❌ 封在 exe 里 |
| 能否连 exe 一起移动 | 整个文件夹 | ✅ |

单文件版每次启动都要把 200 MB 的运行库解压到 `%TEMP%`，所以慢。**如果你只是自己用，用文件夹版**；如果要发给别人，单文件版省事。

## 为什么要有 `--selftest`

冻结之后的 exe 没有 Python 环境可以退回去排查——出了问题就是"双击没反应"。

所以程序内置了一个自检模式，会真的把整条链路跑一遍：

```
1. 依赖包        numpy / matplotlib / Qt 后端 / PySide6 / radar_sim
2. 仿真内核      雷达配置 → 场景离散化 → 射线追踪（含多径）→ IQ 合成
                 → 导出数据集 → 回读校验字节布局
3. 图形界面      离屏建窗口 → 画布放图元 → 跑完 50 帧仿真
4. 数据处理模板  用 templates/python 读回 signal.bin → 距离-多普勒 → CFAR
```

任一项失败就返回非 0 退出码，并把原因和异常堆栈写进报告。

```bash
# 手动验证一个 exe
"dist/雷达仿真器/雷达仿真器.exe" --selftest 报告.txt
```

普通用户不用记这些——双击 `运行自检.cmd` 就行。

## 自检覆盖的是什么

自检用的场景是**刻意构造**的，不是随便摆的：

```
     y
     ↑
  26 │        ○ 植被(r=2.5)
     │
  14 │        ┃
     │        ┃ 金属反射墙 x=30
   8 │  ●────━┛
     │  雷达   ◆ 标定点(20,8)
   4 │
     └────────────────────────→ x
```

雷达在 `(2,8)`，点目标在 `(20,8)`，墙在 `x=30` 贯穿 `y∈[4,14]`。

墙关于雷达的镜像点是 `(58,8)`；从镜像点到目标的连线在 `y=8` 处穿过 `x=30`，落在墙的范围内——**所以一条镜面一次反射路径必然存在**。

如果哪天射线追踪退化到找不到多径，自检会直接报 FAIL。这正是 §15.3 里第 4 号缺陷（多径静默失效）的回归防线。

## 手工构建（不用脚本）

```bash
python packaging/make_icon.py
python -m PyInstaller packaging/radar_sim.spec --noconfirm --distpath dist --workpath build
```

spec 通过环境变量切换形态，因为 PyInstaller 在给了 `.spec` 之后会忽略 `--onefile` / `--name` / `--icon`：

```bash
RADAR_SIM_ONEFILE=1 RADAR_SIM_DIST_NAME=我的名字 python -m PyInstaller packaging/radar_sim.spec ...
```

## 裁剪说明

`radar_sim.spec` 里的 `EXCLUDES` 砍掉了 `tkinter`、`pandas`、`scipy`、`IPython`、`jupyter`、`pytest` 等。

**这里要小心**：排掉一个 matplotlib 或 Qt 懒加载时会用到的包，做出来的 exe 能正常启动，然后在用户点某个按钮时才崩——最难查的那种故障。所以清单只保留了确定可达性为零的项，构建后的自检就是这道防线。

`HIDDEN` 里同时列了 `backend_qtagg` / `backend_qt` / `backend_qt5agg`，因为 matplotlib 选 Qt 后端是动态导入的。

没有开 UPX：它压缩 Qt 的 DLL 时经常压坏，省下的那点体积不值得。

## 已知限制

- **只在 Windows 上验证过**。PyInstaller 的 spec 本身跨平台，但没在 Linux/macOS 上跑过。
- **没有代码签名**。Windows SmartScreen 会拦一次，点「更多信息 → 仍要运行」即可。要免掉得买证书。
- 中文字体走系统字体（微软雅黑等，见 `src/radar_sim/style.py`），没有内嵌字体。系统缺中文字体会显示成方块。
