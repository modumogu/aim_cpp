# 交接文档：Windows 侧进度与 Linux 侧待办

> 写给在 Linux 上接手的 DSH / 队友。
> 最后更新：2026-11-23
> 仓库：https://github.com/modumogu/aim_cpp

---

## 0. 一句话概括

**算法核心已经写完并在 Windows 上用合成数据验证通过（1968 行，6 套自查全过）。
缺的是"接进 ROS 2"这一层：一个节点、一个射击状态机、以及一个必须改的仿真 SDF。**

**不要重新写算法。** 先读第 1 节确认哪些已完成，再按第 3 节的清单做事。

---

## 1. 已完成的部分（请先确认，避免重复劳动）

### 1.1 算法链路（`aim.cpp`，1968 行）

```
相机/视频 → 颜色掩膜 → 轮廓 → 质心 → 槽位归组 → 相位解缠
        → θ(t) 拟合 → 预测 → solvePnP → 命中点迭代 → yaw/pitch 指令
```

| 模块 | 函数 | 状态 |
|---|---|---|
| 颜色掩膜（含 ROI、亮度补偿、饱和度自适应） | `buildMask` | ✅ |
| 灯条检测（面积/长宽比/填充度/色相/半径五级过滤） | `detectLeds` | ✅ |
| 槽位归组（角度聚类 + 3 灯兜底） | `assignSlots` | ✅ |
| 相位解缠（×5 消槽位、跳变拒绝、模型门限） | `estimatePhase` / `PhaseTracker` | ✅ |
| θ(t) 拟合（网格搜 ω + 4 参数线性最小二乘） | `fitMotion` | ✅ |
| 目标预测（增量式） | `predictTarget` / `thetaIncrement` | ✅ |
| 位姿解算（IPPE + 反投影校验 + 位姿沿用） | `solvePlatePose` | ✅ |
| 命中点迭代（弹道 ⇄ 飞行时间自洽） | `solveHit` | ✅ |
| 参数外置（三套 config，46~49 个键） | `loadParams` / `printParams` | ✅ |

### 1.2 验证结果（合成数据，全部可复现）

| 指标 | 实测 |
|---|---|
| 检出率（正常光照，四种运动模式 × 红蓝两色） | 100% |
| PnP 距离误差 | 1.3 mm（理论 √(1950²+350²)=1981mm） |
| PnP 反投影误差 | 0.10 px |
| 拟合残差 | 0.01~0.02° |
| 预测角误差 | 0.15°（预热后 0.03°） |
| 单帧耗时（纯相机模式） | **5.79 ms = 173 fps** |
| 单帧耗时（调试模式，含合成图） | 10.77 ms = 93 fps |

### 1.3 压力测试（23 项，17 通过）

用合成视频逐项加压，详见第 7 节。**已修复 4 个导致全灭的脆弱点**：
曝光自适应、饱和度自适应、冷光色相漂移、遮挡 3 灯兜底。

---

## 2. ⚠️ 第一件事：改仿真 SDF，把相机固定在 base_link

### 为什么必须改

现在官方仿真把相机挂在 `pitch_link` 上（跟着炮管转）。**这会让我们的相位估计失效**：

```
相机跟着 yaw 转 → 整个画面平移 → 转盘圆心在图像里跑
→ 我们算的"灯条相对轮心的方位角"全错
```

量化：云台偏 10° 时画面平移约 `10° × fx`。而转盘半径才 305px，
**圆心跑 297px、半径 305px —— 角度全是错的，θ 会被云台自己的运动污染。**

浙大 RM2026 开源里也明确写了这一点（原文）：

> 如果你的机器人云台/底盘会转动，需要把相机坐标系下的位姿变换到一个相对静止的参考系，
> 否则跟踪器无法正确区分"目标运动"和"自身云台运动"。

**我们的真机方案就是相机固定在基座**，所以仿真必须一致，否则题4 的调参对真机没有意义。

### 怎么改

文件：`assets/models/launcher/model.sdf`

把 `<sensor name="camera" type="camera">` 整段从 `pitch_link` 里剪出来，粘到 `base_link`：

```xml
<!-- base_link 在 world 里位于 (0, -2.07, 0.31)，+Y 朝机关，+Z 朝上 -->
<!-- 下面这个 pose：离基座 0.30m（世界高度 0.61m）、抬头 9.5° -->
<pose>0 0.02 0.30 0 -0.166 1.570796</pose>
```

- `z = 0.30` → 世界高度 0.61m
- `pitch = -0.166` → **抬头 9.5°**（SDF 里 pitch 取负值 = 抬头）
- `yaw = 1.570796` → 朝 +Y（对着机关）

抬头角的算法：

```
抬头角 = atan((转盘中心高 - 相机高) / 水平距离)
       = atan((0.928 - 0.61) / 1.954) = 9.3°
```

**如果实际装多高就按这个式子重算**，不要照抄 9.5。

启动时用 launch 参数覆盖桥接话题（`ros2_ws` 一行代码都不用改）：

```bash
ros2 launch robomini_sim sim.launch.py \
  camera_image_topic:=/world/robomini/model/launcher_blue/link/base_link/sensor/camera/image \
  camera_info_topic:=/world/robomini/model/launcher_blue/link/base_link/sensor/camera/camera_info
```

### 改完要同步 config

```ini
# config_robomini.txt
camera_pitch_deg = 9.3      # 必须和 SDF 里的抬头角一致
```

**填 0 的后果**：最上面那块靶会出画面（0.61m 高时差 4.6°）。
**填过大的后果**：竖直方向被过度压缩，靶的角度有系统偏差。

---

## 3. 待办清单（按依赖顺序）

### 3.1 编写 ROS 2 节点（最大的一块）

新建一个包，把 `aim.cpp` 的算法核心包起来。**建议不要改 `aim.cpp` 的算法函数**，
只新建一个 `ros_node.cpp` 做订阅/发布，然后把它链接进去。

#### 订阅

| 话题 | 类型 | 用途 |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | 图像 |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | **内参（fx/fy/cx/cy + 畸变），不要写死** |
| `/gimbal_to_vision` | `robomini_interfaces/GimbalToVision` | 云台反馈 |

`GimbalToVision` 的字段（来自官方 README）：

| 字段 | 说明 |
|---|---|
| `mode` | 0 空闲 / 1 自瞄 |
| `q` | IMU 四元数（w,x,y,z） |
| `yaw`, `yaw_vel` | 实测偏航角（rad）、角速度 |
| `pitch`, `pitch_vel` | 实测俯仰角、角速度 |
| **`bullet_speed`** | **弹速（m/s），直接用它，不要写死** |
| **`bullet_count`** | 累计发弹数（uint16，会回绕）—— 做弹量保护 |

#### 发布

| 话题 | 类型 | 说明 |
|---|---|---|
| `/vision_to_gimbal` | `robomini_interfaces/VisionToGimbal` | 云台指令 + 开火 |

字段：

| 字段 | 说明 |
|---|---|
| `mode` | **0 不控制 / 1 控制云台但不开火 / 2 控制云台且开火** |
| `yaw`, `yaw_vel`, `yaw_acc` | 目标偏航角（rad）、角速度、角加速度 |
| `pitch`, `pitch_vel`, `pitch_acc` | 目标俯仰角、角速度、角加速度 |

#### 必须做的三件事

**① 内参和弹速从话题读，不要写死**

```cpp
P.fx = info->k[0];  P.cx = info->k[2];
P.fy = info->k[4];  P.cy = info->k[5];
P.ball_speed = gimbal->bullet_speed;
```

这样 `config_robomini.txt` 里那 9 行内参和 `ball_speed` 就不用维护了。

**② 速度前馈**

消息里有 `yaw_vel` / `pitch_vel` / `yaw_acc` / `pitch_acc`，**不给的话云台会滞后**。
我们的模型本来就能解析算出角速度：

```cpp
// θ(t) = c0 + c1·t + c2·cos(ωt) + c3·sin(ωt)
// spd(t) = c1 + ω·(−c2·sin(ωt) + c3·cos(ωt))
double dtheta = model.c[1] + model.omega *
                (model.c[3] * std::cos(model.omega * t) - model.c[2] * std::sin(model.omega * t));
msg.yaw_vel = dtheta;   // rad/s
```

**③ 射击状态机**（不要用 "READY 就开火"）

`mode` 字段填 1 还是 2 是我们的决策。**绝对不能每帧都填 2**，否则会连发、超额扣分
（规则 7.3：每超额发射 1 颗扣 20 分）。

建议状态机：

```
IDLE ──(hit.ok && 待激活靶唯一 && 弹量>0)──> ALIGNING
ALIGNING ──(|指令角 − 实测角| < 0.5° 连续 5 帧)──> FIRE
FIRE ──(发 1 颗，进入 COOLDOWN)──> COOLDOWN
COOLDOWN ──(靶从 4 灯变 2 灯 = 命中，或超时 2s)──> IDLE
```

弹量保护：

```cpp
if (gimbal->bullet_count >= quota) mode = 1;   // 锁死，绝不多发
```

### 3.2 联调（Linux 上第一次真正跑起来）

**验收顺序**（每一步确认了再做下一步）：

1. **改完 SDF 后**：`ros2 topic echo /camera/image_raw --once` 有画面
2. **相机画面**：`ros2 run rqt_image_view rqt_image_view` 能看到转盘，**轮心在画面正中**
3. **跑算法**：节点启动后看 `[参数]` 那几行，确认 `wheel=(-1,-1)`（用画面中心）
4. **看 maskH**：红方应接近 0 或 179，蓝方应接近 103~112
   ```
   [参数] ... gate拒 0  maskH 3
   ```
   **如果 maskH = -1 且灯条 0 → 颜色/阈值问题，先解决这个，别往下走**
5. **看检出**：`灯条 8~12 条，靶 3~5` 才正常
6. **看 READY**：出现 `pred xxx READY` 说明模型收敛了
7. **看命中**：先用 `mode=2` 手动连发几颗，确认能打中

### 3.3 四种模式都要验（仿真可以程序化切换）

```bash
ros2 service call /rune/set_mode robomini_interfaces/srv/SetRuneMode "{mode: fixed}"
ros2 service call /rune/set_mode robomini_interfaces/srv/SetRuneMode "{mode: random_fixed}"
ros2 service call /rune/set_mode robomini_interfaces/srv/SetRuneMode "{mode: small}"
ros2 service call /rune/set_mode robomini_interfaces/srv/SetRuneMode "{mode: big}"
```

**四种都要能打**，不要只验小符。我们的算法四种都支持：
- 固定靶：θ 恒定，预测=当前靶角
- 小机关：定角速度 1.047 rad/s，靠模型退化自动支持
- 大机关：正弦摆动
- 随机固定靶：靠 `tracker.reset()` 从大跳变恢复（约 1.3 秒）

---

## 4. 参数与配置

### 4.1 三套配置的分工

| 文件 | 用在 | fx | 分辨率 | 畸变 |
|---|---|---|---|---|
| `config.txt` | **真机** | 1700.4 | 1280×1024 | 有 |
| **`config_robomini.txt`** | **仿真（用这个）** | **1108.5** | **1280×720** | **全 0** |
| `config_sim.txt` | 合成测试视频 | 1906.9 | 1280×960 | 全 0 |

### 4.2 仿真相机的内参是白给的

```
SDF: horizontal_fov = 1.047198 rad (60°), 1280x720
fx = fy = (1280/2) / tan(30°) = 640 / 0.57735 = 1108.5
cx = 640, cy = 360
畸变全 0（Gazebo 理想针孔）
```

### 4.3 必须现场标定的两个数

| 参数 | 含义 | 不标定的后果 |
|---|---|---|
| **`bar_offset_mm`** | 灯条质心到靶盘中心的距离 | 距离成比例偏 → 弹道全错 |
| **`camera_pitch_deg`** | 相机抬头角 | pitch 偏 + 角度各向异性 |

`bar_offset_mm` 的标定方法：

```
1. 靶放一个已知距离 D（卷尺量，比如 2.000m）
2. 看界面上的 d= 是多少，记为 D'
3. 新的 bar_offset = 当前值 × D' / D
```

官方 SDF 给的是 **47.5mm**。**真机要用卡尺量灯条质心到盘心的距离**，不要照抄。

> 注意：早期测试视频用的是旧几何，灯条质心在 **45mm**，用 47.5 会让距离偏大 5.5%。
> 这不是 bug，是模型和视频不匹配。

### 4.4 颜色参数

```bash
--color red      # 红方
--color blue     # 蓝方
--color yellow   # 黄色（测试用）
```

**配错的症状**：掩膜全空、`灯条 0`、`maskH -1`，界面上只剩轮心十字和半径圆。

程序会打印警告：

```
[警告] 掩膜为空！画面亮部主色相 H=27，看起来是 yellow
[警告] 当前位置 --color red 与该颜色不符，试试 --color yellow
```

---

## 5. 命令速查

```bash
# 仿真（纯相机画面，比赛用，173fps）
./aim --config config_robomini.txt --raw

# 调试模式（原图 | 掩膜叠加）
./aim --config config_robomini.txt

# 不显示窗口（自动化脚本用）
./aim --config config_robomini.txt --headless

# 跑视频、只跑前 100 帧、存图
./aim --config config_sim.txt --video xxx.mp4 --frames 100 --save out.png

# 看耗时分解（退出时打印）
./aim --config config_sim.txt --video xxx.mp4 --headless
```

按键：`ESC/q` 退出，`p` 切换纯相机画面，`s` 截图。

---

## 6. 参数一览（46 个键，`config_robomini.txt`）

```
相机      exposure wb gain
转盘      wheel_x wheel_y px_per_mm wheel_rpx auto_rpx roi_scale
灯条      min_area max_area min_aspect min_extent max_hue_std
归组      group_gap_deg radius_tol
拟合      fit_window_s min_fit_samples omega_min omega_max omega_step
          max_fit_rms min_osc_amp ab_tol
预测      lead_s max_pred_age history_reset_age model_gate
内参      fx fy cx cy k1 k2 p1 p2 k3
几何      bar_offset_mm
弹道      ball_speed muzzle_offset_mm pipeline_delay pitch_bias_deg g_acc
安装      camera_pitch_deg camera_yaw_deg
自适应    adapt_sv adapt_gain_below adapt_max_gain adapt_ratio adapt_min
```

改动后**重开程序即可生效，不需要重新编译**。启动时会打印全部生效值：

```
[参数] exposure=-8.00 wb=4600 gain=0.00
[参数] wheel=(-1,-1) px_per_mm=0.5670 R_px=198.4 roi=1.60 autoR=on
[参数] area=[20,2500] aspect>=1.20 extent>=0.65 hue_sd<=6.0
[参数] K: fx=1108.50 fy=1108.50 cx=640.00 cy=360.00
[参数] bar_offset=47.50mm ball_v=12.00m/s muzzle_dy=75.0mm
[参数] pipeline=0.080s bias=0.00deg cam_pitch=9.53deg cam_yaw=0.00deg y_scale=1.0139
```

**每次启动先看这几行**，确认配置读对了。

---

## 7. 压力测试套件（在 Linux 上也能跑）

### 7.1 位置

```
py_sim/gen_dataset.py      合成数据生成器（可加压力参数）
stress_test.py             23 项压力测试，自动出表
validate_all.py            四模式 × 两色回归
audit_config.py            配置键静态检查
audit_runtime.py           每个键是否真的生效
audit_features.py          18 项功能断言
audit_hit.py               命中点迭代验证
audit_ballistic.py         弹道逐帧重算
```

都是纯 Python + OpenCV，**不依赖 ROS**。改一下 `EXE` 路径就能在 Linux 上跑。

### 7.2 压力测试的 23 项

```
基线 / 欠曝 0.3 / 过曝 2.5 / 暖光 3200K / 冷光 6500K / 褪色 0.3
模糊 8px / 模糊 15px / 眩光 3 斑 / 眩光 6 斑
遮挡 1 块 / 遮挡 2 块 / 丢帧 20% / 丢帧 40% / 噪声 20 / 综合恶劣
环境光 0.35 / 环境光 0.60 / 环境光+暖光
顶灯 3 条 / 顶灯 6 条 / 顶灯+眩光 / 光照全乱
```

**当前 17 项通过**，失败的 6 项全部是"极强环境光导致检出率跌破 90%"，
而**所有工况的预测误差都在 0.14~0.39°**（最大 1.69°）。

### 7.3 判定标准（重要）

```
检出率 >= 90%（注入丢帧时按比例放宽）
且 最长连续 READY >= 1.0 秒
```

**不要用 "READY 帧占比" 判定**：失效是聚集的，"顶灯 3 条"READY 只有 76%，
但最长连续 3.02 秒，实战完全够开火。

### 7.4 生成器可用的压力参数

```bash
--brightness 0.3~3.0    曝光
--temperature 3200~6500 色温（白平衡失配）
--saturation 0.3~1.0    褪色
--blur 0~15px           运动模糊
--glare 0~6             随机白色高光斑
--ambient 0~1           整体环境光抬升（整个场地被照亮）
--ceiling 0~6           顶部条形顶灯（放在画面上 1/3）
--occlude 0~3           靶环遮挡
--dropout 0~0.5         丢帧（整帧压暗）
--noise 0~25            灰度噪声
```

### 7.5 现场调参建议

**曝光宁可偏暗**：欠曝 0.3 检出 100%，过曝 2.5 只有 81%。

**如果现场光很乱打不到，按这个顺序调**：

```
1. roi_scale  1.6 → 1.3，把顶灯排除在 ROI 之外
2. 曝光往暗调
3. 上面两步不行，再考虑改算法（加"被遮挡时放宽填充度"）
```

前两步**改配置就行，不用重新编译**。

---

## 8. 已知的坑（踩过的，别重踩）

| 坑 | 症状 | 处理 |
|---|---|---|
| **`--color` 配错** | 掩膜全空、灯条 0、maskH -1 | 看警告，换 `--color` |
| **相机没固定在基座** | θ 一直不涨或乱跳 | 改 SDF（第 2 节） |
| **`camera_pitch_deg` 和 SDF 不一致** | pitch 系统性偏、靶角有偏差 | 两边改成同一个值 |
| **`bar_offset_mm` 没标定** | 距离成比例偏 | 按 4.3 标定 |
| **`camera_pitch_deg` 填 0** | 最上面的靶出画面 | 抬头 9~15° |
| **`--frames` 在旧版本窗口模式失效** | 以为只跑 1 帧、实际全跑完 | 已修（`--frames` 两种模式都生效） |
| **`cv::imwrite` 中文路径静默失败** | 打印"已存"但文件不存在 | 已加 `saveImage()` 兜底；Linux 无此问题 |
| 启动前 19~25 帧没有预测 | 正常现象（模型需要积累 20 个样本） | 不是 bug |

---

## 9. 性能预算

```
纯相机模式（--raw）  5.79 ms/帧 = 173 fps   ← 比赛用这个
调试模式（合成图）   10.77 ms/帧 =  93 fps
纯视频解码基线       2.20 ms/帧
```

其中掩膜 4.5ms 是最大项（HSV 转换 + inRange + 形态学）。

**真机相机 100fps = 每帧 10ms 预算，纯相机模式有余量。**
如果仿真里 fps 不够，先确认不是在做合成图（该用 `--raw` 或 `--headless`）。

---

## 10. 不要做的事

| 别做 | 原因 |
|---|---|
| 重写检测/拟合/PnP | 已经验证到 100% 检出、0.10px 反投影、0.03° 预测误差 |
| 引入 Ceres / EKF | 我们的网格搜 ω + 线性最小二乘已经够用，残差 0.01~0.02° |
| 把 `b = 2.090 − a` 当硬约束 | 深大 RM2026 开源明确说"未把规则参数关系作为硬约束"，实测也没必要 |
| 用 READY 帧占比做验收 | 见 7.3 |
| 把内参/弹速写死 | 仿真通过话题给了，直接读 |

---

## 11. 交接清单

- [ ] 改 SDF：相机从 `pitch_link` 挪到 `base_link`，抬头约 9.5°
- [ ] 启动时覆盖 `camera_image_topic` / `camera_info_topic`
- [ ] `config_robomini.txt` 里 `camera_pitch_deg` 与 SDF 一致
- [ ] 写 ROS 2 节点：订阅 image + CameraInfo + GimbalToVision
- [ ] 内参、弹速从话题读
- [ ] 发布 VisionToGimbal：yaw/pitch + 速度前馈
- [ ] 写射击状态机：冷却 + 弹量保护 + 命中反馈
- [ ] 联调：按 3.2 的七步走
- [ ] 四种模式（fixed / random_fixed / small / big）都验一遍
- [ ] 标定 `bar_offset_mm`（真机）
- [ ] 压力测试套件在 Linux 上跑一遍确认没退

---

## 12. 参考

- 官方仿真仓库：https://github.com/tianze-dev/robomini_sim
- 深大 RM2026 能量机关开源（状态估计与拟合思路）：https://github.com/SZURPVision/RP-26Rune
- 浙大 RM2026 buff 算法（PnP 双解消歧、RANSAC 拟合）：https://github.com/IC-Alan/HWauto_buff2026
- 武科大 awakening（误差状态 EKF，装甲板方向）：https://github.com/WUST-RM/awakening

---

## 附：关键数值表

### 几何（官方 SDF）

```
转盘中心        (0, -0.116, 0.928)
转盘朝向        +Y（面朝发射架）
5 个靶位        半径 350mm，间隔 72°
靶盘直径        195mm
灯条            41 × 20 × 10 mm，4 条成十字
灯条质心距盘心  47.5mm
射距            1950mm（枪口到靶面）
```

### 运动（规则 5.5.2）

```
spd(t) = a·sin(ωt) + b,  b = 2.090 − a
a ∈ [0.780, 1.045] rad/s
ω ∈ [1.884, 2.000] rad/s
小机关   ω = π/3 = 1.047 rad/s（定速）
固定靶   spd = 0
```

### 弹道（v = 12 m/s，d = 1.954 m，含空气阻力）

```
飞行时间    0.195 s
下坠        187 mm
相对直瞄多抬 4.69°
靶盘半张角  2.86°
```

### 时延预算

```
视觉处理      约 0.03~0.08 s（软件侧，其余是链路）
弹丸飞行      0.17~0.20 s
总提前量      约 0.25 s
```
