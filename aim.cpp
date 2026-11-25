// ============================================================================
//  ROBOMINI 能量机关自瞄 —— 节点 5：拟合 θ(t) + 预测
// ============================================================================
//
//  节点 4 得到的是"转盘转了多少"（θ），这一节点要做的是"100ms 之后它转到哪"。
//
//  模型：
//      θ(t) = c0 + c1·t + c2·cos(ωt) + c3·sin(ωt)
//             └──┬──┘   └───────┬───────┘
//             匀速部分        正弦摆动部分
//
//      · 固定靶 / 随机固定靶 / 小机关  -> c1 起作用，c2≈c3≈0
//      · 大机关                        -> c1、c2、c3 一起起作用
//
//  ω 不用估，直接网格搜索：手册 5.5.2 给了 ω ∈ [1.884, 2.000]，
//  每个候选 ω 用最小二乘解 c0..c3，取残差最小的那个。
//
//  残差超限（>3°）就判定拟合不可信 -> HOLD。宁可不开火，也不乱打。
//
//  预测（关键）：
//      target_pred = angle_now + (θ_future − θ_now)
//  用的是"增量"，不是"绝对角度"。因为靶角 = θ + k×72°，
//  换靶瞬间 k 会跳变，但只要做差，k 就被消掉了。这是整个算法里最巧的一处。
//
//  用法：同节点 4
//
//  验证标准：
//      · 合成数据集：w 落在 1.88~2.00，rms < 0.05°，状态 READY
//      · 与 truth.csv 的"100ms 后靶角"对比，误差 < 0.2°
//
// ----------------------------------------------------------------------------
//  命令行参数
//      aim.exe                                自动挑相机（默认红色灯条）
//      aim.exe --color blue                   蓝色灯条（比赛分边随机，必须能切）
//      aim.exe --color yellow                 黄色灯条
//      aim.exe --index 1                      强制某个索引的相机
//      aim.exe --expo -6                      曝光（-8 最暗，值越大越亮）
//      aim.exe --wb 4600                      色温
//      aim.exe --wheel 640 480                转盘圆心
//      aim.exe --rpx 305                      转盘半径（像素）
//      aim.exe --raw                          启动即纯相机画面
//      aim.exe --video xxx.mp4                用视频文件
//      aim.exe --video xxx.mp4 --save out.png 不弹窗，存一张图
//
//  按键：ESC/q 退出    p 切纯相机画面    s 截图
//
// ============================================================================

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX          // 否则 windows.h 的 min/max 宏会打散 std::min/std::max
#include <windows.h>
#endif

namespace {

// ---- 生产参数：比赛时就是这一套 --------------------------------------------
constexpr int kWantW = 1280;        // 分辨率（竖向要装下 895mm 的转盘）
constexpr int kWantH = 1024;
constexpr double kDefExposure = -8; // DirectShow 的 log2 刻度，-8 ≈ 3.9ms
constexpr double kDefWbTemp = 4600; // 手动色温，必须关自动白平衡

// ---- 灯条形状的物理范围 ----------------------------------------------------
// 依据官方仿真 energy_rune/model.sdf：
//   <geometry><box><size>0.0410 0.0100 0.0200</size></box></geometry>
//   可见面 41mm x 20mm，厚 10mm，质心距盘心 47.5mm
// 在 1950mm、fx=1700 下约 36 x 17 px；光晕会把它撑成 ~45 x 25 px。
//
// 实测（官方几何、带光晕）：
//   面积   1006 ~ 1071 px²
//   填充度 0.873 ~ 0.915
//   长宽比 1.73 ~ 1.80      <- 注意：不是理论上的 2.05，光晕把宽度撑大了
constexpr double kMinLedArea = 20.0;    // 远距离小目标也要能过
constexpr double kMaxLedArea = 2500.0;  // 大于这个：光斑 / 整片红色
constexpr double kMinAspect = 1.2;      // 实测 1.73，光晕更重时会掉到 1.5，留足余量

// ---- 转盘几何 --------------------------------------------------------------
constexpr int kSlotCount = 5;                     // 5 个靶位，间隔 72°
constexpr double kPxPerMm = 0.8720;               // 标定得到：fx 1700 / 1950mm
constexpr double kDefWheelRpx = 350.0 * kPxPerMm; // ≈ 305 像素
constexpr double kGroupGapDeg = 25.0;             // 角度聚类阈值，见下方推导
constexpr double kRadiusTol = 0.45;               // 半径容差，±45%
// 聚类阈值的推导（不能乱设）：
//   灯条质心距盘心 47.5mm，盘心半径 350mm
//   · 组内最大间隔：切向两条灯条相差 2*atan(47.5/350) = 15.46°
//     （中间那两条径向灯没检出时，剩下的两条就隔这么远）
//   · 组间最小间隔：72° - 15.46° = 56.54°
//   阈值必须落在 (15.5°, 56.5°) 之间。取 25°，两边都有余量。
//   设成 15° 会在"径向灯漏检"时把一块靶切成两半，导致整块靶丢失。

// ---- 拟合参数 --------------------------------------------------------------
constexpr double kFitWindowS = 1.5;    // 只用最近 1.5 秒的数据
constexpr int kMinFitSamples = 20;     // 至少 20 个采样点才拟合
constexpr double kOmegaMin = 1.884;    // 手册 5.5.2
constexpr double kOmegaMax = 2.000;
constexpr double kOmegaStep = 0.004;
constexpr double kMaxFitRms = 0.05;    // 弧度，约 3°；超过就 HOLD
constexpr double kLeadS = 0.10;        // 预测提前量：100ms

// ---- 手册给的运动学约束 ----------------------------------------------------
//   spd(t) = a·sin(ωt) + b,   b = 2.090 − a
//   积分到角度后： b = c1（一次项系数）， a = ω·√(c2²+c3²)（振幅）
//   所以理论上 a + b 恒等于 2.090。用它做一个免费的物理合理性检查。
//   实测（881 帧干净数据）：|a+b−2.090| 最大 0.0032 rad，取 0.05 有 15 倍余量。
// 实测（红蓝各 881 帧、真实 C++ 管线）：|a+b−2.090| 的 P99 = 0.104，最大 0.144。
// 这个散布不是代码错，是"过零点附近条件数爆炸"：
//   转盘在折返点速度接近 0，1.5s 窗口只覆盖一小段弧，
//   偏置 c0 和斜率 c1 高度相关，a、b 各自估不准（残差仍只有 0.02°）。
// 所以门限按噪声定，不能按"理论应该严格相等"定。
// 0.20 rad 在本数据集上误杀率 0%，同时还能挡住"拟合到完全不同的运动"。
constexpr double kAbSum = 2.090;
constexpr double kAbTol = 0.20;        // 弧度，--abtol 可现场调整
// 只有"确实在正弦摆动"时才检查。小机关(a=0,b=1.047) 和固定靶(a=b=0)
// 的 a+b 远小于 2.090，无条件检查会把它们全部误杀。
constexpr double kMinOscAmp = 0.05;    // 振幅低于这个就当是匀速，不检查约束

// ---- ROI -------------------------------------------------------------------
// 只在转盘附近找灯条。顶灯、墙面反光都在画面边缘，这一步直接排掉。
// 1.6 倍半径 = 488 像素（R=305 时），转盘直径 610，留了足够余量。
constexpr double kRoiScale = 1.6;

// ---- 丢帧处理 --------------------------------------------------------------
//   0 ~ 50ms   （1~3 帧）  正常：θ 不更新，模型继续用，预测照常输出
//   50 ~ 250ms （4~15 帧） 降级：不再输出新预测（HOLD），但保留 history
//   > 250ms                 HOLD，停止预测
//   > 1000ms                清空 history，等重新积累（转盘可能已经换了模式）
constexpr double kMaxPredAge = 0.25;      // 数据超过这个年龄就不预测了
constexpr double kHistoryResetAge = 1.0;  // 超过这个年龄清空历史

// ---- 模型门限 --------------------------------------------------------------
// theta 是累加的：某帧误检让相位偏了 10 度（不到 PhaseTracker 的 28.8 度拒绝线），
// theta 就永久偏 10 度，之后所有预测都偏 10 度 —— 在 1950mm 处是 340mm，必脱靶。
// 而且拟合发现不了：10 度的阶跃会被 c0 吸收，残差照样 0.02 度。
//
// 办法：用上一帧的模型外推本帧的 theta，对不上就不采用这帧。
// 模型只外推 1 帧（16.7ms），外推误差约 0.5*w^2*A*dt^2 = 0.03 度，
// 所以 1.5 度的门限有 50 倍余量，同时能挡住 5 度以上的误检。
constexpr double kModelGateRad = 1.5 * CV_PI / 180.0;

// PnP 反投影误差上限（像素）。超过说明 4 个点的对应关系错了或位姿不可信。
constexpr double kMaxPnPReproj = 3.0;

// PnP 失败后允许沿用上次成功位姿的最长时间（秒）
constexpr double kPoseReuseAge = 0.20;

// ---- 形状/颜色一致性过滤 ---------------------------------------------------
// 灯条是矩形（官方几何 41x20mm，长宽比 2.05）。
// 实测（带光晕）填充度 0.873~0.915；1px 抗锯齿的理论下限约 0.87。
// 眩光、反光点、彩色边缘形状不规则，填充度通常 < 0.6。
// 取 0.70：比实测最小 0.873 留 0.17 余量，同时和眩光分得开。
constexpr double kMinExtent = 0.70;
// 真实 LED 单一色相，H 的环形标准差只有 1~3°；
// 白色高光周围的色差边缘（紫边/红边）H 分布很散。
constexpr double kMaxHueStdDeg = 6.0;

// ---- 颜色阈值 --------------------------------------------------------------
// 红色在 HSV 的 H 上跨 0 和 179 两端，所以要两段。
struct HsvRange {
    cv::Scalar lo;
    cv::Scalar hi;
};

// ============================================================================
//  可调参数
//    上面的 constexpr 是"出厂默认值"，这里的 Params 是"本次运行实际用的值"。
//    优先级： 命令行 > config.txt > 上面的默认值
//    这样现场改个数字不用重新编译。
// ============================================================================
struct Params {
    // 相机
    double exposure = kDefExposure;
    double wb_temp = kDefWbTemp;
    double gain = 0.0;

    // 转盘在画面里的位置与尺度（现场标定）
    double wheel_x = -1.0;                       // -1 表示用画面中心
    double wheel_y = -1.0;
    double px_per_mm = kPxPerMm;                 // 每毫米多少像素
    double wheel_rpx = kDefWheelRpx;             // 由 px_per_mm 推出，可单独覆盖
    double auto_rpx = 1.0;                       // 1 = 用检出的灯条自动估计半径
    double roi_scale = kRoiScale;

    // 灯条形状
    double min_area = kMinLedArea;
    double max_area = kMaxLedArea;
    double min_aspect = kMinAspect;
    double min_extent = kMinExtent;
    double max_hue_std = kMaxHueStdDeg;

    // 槽位归组
    double group_gap_deg = kGroupGapDeg;
    double radius_tol = kRadiusTol;

    // 拟合
    double fit_window_s = kFitWindowS;
    int min_fit_samples = kMinFitSamples;
    // 最小拟合时间跨度（秒）。只有样本数不够，高帧率下 20 帧可能只覆盖 0.16 秒，
    // 正弦拟合会欠定（c0 与 c1 共线），初始模型不可信。必须同时满足跨度。
    double min_fit_span_s = 0.30;

    // 运行时帧率看门狗（只对相机模式生效；视频模式的帧率由文件决定）
    // 做成配置项而不是常量，一是现场可调，二是能在测试里把门槛调高来验证告警。
    double fps_warn_low = 50.0;      // 低于这个持续 2 秒就报警
    double fps_warn_high = 65.0;     // 高于这个算恢复，重新布防
    double omega_min = kOmegaMin;
    double omega_max = kOmegaMax;
    double omega_step = kOmegaStep;
    double max_fit_rms = kMaxFitRms;
    double min_osc_amp = kMinOscAmp;
    double ab_tol = kAbTol;

    // 预测与门限
    double lead_s = kLeadS;
    double max_pred_age = kMaxPredAge;
    double history_reset_age = kHistoryResetAge;
    double model_gate = kModelGateRad;           // 弧度

    // 相机内参（棋盘格标定结果）
    double fx = 1700.37, fy = 1701.76;
    double cx = 647.93, cy = 490.34;
    double k1 = -0.1262, k2 = 0.5895, p1 = -0.0011, p2 = 0.0009, k3 = -1.0703;

    // 靶的几何（官方 SDF）：4 条灯条距盘心 47.5mm，成十字
    double bar_offset_mm = 47.5;

    // 弹道
    double ball_speed = 12.0;        // 乒乓球初速 m/s（仿真默认 12）
    double muzzle_offset_mm = 75.0;  // 枪管轴比相机光轴低多少（相机装在枪管上方）
    double pipeline_delay = 0.10;    // 视觉+串口+云台响应，不含飞行时间
    double g_acc = 9.8;
    double pitch_bias_deg = 0.0;     // 实弹标定后的经验修正

    // 相机安装姿态（相对云台零位）
    //   camera_pitch_deg：相机光轴相对水平面向上抬了多少。
    //     相机固定在基座上时通常要抬一点，才能把整个转盘收进画面。
    //     这个值不补的话，弹道高度差会算错，pitch 也就错了。
    //   camera_yaw_deg：相机光轴相对云台零位的水平夹角，通常为 0。
    double camera_pitch_deg = 0.0;
    double camera_yaw_deg = 0.0;
    double y_scale = 1.0;            // 竖直补偿系数 = 1/cos(camera_pitch_deg)，派生量

    // S/V 阈值自适应：解决"曝光/白平衡一变就全灭"
    //   adaptive = clamp(ROI内P99 × adapt_ratio, adapt_min, 255)
    //   实际下限 = min(配置里的原阈值, adaptive)  —— 只会放宽，永不收紧
    double adapt_sv = 1.0;           // 0 关闭，回到固定阈值
    double adapt_ratio = 0.45;
    double adapt_min = 45.0;
    double adapt_max_gain = 4.0;     // 亮度补偿的最大倍数
    double adapt_gain_below = 110.0; // 只有 P99 低于这个值才补增益（真暗才补）
};

std::string trimStr(const std::string& s) {
    const char* ws = " \t\r\n";
    size_t a = s.find_first_not_of(ws);
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(ws);
    return s.substr(a, b - a + 1);
}

// 读 config.txt。
//   · 支持 UTF-8 BOM
//   · '#' 之后整行当注释，所以可以写中文注释
//   · 只认 "键 = 值"，键名不区分大小写
//   · 不认识的键会打警告，打字错误不会被静默吞掉
bool loadParams(const std::string& path, Params* P) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;

    int applied = 0;
    std::string line;
    bool first = true;
    while (std::getline(f, line)) {
        if (first) {
            first = false;
            if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF &&
                static_cast<unsigned char>(line[1]) == 0xBB &&
                static_cast<unsigned char>(line[2]) == 0xBF) {
                line.erase(0, 3);   // UTF-8 BOM
            }
        }
        size_t h = line.find('#');
        if (h != std::string::npos) line = line.substr(0, h);
        line = trimStr(line);
        if (line.empty()) continue;

        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;
        std::string k = trimStr(line.substr(0, eq));
        std::string v = trimStr(line.substr(eq + 1));
        if (k.empty() || v.empty()) continue;
        for (auto& c : k) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

        try {
            if (k == "exposure")            P->exposure = std::stod(v);
            else if (k == "wb")             P->wb_temp = std::stod(v);
            else if (k == "gain")           P->gain = std::stod(v);
            else if (k == "wheel_x")        P->wheel_x = std::stod(v);
            else if (k == "wheel_y")        P->wheel_y = std::stod(v);
            else if (k == "px_per_mm")      { P->px_per_mm = std::stod(v);
                                              P->wheel_rpx = 350.0 * P->px_per_mm; }
            else if (k == "wheel_rpx")      P->wheel_rpx = std::stod(v);
            else if (k == "auto_rpx")       P->auto_rpx = std::stod(v);
            else if (k == "roi_scale")      P->roi_scale = std::stod(v);
            else if (k == "min_area")       P->min_area = std::stod(v);
            else if (k == "max_area")       P->max_area = std::stod(v);
            else if (k == "min_aspect")     P->min_aspect = std::stod(v);
            else if (k == "min_extent")     P->min_extent = std::stod(v);
            else if (k == "max_hue_std")    P->max_hue_std = std::stod(v);
            else if (k == "group_gap_deg")  P->group_gap_deg = std::stod(v);
            else if (k == "radius_tol")     P->radius_tol = std::stod(v);
            else if (k == "fit_window_s")   P->fit_window_s = std::stod(v);
            else if (k == "min_fit_samples")P->min_fit_samples = std::stoi(v);
            else if (k == "min_fit_span_s") P->min_fit_span_s = std::stod(v);
            else if (k == "fps_warn_low")  P->fps_warn_low = std::stod(v);
            else if (k == "fps_warn_high") P->fps_warn_high = std::stod(v);
            else if (k == "omega_min")      P->omega_min = std::stod(v);
            else if (k == "omega_max")      P->omega_max = std::stod(v);
            else if (k == "omega_step")     P->omega_step = std::stod(v);
            else if (k == "max_fit_rms")    P->max_fit_rms = std::stod(v);
            else if (k == "min_osc_amp")    P->min_osc_amp = std::stod(v);
            else if (k == "ab_tol")         P->ab_tol = std::stod(v);
            else if (k == "lead_s")         P->lead_s = std::stod(v);
            else if (k == "max_pred_age")   P->max_pred_age = std::stod(v);
            else if (k == "history_reset_age") P->history_reset_age = std::stod(v);
            else if (k == "model_gate")     P->model_gate = std::stod(v) * CV_PI / 180.0;
            else if (k == "fx")             P->fx = std::stod(v);
            else if (k == "fy")             P->fy = std::stod(v);
            else if (k == "cx")             P->cx = std::stod(v);
            else if (k == "cy")             P->cy = std::stod(v);
            else if (k == "k1")             P->k1 = std::stod(v);
            else if (k == "k2")             P->k2 = std::stod(v);
            else if (k == "p1")             P->p1 = std::stod(v);
            else if (k == "p2")             P->p2 = std::stod(v);
            else if (k == "k3")             P->k3 = std::stod(v);
            else if (k == "bar_offset_mm")  P->bar_offset_mm = std::stod(v);
            else if (k == "ball_speed")     P->ball_speed = std::stod(v);
            else if (k == "muzzle_offset_mm") P->muzzle_offset_mm = std::stod(v);
            else if (k == "pipeline_delay") P->pipeline_delay = std::stod(v);
            else if (k == "pitch_bias_deg") P->pitch_bias_deg = std::stod(v);
            else if (k == "g_acc")          P->g_acc = std::stod(v);
            else if (k == "adapt_sv")       P->adapt_sv = std::stod(v);
            else if (k == "adapt_ratio")    P->adapt_ratio = std::stod(v);
            else if (k == "adapt_min")      P->adapt_min = std::stod(v);
            else if (k == "adapt_max_gain") P->adapt_max_gain = std::stod(v);
            else if (k == "adapt_gain_below") P->adapt_gain_below = std::stod(v);
            else if (k == "camera_pitch_deg") P->camera_pitch_deg = std::stod(v);
            else if (k == "camera_yaw_deg")   P->camera_yaw_deg = std::stod(v);
            else {
                std::cout << "[config] 未知键 \"" << k << "\"，已忽略\n";
                continue;
            }
            ++applied;
        } catch (...) {
            std::cout << "[config] 值解析失败: " << k << " = " << v << "\n";
        }
    }
    std::cout << "[config] 从 " << path << " 读取了 " << applied << " 项\n";
    return applied > 0;
}

void printParams(const Params& P) {
    std::printf("[参数] exposure=%.2f wb=%.0f gain=%.2f\n", P.exposure, P.wb_temp, P.gain);
    std::printf("[参数] wheel=(%.0f,%.0f) px_per_mm=%.4f R_px=%.1f roi=%.2f autoR=%s\n",
                P.wheel_x, P.wheel_y, P.px_per_mm, P.wheel_rpx, P.roi_scale,
                P.auto_rpx > 0.5 ? "on" : "off");
    std::printf("[参数] area=[%.0f,%.0f] aspect>=%.2f extent>=%.2f hue_sd<=%.1f\n",
                P.min_area, P.max_area, P.min_aspect, P.min_extent, P.max_hue_std);
    std::printf("[参数] gap=%.1fdeg radius_tol=%.2f\n", P.group_gap_deg, P.radius_tol);
    std::printf("[参数] fit: win=%.2fs n>=%d span>=%.2fs w=[%.3f,%.3f]/%.3f "
                "rms<=%.3f osc>=%.2f\n",
                P.fit_window_s, P.min_fit_samples, P.min_fit_span_s,
                P.omega_min, P.omega_max, P.omega_step, P.max_fit_rms, P.min_osc_amp);
    std::printf("[参数] 自适应: sv=%.1f gain_below=%.0f max_gain=%.1f s_ratio=%.2f s_min=%.0f\n",
                P.adapt_sv, P.adapt_gain_below, P.adapt_max_gain,
                P.adapt_ratio, P.adapt_min);
    std::printf("[参数] 看门狗: low=%.0f high=%.0f\n", P.fps_warn_low, P.fps_warn_high);
    std::printf("[参数] lead=%.3fs age<=%.3fs reset=%.1fs gate=%.2fdeg ab_tol=%.3f\n",
                P.lead_s, P.max_pred_age, P.history_reset_age,
                P.model_gate * 180.0 / CV_PI, P.ab_tol);
    // 相机内参：PnP 全靠它，改没改成功必须能一眼看出来
    std::printf("[参数] K: fx=%.2f fy=%.2f cx=%.2f cy=%.2f\n",
                P.fx, P.fy, P.cx, P.cy);
    std::printf("[参数] D: k1=%.4f k2=%.4f p1=%.4f p2=%.4f k3=%.4f\n",
                P.k1, P.k2, P.p1, P.p2, P.k3);
    // 靶几何与弹道
    std::printf("[参数] bar_offset=%.2fmm ball_v=%.2fm/s muzzle_dy=%.1fmm\n",
                P.bar_offset_mm, P.ball_speed, P.muzzle_offset_mm);
    std::printf("[参数] pipeline=%.3fs bias=%.2fdeg cam_pitch=%.2fdeg cam_yaw=%.2fdeg y_scale=%.4f\n",
                P.pipeline_delay, P.pitch_bias_deg, P.camera_pitch_deg,
                P.camera_yaw_deg, P.y_scale);
}

// 三种颜色各自的 HSV 区间。
//   红色：H 跨 0 和 179 两端，必须两段
//   蓝色：H≈120 一段就行
//     但官方仿真蓝方 emissive 是 (0, 0.15, 0.6)，换算成 sRGB 约 (0,108,203)，
//     对应 H≈103.5 —— 距离原来的下界 100 只有 3.5，太危险。
//     不同 gamma / 色调映射会让这个值上下浮动十几，所以放宽到 [92,138]。
//   黄色：H≈30 一段；实测黄色 LED 饱和度偏低（S≈73），所以 S 下限放到 50
// 比赛分边是随机的，红蓝都要能用，所以必须可切换而不是写死。
std::vector<HsvRange> rangesForColor(const std::string& color) {
    std::vector<HsvRange> v;
    if (color == "blue") {
        v.push_back({cv::Scalar(92, 110, 110), cv::Scalar(138, 255, 255)});
    } else if (color == "yellow") {
        v.push_back({cv::Scalar(15, 50, 150), cv::Scalar(45, 255, 255)});
    } else {   // red（默认）
        // 压力测试：光源 6500K、白平衡仍锁 4600K 时，红条 H 会从 ~3 漂到 ~139
        // （偏品红）。原区间 [0,10]+[170,179] 完全覆盖不到，直接全灭。
        // 放宽到 [0,15]+[158,179]，可容纳 ±1500K 的白平衡失配。
        // 与蓝色区间 [92,138] 不重叠，不会互相抢。
        // 第一段 [0,12]：实测暖光下红灯核心 H≈4，而灯条光晕是 H≈13~15 的橙色，
        //   放宽到 15 会把光晕放进来（暖光工况下灯条数从 4 涨到 16）。
        // 第二段 [158,179]：冷光 6500K 时红灯核心会漂到 H≈160，靠这一段兜住。
        v.push_back({cv::Scalar(0, 120, 120), cv::Scalar(12, 255, 255)});
        v.push_back({cv::Scalar(158, 120, 120), cv::Scalar(179, 255, 255)});
    }
    return v;
}

// 在 ROI 圆内统计某通道的高分位数（P99），用 256 桶直方图 + 4 像素采样。
//   chan: 1 = S，2 = V
int roiChannelP99(const cv::Mat& hsv, cv::Point2f wheel, double roi_r, int chan) {
    int hist[256] = {0};
    const int step = 4;
    const double r2 = roi_r * roi_r;
    int n = 0;
    for (int y = 0; y < hsv.rows; y += step) {
        const uchar* row = hsv.ptr<uchar>(y);
        const double dy = static_cast<double>(y) - wheel.y;
        for (int x = 0; x < hsv.cols; x += step) {
            const double dx = static_cast<double>(x) - wheel.x;
            if (dx * dx + dy * dy > r2) continue;
            ++hist[row[x * 3 + chan]];
            ++n;
        }
    }
    if (n < 16) return 255;
    const int target = static_cast<int>(n * 0.99);
    int acc = 0;
    for (int v = 0; v < 256; ++v) {
        acc += hist[v];
        if (acc >= target) return v;
    }
    return 255;
}

// ============================================================================
//  ① 颜色掩膜
//     ROI 圆：只保留转盘周围的像素，顶灯/墙面反光在圆外，一步排掉。
//     同时把 H 通道回传出去，后面算色相一致性要用。
// ============================================================================
// V = max(B,G,R)，在 ROI 内取 P99。用来判断画面是不是整体偏暗。
int roiBrightP99(const cv::Mat& bgr, cv::Point2f wheel, double roi_r) {
    int hist[256] = {0};
    const int step = 4;
    const double r2 = roi_r * roi_r;
    int n = 0;
    for (int y = 0; y < bgr.rows; y += step) {
        const uchar* row = bgr.ptr<uchar>(y);
        const double dy = static_cast<double>(y) - wheel.y;
        for (int x = 0; x < bgr.cols; x += step) {
            const double dx = static_cast<double>(x) - wheel.x;
            if (dx * dx + dy * dy > r2) continue;
            const uchar b = row[x * 3], g = row[x * 3 + 1], r = row[x * 3 + 2];
            ++hist[std::max(b, std::max(g, r))];
            ++n;
        }
    }
    if (n < 16) return 255;
    const int target = static_cast<int>(n * 0.99);
    int acc = 0;
    for (int v = 0; v < 256; ++v) {
        acc += hist[v];
        if (acc >= target) return v;
    }
    return 255;
}

// 统计"最亮那批像素"的 S 中位数。
//   v_min 取 Vp99 的 85%，基本只保留灯条核心，避免背景高光把中位数拉低。
int roiBrightSMedian(const cv::Mat& hsv, cv::Point2f wheel, double roi_r, int v_min) {
    int hist[256] = {0};
    const int step = 4;
    const double r2 = roi_r * roi_r;
    int n = 0;
    for (int y = 0; y < hsv.rows; y += step) {
        const uchar* row = hsv.ptr<uchar>(y);
        const double dy = static_cast<double>(y) - wheel.y;
        for (int x = 0; x < hsv.cols; x += step) {
            const double dx = static_cast<double>(x) - wheel.x;
            if (dx * dx + dy * dy > r2) continue;
            if (row[x * 3 + 2] < v_min) continue;      // V 不够亮，跳过
            ++hist[row[x * 3 + 1]];                    // 记 S
            ++n;
        }
    }
    if (n < 8) return 255;
    const int target = n / 2;
    int acc = 0;
    for (int s = 0; s < 256; ++s) {
        acc += hist[s];
        if (acc >= target) return s;
    }
    return 255;
}

cv::Mat buildMask(const cv::Mat& bgr, const std::vector<HsvRange>& ranges,
                  cv::Point2f wheel, double roi_radius, cv::Mat* h_out,
                  const Params& P_adapt) {
    // ---- 亮度补偿（软件自动曝光）----
    // 画面对暗时，直接压低 V 阈值会把背景噪声一起放进来（实测轮廓从 10 涨到 2242）。
    // 正确做法是先把整幅图补到正常亮度，V 阈值保持不动。
    cv::Mat src = bgr;
    cv::Mat gained;
    int s_floor = 0;
    if (P_adapt.adapt_sv > 0.5) {
        const double rr = (roi_radius > 1.0) ? roi_radius
                                             : std::hypot(bgr.cols, bgr.rows);
        const int vp = roiBrightP99(bgr, wheel, rr);
        if (vp > 10 && vp < P_adapt.adapt_gain_below) {
            const double gain = std::min(P_adapt.adapt_max_gain, 200.0 / vp);
            if (gain > 1.02) {
                cv::convertScaleAbs(bgr, gained, gain, 0.0);
                src = gained;
            }
        }
    }

    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);

    // ---- 饱和度下限自适应 ----
    // 只动 S：褪色时红灯会变成粉色，S 掉到 70 上下，固定 120 就全灭。
    // V 阈值不参与自适应，避免暗噪声涌入。
    if (P_adapt.adapt_sv > 0.5) {
        const double rr = (roi_radius > 1.0) ? roi_radius
                                             : std::hypot(hsv.cols, hsv.rows);
        const int vp = roiBrightP99(src, wheel, rr);
        const int v_thr = static_cast<int>(vp * 0.85);
        const int sb = roiBrightSMedian(hsv, wheel, rr, v_thr);
        // 只有"最亮的东西本身就偏淡"时才放宽 S（褪色的情形）。
        // 暖光/正常光下灯条 S 都在 160 以上，此时绝不动阈值，
        // 否则会把暖色调的背景放进掩膜。
        if (sb < 120) {
            s_floor = std::max(static_cast<int>(P_adapt.adapt_min),
                               static_cast<int>(sb * 0.8));
            s_floor = std::min(s_floor, 119);
        }
    }

    cv::Mat mask = cv::Mat::zeros(hsv.size(), CV_8UC1);
    for (const auto& r : ranges) {
        cv::Scalar lo = r.lo;
        if (s_floor > 0) lo[1] = std::min(static_cast<double>(lo[1]),
                                          static_cast<double>(s_floor));
        cv::Mat m;
        cv::inRange(hsv, lo, r.hi, m);
        cv::bitwise_or(mask, m, mask);
    }

    if (h_out) {
        cv::extractChannel(hsv, *h_out, 0);   // H 通道，色相一致性检查要用
    }

    // 圆形 ROI：转盘之外的红色一律不要（顶灯、衣服、横幅、广告牌）
    if (roi_radius > 1.0) {
        cv::Mat roi = cv::Mat::zeros(mask.size(), CV_8UC1);
        cv::circle(roi, wheel, static_cast<int>(roi_radius), cv::Scalar(255), -1);
        cv::bitwise_and(mask, roi, mask);
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    return mask;
}

// ============================================================================
//  ② 灯条
//     三级过滤，并且统计每一级拒了多少个——现场调参就靠这三个数。
// ============================================================================
struct Led {
    cv::Point2f center;   // 原始质心（画图用）
    cv::Point2f geom;     // 做过透视压缩补偿的点（算角度和半径用）
    double area = 0.0;
    double aspect = 0.0;
};

struct RejectStats {
    int raw = 0;          // 轮廓总数
    int by_area = 0;      // 面积不合格
    int by_aspect = 0;    // 长宽比不合格（圆斑/光晕）
    int by_extent = 0;    // 填充度不合格（形状不规则，眩光的典型特征）
    int by_hue = 0;       // 色相不一致（彩色边缘/色差）
    int by_radius = 0;    // 半径不对（不在转盘上）
    int kept = 0;
};

// 色相的环形标准差（度）。
// H 在 0/179 处回绕，直接算标准差会把"红色跨两端"算成巨大方差，
// 所以先把 H 映射到复平面求合成向量长度 R，再换算：sd = sqrt(-2 ln R)。
//   真实 LED：同一色相，R 接近 1，sd 只有 1~3°
//   彩色边缘/眩光：色相分散，R 小，sd 会超过 10°
double hueCircularStdDeg(const cv::Mat& hch, const std::vector<cv::Point>& c) {
    cv::Rect bb = cv::boundingRect(c);
    if (bb.width < 2 || bb.height < 2) return 0.0;

    cv::Mat local = cv::Mat::zeros(bb.size(), CV_8UC1);
    std::vector<cv::Point> shifted;
    shifted.reserve(c.size());
    for (const auto& p : c) shifted.emplace_back(p.x - bb.x, p.y - bb.y);
    std::vector<std::vector<cv::Point>> polys{shifted};
    cv::fillPoly(local, polys, cv::Scalar(255));

    cv::Mat hroi = hch(bb);
    double sc = 0.0, ss = 0.0;
    int n = 0;
    for (int y = 0; y < local.rows; ++y) {
        const uchar* lm = local.ptr<uchar>(y);
        const uchar* hp = hroi.ptr<uchar>(y);
        for (int x = 0; x < local.cols; ++x) {
            if (!lm[x]) continue;
            const double a = hp[x] * (CV_PI / 90.0);   // H/180*2π
            sc += std::cos(a);
            ss += std::sin(a);
            ++n;
        }
    }
    if (n < 10) return 0.0;
    const double R = std::hypot(sc, ss) / n;
    if (R < 1e-9) return 180.0;
    return std::sqrt(-2.0 * std::log(R)) * 180.0 / CV_PI;
}

std::vector<Led> detectLeds(const cv::Mat& mask, const cv::Mat& hch, cv::Point2f wheel,
                            double R_px, const Params& P, RejectStats* st) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (st) st->raw = static_cast<int>(contours.size());

    std::vector<Led> leds;
    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < P.min_area || area > P.max_area) {
            if (st) ++st->by_area;
            continue;
        }

        cv::RotatedRect rr = cv::minAreaRect(c);
        double w = rr.size.width, h = rr.size.height;
        double lo = std::min(w, h), hi = std::max(w, h);
        if (lo <= 0.0) {
            if (st) ++st->by_aspect;
            continue;
        }
        double aspect = hi / lo;
        if (aspect < P.min_aspect) {
            if (st) ++st->by_aspect;
            continue;
        }

        // 填充度 = 轮廓面积 / 最小外接矩形面积。
        // 圆角矩形实测 0.873~0.915；眩光、反光点形状不规则，明显更低。
        const double extent = area / (w * h);
        if (extent < P.min_extent) {
            if (st) ++st->by_extent;
            continue;
        }

        // 色相一致性：真实 LED 是单色，H 的环形标准差很小
        if (!hch.empty()) {
            const double sd = hueCircularStdDeg(hch, c);
            if (sd > P.max_hue_std) {
                if (st) ++st->by_hue;
                continue;
            }
        }

        cv::Moments M = cv::moments(c);
        if (M.m00 <= 0.0) {
            if (st) ++st->by_area;
            continue;
        }

        Led led;
        led.center = cv::Point2f(static_cast<float>(M.m10 / M.m00),
                                 static_cast<float>(M.m01 / M.m00));
        led.area = area;
        led.aspect = aspect;

        // 透视压缩补偿：相机仰角 t 时，竖直方向被压缩 cos(t)，
        // 正圆看起来是竖着压扁的椭圆。把竖直偏移除以 cos(t) 还原。
        // 不补的话，靶的角度会产生最多约 1.9° 的系统偏差（t=15° 时），
        // 比我们 0.1° 的精度要求大 20 倍。
        led.geom = led.center;
        led.geom.y = wheel.y + (led.center.y - wheel.y) * P.y_scale;

        // 半径检查放在这里（而不是等到槽位归组）：
        // 离轮心太远的红色块一定不是灯条，早点扔掉还能少一次聚簇运算。
        cv::Point2f d = led.geom - wheel;
        double r = std::hypot(d.x, d.y);
        if (std::fabs(r - R_px) > P.radius_tol * R_px) {
            if (st) ++st->by_radius;
            continue;
        }

        if (st) ++st->kept;
        leds.push_back(led);
    }
    return leds;
}

// ============================================================================
//  ③ 槽位归组
// ============================================================================
struct Plate {
    std::vector<int> bars;   // 属于这块靶的灯条下标
    cv::Point2f center;      // 靶中心（像素）
    double angle_deg = 0.0;  // 相对转盘圆心的角度 [0, 360)
    int state = 0;           // 4 = 待激活，2 = 已激活
};

std::vector<Plate> assignSlots(const std::vector<Led>& leds, cv::Point2f wheel,
                               double R_px, const Params& P) {
    struct Item {
        int idx;
        double ang;
    };
    std::vector<Item> items;
    for (int i = 0; i < static_cast<int>(leds.size()); ++i) {
        cv::Point2f d = leds[i].geom - wheel;
        double r = std::hypot(d.x, d.y);
        if (std::fabs(r - R_px) > kRadiusTol * R_px) continue;
        double a = std::atan2(d.y, d.x) * 180.0 / CV_PI;
        if (a < 0.0) a += 360.0;
        items.push_back({i, a});
    }
    if (items.empty()) return {};

    std::sort(items.begin(), items.end(),
              [](const Item& a, const Item& b) { return a.ang < b.ang; });

    // 在最大角度空隙处剪开，避免 0°/360° 交界被切成两块靶
    int cut = 0;
    double max_gap = -1.0;
    for (size_t k = 0; k < items.size(); ++k) {
        double cur = items[k].ang;
        double nxt = (k + 1 < items.size()) ? items[k + 1].ang : items[0].ang + 360.0;
        if (nxt - cur > max_gap) {
            max_gap = nxt - cur;
            cut = static_cast<int>((k + 1) % items.size());
        }
    }
    std::rotate(items.begin(), items.begin() + cut, items.end());

    double base = items[0].ang;
    for (auto& it : items) {
        if (it.ang < base) it.ang += 360.0;
    }

    std::vector<Plate> plates;
    std::vector<Item> cur{items[0]};
    auto flush = [&]() {
        if (cur.size() >= 2) {
            Plate p;
            cv::Point2f c(0.0f, 0.0f);
            for (const auto& it : cur) {
                p.bars.push_back(it.idx);
                c += leds[it.idx].geom;
            }
            c *= 1.0f / static_cast<float>(cur.size());
            // 只有 3 条灯时，质心不是盘心。取"相距最远的两条"的中点：
            // 相对的两条灯相距 2×47.5mm，共线且中点即盘心。
            if (cur.size() == 3) {
                int bi = 0, bj = 1;
                float best = -1.0f;
                for (int i = 0; i < 3; ++i) {
                    for (int j = i + 1; j < 3; ++j) {
                        const cv::Point2f d = leds[cur[i].idx].geom - leds[cur[j].idx].geom;
                        const float dd = d.x * d.x + d.y * d.y;
                        if (dd > best) {
                            best = dd;
                            bi = i;
                            bj = j;
                        }
                    }
                }
                c = (leds[cur[bi].idx].geom + leds[cur[bj].idx].geom) * 0.5f;
            }
            p.center = c;
            cv::Point2f d = c - wheel;
            double a = std::atan2(d.y, d.x) * 180.0 / CV_PI;
            if (a < 0.0) a += 360.0;
            p.angle_deg = a;
            p.state = static_cast<int>(cur.size());
            plates.push_back(p);
        }
        cur.clear();
    };
    for (size_t k = 1; k < items.size(); ++k) {
        if (items[k].ang - items[k - 1].ang < P.group_gap_deg) {
            cur.push_back(items[k]);
        } else {
            flush();
            cur.push_back(items[k]);
        }
    }
    flush();
    return plates;
}

// ============================================================================
//  ④ 相位解缠
// ============================================================================
bool estimatePhase(const std::vector<Plate>& plates, double* raw_out) {
    if (plates.empty()) return false;

    double sx = 0.0, sy = 0.0;
    for (const auto& p : plates) {
        double a = p.angle_deg * CV_PI / 180.0;
        sx += std::cos(kSlotCount * a);
        sy += std::sin(kSlotCount * a);
    }
    if (std::hypot(sx, sy) < 1e-6) return false;

    *raw_out = std::atan2(sy, sx) / kSlotCount;   // 弧度，落在 (-36°, +36°]
    return true;
}

class PhaseTracker {
public:
    explicit PhaseTracker(int slot_count) : half_slot_(CV_PI / slot_count) {}

    // 算出"如果采用这帧，theta 会变成多少"，但不提交。
    // 给模型门限做判断用：先看看结果合不合理，再决定要不要写进去。
    double peek(bool has_raw, double raw) const {
        if (!has_raw) return theta_;
        if (!has_theta_) return raw;
        double d = 0.0;
        if (!foldDelta(raw, d)) return theta_;   // 跳变，本来就不会采用
        return theta_ + d;
    }

    double update(bool has_raw, double raw) {
        if (!has_raw) return theta_;
        if (!has_theta_) {
            theta_ = raw;
            last_raw_ = raw;
            has_theta_ = true;
            return theta_;
        }
        double d = 0.0;
        if (!foldDelta(raw, d)) {
            return theta_;      // 跳变拒绝，防 72° 滑移
        }
        theta_ += d;
        last_raw_ = raw;
        return theta_;
    }

    bool has() const { return has_theta_; }
    double theta() const { return theta_; }
    double thetaDeg() const { return theta_ * 180.0 / CV_PI; }

    // 丢弃连续性，把下一帧当作新起点。
    // 必须提供这个：随机固定靶会突然跳到新位置。跳变超过 28.8° 时每一帧都会被
    // 跳变拒绝，last_raw_ 不更新，theta 就永久冻结、再也救不回来。
    void reset() { has_theta_ = false; }

private:
    // 把两帧原始相位之差折叠到 ±半个槽。返回 false 表示跳变太大，应当丢弃。
    bool foldDelta(double raw, double& d) const {
        const double two = 2.0 * half_slot_;
        d = std::fmod(raw - last_raw_ + half_slot_, two);
        if (d < 0.0) d += two;
        d -= half_slot_;
        return std::fabs(d) <= 0.8 * half_slot_;
    }

    double half_slot_;
    double theta_ = 0.0;
    double last_raw_ = 0.0;
    bool has_theta_ = false;
};

// ============================================================================
//  ⑤ 拟合 θ(t)
// ============================================================================
struct MotionModel {
    double omega = 0.0;
    double c[4] = {0.0, 0.0, 0.0, 0.0};
    double rms = 0.0;
    double a = 0.0;      // 摆动振幅（rad/s），= ω·√(c2²+c3²)
    double b = 0.0;      // 平均角速度（rad/s），= c1
};

// 4x4 线性方程组，高斯消元（部分主元）
bool solve4(double A[4][4], const double b[4], double x[4]) {
    double M[4][5];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) M[r][c] = A[r][c];
        M[r][4] = b[r];
    }
    for (int col = 0; col < 4; ++col) {
        int piv = col;
        for (int r = col + 1; r < 4; ++r) {
            if (std::fabs(M[r][col]) > std::fabs(M[piv][col])) piv = r;
        }
        if (std::fabs(M[piv][col]) < 1e-12) return false;   // 奇异
        if (piv != col) {
            for (int c = 0; c < 5; ++c) std::swap(M[col][c], M[piv][c]);
        }
        for (int r = 0; r < 4; ++r) {
            if (r == col) continue;
            double f = M[r][col] / M[col][col];
            for (int c = col; c < 5; ++c) M[r][c] -= f * M[col][c];
        }
    }
    for (int r = 0; r < 4; ++r) x[r] = M[r][4] / M[r][r];
    return true;
}

// t_now 必须由调用方传"当前时刻"，不能用 history.back().first。
// 否则丢帧时拟合还以为现在停在最后一次成功的那一帧，
// 会拿过期的模型外推，预测出来的角度对应的是几百毫秒前。
bool fitMotion(const std::vector<std::pair<double, double>>& history, double t_now,
               const Params& P, MotionModel* out) {
    if (static_cast<int>(history.size()) < P.min_fit_samples) return false;

    std::vector<double> ts, ys;
    for (const auto& p : history) {
        if (p.first >= t_now - P.fit_window_s && p.first <= t_now) {
            ts.push_back(p.first - t_now);   // 以"现在"为原点，数值更稳
            ys.push_back(p.second);
        }
    }
    if (static_cast<int>(ts.size()) < P.min_fit_samples) return false;
    // 样本足够但跨度太短一样不能拟合：高帧率下 20 个样本可能只覆盖 0.16 秒，
    // 这么短的弧段上 c0（偏置）与 c1（斜率）几乎共线，解出来的参数不可信。
    // 仿真和视频都是 60fps，这个差异在那边永远暴露不出来。
    if (ts.back() - ts.front() < P.min_fit_span_s) return false;

    double best_rms = 1e18;
    double best_w = 0.0;
    double best_c[4] = {0.0, 0.0, 0.0, 0.0};

    // ω 网格搜索：每个候选解一次最小二乘，留残差最小的
    for (double w = P.omega_min; w <= P.omega_max + 1e-9; w += P.omega_step) {
        double A[4][4] = {{0}};
        double b[4] = {0, 0, 0, 0};
        for (size_t i = 0; i < ts.size(); ++i) {
            double row[4] = {1.0, ts[i], std::cos(w * ts[i]), std::sin(w * ts[i])};
            for (int r = 0; r < 4; ++r) {
                for (int c = 0; c < 4; ++c) A[r][c] += row[r] * row[c];
                b[r] += row[r] * ys[i];
            }
        }
        double c[4];
        if (!solve4(A, b, c)) continue;

        double sse = 0.0;
        for (size_t i = 0; i < ts.size(); ++i) {
            double row[4] = {1.0, ts[i], std::cos(w * ts[i]), std::sin(w * ts[i])};
            double pred = row[0] * c[0] + row[1] * c[1] + row[2] * c[2] + row[3] * c[3];
            double e = ys[i] - pred;
            sse += e * e;
        }
        double rms = std::sqrt(sse / static_cast<double>(ts.size()));
        if (rms < best_rms) {
            best_rms = rms;
            best_w = w;
            for (int k = 0; k < 4; ++k) best_c[k] = c[k];
        }
    }

    if (best_rms > P.max_fit_rms) return false;   // 残差大 -> 不可信，HOLD

    out->omega = best_w;
    out->rms = best_rms;
    for (int k = 0; k < 4; ++k) out->c[k] = best_c[k];

    // 从拟合系数反推出物理量：
    //   θ(t) = c0 + c1·t + c2·cos(ωt) + c3·sin(ωt)
    //   求导 => spd(t) = c1 + ω·(−c2·sin + c3·cos)
    //   所以 b = c1，振幅 a = ω·√(c2²+c3²)
    out->b = best_c[1];
    out->a = best_w * std::hypot(best_c[2], best_c[3]);

    // 手册约束：b = 2.090 − a，即 a + b 恒等于 2.090。
    // 只在确实存在正弦摆动时检查——小机关(a=0,b≈1.047)、固定靶(a=b=0)
    // 的 a+b 远不是 2.090，无条件检查会全部误杀。
    if (out->a > P.min_osc_amp &&
        std::fabs(out->a + out->b - kAbSum) > P.ab_tol) {
        return false;   // 物理上不成立，说明拟合到了错误的对象
    }
    return true;
}

// ============================================================================
//  ⑥ 预测：增量式
//      target_pred = angle_now + (θ_future − θ_now)
//  靶角 = θ + k×72°，换靶时 k 跳变，但做差就消掉了
// ============================================================================
double predictTarget(const MotionModel& m, double theta_now, double angle_now,
                     double lead) {
    double theta_future = m.c[0] + m.c[1] * lead + m.c[2] * std::cos(m.omega * lead) +
                          m.c[3] * std::sin(m.omega * lead);
    return angle_now + (theta_future - theta_now);
}

// θ(t_now + dt) − θ(t_now)。模型以"现在"为时间原点，所以 c0 就是 θ(t_now)。
//   = c1·dt + c2·(cos(ω·dt) − 1) + c3·sin(ω·dt)
double thetaIncrement(const MotionModel& m, double dt) {
    return m.c[1] * dt + m.c[2] * (std::cos(m.omega * dt) - 1.0) +
           m.c[3] * std::sin(m.omega * dt);
}

// ============================================================================
//  ⑧ 位姿解算（solvePnP）+ 弹道补偿
//
//  靶的 3D 模型（官方 SDF，已验证是正十字）：
//      4 条灯条质心距盘心 47.5mm，分别在径向 ±47.5 与切向 ±47.5
//      放在靶平面局部坐标系里就是 (±47.5, 0, 0) 和 (0, ±47.5, 0)
//
//  注意：这个十字在 90° 旋转和镜像下都对称，所以"哪条灯对应哪个 3D 点"
//  的循环顺序不影响解出的中心位置 tvec —— 只要每帧顺序一致就行。
//
//  PnP 给出 tvec = 靶心在相机坐标系中的 (X, Y, Z)，单位 mm：
//      X 右、Y 下、Z 前方（OpenCV 相机坐标系）
//  Z 就是"距离"，这是弹道补偿和飞行时间唯一的数据来源。
// ============================================================================
struct PoseResult {
    bool ok = false;
    double x = 0.0, y = 0.0, z = 0.0;   // mm，相机坐标系
    double dist_mm = 0.0;
    double yaw_deg = 0.0;               // 水平偏角，右正
    double los_pitch_deg = 0.0;         // 视线俯仰角，上正
    double reproj_px = 0.0;
};

PoseResult solvePlatePose(const std::vector<cv::Point2f>& pts, const Params& P) {
    PoseResult r;
    if (pts.size() < 4) return r;

    // 按"绕质心的极角"排序，保证每帧的循环顺序一致
    cv::Point2f c(0.0f, 0.0f);
    for (const auto& p : pts) c += p;
    c *= 1.0f / static_cast<float>(pts.size());

    std::vector<cv::Point2f> ordered(pts.begin(), pts.begin() + 4);
    std::sort(ordered.begin(), ordered.end(),
              [&](const cv::Point2f& a, const cv::Point2f& b) {
                  return std::atan2(a.y - c.y, a.x - c.x) <
                         std::atan2(b.y - c.y, b.x - c.x);
              });

    const double r_off = P.bar_offset_mm;
    std::vector<cv::Point3f> obj = {
        {static_cast<float>(r_off), 0.0f, 0.0f},
        {0.0f, static_cast<float>(r_off), 0.0f},
        {static_cast<float>(-r_off), 0.0f, 0.0f},
        {0.0f, static_cast<float>(-r_off), 0.0f},
    };

    cv::Mat K = (cv::Mat_<double>(3, 3) << P.fx, 0.0, P.cx,
                                          0.0, P.fy, P.cy,
                                          0.0, 0.0, 1.0);
    cv::Mat D = (cv::Mat_<double>(1, 5) << P.k1, P.k2, P.p1, P.p2, P.k3);

    cv::Vec3d rvec, tvec;
    bool solved = false;
    try {
        // 4 点共面，用专门为此设计的 IPPE；失败再退回通用迭代法
        solved = cv::solvePnP(obj, ordered, K, D, rvec, tvec, false,
                              cv::SOLVEPNP_IPPE);
    } catch (...) {
        solved = false;
    }
    if (!solved) {
        try {
            solved = cv::solvePnP(obj, ordered, K, D, rvec, tvec, false,
                                  cv::SOLVEPNP_ITERATIVE);
        } catch (...) {
            return r;
        }
    }
    if (!solved || tvec[2] <= 0.0) return r;

    // 用解出来的位姿反投影，检验是否可信
    std::vector<cv::Point2f> proj;
    cv::projectPoints(obj, rvec, tvec, K, D, proj);
    double sse = 0.0;
    for (size_t i = 0; i < proj.size(); ++i) {
        double dx = proj[i].x - ordered[i].x, dy = proj[i].y - ordered[i].y;
        sse += dx * dx + dy * dy;
    }
    r.reproj_px = std::sqrt(sse / static_cast<double>(proj.size()));
    if (r.reproj_px > kMaxPnPReproj) return r;    // 解不可信

    r.ok = true;
    r.x = tvec[0];
    r.y = tvec[1];
    r.z = tvec[2];
    r.dist_mm = std::hypot(tvec[0], tvec[1], tvec[2]);
    r.yaw_deg = std::atan2(tvec[0], tvec[2]) * 180.0 / CV_PI;
    // 相机 y 向下，所以靶在上方时 tvec[1] < 0；取负号让"上"为正
    r.los_pitch_deg = -std::atan2(tvec[1], std::hypot(tvec[0], tvec[2])) * 180.0 / CV_PI;
    return r;
}

// 低伸弹道：给定水平距离 d 与相对枪口的高度差 h，求发射角（弧度）。
// tan(theta) = (v^2 - sqrt(v^4 - g(g d^2 + 2 h v^2))) / (g d)
// 返回 false 表示这个距离打不到（判别式 < 0）。
bool ballisticPitch(double v, double d, double h, double g, double* theta_rad) {
    if (d <= 1e-6 || v <= 1e-6) return false;
    const double v2 = v * v;
    const double disc = v2 * v2 - g * (g * d * d + 2.0 * h * v2);
    if (disc < 0.0) return false;
    *theta_rad = std::atan((v2 - std::sqrt(disc)) / (g * d));
    return true;
}

// 飞行时间：t = d / (v cos(theta))
double flightTime(double v, double d, double theta_rad) {
    if (v <= 1e-6) return 0.0;
    return d / (v * std::cos(theta_rad));
}

// ============================================================================
//  ⑦ 命中点解算（迭代）
//
//  为什么必须迭代：
//     pitch  依赖 靶在命中时刻的高度
//     靶高    依赖 弹丸飞行期间转过多少角度
//     转过多 依赖 飞行时间
//     飞行时间 依赖 pitch
//  这是个循环依赖。只算一遍等于假设靶在飞行期间不动。
//  实测（1950mm、110°/s、弹速 12）：这样会带来最大 5.3° 的 pitch 偏差，
//  而靶盘在 1950mm 处的半张角只有 2.86°——偏出 186%，必脱靶。
//
//  同一套迭代还顺手修掉了 yaw：原来 yaw 用的是【当前】方位角，没有提前量。
//  改成从命中时刻的 3D 位置反推，两者同时正确。
// ============================================================================
constexpr int kHitIters = 4;          // 迭代次数，3~4 次即收敛
constexpr double kHitLeadTol = 1e-4;  // 提前量收敛判据（秒）

struct HitSolution {
    bool ok = false;
    double yaw_deg = 0.0;
    double pitch_deg = 0.0;
    double t_fly = 0.0;
    double lead = 0.0;
    cv::Point3d p_hit;      // 命中时刻靶心，相机系 mm
    double dtheta_deg = 0.0;  // 最后一次迭代用到的相位增量（诊断用）
    int iters = 0;
    double residual = 0.0;  // 最后一次迭代提前量的变化量
};

// 轮盘是竖直的、正对相机，所以"轮心 → 靶心"这个径向向量就在图像平面内。
// 于是命中时刻的靶心 = 轮心 + Rot_z(Δθ) · (靶心 − 轮心)。
//
// 入参：
//   pose      当前靶心的 PnP 结果（相机系 mm）
//   wheel_px  轮心的图像坐标
//   plate_px  当前靶心的图像坐标
//   model     运动模型；传 nullptr 表示没有模型（固定不动，Δθ = 0）
HitSolution solveHit(const PoseResult& pose, cv::Point2f wheel_px, cv::Point2f plate_px,
                     const MotionModel* model, const Params& P) {
    HitSolution r;
    if (!pose.ok || P.fx <= 1.0 || P.fy <= 1.0) return r;

    // ---- 轮心 → 靶心 的 3D 径向向量（mm）----
    // 图像上差 dn 像素，在深度 Z 处对应 dn·Z/f 毫米。
    const double dx = plate_px.x - wheel_px.x;
    const double dy = plate_px.y - wheel_px.y;
    const double vx = dx * pose.z / P.fx;
    const double vy = dy * pose.z / P.fy;
    if (std::hypot(vx, vy) < 1.0) return r;   // 靶心几乎压在轮心上，退化

    const cv::Point3d Pcur(pose.x, pose.y, pose.z);
    const cv::Point3d wheel3(Pcur.x - vx, Pcur.y - vy, Pcur.z);

    // ---- 迭代 ----
    double lead = P.pipeline_delay;
    double prev_lead = -1.0;
    for (int it = 0; it < kHitIters; ++it) {
        // 命中时刻相对现在转过了多少相位
        const double dth = model ? thetaIncrement(*model, lead) : 0.0;
        const double c = std::cos(dth), s = std::sin(dth);
        const cv::Point3d Phit(wheel3.x + vx * c - vy * s,
                               wheel3.y + vx * s + vy * c,
                               wheel3.z);

        // 由命中点算视线角：相机 y 向下，所以靶在上方时 y < 0
        const double dist3 = std::sqrt(Phit.x * Phit.x + Phit.y * Phit.y + Phit.z * Phit.z);
        const double los = std::atan2(-Phit.y, std::hypot(Phit.x, Phit.z));
        const double elev = los + P.camera_pitch_deg * CV_PI / 180.0;
        const double d_h = (dist3 / 1000.0) * std::cos(elev);
        const double h_rel = d_h * std::tan(elev) + P.muzzle_offset_mm / 1000.0;

        double theta_rad = 0.0;
        if (!ballisticPitch(P.ball_speed, d_h, h_rel, P.g_acc, &theta_rad)) {
            return r;   // 这个距离打不到
        }
        const double t_fly = flightTime(P.ball_speed, d_h, theta_rad);

        r.p_hit = Phit;
        r.dtheta_deg = dth * 180.0 / CV_PI;
        r.iters = it + 1;
        r.t_fly = t_fly;
        r.pitch_deg = theta_rad * 180.0 / CV_PI + P.pitch_bias_deg;
        r.yaw_deg = std::atan2(Phit.x, Phit.z) * 180.0 / CV_PI + P.camera_yaw_deg;

        const double new_lead = P.pipeline_delay + t_fly;
        r.residual = std::fabs(new_lead - lead);
        lead = new_lead;
        r.lead = lead;
        r.ok = true;
        if (r.residual < kHitLeadTol) break;
        (void)prev_lead;
        prev_lead = lead;
    }
    return r;
}


// 存图。cv::imwrite 在中文路径下会静默失败（返回 false 但不报错，也不建文件），
// 所以失败时退回 imencode + ofstream 的写法。项目路径全是中文，必须走这条路。
bool saveImage(const std::string& path, const cv::Mat& img) {
    if (img.empty() || path.empty()) return false;
    try {
        if (cv::imwrite(path, img)) return true;
    } catch (...) {
        // imwrite 对中文路径可能直接抛异常，继续走后备方案
    }
    try {
        std::string ext = ".png";
        const size_t dot = path.find_last_of('.');
        if (dot != std::string::npos) ext = path.substr(dot);
        std::vector<uchar> buf;
        if (!cv::imencode(ext, img, buf)) return false;
        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(buf.data()),
                static_cast<std::streamsize>(buf.size()));
        return f.good();
    } catch (...) {
        return false;
    }
}

// ============================================================================
//  打开相机
// ============================================================================
// ---------------------------------------------------------------------------
//  相机模式候选
//
//  实测（2026-11-25，LRCP S200 / VID_0BDA&PID_5882，YUY2）：
//      请求 1280x1024 -> 实际 1280x1024，只有 5.0 fps   <- 驱动这条直连路径是坏的
//      请求 1280x960  -> 实际 1280x960，138 fps         <- 干净，但垂直裁掉 64 行
//      请求 1920x1080 -> 实际 1280x1024，130 fps        <- 同一模式走了另一条路径
//  ORB 单应验证三种方式视野一致（缩放 1.007~1.011），所以第 2 条路
//  拿到的是与标定完全相同的视野，不需要重新标定。
//
//  顺序有意为之：优先要"尺寸恰好是标定尺寸"的模式，
//  垂直裁切的那个放最后（需要重标定或改 cy）。
// ---------------------------------------------------------------------------
struct CamMode {
    int req_w, req_h;
    const char* note;
};

constexpr CamMode kCamModes[] = {
    {1280, 1024, "标准"},
    {1920, 1080, "驱动回落到 1280x1024"},
    {1280, 960,  "垂直裁切 64 行，需重标定"},
};
constexpr int kCamModeCount = static_cast<int>(sizeof(kCamModes) / sizeof(kCamModes[0]));

// --frames 之类用到；打开成功后由这里写出去，供启动信息打印
int g_cam_w = 0;
int g_cam_h = 0;
double g_cam_fps = 0.0;

// 量实际出帧速率。
//   必须用 read() 而不是 grab()：grab() 会直接从内部缓冲返回已有帧，
//   实测会算出 6.7e6 fps 这种离谱值，把 5fps 的坏模式误判成合格。
//   read() 要真正取到一帧才返回，才反映相机真实速率。
double measureReadFps(cv::VideoCapture& cap, int n) {
    cv::Mat tmp;
    for (int i = 0; i < 3; ++i) cap.read(tmp);      // 丢掉启动期的缓冲帧
    const auto t0 = std::chrono::high_resolution_clock::now();
    int got = 0;
    for (int i = 0; i < n; ++i) {
        if (cap.read(tmp) && !tmp.empty()) ++got;
    }
    const double dt = std::chrono::duration<double>(
                          std::chrono::high_resolution_clock::now() - t0).count();
    return dt > 1e-6 ? static_cast<double>(got) / dt : 0.0;
}

int openCamera(cv::VideoCapture& cap, int force_index, double exposure, double wb_temp,
               double gain) {
    for (int idx = 0; idx < 5; ++idx) {
        if (force_index >= 0 && idx != force_index) continue;

        for (int m = 0; m < kCamModeCount; ++m) {
            const CamMode& mode = kCamModes[m];
            cap.open(idx, cv::CAP_DSHOW);
            if (!cap.isOpened()) {
                cap.release();
                break;                          // 这个索引没有相机，换索引
            }

            cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', '2'));
            cap.set(cv::CAP_PROP_FRAME_WIDTH, mode.req_w);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, mode.req_h);
            cap.set(cv::CAP_PROP_FPS, 100);
            cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);
            cap.set(cv::CAP_PROP_EXPOSURE, exposure);
            cap.set(cv::CAP_PROP_AUTO_WB, 0);
            cap.set(cv::CAP_PROP_WB_TEMPERATURE, wb_temp);
            // 增益必须钉死。自动增益在暗处会把噪声放大，
            // 直接表现为 mask 里冒出一堆小碎块，面积过滤都救不回来。
            cap.set(cv::CAP_PROP_GAIN, gain);

            cv::Mat probe;
            cap >> probe;
            const int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
            const int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
            if (probe.empty() || h < 960) {
                std::cout << "  索引 " << idx << " 请求 " << mode.req_w << "x" << mode.req_h
                          << " 拿到 " << w << "x" << h << "，太小，跳过\n";
                cap.release();
                continue;
            }

            const double fps = measureReadFps(cap, 6);
            if (fps < 40.0) {
                // 这一条路能出图但跑不快（本机 1280x1024 直连只有 5fps）
                std::cout << "  索引 " << idx << " 请求 " << mode.req_w << "x" << mode.req_h
                          << " 只有 " << fps << " fps，换下一个模式\n";
                cap.release();
                continue;
            }

            g_cam_w = w;
            g_cam_h = h;
            g_cam_fps = fps;
            std::cout << "  相机就绪：索引 " << idx << "，请求 " << mode.req_w << "x"
                      << mode.req_h << " -> 实际 " << w << "x" << h << "，" << fps
                      << " fps（" << mode.note << "）\n";
            if (fps < 80.0) {
                // 实测本机：曝光 -6(15.6ms) 只有 63.8fps，-7/-8 有 126~129fps。
                // 曝光时间超过帧周期时帧率被直接封顶，而且不报错。
                std::cout << "  [警告] 帧率只有 " << fps
                          << " fps。如果这不是相机本身的上限，"
                          << "很可能是曝光时间超过了帧周期（曝光越大帧率越低），"
                          << "或是被其他 USB 设备抢了带宽。\n";
            }
            if (w != kWantW || h != kWantH) {
                std::cout << "  [注意] 实际分辨率 " << w << "x" << h << " 与标定用的 "
                          << kWantW << "x" << kWantH << " 不同，"
                          << "内参（cx/cy）需要重新标定！\n";
            }
            return idx;
        }
    }
    return -1;
}

}  // namespace

#ifndef AIM_NO_MAIN   // 由 ROS 节点以 #define AIM_NO_MAIN 后 #include 本文件时，跳过 main()
int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    bool use_camera = true;
    bool headless = false;
    bool show_raw = false;      // 纯相机画面（按 p 切换）
    int want_index = -1;
    int max_frames = 0;         // 0 = 不限；>0 时跑够这么多帧就退出（--frames）
    std::string color = "red";      // red / blue / yellow
    std::string video_path, save_path, log_path;
    std::string config_path = "config.txt";   // 默认读同目录下的 config.txt

    // 先扫一遍命令行，只为了找 --config：配置文件要最先读，
    // 这样后面的命令行参数才能覆盖它（优先级 命令行 > config > 默认值）
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--config" && i + 1 < argc) {
            config_path = argv[++i];
        }
    }

    Params P;
    if (loadParams(config_path, &P)) {
        // 读到了
    } else {
        std::cout << "[config] 未找到或为空: " << config_path << "，使用默认值\n";
    }

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--video" && i + 1 < argc) {
            use_camera = false;
            video_path = argv[++i];
        } else if (a == "--index" && i + 1 < argc) {
            want_index = std::stoi(argv[++i]);
        } else if (a == "--expo" && i + 1 < argc) {
            P.exposure = std::stod(argv[++i]);
        } else if (a == "--wb" && i + 1 < argc) {
            P.wb_temp = std::stod(argv[++i]);
        } else if (a == "--wheel" && i + 2 < argc) {
            P.wheel_x = std::stod(argv[++i]);
            P.wheel_y = std::stod(argv[++i]);
        } else if (a == "--rpx" && i + 1 < argc) {
            P.wheel_rpx = std::stod(argv[++i]);
        } else if (a == "--save" && i + 1 < argc) {
            headless = true;
            save_path = argv[++i];
        } else if (a == "--frames" && i + 1 < argc) {
            max_frames = std::stoi(argv[++i]);
        } else if (a == "--headless") {
            headless = true;      // 不显示窗口（自查脚本用，避免误弹出画面）
        } else if (a == "--raw") {
            show_raw = true;        // 启动时就是纯相机画面
        } else if (a == "--color" && i + 1 < argc) {
            color = argv[++i];
        } else if (a == "--log" && i + 1 < argc) {
            log_path = argv[++i];
        } else if (a == "--gain" && i + 1 < argc) {
            P.gain = std::stod(argv[++i]);
        } else if (a == "--roi" && i + 1 < argc) {
            P.roi_scale = std::stod(argv[++i]);
        } else if (a == "--abtol" && i + 1 < argc) {
            P.ab_tol = std::stod(argv[++i]);
        } else if (a == "--gate" && i + 1 < argc) {
            P.model_gate = std::stod(argv[++i]) * CV_PI / 180.0;   // 输入是度
        }
    }

    // 由相机仰角推出竖直补偿系数（透视压缩的逆）
    if (std::fabs(P.camera_pitch_deg) < 80.0) {
        P.y_scale = 1.0 / std::cos(P.camera_pitch_deg * CV_PI / 180.0);
    }
    printParams(P);

    cv::VideoCapture cap;
    if (use_camera) {
        int idx = openCamera(cap, want_index, P.exposure, P.wb_temp, P.gain);
        if (idx < 0) {
            std::cerr << "没找到能跑 1280x1024 的相机。\n"
                         "检查：1) USB 线插紧  2) 拔掉无线键鼠接收器（会抢带宽）"
                         "  3) 关掉占用相机的程序\n";
            return 1;
        }
        std::cout << "选中相机索引 " << idx << "\n";
    } else {
        cap.open(video_path);
        if (!cap.isOpened()) {
            std::cerr << "打不开视频: " << video_path << "\n";
            return 1;
        }
    }

    std::cout << "分辨率 " << cap.get(cv::CAP_PROP_FRAME_WIDTH) << "x"
              << cap.get(cv::CAP_PROP_FRAME_HEIGHT)
              << "   实测帧率 " << g_cam_fps
              << "   曝光 " << cap.get(cv::CAP_PROP_EXPOSURE)
              << "   色温 " << cap.get(cv::CAP_PROP_WB_TEMPERATURE) << "\n";

    // 颜色：比赛分边随机，红蓝都要能用
    const std::vector<HsvRange> ranges = rangesForColor(color);
    std::cout << "灯条颜色 " << color << "（" << ranges.size() << " 段 HSV 区间）\n";
    if (color != "red" && color != "blue" && color != "yellow") {
        std::cout << "[WARN] 未知颜色 \"" << color << "\"，已按红色处理\n";
    }

    std::cout << "按键：ESC/q 退出   p 切纯相机画面   s 截图\n";

    int idx = 0;
    bool warned_dark = false;
    cv::Point2f wheel(0.0f, 0.0f);
    bool wheel_set = false;
    PhaseTracker tracker(kSlotCount);
    std::vector<std::pair<double, double>> history;   // (t, theta)
    double last_valid_t = -1.0;   // 最后一次成功测到相位的时刻
    double data_age = 1e9;        // 当前数据有多旧（秒）
    MotionModel last_model;       // 上一帧的模型，给模型门限用
    double last_model_t = 0.0;
    bool has_last_model = false;
    int gate_rejects = 0;         // 被模型门限拒掉的帧数
    double r_est = -1.0;          // 自动估计的转盘半径（-1 = 还没估出来）
    bool mask_warned = false;     // 掩膜为空的警告已经报过了（避免刷屏）
    int slow_ticks = 0;           // 帧率看门狗：连续多少个 0.5 秒窗口偏低
    bool slow_warned = false;     // 已经报过，避免刷屏
    PoseResult last_pose;         // 最近一次成功的 PnP 位姿
    double last_pose_t = -1.0;
    int pose_reuse = 0;           // 沿用了多少次上次位姿

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 1.0) fps = 30.0;
    const double t_start = cv::getTickCount() / cv::getTickFrequency();

    // 实测帧率（驱动报的 fps 不一定准，尤其相机被抢带宽的时候）
    double fps_shown = 0.0;
    int frames_since = 0;
    double fps_t0 = cv::getTickCount() / cv::getTickFrequency();
    bool raw_window_open = false;
    const std::string win_name = "aim | " + color + " | frame+mask";

    // 逐帧日志：调试和技术文档的数据来源
    std::ofstream logf;
    if (!log_path.empty()) {
        logf.open(log_path);
        logf << "frame,t,theta_deg,w,rms_deg,pred_deg,armed_angle,plates,leds,"
                "mask_px,a,b,data_age,dist_mm,yaw_deg,pitch_cmd,reproj_px,lead_s,dtheta_deg,hit_y_mm,pred_minus_armed\n";
        std::cout << "日志写入 " << log_path << "\n";
    }

    // 分段计时（毫秒），结束时打印，用来找性能瓶颈
    using HrClock = std::chrono::high_resolution_clock;
    double t_mask = 0, t_detect = 0, t_slots = 0, t_phase = 0, t_fit = 0;
    double t_pnp = 0, t_pred = 0, t_draw = 0, t_total = 0;
    auto tick = []() { return HrClock::now(); };
    auto ms = [](HrClock::time_point a) {
        return std::chrono::duration<double, std::milli>(HrClock::now() - a).count();
    };
    bool did_save = false;     // --save 是否已经存过图
    cv::Mat last_both;         // 最后一帧的合成画面，供退出时补存
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // 实测帧率：每 0.5 秒刷新一次
        ++frames_since;
        {
            const double now = cv::getTickCount() / cv::getTickFrequency();
            if (now - fps_t0 >= 0.5) {
                fps_shown = frames_since / (now - fps_t0);
                frames_since = 0;
                fps_t0 = now;

                // ---- 运行时帧率看门狗 ----
                // 启动时的检查只能发现"一开始就慢"。现场更常见的是跑着跑着掉下去
                // （中途插了 USB 设备抢带宽、笔记本热降频），程序不会报错，
                // 但延迟翻几倍、命中率掉光。这里连续 2 秒偏低就提示一次。
                if (use_camera && fps_shown > 0.0) {
                    if (fps_shown < P.fps_warn_low) {
                        ++slow_ticks;
                        if (slow_ticks >= 4 && !slow_warned) {
                            std::cout << "[警告] 帧率持续低于 " << P.fps_warn_low
                                      << " fps（当前 " << fps_shown
                                      << "）。常见原因：中途插入了别的 USB 设备抢带宽、"
                                      << "曝光时间超过帧周期、笔记本降频。"
                                      << "此时延迟会明显变大，建议把 pipeline_delay 相应加大。\n";
                            slow_warned = true;
                        }
                    } else if (fps_shown > P.fps_warn_high) {
                        slow_ticks = 0;
                        slow_warned = false;   // 恢复了，重新布防
                    }
                }
            }
        }

        if (!wheel_set) {
            if (P.wheel_x < 0 || P.wheel_y < 0) {
                wheel = cv::Point2f(frame.cols * 0.5f, frame.rows * 0.5f);
            } else {
                wheel = cv::Point2f(static_cast<float>(P.wheel_x),
                                    static_cast<float>(P.wheel_y));
            }
            wheel_set = true;
            const double roi_r = P.roi_scale * P.wheel_rpx;
            std::cout << "轮心 (" << wheel.x << ", " << wheel.y
                      << ")   半径 " << P.wheel_rpx << " px"
                      << "   ROI " << (roi_r > 1.0 ? std::to_string((int)roi_r) + " px" : "关闭")
                      << "\n";
        }

        if (!warned_dark && cv::mean(frame)[0] < 5.0) {
            std::cout << "[WARN] 画面几乎全黑（平均灰度 < 5）\n"
                         "       可能原因：曝光太短 / 镜头盖没摘 / 环境太暗\n"
                         "       试试 --expo -6 或者开灯\n";
            warned_dark = true;
        }

        RejectStats st;
        cv::Mat hch;
        auto tp_frame = tick();
        auto tp = tick();
        cv::Mat mask = buildMask(frame, ranges, wheel, P.roi_scale * P.wheel_rpx, &hch, P);
        t_mask += ms(tp);
        const int mask_px = cv::countNonZero(mask);   // 记进日志，用来查灯光闪烁
        tp = tick();
        std::vector<Led> leds = detectLeds(mask, hch, wheel, P.wheel_rpx, P, &st);
        t_detect += ms(tp);

        // 自动估计转盘半径：灯条质心相对轮心的半径，中位数就约等于 R。
        //   4 条灯对称分布，平均半径正好是 R（径向两条在 R±47.5，切向两条在 R±3.2）
        // 这样换相机、换距离、甚至拿测试视频跑，都不用改 px_per_mm。
        // 只做缓慢收敛，避免单帧噪声让圆圈抖动。
        if (P.auto_rpx && leds.size() >= 4) {
            std::vector<double> rr;
            rr.reserve(leds.size());
            for (const auto& l : leds) {
                cv::Point2f d = l.center - wheel;
                rr.push_back(std::hypot(d.x, d.y));
            }
            std::nth_element(rr.begin(), rr.begin() + rr.size() / 2, rr.end());
            const double med = rr[rr.size() / 2];
            if (r_est < 0.0) {
                r_est = med;
            } else {
                r_est += 0.08 * (med - r_est);   // 一阶低通
            }
            P.wheel_rpx = r_est;
        }

        tp = tick();
        std::vector<Plate> plates = assignSlots(leds, wheel, P.wheel_rpx, P);
        t_slots += ms(tp);

        // 待激活靶必须唯一
        // 用 >= 4 而不是 == 4：噪点可能让一块靶聚出 5 条灯，
        // Python 版也是 n >= armed_bars，不能写死成等于。
        int armed = -1;
        int armed_count = 0;
        for (size_t k = 0; k < plates.size(); ++k) {
            if (plates[k].state >= 4) {
                if (armed < 0) armed = static_cast<int>(k);
                ++armed_count;
            }
        }
        // 一块 4 灯都没有时退一步：唯一一块 3 灯大概率是"4 灯被挡掉一条"。
        // 压力测试：只遮住一条灯条，原逻辑会直接找不到待激活靶，
        // 整条链路（相位、预测、PnP）全部失效。
        if (armed_count != 1) {
            armed = -1;
            int c3 = 0;
            for (size_t k = 0; k < plates.size(); ++k) {
                if (plates[k].state == 3) {
                    if (armed < 0) armed = static_cast<int>(k);
                    ++c3;
                }
            }
            if (c3 != 1) armed = -1;
        }

        // ---- ④ 相位解缠 ----
        double raw_phase = 0.0;
        tp = tick();
        bool has_phase = estimatePhase(plates, &raw_phase);
        t_phase += ms(tp);

        // 时间戳：视频按帧号/帧率，相机用真实时钟
        const double t_now = use_camera
                                 ? (cv::getTickCount() / cv::getTickFrequency() - t_start)
                                 : (idx / fps);

        // ---- 模型门限 ----
        // 用上一帧的模型外推本帧的 theta，对不上就作废这帧的相位，
        // 防止单帧误检把累加的 theta 永久带偏（PhaseTracker 只拦 >28.8 度的大跳）
        if (has_phase && has_last_model && last_valid_t >= 0.0 &&
            (t_now - last_valid_t) <= P.max_pred_age) {
            const double dt = t_now - last_model_t;
            const double th_pred =
                last_model.c[0] + last_model.c[1] * dt +
                last_model.c[2] * std::cos(last_model.omega * dt) +
                last_model.c[3] * std::sin(last_model.omega * dt);
            const double candidate = tracker.peek(has_phase, raw_phase);
            if (std::fabs(candidate - th_pred) > P.model_gate) {
                has_phase = false;      // 这帧不可信，本帧不更新 theta
                ++gate_rejects;
            }
        }

        double theta = tracker.update(has_phase, raw_phase);

        if (has_phase && tracker.has()) {
            history.emplace_back(t_now, theta);
            last_valid_t = t_now;
            data_age = 0.0;
        } else {
            // 丢帧：数据年龄随时间增长
            data_age = (last_valid_t < 0.0) ? 1e9 : (t_now - last_valid_t);
        }
        // 丢帧太久，历史已经不可信（转盘可能换了模式/被复位），清空重新积累
        if (data_age > P.history_reset_age) {
            history.clear();
            has_last_model = false;     // 模型也一起作废
            tracker.reset();            // 相位连续性也丢掉，否则大跳变会永久卡死
        }
        // 顺便剪掉过老的样本，防止内存无限增长
        while (!history.empty() && history.front().first < t_now - 3.0 * P.fit_window_s) {
            history.erase(history.begin());
        }

        // ---- ⑤ 拟合（用当前时刻做时间原点，不用最后一个样本的时间）----
        MotionModel model;
        tp = tick();
        bool has_model = fitMotion(history, t_now, P, &model);
        t_fit += ms(tp);
        if (has_model) {
            last_model = model;
            last_model_t = t_now;
            has_last_model = true;
        }

        // ---- ⑧ 位姿解算 + 命中点迭代 ----
        // 只用"待激活靶"的 4 条灯做 PnP：2 条灯点不够，解不出来。
        PoseResult pose;
        HitSolution hit;
        double lead_total = P.lead_s;        // 没有 PnP 时的兜底
        double pitch_cmd_deg = 0.0;
        double yaw_cmd_deg = 0.0;
        if (armed >= 0 && plates[armed].bars.size() == 4) {
            std::vector<cv::Point2f> pts;
            pts.reserve(4);
            for (int bi : plates[armed].bars) pts.push_back(leds[bi].geom);   // 用补偿后的点，否则倾斜会让深度偏

            tp = tick();
            pose = solvePlatePose(pts, P);
            t_pnp += ms(tp);

            // 遮挡兜底：IPPE 必须 4 个共面点，被挡掉一条就解不出来。
            // 此时沿用最近一次成功的位姿（限 0.2 秒内）。靶的距离在几十毫秒内
            // 几乎不变，沿用比整帧丢掉瞄准解要好得多。
            if (!pose.ok && last_pose.ok && last_pose_t >= 0.0 &&
                (t_now - last_pose_t) < kPoseReuseAge) {
                pose = last_pose;
                ++pose_reuse;
            }
            if (pose.ok) {
                last_pose = pose;
                last_pose_t = t_now;
            }

            // 命中时刻的靶位置：迭代求解 pitch ⇄ 飞行时间 的循环依赖，
            // 同时把 yaw 也从命中位置算出来（原来 yaw 用的是当前方位，没有提前量）
            hit = solveHit(pose, wheel, plates[armed].center, has_model ? &model : nullptr, P);
            if (hit.ok) {
                yaw_cmd_deg = hit.yaw_deg;
                pitch_cmd_deg = hit.pitch_deg;
                lead_total = hit.lead;
            }
        }

        // ---- ⑥ 预测 ----
        // 数据过期就不预测了：模型是按几百毫秒前拟合的，外推出去是瞎猜
        bool fresh = (data_age <= P.max_pred_age);
        bool ready = false;
        double pred = 0.0;
        if (has_model && fresh && tracker.has() && armed >= 0) {
            pred = predictTarget(model, theta,
                                 plates[armed].angle_deg * CV_PI / 180.0,
                                 lead_total) * 180.0 / CV_PI;
            ready = true;
        }

        // ---- 逐帧日志 ----
        if (logf.is_open()) {
            logf << idx << ',' << t_now << ','
                 << (tracker.has() ? tracker.thetaDeg() : 0.0) << ','
                 << (has_model ? model.omega : 0.0) << ','
                 << (has_model ? model.rms * 180.0 / CV_PI : 0.0) << ','
                 << (ready ? pred : 0.0) << ','
                 << (armed >= 0 ? plates[armed].angle_deg : 0.0) << ','
                 << plates.size() << ',' << leds.size() << ',' << mask_px << ','
                 << (has_model ? model.a : 0.0) << ','
                 << (has_model ? model.b : 0.0) << ',' << data_age << ','
                 << (pose.ok ? pose.dist_mm : 0.0) << ','
                 << (pose.ok ? pose.yaw_deg : 0.0) << ','
                 << (pose.ok ? pitch_cmd_deg : 0.0) << ','
                 << (pose.ok ? pose.reproj_px : 0.0) << ','
                 << lead_total << ','
                 << hit.dtheta_deg << ','
                 << (hit.ok ? hit.p_hit.y : 0.0) << ','
                 << (ready && armed >= 0 ? pred - plates[armed].angle_deg : 0.0)
                 << '\n';
        }

        // 掩膜的主色相（诊断用）。换相机/换环境时先看这个数对不对：
        //   红方应接近 0 或 179，蓝方应接近 103~112，黄色应接近 30。
        // 对不上说明 HSV 区间没覆盖住，灯条会整条检不出来（而拒绝计数器全是 0）。
        double mask_hue = -1.0;
        if (idx % 30 == 0 && mask_px > 0 && !hch.empty()) {
            double sx = 0.0, sy = 0.0;
            int cnt = 0;
            for (int y = 0; y < mask.rows; ++y) {
                const uchar* mrow = mask.ptr<uchar>(y);
                const uchar* hrow = hch.ptr<uchar>(y);
                for (int x = 0; x < mask.cols; ++x) {
                    if (!mrow[x]) continue;
                    const double a = hrow[x] * 2.0 * CV_PI / 180.0;
                    sx += std::cos(a);
                    sy += std::sin(a);
                    ++cnt;
                }
            }
            if (cnt > 0) {
                double d = std::atan2(sy, sx) * 180.0 / CV_PI;
                if (d < 0.0) d += 360.0;
                mask_hue = d / 2.0;   // 换回 OpenCV 的 0~179 刻度
            }
        }

        // 掩膜是空的 = 一个灯条都没进掩膜。最常见的原因就是 --color 配错。
        // 这里统计整幅图里"亮像素"的主色相，直接提示该用哪个颜色。
        // 掩膜恢复了就复位告警标志，下次再空时重新提示一次
        if (mask_px > 0) mask_warned = false;

        if (mask_px == 0 && idx % 30 == 0) {
            cv::Mat hsv_all;
            cv::cvtColor(frame, hsv_all, cv::COLOR_BGR2HSV);
            double bx = 0.0, by = 0.0;
            int cnt = 0;
            for (int y = 0; y < hsv_all.rows; y += 2) {
                const uchar* hr = hsv_all.ptr<uchar>(y);
                for (int x = 0; x < hsv_all.cols; x += 2) {
                    const uchar H = hr[x * 3], S = hr[x * 3 + 1], V = hr[x * 3 + 2];
                    if (V < 110 || S < 50) continue;   // 只看又亮又有色的像素
                    const double a = H * 2.0 * CV_PI / 180.0;
                    bx += std::cos(a);
                    by += std::sin(a);
                    ++cnt;
                }
            }
            if (cnt > 50) {
                double d = std::atan2(by, bx) * 180.0 / CV_PI;
                if (d < 0.0) d += 360.0;
                const double H = d / 2.0;   // 回到 0~179 刻度
                const char* guess = "未知";
                if (H < 13.0 || H > 166.0) guess = "red";
                else if (H >= 13.0 && H <= 46.0) guess = "yellow";
                else if (H >= 88.0 && H <= 142.0) guess = "blue";
                // 只在"刚进入掩膜为空"这个状态时报一次。
                // 原来每 30 帧刷一遍，120fps 下每秒 4 次，控制台 IO 会拖慢主循环。
                if (!mask_warned) {
                    std::printf("[警告] 掩膜为空！画面亮部主色相 H=%.0f，看起来是 %s\n",
                                H, guess);
                    if (color != guess) {
                        std::printf("[警告] 当前位置 --color %s 与该颜色不符，试试 --color %s\n",
                                    color.c_str(), guess);
                    } else {
                        std::printf("[警告] 颜色参数是对的，说明是饱和度/亮度阈值卡住了，"
                                    "把 S、V 下限调低试试\n");
                    }
                    mask_warned = true;
                }
            } else {
                if (!mask_warned) {
                    std::printf("[警告] 掩膜为空，且画面里几乎没有又亮又有色的像素"
                                "（检查曝光/增益/镜头盖）\n");
                    mask_warned = true;
                }
            }
        }

        if (idx % 30 == 0) {
            std::printf("frame %d  R=%.1f  轮廓 %d → 面积 %d 长宽比 %d 填充度 %d "
                        "色相 %d 半径 %d → 灯条 %d  靶 %d",
                        idx, P.wheel_rpx, st.raw, st.by_area, st.by_aspect, st.by_extent,
                        st.by_hue, st.by_radius, static_cast<int>(leds.size()),
                        static_cast<int>(plates.size()));
            if (tracker.has()) std::printf("  theta %.1f", tracker.thetaDeg());
            if (has_model) {
                std::printf("  w=%.3f a=%.3f b=%.3f rms=%.2f", model.omega, model.a,
                            model.b, model.rms * 180.0 / CV_PI);
            }
            if (data_age > P.max_pred_age) {
                std::printf("  age=%.0fms  HOLD", data_age * 1000.0);
            } else if (ready) {
                std::printf("  age=%.0fms  pred %.1f  READY", data_age * 1000.0, pred);
            } else {
                std::printf("  age=%.0fms  HOLD", data_age * 1000.0);
            }
            std::printf("  gate拒 %d  maskH %.0f", gate_rejects, mask_hue);
            if (pose_reuse > 0) std::printf("  位姿沿用 %d", pose_reuse);
            std::printf("\n");
        }

        // 纯相机模式（比赛用）不需要合成图，也不需要任何叠加绘制。
        // 这一段原本每帧要 cvtColor + hconcat 出一张 2560x960 的图，实测 1.7ms。
        cv::Mat both;
        if (show_raw) {
            last_both = frame;      // 这种模式下 --save 存的就是原图
        } else {
        cv::Mat mask_bgr;
        cv::cvtColor(mask, mask_bgr, cv::COLOR_GRAY2BGR);
        tp = tick();
        cv::hconcat(frame, mask_bgr, both);
        t_draw += ms(tp);
        last_both = both;

        for (size_t k = 0; k < leds.size(); ++k) {
            const cv::Point2f& p = leds[k].center;
            cv::circle(both, p, 5, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::circle(both, p + cv::Point2f(static_cast<float>(frame.cols), 0.0f),
                       5, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }

        for (int half = 0; half < 2; ++half) {
            cv::Point2f off(static_cast<float>(half) * frame.cols, 0.0f);
            cv::Point2f wc = wheel + off;
            cv::drawMarker(both, wc, cv::Scalar(255, 0, 255), cv::MARKER_CROSS, 24, 2);
            cv::circle(both, wc, static_cast<int>(P.wheel_rpx), cv::Scalar(90, 90, 90), 1,
                       cv::LINE_AA);
            // 槽位射线：槽位在 theta + i×72° 上，所以必须从 theta 起画，
            // 否则五条线固定不动，跟转动的靶子对不上。
            const double slot_base = tracker.has() ? tracker.thetaDeg() : 0.0;
            for (int s = 0; s < kSlotCount; ++s) {
                double a = (slot_base + s * (360.0 / kSlotCount)) * CV_PI / 180.0;
                cv::Point2f e(wc.x + static_cast<float>(P.wheel_rpx * std::cos(a)),
                              wc.y + static_cast<float>(P.wheel_rpx * std::sin(a)));
                cv::line(both, wc, e, cv::Scalar(70, 70, 70), 1, cv::LINE_AA);
            }
        }

        for (size_t k = 0; k < plates.size(); ++k) {
            bool is_armed = (static_cast<int>(k) == armed);
            cv::Scalar col = is_armed ? cv::Scalar(0, 0, 255) : cv::Scalar(255, 160, 0);
            for (int half = 0; half < 2; ++half) {
                cv::Point2f off(static_cast<float>(half) * frame.cols, 0.0f);
                cv::Point2f c = plates[k].center + off;
                cv::circle(both, c, 14, col, 2, cv::LINE_AA);
                char tag[64];
                std::snprintf(tag, sizeof(tag), "%d bars %.1f",
                              plates[k].state, plates[k].angle_deg);
                cv::putText(both, tag, c + cv::Point2f(16, 6),
                            cv::FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv::LINE_AA);
            }
        }

        // 命中点：直接把 solveHit 算出的命中时刻 3D 位置投影回图像画标记。
        // 这样标记画的就是"真正要打的那一点"，不再是当前靶心。
        if (hit.ok && P.fx > 1.0) {
            const double zz = hit.p_hit.z;
            if (zz > 1.0) {
                const float u = static_cast<float>(P.fx * hit.p_hit.x / zz + P.cx);
                const float v = static_cast<float>(P.fy * hit.p_hit.y / zz + P.cy);
                for (int half = 0; half < 2; ++half) {
                    cv::Point2f off(static_cast<float>(half) * frame.cols, 0.0f);
                    cv::drawMarker(both, cv::Point2f(u, v) + off, cv::Scalar(0, 255, 255),
                                   cv::MARKER_DIAMOND, 22, 2);
                }
            }
        } else if (ready && armed >= 0) {
            cv::Point2f rel = plates[armed].center - wheel;
            const double r_now = std::hypot(rel.x, rel.y);
            const double ang_now = std::atan2(rel.y, rel.x);
            const double d_ang = (pred - plates[armed].angle_deg) * CV_PI / 180.0;
            for (int half = 0; half < 2; ++half) {
                cv::Point2f off(static_cast<float>(half) * frame.cols, 0.0f);
                cv::Point2f p(wheel.x + static_cast<float>(r_now * std::cos(ang_now + d_ang)) + off.x,
                              wheel.y + static_cast<float>(r_now * std::sin(ang_now + d_ang)) + off.y);
                cv::drawMarker(both, p, cv::Scalar(0, 255, 255), cv::MARKER_DIAMOND, 22, 2);
            }
        }

        char text[256];
        if (ready) {
            std::snprintf(text, sizeof(text),
                          "frame %d  plates %d  w=%.3f  rms=%.2f  pred %.1f  READY",
                          idx, static_cast<int>(plates.size()), model.omega,
                          model.rms * 180.0 / CV_PI, pred);
        } else {
            std::snprintf(text, sizeof(text),
                          "frame %d  plates %d  theta %.1f  HOLD",
                          idx, static_cast<int>(plates.size()), tracker.thetaDeg());
        }
        cv::putText(both, text, cv::Point(15, 35), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    ready ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 200, 255), 2,
                    cv::LINE_AA);

        // 实测帧率：掉到 50 以下通常是被别的 USB 设备抢了带宽
        char fz[128];
        std::snprintf(fz, sizeof(fz), "color %s   fps %.1f  %s", color.c_str(),
                      fps_shown,
                      (fps_shown > 0.0 && fps_shown < 50.0) ? "(LOW!)" : "");
        cv::putText(both, fz, cv::Point(15, 105), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    (fps_shown > 0.0 && fps_shown < 50.0) ? cv::Scalar(0, 0, 255)
                                                          : cv::Scalar(200, 255, 200),
                    2, cv::LINE_AA);

        char th[160];
        if (armed >= 0) {
            std::snprintf(th, sizeof(th), "theta %+9.2f   armed %.1f",
                          tracker.thetaDeg(), plates[armed].angle_deg);
        } else {
            std::snprintf(th, sizeof(th), "theta %+9.2f   no unique armed plate",
                          tracker.thetaDeg());
        }
        cv::putText(both, th, cv::Point(15, 70), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

        // 位姿与弹道
        char pq[192];
        if (pose.ok) {
            std::snprintf(pq, sizeof(pq),
                          "d=%.0fmm  yaw=%+.2f  pitch=%.2f  lead=%.3fs  rep=%.2fpx",
                          pose.dist_mm, yaw_cmd_deg, pitch_cmd_deg, lead_total,
                          pose.reproj_px);
        } else {
            std::snprintf(pq, sizeof(pq), "PnP invalid (need 4 bars)");
        }
        cv::putText(both, pq, cv::Point(15, 140), cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    pose.ok ? cv::Scalar(255, 200, 0) : cv::Scalar(0, 128, 255), 2,
                    cv::LINE_AA);
        }   // end if (!show_raw)

        // ---- 显示 ----
        //   默认显示"原图 | 掩膜"复合画面
        //   按 p 切到纯相机画面（比赛/录像时用，就看镜头拍到什么）
        if (!headless) {
            if (show_raw) {
                if (!raw_window_open) {
                    cv::namedWindow("camera", cv::WINDOW_NORMAL);
                    cv::resizeWindow("camera", 960, 768);
                    raw_window_open = true;
                }
                cv::imshow("camera", frame);        // 纯画面，不叠任何东西
            } else {
                if (raw_window_open) {
                    cv::destroyWindow("camera");
                    raw_window_open = false;
                }
                cv::imshow(win_name, both);
            }
        }
        t_total += ms(tp_frame);
        ++idx;

        // --frames 必须在两种模式下都生效。
        // 之前只在 headless 分支里检查，带窗口运行时该参数被完全忽略，
        // 导致自查脚本以为只跑 1 帧、实际把整段视频放完了（窗口一直开着）。
        if (max_frames > 0 && idx >= max_frames) {
            if (!save_path.empty()) {
                const cv::Mat& out_img = show_raw ? frame : both;
                if (saveImage(save_path, out_img)) {
                    std::cout << "已存 " << save_path << "\n";
                } else {
                    std::cout << "存图失败: " << save_path << "\n";
                }
                did_save = true;
            }
            break;
        }

        if (headless) {
            continue;
        }

        int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') break;
        if (key == 'p' || key == 'P') show_raw = !show_raw;
        if (key == 's' || key == 'S') {          // 按 s 存一张当前画面
            char name[64];
            std::snprintf(name, sizeof(name), "shot_%04d.png", idx);
            if (!saveImage(name, show_raw ? frame : both)) {
                std::cout << "截图失败\n";
            }
            std::cout << "已截图 " << name << "\n";
        }
    }

    // 走到这里说明视频读完了或被 ESC 提前退出。
    // 如果指定了 --save 但还没在帧数上限处存过，就把最后一帧存下来，
    // 这样"只给 --save 不给 --frames"也能得到一张图。
    if (!save_path.empty() && !did_save && !last_both.empty()) {
        if (saveImage(save_path, last_both)) {
            std::cout << "已存 " << save_path << "\n";
        } else {
            std::cout << "存图失败: " << save_path << "\n";
        }
    }

    cap.release();
    cv::destroyAllWindows();
    if (idx > 0) {
        const double n = static_cast<double>(idx);
        std::printf("[耗时] 平均每帧 %.2f ms (=%.0f fps)\n", t_total / n, 1000.0 * n / t_total);
        std::printf("[耗时]   掩膜 %.2f  灯条 %.2f  归组 %.2f  相位 %.2f  拟合 %.2f  "
                    "PnP %.2f  绘制 %.2f\n",
                    t_mask / n, t_detect / n, t_slots / n, t_phase / n, t_fit / n,
                    t_pnp / n, t_draw / n);
    }
    std::cout << "处理了 " << idx << " 帧\n";
    return 0;
}
#endif  // AIM_NO_MAIN
