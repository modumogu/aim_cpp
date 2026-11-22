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
    std::printf("[参数] wheel=(%.0f,%.0f) px_per_mm=%.4f R_px=%.1f roi=%.2f\n",
                P.wheel_x, P.wheel_y, P.px_per_mm, P.wheel_rpx, P.roi_scale);
    std::printf("[参数] area=[%.0f,%.0f] aspect>=%.2f extent>=%.2f hue_sd<=%.1f\n",
                P.min_area, P.max_area, P.min_aspect, P.min_extent, P.max_hue_std);
    std::printf("[参数] gap=%.1fdeg radius_tol=%.2f\n", P.group_gap_deg, P.radius_tol);
    std::printf("[参数] fit: win=%.2fs n>=%d w=[%.3f,%.3f]/%.3f rms<=%.3f osc>=%.2f\n",
                P.fit_window_s, P.min_fit_samples, P.omega_min, P.omega_max,
                P.omega_step, P.max_fit_rms, P.min_osc_amp);
    std::printf("[参数] lead=%.3fs age<=%.3fs reset=%.1fs gate=%.2fdeg ab_tol=%.3f\n",
                P.lead_s, P.max_pred_age, P.history_reset_age,
                P.model_gate * 180.0 / CV_PI, P.ab_tol);
}

// 三种颜色各自的 HSV 区间。
//   红色：H 跨 0 和 179 两端，必须两段
//   蓝色：H≈120 一段就行
//   黄色：H≈30 一段；实测黄色 LED 饱和度偏低（S≈73），所以 S 下限放到 50
// 比赛分边是随机的，红蓝都要能用，所以必须可切换而不是写死。
std::vector<HsvRange> rangesForColor(const std::string& color) {
    std::vector<HsvRange> v;
    if (color == "blue") {
        v.push_back({cv::Scalar(100, 120, 120), cv::Scalar(130, 255, 255)});
    } else if (color == "yellow") {
        v.push_back({cv::Scalar(15, 50, 150), cv::Scalar(45, 255, 255)});
    } else {   // red（默认）
        v.push_back({cv::Scalar(0, 120, 120), cv::Scalar(10, 255, 255)});
        v.push_back({cv::Scalar(170, 120, 120), cv::Scalar(179, 255, 255)});
    }
    return v;
}

// ============================================================================
//  ① 颜色掩膜
//     ROI 圆：只保留转盘周围的像素，顶灯/墙面反光在圆外，一步排掉。
//     同时把 H 通道回传出去，后面算色相一致性要用。
// ============================================================================
cv::Mat buildMask(const cv::Mat& bgr, const std::vector<HsvRange>& ranges,
                  cv::Point2f wheel, double roi_radius, cv::Mat* h_out) {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask = cv::Mat::zeros(hsv.size(), CV_8UC1);
    for (const auto& r : ranges) {
        cv::Mat m;
        cv::inRange(hsv, r.lo, r.hi, m);
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
    cv::Point2f center;
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

        // 半径检查放在这里（而不是等到槽位归组）：
        // 离轮心太远的红色块一定不是灯条，早点扔掉还能少一次聚簇运算。
        cv::Point2f d = led.center - wheel;
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
        cv::Point2f d = leds[i].center - wheel;
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
                c += leds[it.idx].center;
            }
            c *= 1.0f / static_cast<float>(cur.size());
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

// ============================================================================
//  打开相机
// ============================================================================
int openCamera(cv::VideoCapture& cap, int force_index, double exposure, double wb_temp,
               double gain) {
    for (int idx = 0; idx < 5; ++idx) {
        if (force_index >= 0 && idx != force_index) continue;

        cap.open(idx, cv::CAP_DSHOW);
        if (!cap.isOpened()) {
            cap.release();
            continue;
        }

        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('Y', 'U', 'Y', '2'));
        cap.set(cv::CAP_PROP_FRAME_WIDTH, kWantW);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, kWantH);
        cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);
        cap.set(cv::CAP_PROP_EXPOSURE, exposure);
        cap.set(cv::CAP_PROP_AUTO_WB, 0);
        cap.set(cv::CAP_PROP_WB_TEMPERATURE, wb_temp);
        // 增益必须钉死。自动增益在暗处会把噪声放大，
        // 直接表现为 mask 里冒出一堆小碎块，面积过滤都救不回来。
        cap.set(cv::CAP_PROP_GAIN, gain);

        cv::Mat probe;
        cap >> probe;
        int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        if (!probe.empty() && w == kWantW && h == kWantH) {
            return idx;
        }
        std::cout << "  索引 " << idx << " 不行（" << w << "x" << h << "），跳过\n";
        cap.release();
    }
    return -1;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif

    bool use_camera = true;
    bool headless = false;
    bool show_raw = false;      // 纯相机画面（按 p 切换）
    int want_index = -1;
    int max_frames = 1;
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
              << "   帧率 " << cap.get(cv::CAP_PROP_FPS)
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
                "mask_px,a,b,data_age\n";
        std::cout << "日志写入 " << log_path << "\n";
    }

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
        cv::Mat mask = buildMask(frame, ranges, wheel, P.roi_scale * P.wheel_rpx, &hch);
        const int mask_px = cv::countNonZero(mask);   // 记进日志，用来查灯光闪烁
        std::vector<Led> leds = detectLeds(mask, hch, wheel, P.wheel_rpx, P, &st);
        std::vector<Plate> plates = assignSlots(leds, wheel, P.wheel_rpx, P);

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
        if (armed_count != 1) armed = -1;

        // ---- ④ 相位解缠 ----
        double raw_phase = 0.0;
        bool has_phase = estimatePhase(plates, &raw_phase);

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
        }
        // 顺便剪掉过老的样本，防止内存无限增长
        while (!history.empty() && history.front().first < t_now - 3.0 * P.fit_window_s) {
            history.erase(history.begin());
        }

        // ---- ⑤ 拟合（用当前时刻做时间原点，不用最后一个样本的时间）----
        MotionModel model;
        bool has_model = fitMotion(history, t_now, P, &model);
        if (has_model) {
            last_model = model;
            last_model_t = t_now;
            has_last_model = true;
        }

        // ---- ⑥ 预测 ----
        // 数据过期就不预测了：模型是按几百毫秒前拟合的，外推出去是瞎猜
        bool fresh = (data_age <= P.max_pred_age);
        bool ready = false;
        double pred = 0.0;
        if (has_model && fresh && tracker.has() && armed >= 0) {
            pred = predictTarget(model, theta,
                                 plates[armed].angle_deg * CV_PI / 180.0,
                                 P.lead_s) * 180.0 / CV_PI;
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
                 << (has_model ? model.b : 0.0) << ',' << data_age << '\n';
        }

        if (idx % 30 == 0) {
            std::printf("frame %d  轮廓 %d → 面积 %d 长宽比 %d 填充度 %d 色相 %d 半径 %d"
                        " → 灯条 %d  靶 %d",
                        idx, st.raw, st.by_area, st.by_aspect, st.by_extent, st.by_hue,
                        st.by_radius, static_cast<int>(leds.size()),
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
            std::printf("  gate拒 %d\n", gate_rejects);
        }

        cv::Mat mask_bgr;
        cv::cvtColor(mask, mask_bgr, cv::COLOR_GRAY2BGR);
        cv::Mat both;
        cv::hconcat(frame, mask_bgr, both);

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
            for (int s = 0; s < kSlotCount; ++s) {
                double a = s * (360.0 / kSlotCount) * CV_PI / 180.0;
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

        // 预测位置：在圆周上画一个小方块，直观看到"打向哪"
        if (ready) {
            for (int half = 0; half < 2; ++half) {
                cv::Point2f off(static_cast<float>(half) * frame.cols, 0.0f);
                double a = pred * CV_PI / 180.0;
                cv::Point2f p(wheel.x + static_cast<float>(P.wheel_rpx * std::cos(a)) + off.x,
                              wheel.y + static_cast<float>(P.wheel_rpx * std::sin(a)) + off.y);
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
        ++idx;

        if (headless) {
            if (idx >= max_frames) {
                cv::imwrite(save_path, both);
                std::cout << "已存 " << save_path << "\n";
                break;
            }
            continue;
        }

        int key = cv::waitKey(1) & 0xFF;
        if (key == 27 || key == 'q') break;
        if (key == 'p' || key == 'P') show_raw = !show_raw;
        if (key == 's' || key == 'S') {          // 按 s 存一张当前画面
            char name[64];
            std::snprintf(name, sizeof(name), "shot_%04d.png", idx);
            cv::imwrite(name, show_raw ? frame : both);
            std::cout << "已截图 " << name << "\n";
        }
    }

    cap.release();
    cv::destroyAllWindows();
    std::cout << "处理了 " << idx << " 帧\n";
    return 0;
}
