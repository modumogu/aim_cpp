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
// ============================================================================

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
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

// ---- 灯条形状的物理范围（1280x1024、2 米距离下的经验值）---------------------
constexpr double kMinLedArea = 20.0;    // 小于这个：噪点
constexpr double kMaxLedArea = 2500.0;  // 大于这个：光斑 / 整片红色
constexpr double kMinAspect = 1.3;      // 长宽比，滤掉圆形的反光点

// ---- 转盘几何 --------------------------------------------------------------
constexpr int kSlotCount = 5;                     // 5 个靶位，间隔 72°
constexpr double kPxPerMm = 0.8720;               // 标定得到：fx 1700 / 1950mm
constexpr double kDefWheelRpx = 350.0 * kPxPerMm; // ≈ 305 像素
constexpr double kGroupGapDeg = 15.0;             // 角度聚类阈值（远小于 72°）
constexpr double kRadiusTol = 0.45;               // 半径容差，±45%

// ---- 拟合参数 --------------------------------------------------------------
constexpr double kFitWindowS = 1.5;    // 只用最近 1.5 秒的数据
constexpr int kMinFitSamples = 20;     // 至少 20 个采样点才拟合
constexpr double kOmegaMin = 1.884;    // 手册 5.5.2
constexpr double kOmegaMax = 2.000;
constexpr double kOmegaStep = 0.004;
constexpr double kMaxFitRms = 0.05;    // 弧度，约 3°；超过就 HOLD
constexpr double kLeadS = 0.10;        // 预测提前量：100ms

// ---- 颜色阈值 --------------------------------------------------------------
// 红色在 HSV 的 H 上跨 0 和 179 两端，所以要两段。
struct HsvRange {
    cv::Scalar lo;
    cv::Scalar hi;
};

const HsvRange kRedRanges[] = {
    {cv::Scalar(0, 120, 120), cv::Scalar(10, 255, 255)},
    {cv::Scalar(170, 120, 120), cv::Scalar(179, 255, 255)},
};

// ============================================================================
//  ① 颜色掩膜
// ============================================================================
cv::Mat buildRedMask(const cv::Mat& bgr) {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask = cv::Mat::zeros(hsv.size(), CV_8UC1);
    for (const auto& r : kRedRanges) {
        cv::Mat m;
        cv::inRange(hsv, r.lo, r.hi, m);
        cv::bitwise_or(mask, m, mask);
    }

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    return mask;
}

// ============================================================================
//  ② 灯条
// ============================================================================
struct Led {
    cv::Point2f center;
    double area = 0.0;
    double aspect = 0.0;
};

std::vector<Led> detectLeds(const cv::Mat& mask, int* raw_count) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (raw_count) *raw_count = static_cast<int>(contours.size());

    std::vector<Led> leds;
    for (const auto& c : contours) {
        double area = cv::contourArea(c);
        if (area < kMinLedArea || area > kMaxLedArea) continue;

        cv::RotatedRect rr = cv::minAreaRect(c);
        double w = rr.size.width, h = rr.size.height;
        double lo = std::min(w, h), hi = std::max(w, h);
        if (lo <= 0.0) continue;
        double aspect = hi / lo;
        if (aspect < kMinAspect) continue;

        cv::Moments M = cv::moments(c);
        if (M.m00 <= 0.0) continue;

        Led led;
        led.center = cv::Point2f(static_cast<float>(M.m10 / M.m00),
                                 static_cast<float>(M.m01 / M.m00));
        led.area = area;
        led.aspect = aspect;
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
                               double R_px) {
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
        if (items[k].ang - items[k - 1].ang < kGroupGapDeg) {
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

    double update(bool has_raw, double raw) {
        if (!has_raw) return theta_;
        if (!has_theta_) {
            theta_ = raw;
            last_raw_ = raw;
            has_theta_ = true;
            return theta_;
        }

        const double two = 2.0 * half_slot_;
        double d = std::fmod(raw - last_raw_ + half_slot_, two);
        if (d < 0.0) d += two;
        d -= half_slot_;

        if (std::fabs(d) > 0.8 * half_slot_) {
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

bool fitMotion(const std::vector<std::pair<double, double>>& history, MotionModel* out) {
    if (static_cast<int>(history.size()) < kMinFitSamples) return false;

    const double t_now = history.back().first;
    std::vector<double> ts, ys;
    for (const auto& p : history) {
        if (p.first >= t_now - kFitWindowS) {
            ts.push_back(p.first - t_now);   // 以"现在"为原点，数值更稳
            ys.push_back(p.second);
        }
    }
    if (static_cast<int>(ts.size()) < kMinFitSamples) return false;

    double best_rms = 1e18;
    double best_w = 0.0;
    double best_c[4] = {0.0, 0.0, 0.0, 0.0};

    // ω 网格搜索：每个候选解一次最小二乘，留残差最小的
    for (double w = kOmegaMin; w <= kOmegaMax + 1e-9; w += kOmegaStep) {
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

    if (best_rms > kMaxFitRms) return false;   // 不可信 -> HOLD

    out->omega = best_w;
    out->rms = best_rms;
    for (int k = 0; k < 4; ++k) out->c[k] = best_c[k];
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
int openCamera(cv::VideoCapture& cap, int force_index, double exposure, double wb_temp) {
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
    double exposure = kDefExposure;
    double wb_temp = kDefWbTemp;
    double wheel_x = -1.0;
    double wheel_y = -1.0;
    double R_px = kDefWheelRpx;
    std::string video_path, save_path;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--video" && i + 1 < argc) {
            use_camera = false;
            video_path = argv[++i];
        } else if (a == "--index" && i + 1 < argc) {
            want_index = std::stoi(argv[++i]);
        } else if (a == "--expo" && i + 1 < argc) {
            exposure = std::stod(argv[++i]);
        } else if (a == "--wb" && i + 1 < argc) {
            wb_temp = std::stod(argv[++i]);
        } else if (a == "--wheel" && i + 2 < argc) {
            wheel_x = std::stod(argv[++i]);
            wheel_y = std::stod(argv[++i]);
        } else if (a == "--rpx" && i + 1 < argc) {
            R_px = std::stod(argv[++i]);
        } else if (a == "--save" && i + 1 < argc) {
            headless = true;
            save_path = argv[++i];
        } else if (a == "--frames" && i + 1 < argc) {
            max_frames = std::stoi(argv[++i]);
        } else if (a == "--raw") {
            show_raw = true;        // 启动时就是纯相机画面
        }
    }

    cv::VideoCapture cap;
    if (use_camera) {
        int idx = openCamera(cap, want_index, exposure, wb_temp);
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
    std::cout << "ESC / q 退出\n";

    int idx = 0;
    bool warned_dark = false;
    cv::Point2f wheel(0.0f, 0.0f);
    bool wheel_set = false;
    PhaseTracker tracker(kSlotCount);
    std::vector<std::pair<double, double>> history;   // (t, theta)

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 1.0) fps = 30.0;
    const double t_start = cv::getTickCount() / cv::getTickFrequency();

    // 实测帧率（驱动报的 fps 不一定准，尤其相机被抢带宽的时候）
    double fps_shown = 0.0;
    int frames_since = 0;
    double fps_t0 = cv::getTickCount() / cv::getTickFrequency();
    bool raw_window_open = false;

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
            if (wheel_x < 0 || wheel_y < 0) {
                wheel = cv::Point2f(frame.cols * 0.5f, frame.rows * 0.5f);
            } else {
                wheel = cv::Point2f(static_cast<float>(wheel_x),
                                    static_cast<float>(wheel_y));
            }
            wheel_set = true;
            std::cout << "轮心 (" << wheel.x << ", " << wheel.y
                      << ")   半径 " << R_px << " px\n";
        }

        if (!warned_dark && cv::mean(frame)[0] < 5.0) {
            std::cout << "[WARN] 画面几乎全黑（平均灰度 < 5）\n"
                         "       可能原因：曝光太短 / 镜头盖没摘 / 环境太暗\n"
                         "       试试 --expo -6 或者开灯\n";
            warned_dark = true;
        }

        cv::Mat mask = buildRedMask(frame);
        int raw_contours = 0;
        std::vector<Led> leds = detectLeds(mask, &raw_contours);
        std::vector<Plate> plates = assignSlots(leds, wheel, R_px);

        int armed = -1;
        int armed_count = 0;
        for (size_t k = 0; k < plates.size(); ++k) {
            if (plates[k].state == 4) {
                if (armed < 0) armed = static_cast<int>(k);
                ++armed_count;
            }
        }
        if (armed_count != 1) armed = -1;

        // ---- ④ 相位解缠 ----
        double raw_phase = 0.0;
        bool has_phase = estimatePhase(plates, &raw_phase);
        double theta = tracker.update(has_phase, raw_phase);

        // 时间戳：视频按帧号/帧率，相机用真实时钟
        const double t_now = use_camera
                                 ? (cv::getTickCount() / cv::getTickFrequency() - t_start)
                                 : (idx / fps);

        if (has_phase && tracker.has()) {
            history.emplace_back(t_now, theta);
            while (!history.empty() && history.front().first < t_now - 3.0 * kFitWindowS) {
                history.erase(history.begin());
            }
        }

        // ---- ⑤ 拟合 ----
        MotionModel model;
        bool has_model = fitMotion(history, &model);

        // ---- ⑥ 预测 ----
        bool ready = false;
        double pred = 0.0;
        if (has_model && tracker.has() && armed >= 0) {
            pred = predictTarget(model, theta,
                                 plates[armed].angle_deg * CV_PI / 180.0,
                                 kLeadS) * 180.0 / CV_PI;
            ready = true;
        }

        if (idx % 30 == 0) {
            std::printf("frame %d  灯条 %d  靶 %d", idx,
                        static_cast<int>(leds.size()), static_cast<int>(plates.size()));
            if (tracker.has()) std::printf("  theta %.1f", tracker.thetaDeg());
            if (has_model)
                std::printf("  w=%.3f  rms=%.2f", model.omega,
                            model.rms * 180.0 / CV_PI);
            if (ready)
                std::printf("  pred %.1f  READY", pred);
            else
                std::printf("  HOLD");
            std::printf("\n");
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
            cv::circle(both, wc, static_cast<int>(R_px), cv::Scalar(90, 90, 90), 1,
                       cv::LINE_AA);
            for (int s = 0; s < kSlotCount; ++s) {
                double a = s * (360.0 / kSlotCount) * CV_PI / 180.0;
                cv::Point2f e(wc.x + static_cast<float>(R_px * std::cos(a)),
                              wc.y + static_cast<float>(R_px * std::sin(a)));
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
                cv::Point2f p(wheel.x + static_cast<float>(R_px * std::cos(a)) + off.x,
                              wheel.y + static_cast<float>(R_px * std::sin(a)) + off.y);
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
        char fz[96];
        std::snprintf(fz, sizeof(fz), "fps %.1f  %s", fps_shown,
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
                cv::imshow("frame | red mask", both);
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
