// ============================================================================
//  ROBOMINI 能量机关自瞄 —— 节点 4：相位解缠（mod 72°）
// ============================================================================
//
//  节点 3 得到的是"每块靶各自的角度"，但它们都等于 θ + k×72°。
//  这一节点把它们合成一个数：转盘转了多少（θ）。
//
//  两步：
//      ① 乘 5 消掉槽位号 —— estimatePhase()
//         5 块靶的方位角分别是 θ+0×72°、θ+1×72°、…、θ+4×72°
//         角度乘 5 后，k×72°×5 = k×360° 全部绕回原点，只剩 5θ
//         所有靶乘 5 后做向量平均，再除以 5，就得到 θ mod 72°
//
//      ② 折叠累加成连续角度 —— PhaseTracker
//         mod 72° 的值每转 72° 就跳一次，没法直接拟合。
//         把相邻两帧之差折叠到 ±半个槽，得到"这一帧转了多少"，累加起来。
//         单帧转动超过 0.8 个槽（约 57.6°）物理上不可能，当这帧无效。
//
//  为什么第 ② 步的"跳变拒绝"这么重要：
//      如果某帧把靶认错成隔壁那块，原始相位会瞬间跳 72°。
//      不作废它，θ 就永久偏了 72°，之后所有预测全废。
//
//  用法：同节点 3
//
//  验证标准：
//      · 合成数据集：theta 连续递增（10° → 44° → 133° → 234°），不在 72° 处回落
//      · 与 truth.csv 的 theta 对比，误差应 < 0.1°
//
// ============================================================================

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
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

// ① 乘 5 消掉槽位号：(θ + k×72°) × 5 = 5θ + k×360° ≡ 5θ
bool estimatePhase(const std::vector<Plate>& plates, double* raw_out) {
    if (plates.empty()) return false;

    double sx = 0.0, sy = 0.0;
    for (const auto& p : plates) {
        double a = p.angle_deg * CV_PI / 180.0;   // 转成弧度再乘 5
        sx += std::cos(kSlotCount * a);
        sy += std::sin(kSlotCount * a);
    }
    // 向量互相抵消 -> 方向不可信（比如只剩一块靶、或者分布太乱）
    if (std::hypot(sx, sy) < 1e-6) return false;

    *raw_out = std::atan2(sy, sx) / kSlotCount;   // 弧度，落在 (-36°, +36°]
    return true;
}

// ② 折叠累加成连续角度，并拒绝可疑帧（防 72° 滑移）
class PhaseTracker {
public:
    explicit PhaseTracker(int slot_count) : half_slot_(CV_PI / slot_count) {}

    double update(bool has_raw, double raw) {
        if (!has_raw) return theta_;

        if (!has_theta_) {          // 第一帧：直接作为起点
            theta_ = raw;
            last_raw_ = raw;
            has_theta_ = true;
            return theta_;
        }

        // 把两帧之差折叠到 ±半个槽：这才是"这一帧到底转了多少"
        const double two = 2.0 * half_slot_;
        double d = std::fmod(raw - last_raw_ + half_slot_, two);
        if (d < 0.0) d += two;
        d -= half_slot_;

        // 一帧转超过 0.8 个槽（约 57.6°）——物理上不可能，当这帧无效
        if (std::fabs(d) > 0.8 * half_slot_) {
            return theta_;
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
    int want_index = -1;
    int max_frames = 1;
    double exposure = kDefExposure;
    double wb_temp = kDefWbTemp;
    double wheel_x = -1.0;       // -1 表示"用画面中心"
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

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

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

        // 待激活靶必须唯一
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
        tracker.update(has_phase, raw_phase);
        double theta = tracker.theta();

        if (idx % 30 == 0) {
            std::cout << "frame " << idx
                      << "  灯条 " << leds.size()
                      << "  靶 " << plates.size();
            if (armed >= 0) {
                std::cout << "  待激活角度 " << plates[armed].angle_deg;
            }
            if (tracker.has()) {
                std::printf("  theta %.2f (mod72 %+.1f)", theta * 180.0 / CV_PI,
                            std::fmod(theta * 180.0 / CV_PI, 72.0));
            }
            std::cout << "\n";
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

        char text[224];
        if (armed >= 0) {
            std::snprintf(text, sizeof(text),
                          "frame %d  leds %d  plates %d  ARMED %.1f  theta %.1f",
                          idx, static_cast<int>(leds.size()),
                          static_cast<int>(plates.size()),
                          plates[armed].angle_deg, tracker.thetaDeg());
        } else {
            std::snprintf(text, sizeof(text),
                          "frame %d  leds %d  plates %d  no unique armed  theta %.1f",
                          idx, static_cast<int>(leds.size()),
                          static_cast<int>(plates.size()), tracker.thetaDeg());
        }
        cv::putText(both, text, cv::Point(15, 35), cv::FONT_HERSHEY_SIMPLEX,
                    0.72, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        // 单独一行显示连续相位（一直累加，不是折叠值）
        char th[96];
        std::snprintf(th, sizeof(th), "theta %+9.2f deg", tracker.thetaDeg());
        cv::putText(both, th, cv::Point(15, 70), cv::FONT_HERSHEY_SIMPLEX,
                    0.7, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

        cv::imshow("frame | red mask", both);
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
    }

    cap.release();
    cv::destroyAllWindows();
    std::cout << "处理了 " << idx << " 帧\n";
    return 0;
}
