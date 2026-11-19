// ============================================================================
//  ROBOMINI 能量机关自瞄 —— 节点 2：轮廓过滤 + 质心
// ============================================================================
//
//  在节点 1（颜色掩膜）的基础上，把掩膜里的白色块变成"点"：
//      掩膜 -> findContours -> 面积/长宽比过滤 -> moments 求质心
//
//  为什么要过滤：
//      · 面积太小  -> 噪点、反光
//      · 面积太大  -> 整片光斑、被照亮的墙面
//      · 长宽比太方 -> 圆形光斑，不是灯条
//
//  用法：
//      aim.exe                                自动挑相机
//      aim.exe --index 1                      强制某个索引
//      aim.exe --expo -6                      曝光（-8 最暗，值越大越亮）
//      aim.exe --wb 4600                      色温
//      aim.exe --video xxx.mp4                用视频文件
//      aim.exe --video xxx.mp4 --save out.png 不弹窗，存一张图
//
//  按键：ESC 或 q 退出
//
//  验证标准：
//      · 合成数据集 red_var  -> "灯条 4"，画面上 4 个绿圈套在 4 条灯上
//      · 纯红屏幕（整块）    -> "灯条 0"（面积 25 万像素，被上限滤掉，正常）
//      · 桌面杂物            -> "灯条 0"（没有细长的红色块）
//
// ============================================================================

#include <opencv2/opencv.hpp>

#include <algorithm>
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
// 灯条实测：宽 6~7 px，长 40~70 px，面积 300~800 px²
constexpr double kMinLedArea = 20.0;    // 小于这个：噪点
constexpr double kMaxLedArea = 2500.0;  // 大于这个：光斑 / 整片红色
constexpr double kMinAspect = 1.3;      // 长宽比，滤掉圆形的反光点

// ---- 颜色阈值 --------------------------------------------------------------
// 红色在 HSV 的 H 上跨 0 和 179 两端，所以要两段。
// S/V 下限 120：既滤掉灰白的背景，又不会漏掉灯条。
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
//      BGR -> HSV -> 两段 inRange -> 位或合成 -> 3x3 椭圆开运算
// ============================================================================
cv::Mat buildRedMask(const cv::Mat& bgr) {
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    cv::Mat mask = cv::Mat::zeros(hsv.size(), CV_8UC1);
    for (const auto& r : kRedRanges) {
        cv::Mat m;
        cv::inRange(hsv, r.lo, r.hi, m);
        cv::bitwise_or(mask, m, mask);      // C++ 的 bitwise_or 写进第三个参数
    }

    // 3x3 椭圆开运算去孤立噪点。灯条只有 6~7 像素宽，核不能大。
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    return mask;
}

// ============================================================================
//  ② 灯条：一条灯（可能是一块靶上的 4 条之一）
// ============================================================================
struct Led {
    cv::Point2f center;
    double area = 0.0;
    double aspect = 0.0;
};

// ============================================================================
//  ②③ 轮廓过滤 + 质心
//
//      raw_count 回传"过滤前有多少个轮廓"，用来区分
//      "根本找不到东西" 和 "找到了但都被形状条件刷掉"。
// ============================================================================
std::vector<Led> detectLeds(const cv::Mat& mask, int* raw_count) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (raw_count) *raw_count = static_cast<int>(contours.size());

    std::vector<Led> leds;
    leds.reserve(contours.size());

    for (const auto& c : contours) {
        // --- 面积过滤 ---
        double area = cv::contourArea(c);
        if (area < kMinLedArea || area > kMaxLedArea) {
            continue;
        }

        // --- 长宽比过滤：灯条一定是"细长"的 ---
        cv::RotatedRect rr = cv::minAreaRect(c);
        double w = rr.size.width, h = rr.size.height;
        double lo = std::min(w, h), hi = std::max(w, h);
        if (lo <= 0.0) continue;
        double aspect = hi / lo;
        if (aspect < kMinAspect) {
            continue;
        }

        // --- 质心：用一阶矩 / 零阶矩 ---
        //     binaryImage 默认 false，但这里传的是轮廓不是灰度图，
        //     等价于把轮廓内部当作权重 1，和 Python 版一致。
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
//  打开相机
//      笔记本内置摄像头通常是索引 0、最高 640x480；USB 工业相机在后面。
//      所以逐个试，谁能给出 1280x1024 就用谁。
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
        cap.set(cv::CAP_PROP_AUTO_EXPOSURE, 0.25);      // 关自动曝光
        cap.set(cv::CAP_PROP_EXPOSURE, exposure);
        cap.set(cv::CAP_PROP_AUTO_WB, 0);               // 关自动白平衡（关键！）
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
    // 源码是 UTF-8，控制台默认 GBK，不设这个中文输出就是乱码
    SetConsoleOutputCP(CP_UTF8);
#endif

    bool use_camera = true;
    bool headless = false;
    int want_index = -1;
    int max_frames = 1;
    double exposure = kDefExposure;
    double wb_temp = kDefWbTemp;
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

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;   // 视频放完

        // 画面太黑时提醒一次（曝光太短 / 镜头盖 / 环境太暗）
        if (!warned_dark && cv::mean(frame)[0] < 5.0) {
            std::cout << "[WARN] 画面几乎全黑（平均灰度 < 5）\n"
                         "       可能原因：曝光太短 / 镜头盖没摘 / 环境太暗\n"
                         "       试试 --expo -6 或者开灯\n";
            warned_dark = true;
        }

        cv::Mat mask = buildRedMask(frame);

        int raw_contours = 0;
        std::vector<Led> leds = detectLeds(mask, &raw_contours);

        if (idx % 30 == 0) {
            std::cout << "frame " << idx
                      << "  轮廓 " << raw_contours
                      << "  灯条 " << leds.size() << "\n";
        }

        cv::Mat mask_bgr;
        cv::cvtColor(mask, mask_bgr, cv::COLOR_GRAY2BGR);
        cv::Mat both;
        cv::hconcat(frame, mask_bgr, both);     // 左：原图   右：掩膜

        // 在原图和掩膜两边都画质心，方便对照"看到的就是找到的"
        for (size_t k = 0; k < leds.size(); ++k) {
            const cv::Point2f& p = leds[k].center;
            cv::circle(both, p, 6, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
            cv::circle(both, p + cv::Point2f(static_cast<float>(kWantW), 0.0f), 6,
                       cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

            char tag[16];
            std::snprintf(tag, sizeof(tag), "%d", static_cast<int>(k));
            cv::putText(both, tag, p + cv::Point2f(9, -9),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6,
                        cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        }

        char text[160];
        std::snprintf(text, sizeof(text),
                      "frame %d   leds %d   contours %d   mask %d",
                      idx, static_cast<int>(leds.size()), raw_contours,
                      cv::countNonZero(mask));
        cv::putText(both, text, cv::Point(15, 35), cv::FONT_HERSHEY_SIMPLEX,
                    0.8, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

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
