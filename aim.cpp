// ============================================================================
//  ROBOMINI 能量机关自瞄 —— 节点 1：相机采集 + 红色颜色掩膜
// ============================================================================
//
//  本节点只做一件事：把画面里的红色灯条分离出来，左右并排显示"原图 | 掩膜"。
//  这是整条链路的第一步，后面所有节点都建立在它之上。
//
//  用法：
//      aim.exe                                自动挑相机（跳过笔记本内置摄像头）
//      aim.exe --index 1                      强制用某个索引的相机
//      aim.exe --expo -6                      曝光（-8 最暗，值越大越亮）
//      aim.exe --wb 4600                      色温
//      aim.exe --video xxx.mp4                用视频文件
//      aim.exe --video xxx.mp4 --save out.png 不弹窗，存一张图
//
//  按键：ESC 或 q 退出
//
//  验证标准：
//      · 对准纯红色屏幕   → mask px 几十万，右半屏整块白
//      · 对准没有红色的场景 → mask px = 0，右半屏全黑
//
// ============================================================================

#include <opencv2/opencv.hpp>

#include <cstdio>
#include <iostream>
#include <string>

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

        cv::Mat mask_bgr;
        cv::cvtColor(mask, mask_bgr, cv::COLOR_GRAY2BGR);
        cv::Mat both;
        cv::hconcat(frame, mask_bgr, both);     // 左：原图   右：掩膜

        char text[128];
        std::snprintf(text, sizeof(text), "frame %d   mask px %d",
                      idx, cv::countNonZero(mask));
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
