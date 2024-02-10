// 用于同时录制屏幕和键鼠
// 三个线程。
// 屏幕录制使用生产者消费者模型
// 键鼠录制使用简单的先存储再处理模型
// 如何结束录制？

#include <iostream>
#include <thread>
#include <queue>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <CLI/CLI.hpp>

#include "ms_kb_utils.h"
#include "bb_utils.h"

// just three function for three thread
// define the needed queues here
std::queue<std::pair<cv::Mat, long long>> mat_queue;
std::mutex mat_mtx;
std::condition_variable mat_cv;

// which can be lock-free ?
// cause basic var type is lock-free ?
// idk, just use std::atomic for trying
std::atomic<bool> is_recording = false;


// thread 0
// bb capture producer
void bb_capture_producer(HWND& window_handle, std::chrono::steady_clock::time_point start_time) {
    auto freme_cnt = 0;
    while (is_recording) {
        cv::Mat frame;
        auto curr_time = std::chrono::high_resolution_clock::now();
        auto curr_time_stamp = std::chrono::duration_cast<std::chrono::nanoseconds>(curr_time - start_time).count();
        bb_capture(window_handle, frame);
        {
            std::lock_guard<std::mutex> lock(mat_mtx);
            mat_queue.push(
                std::make_pair(frame, curr_time_stamp)
            );
        }
        mat_cv.notify_one();
        freme_cnt++;
    }
    // only print in the end
    std::cout<<"bb_capture_producer: "<<freme_cnt<<std::endl;
}

struct BBCaptureConfig
{
    double fps;
    int width;
    int height;
    std::string output_path;
    std::string task_name;
    int episode_id;
};


// thread 1
// bb capture consumer
void bb_capture_consumer(BBCaptureConfig& config) {
    auto freme_cnt = 0;
    while (is_recording) {
        std::unique_lock<std::mutex> lock(mat_mtx);
        mat_cv.wait(lock, []{return !mat_queue.empty();});
        auto frame = mat_queue.front();

        // 

        mat_queue.pop();
        lock.unlock();
        // do something with frame
        freme_cnt++;
    }
    // only print in the end
    std::cout<<"bb_capture_consumer: "<<freme_cnt<<std::endl;
}

// read config from argv
void read_config(int argc, char const *argv[], BBCaptureConfig& config) {
    // read bb capture config from argv using CLI11
    CLI::App app{"yaa_recorder"};
    app.add_option("-f,--fps", config.fps, "fps")
        ->required(false)
        ->default_val(60.0);
    app.add_option("--width", config.width, "width")
        ->required(false)
        ->default_val(1600);
    // -h for help, so use --height and --width
    // however, normal guy won't change the height and width
    app.add_option("--height", config.height, "height")
        ->required(false)
        ->default_val(900);
    app.add_option("-o,--output_path", config.output_path, "output folder for recording")
        ->required()
        ->default_val("./");
    app.add_option("-t,--task_name", config.task_name, "task_name")
        ->required();
    app.add_option("-e,--episode_id", config.episode_id, "episode_id")
        ->required();

    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError &e) {
        // std::cout << e.what() << std::endl;
        app.exit(e);
    }
}

void print_config(const BBCaptureConfig& config) {
    std::cout<<"fps: "<<config.fps<<std::endl;
    std::cout<<"width: "<<config.width<<std::endl;
    std::cout<<"height: "<<config.height<<std::endl;
    std::cout<<"output_path: "<<config.output_path<<std::endl;
    std::cout<<"task_name: "<<config.task_name<<std::endl;
    std::cout<<"episode_id: "<<config.episode_id<<std::endl;
}

int main(int argc, char const *argv[]) {
    // read bb capture config from argv using CLI11
    CLI::App app{"yaa_recorder"};
    BBCaptureConfig config;
    read_config(argc, argv, config);

    print_config(config);
    // press F6 to start recording


    return 0;
}
