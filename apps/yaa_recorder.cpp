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

struct BBCaptureConfig
{
    // video config
    double fps;
    int width;
    int height;
    // output capture to:
    // output_path / task_name / 
    //  {episode_id}.mp4 
    //  {episode_id}_alpha.mp4 
    //  {episode_id}.jsonl
    std::string output_path;
    std::string task_name;
    int episode_id;

    // print config
    void print() {
        std::cout<<"fps: "<<fps<<std::endl;
        std::cout<<"width: "<<width<<std::endl;
        std::cout<<"height: "<<height<<std::endl;
        std::cout<<"output_path: "<<output_path<<std::endl;
        std::cout<<"task_name: "<<task_name<<std::endl;
        std::cout<<"episode_id: "<<episode_id<<std::endl;
    }    
};

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

// thread 1
// bb capture consumer
void bb_capture_consumer(BBCaptureConfig& config) {
    auto freme_cnt = 0;
    // init writers
    auto writer_rgb = cv::VideoWriter(
        config.output_path + "/" + config.task_name + "/" + std::to_string(config.episode_id) + ".mp4",
        cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
        config.fps,
        cv::Size(config.width, config.height)
    );
    auto wirter_alpha = cv::VideoWriter(
        config.output_path + "/" + config.task_name + "/" + std::to_string(config.episode_id) + "_alpha.mp4",
        cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
        config.fps,
        cv::Size(config.width, config.height),
        false
    );
    // init jsonl data structure
    std::vector<BBTimeStamp> timestamps;
    // 50fps, 120s, 10 time larger
    timestamps.reserve(50*120*10);

    while (is_recording) {
        std::unique_lock<std::mutex> lock(mat_mtx);
        mat_cv.wait(lock, []{return !mat_queue.empty();});
        auto frame = mat_queue.front().first;
        auto timestamp = mat_queue.front().second;

        // check frame size
        if (frame.cols != config.width || frame.rows != config.height) {
            // sliently resize
            cv::resize(frame, frame, cv::Size(config.width, config.height));
        }

        // split rgb and aplha
        cv::Mat alpha;
        cv::extractChannel(frame, alpha, 3);
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        
        // write to video
        writer_rgb.write(frame);
        wirter_alpha.write(alpha);
        // save timestamp
        timestamps.emplace_back(timestamp);

        mat_queue.pop();
        lock.unlock();
        // do something with frame
        freme_cnt++;
    }
    // release writer
    writer_rgb.release();
    wirter_alpha.release();
    // saving to jsonl
    std::ofstream file(config.output_path + "/" + config.task_name + "/" + std::to_string(config.episode_id) + "_video.jsonl");
    if (!file.is_open()) {
        std::cout<<"Failed to open file."<<std::endl;
    }
    // ts just long long alias
    for (auto& ts : timestamps) {
        // make it json object
        nlohmann::json j = {
            {"timestamp", ts}
        };
        file << j.dump() << std::endl;
    }
    // only print in the end
    std::cout<<"bb_capture_consumer: "<<freme_cnt<<std::endl;
}

// thread 2
// ms kb recorder
void ms_kb_recorder(BBCaptureConfig& config, std::chrono::steady_clock::time_point start_time) {
    bool key_state[1<<16] = {false};
    std::vector<ABEvent> events;
    // 120fps, 120s, 10 time larger
    events.reserve(120*120*10);
    
    InterceptionContext ctx = interception_create_context();
    InterceptionDevice device;
    InterceptionStroke stroke;
    interception_set_filter(ctx, 
                            interception_is_mouse, 
                            INTERCEPTION_FILTER_MOUSE_ALL);
    interception_set_filter(ctx,
                            interception_is_keyboard,
                            INTERCEPTION_FILTER_KEY_DOWN|INTERCEPTION_FILTER_KEY_UP|INTERCEPTION_FILTER_KEY_E0);
    while (is_recording &&
           interception_receive(ctx, device = interception_wait(ctx), &stroke, 1) > 0) {
        ABEvent event;

        auto curr_time = std::chrono::high_resolution_clock::now();
        auto curr_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(curr_time - start_time).count();

        if (interception_is_mouse(device)) {
            interception_send(ctx, device, &stroke, 1);
            InterceptionMouseStroke &mstroke = reinterpret_cast<InterceptionMouseStroke &>(stroke);
            // do something with mouse stroke
            event.timestamp = curr_timestamp;
            event.type = EVENT_TYPE_MOUSE;
            event.mouse.dx = mstroke.x;
            event.mouse.dy = mstroke.y;
            event.mouse.event_type = mstroke.state;
        }
        else if (interception_is_keyboard(device)) {
            interception_send(ctx, device, &stroke, 1);
            InterceptionKeyStroke &kstroke = reinterpret_cast<InterceptionKeyStroke &>(stroke);
            // do something with key stroke
            
            // press F6 to stop
            if (kstroke.code == SCANCODE_F6 && kstroke.state == INTERCEPTION_KEY_DOWN) {
                is_recording = false;
                break;
            }

            // filter continuous key down
            if (key_state[kstroke.code] == true && kstroke.state == INTERCEPTION_KEY_DOWN) {
                continue;
            }
            key_state[kstroke.code] = kstroke.state == INTERCEPTION_KEY_DOWN;

            event.timestamp = curr_timestamp;
            event.type = EVENT_TYPE_KEYBOARD;
            event.keyboard.scancode = kstroke.code;
            event.keyboard.event_type = kstroke.state;
        }
        events.emplace_back(event);
    }
    interception_destroy_context(ctx);
    std::cout << "Saving to file..." << std::endl;
    // saving to jsonl
    mskbevts2jsonl(events, config.output_path + "/" + config.task_name + "/" + std::to_string(config.episode_id) + "_mskb.jsonl");
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
        ->required(false)
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
        exit(1);
    }
}

int main(int argc, char const *argv[]) {
    // read bb capture config from argv using CLI11
    CLI::App app{"yaa_recorder"};
    BBCaptureConfig config;
    read_config(argc, argv, config);
    // config.print();

    // check path exist
    // unwarp_or create
    auto output_folder = config.output_path + "/" + config.task_name;
    if (!std::filesystem::exists(output_folder)) {
        std::filesystem::create_directories(output_folder);
    }
    is_recording = false;

    // init HWND
    HWND ge_win_handle;
    auto found = find_window_local(ge_win_handle);
    if (!found) {
        found = find_window_cloud(ge_win_handle);
        if (!found) {
            std::cout<<"cannot find window"<<std::endl;
            return 0;
        }
    }
    if (!found) {
        std::cout<<"cannot find window"<<std::endl;
        return 0;
    }

    std::cout << "Press F6 to start and stop recording..." << std::endl;
    std::cout << "Waiting for F6..." << std::endl;
    // press F6 to start recording
    wait_untill_press(SCANCODE_F6);

    auto start_time = std::chrono::high_resolution_clock::now();

    // create three threads
    // 1. bb capture producer
    // 2. bb capture consumer
    // 3. ms kb recorder
    is_recording = true;
    std::thread producerThread(bb_capture_producer, std::ref(ge_win_handle), start_time);
    std::thread consumerThread(bb_capture_consumer, std::ref(config));
    std::thread recorderThread(ms_kb_recorder, std::ref(config), start_time);

    producerThread.join();
    consumerThread.join();
    recorderThread.join();
    return 0;
}
