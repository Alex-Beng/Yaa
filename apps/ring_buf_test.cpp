// 测试 ring buf
// 两个线程，一个写，一个读
// 通过对比之前yaa recoder的mutex+cv+queue，验证ring buf的有效性
// 实际对比应该是产出时间戳，然后对比时间戳的均值和方差
// 录制1w帧

#include <iostream>
#include <thread>
#include <queue>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <CLI/CLI.hpp>

#include "ms_kb_utils.h"
#include "bb_utils.h"
#include "ring_buf.h"

// the yaa recoder's mutex+cv+queue
std::queue<std::pair<cv::Mat, long long>> yaa_queue;
std::mutex yaa_mutex;
std::condition_variable yaa_cv;

// the ring buf
// in obs, the video buffer is 16
auto ring_buf = LockFreeRingBuffer<std::pair<cv::Mat, long long>>(320);

std::atomic<bool> is_recording = true;

void yaa_capture_producer(HWND& window_handle, std::chrono::steady_clock::time_point start_time) {
    auto frame_count = 0;
    for (int i=0; i<10000; i++) {
        cv::Mat frame;
        auto curr_time = std::chrono::high_resolution_clock::now();
        auto curr_time_stamp = std::chrono::duration_cast<std::chrono::nanoseconds>(curr_time - start_time).count();
        bb_capture(window_handle, frame);
        {
            std::lock_guard<std::mutex> lock(yaa_mutex);
            yaa_queue.push(
                std::make_pair(frame, curr_time_stamp)
            );
        }
        yaa_cv.notify_one();
        frame_count++;
    }
    std::cout << "yaa capture producer done with " << frame_count << " frames" << std::endl;
}

void yaa_recorder_consumer() {
    // just hardcode the output path in testings
    auto output_path = "./yaa_recorder.mp4";
    // init writers 
    auto writer_rgb = cv::VideoWriter(
        output_path,
        cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
        60.0,
        cv::Size(1600, 900)
    );

    std::vector<BBTimeStamp> timestamps;
    for (int i = 0; i < 10000; i++) {
        std::unique_lock<std::mutex> lock(yaa_mutex);
        yaa_cv.wait(lock, [] { return !yaa_queue.empty(); });

        auto frame = yaa_queue.front().first;
        auto timestamp = yaa_queue.front().second;


        // split the frame from rgba to rgb
        // std::cout<<frame.channels()<<' '<<frame.size()<<std::endl;
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
        
        writer_rgb.write(frame);
        timestamps.emplace_back(timestamp);

        yaa_queue.pop();
        lock.unlock();
    }
    // release the writer
    writer_rgb.release();

    // save to jsonl
    std::ofstream file("./yaa_recorder.jsonl");
    if (!file.is_open()) {
        std::cerr << "failed to open file" << std::endl;
        return;
    }

    for (auto& ts : timestamps) {
        nlohmann::json j = {
            {"timestamp", ts}
        };
        file << j.dump() << std::endl;
    }
    file.close();
}

void ring_buf_capture_producer(HWND& window_handle, std::chrono::steady_clock::time_point start_time) {
    auto frame_count = 0;
    for (int i=0; i<1000; i++) {
        cv::Mat frame;
        auto curr_time = std::chrono::high_resolution_clock::now();
        auto curr_time_stamp = std::chrono::duration_cast<std::chrono::nanoseconds>(curr_time - start_time).count();
        bb_capture(window_handle, frame);
        ring_buf.push(
            std::make_pair(frame, curr_time_stamp)
        );
        // printf("push frame %d\n", i);
        // too fast, need to slow down
        // std::this_thread::sleep_for(std::chrono::milliseconds(5));
        frame_count++;
        if (frame_count >= 10000) {
            break;
        }
    }
    is_recording = false;
    std::cout << "ring buf capture producer done with " << frame_count << " frames" << std::endl;
}

void ring_buf_recorder_consumer() {
    // just hardcode the output path in testings
    auto output_path = "./ring_buf_recorder.mp4";
    // init writers 
    auto writer_rgb = cv::VideoWriter(
        output_path,
        cv::VideoWriter::fourcc('a', 'v', 'c', '1'),
        60.0,
        cv::Size(1600, 900)
    );

    std::vector<BBTimeStamp> timestamps;
    std::pair<cv::Mat, long long> frame;
    auto i = 0;
    while (is_recording) {
        if (!ring_buf.pop(frame)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        i++ ;
        // printf("pop frame %d\n", i);
        auto frame_img = frame.first;
        auto timestamp = frame.second;
        
        cv::cvtColor(frame_img, frame_img, cv::COLOR_BGRA2BGR);
        writer_rgb.write(frame_img);
        timestamps.push_back(timestamp);

    }
    std::cout<<"ring buf recorder consumer done with "<<i<<" frames"<<std::endl;
    // release the writer
    writer_rgb.release();

    // save to jsonl
    std::ofstream file("./ring_buf_recorder.jsonl");
    if (!file.is_open()) {
        std::cerr << "failed to open file" << std::endl;
        return;
    }

    for (auto& ts : timestamps) {
        nlohmann::json j = {
            {"timestamp", ts}
        };
        file << j.dump() << std::endl;
    }
    file.close();
}

int main(int argc, char** argv) {
    // 对比测试
    CLI::App app("yaa recoder vs ring buf test");

    std::string mode;
    app.add_option("-m,--mode", mode, "mode: yaa or ring buf")->required();

    CLI11_PARSE(app, argc, argv);
    HWND ge_win_handle;
    auto found = find_window_local(ge_win_handle);
    if (!found) {
        std::cerr << "failed to find window" << std::endl;
        return -1;
    }

    is_recording = false;
    std::cout << "Press F6 to start and stop recording..." << std::endl;
    std::cout << "Waiting for F6..." << std::endl;
    wait_untill_press(SCANCODE_F6);

    auto start_time = std::chrono::high_resolution_clock::now();

    if (mode == "yaa") {
        is_recording = true;
        std::thread producer(yaa_capture_producer, std::ref(ge_win_handle), start_time);
        std::thread consumer(yaa_recorder_consumer);
        producer.join();
        consumer.join();
    } else if (mode == "ring_buf") {
        is_recording = true;
        std::thread producer(ring_buf_capture_producer, std::ref(ge_win_handle), start_time);
        std::thread consumer(ring_buf_recorder_consumer);
        producer.join();
        consumer.join();
    } else {
        std::cerr << "invalid mode" << std::endl;
        return -1;
    }

    return 0;
}