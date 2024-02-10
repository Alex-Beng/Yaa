// 用于bitblt截屏测试
// 主要测试截屏频率

#include <opencv2/opencv.hpp>
#include <iostream>

#include <windows.h>

#include "bb_utils.h"

// 截屏和对应的时间戳
std::queue<std::pair<cv::Mat, long long>> mat_queue;
// std::queue<cv::Mat> mat_queue;
std::mutex mat_mtx;
std::condition_variable mat_cv;

// 截屏的生产者
void producer(std::chrono::steady_clock::time_point start_time) {
    // 初始化寻找窗口
    HWND window_handle;
    auto found = find_window_local(window_handle);
    if (found) {
        // std::cout<<window_handle<<std::endl;
    }
    else {
        found = find_window_cloud(window_handle);
        if (found) {
            // std::cout<<window_handle<<std::endl;
        }
        else {
            std::cout<<"cannot find window"<<std::endl;
            return;
        }
    }
    // 进行一个时的计
    auto start = start_time;
    auto N = 1000.0;
    long long max_iter = -1;
    // while (true) {
    for (int i = 0; i < N; i++) {
        auto iter_start = std::chrono::high_resolution_clock::now();
        cv::Mat frame;
        auto curr_time = std::chrono::high_resolution_clock::now();
        auto curr_time_stamp = std::chrono::duration_cast<std::chrono::nanoseconds>(curr_time - start_time).count();
        bb_capture(window_handle, frame);
        {
            std::lock_guard<std::mutex> lock(mat_mtx);
            // mat_queue.push(frame);
            mat_queue.push(
                std::make_pair(frame, curr_time_stamp)
            );
        }
        // mat_cv.notify_one();
        auto iter_end = std::chrono::high_resolution_clock::now();
        auto iter_time = std::chrono::duration_cast<std::chrono::milliseconds>(iter_end - iter_start).count();
        if (iter_time > max_iter) {
            max_iter = iter_time;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<<std::endl;
    // 每帧截屏时间
    std::cout<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / N<<std::endl;
    // max iter time
    std::cout<<max_iter<<std::endl;
}


void consumer() {
    cv::namedWindow("frame", cv::WINDOW_NORMAL);
    int frame_count = 0;
    auto N = 1000;

    auto videoname = "output.mp4";
    auto alpha_videoname = "output_alpha.mp4";
    // int fourcc = cv::VideoWriter::fourcc('h', '2',  '6', '4');
    // int fourcc = cv::VideoWriter::fourcc('H', '2',  '6', '4');
    // in BING COPILOT, H264 is same with AVC1
    int fourcc = cv::VideoWriter::fourcc('a', 'v',  'c', '1');
    // int fourcc = cv::VideoWriter::fourcc('M', 'J',  'P', 'G');
    // int fourcc = cv::VideoWriter::fourcc('M', 'P', '4', 'V');

    // H264 10944  KB
    // MJPG 108411 KB
    // MP4V 11875  KB
    // versus JPG per frame 311000 KB = 311 KB * 20s * 50fps
    // 使用H264

    // 另一个writer保存alpha通道
    double fps = 50.0;
    int width = 1600;
    int height = 900;
    cv::VideoWriter writer = cv::VideoWriter(videoname, fourcc, fps, cv::Size(width, height));
    cv::VideoWriter writer_alpha = cv::VideoWriter(alpha_videoname, fourcc, fps, cv::Size(width, height), false);
    if (!writer.isOpened()) {
        std::cerr<<"cannot open video writer"<<std::endl;
        return;
    }
    // while (true) {
    for (int i = 0; i < N; i++) {
        std::unique_lock<std::mutex> lock(mat_mtx);
        mat_cv.wait(lock, []{ return !mat_queue.empty(); });
        // if (mat_queue.empty()) {
        // 	std::this_thread::sleep_for(std::chrono::milliseconds(40));
        // 	continue;
        // }
        // cv::Mat frame = mat_queue.front();
        cv::Mat frame = mat_queue.front().first;
        // check size and make it same with (width, height)
        if (frame.cols != width || frame.rows != height) {
            // resize it silently
            // std::cout<<"need resize"<<std::endl;
            cv::resize(frame, frame, cv::Size(width, height));
        }

        long long time_stamp = mat_queue.front().second;
        // split alpha channel
        cv::Mat alpha;
        cv::extractChannel(frame, alpha, 3);
        writer_alpha.write(alpha);

        // convert 4 channel to 3 channel
        // otherwise, the video writer will not work
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);

        cv::imshow("frame", frame);
        writer.write(frame);

        // std::cout<<frame.size()<<std::endl;
        if (i%100 == 0) {
            std::cout<<"time stamp: "<<time_stamp<<std::endl;
            std::cout<<"frame size: "<<frame.size()<<std::endl;
        }
        cv::waitKey(1);
        mat_queue.pop();
        frame_count++;
        lock.unlock();
    }
    writer.release();
    cv::destroyAllWindows();
    std::cout<<"frame count: "<<frame_count<<std::endl;
}

int main(int argc, char const *argv[]) {
    std::thread producerThread(producer, std::chrono::steady_clock::now());
    std::thread consumerThread(consumer);
    producerThread.join();
    consumerThread.join();
    return 0;
}


int old_main(int argc, char const *argv[]) {
    // 寻找窗口
    HWND window_handle;
    auto found = find_window_local(window_handle);
    if (found) {
        // std::cout<<window_handle<<std::endl;
    }
    else {
        found = find_window_cloud(window_handle);
        if (found) {
            // std::cout<<window_handle<<std::endl;
        }
        else {
            std::cout<<"cannot find window"<<std::endl;
            return 0;
        }
    }
    
    // 截屏
    cv::Mat frame;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 500; i++) {
        bb_capture(window_handle, frame);
        // cv::imshow("frame", frame);
        // cv::waitKey(1);
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::cout<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<<std::endl;
    // 每帧截屏时间
    std::cout<<"每帧截屏时间:";
    std::cout<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 500.0<<std::endl;
    // 8.2421 ms in 10000 frames
    // 7.498 ms in 10000 frames with release mode
    // 测试加上队列的用时
    // 16.9 ms with queue + cv + lock
    // 14.47 ms with queue + no lock + huge memory leak
    // 主要是队列带来的时间开销？
    // nope，是std::cerr的问题
    // 9.54 ms with queue + cv + lock - std::cerr
    
    return 0;
}
