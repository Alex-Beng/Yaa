// 用于bitblt截屏测试
// 主要测试截屏频率

#include <opencv2/opencv.hpp>
#include <iostream>

#include <windows.h>

bool find_window_local(HWND& ret_handle) {
    std::wstring wide = L"原神";
    std::wstring class_name = L"UnityWndClass";
    ret_handle = FindWindowW(class_name.c_str(), wide.c_str());
    
    return ret_handle != NULL;
}

bool find_window_cloud(HWND& ret_handle) {
    std::wstring wide = L"云·原神";
    ret_handle = FindWindowW(NULL, wide.c_str());
    return ret_handle != NULL;
}

// stolen from cvat shamelessly
bool bb_capture(HWND& giHandle, cv::Mat& frame) {
    // auto start = std::chrono::high_resolution_clock::now();
    // 句柄和位图结构体
    // windows 中只能通过句柄操作具体数据
    // 且需要手动管理生命周期
    static HBITMAP	hBmp;
    BITMAP bmp;

    RECT giRect, giClientRect;
    cv::Size giClientSize;

    DeleteObject(hBmp);

    if (giHandle == NULL) {
        // std::cerr<<"无效句柄"<<std::endl;
        std::cerr<<"invalid handle"<<std::endl;
        return false;
    }
    if (!IsWindow(giHandle)) {
        // std::cerr<<"无效句柄或指定句柄所指向窗口不存在"<<std::endl;
        std::cerr<<"invalid handle or window does not exist"<<std::endl;
        return false;
    }
    if (!GetWindowRect(giHandle, &giRect)) {
        // std::cerr<<"无效句柄或指定句柄所指向窗口不存在"<<std::endl;
        std::cerr<<"invalid handle or window does not exist"<<std::endl;
        return false;
    }
    if (!GetClientRect(giHandle, &giClientRect)) {
        // std::cerr<<"无效句柄或指定句柄所指向窗口不存在"<<std::endl;
        std::cerr<<"invalid handle or window does not exist"<<std::endl;
        return false;
    }

    //获取屏幕缩放比例
    HWND hWnd = GetDesktopWindow();
    HMONITOR hMonitor = MonitorFromWindow(hWnd, MONITOR_DEFAULTTONEAREST);

    // 获取监视器逻辑宽度与高度
    MONITORINFOEX miex;
    miex.cbSize = sizeof(miex);
    GetMonitorInfo(hMonitor, &miex);
    int cxLogical = (miex.rcMonitor.right - miex.rcMonitor.left);
    //int cyLogical = (miex.rcMonitor.bottom - miex.rcMonitor.top);

    // 获取监视器物理宽度与高度
    DEVMODE dm;
    dm.dmSize = sizeof(dm);
    dm.dmDriverExtra = 0;
    EnumDisplaySettings(miex.szDevice, ENUM_CURRENT_SETTINGS, &dm);
    int cxPhysical = dm.dmPelsWidth;
    //int cyPhysical = dm.dmPelsHeight;

    double horzScale = ((double)cxPhysical / (double)cxLogical);
    double screen_scale = horzScale;

    giClientSize.width = (int)(screen_scale * (giClientRect.right - giClientRect.left));
    giClientSize.height = (int)(screen_scale * (giClientRect.bottom - giClientRect.top));

    //获取目标句柄的窗口大小RECT
    GetWindowRect(giHandle, &giRect);/* 对原神窗口的操作 */

    //获取目标句柄的DC
    HDC hScreen = GetDC(giHandle);/* 对原神窗口的操作 */
    HDC hCompDC = CreateCompatibleDC(hScreen);

    //获取目标句柄的宽度和高度
    int	nWidth = (int)((screen_scale) * (giRect.right - giRect.left));
    int	nHeight = (int)((screen_scale) * (giRect.bottom - giRect.top));

    //创建Bitmap对象
    hBmp = CreateCompatibleBitmap(hScreen, nWidth, nHeight);//得到位图

    SelectObject(hCompDC, hBmp); //不写就全黑

    BitBlt(hCompDC, 0, 0, nWidth, nHeight, hScreen, 0, 0, SRCCOPY);
    
    ////释放对象
    DeleteDC(hScreen);
    DeleteDC(hCompDC);

    //类型转换
    //这里获取位图的大小信息,事实上也是兼容DC绘图输出的范围
    GetObject(hBmp, sizeof(BITMAP), &bmp);

    int nChannels = bmp.bmBitsPixel == 1 ? 1 : bmp.bmBitsPixel / 8;
    //int depth = bmp.bmBitsPixel == 1 ? 1 : 8;

    //mat操作
    cv::Mat giFrame;
    giFrame.create(cv::Size(bmp.bmWidth, bmp.bmHeight), CV_MAKETYPE(CV_8U, nChannels));
    // std::cout<<bmp.bmWidth<<" "<<bmp.bmHeight<<" "<<nChannels<<std::endl;

    GetBitmapBits(hBmp, bmp.bmHeight * bmp.bmWidth * nChannels, giFrame.data);

    giFrame = giFrame(cv::Rect(giClientRect.left, giClientRect.top, giClientSize.width, giClientSize.height));


    if (giFrame.empty()) {
        // std::cerr<<"窗口画面为空"<<std::endl;
        // std::cerr<<"frame is empty"<<std::endl;
        return false;
    }

    if (giFrame.cols < 480 || giFrame.rows < 360) {
        // err = { 14, "窗口画面大小小于480x360，无法使用" };
        // std::cerr<<"窗口画面大小小于480x360，无法使用"<<std::endl;
        std::cerr<<"window size is too small"<<std::endl;
        return false;
    }
    frame = giFrame;
    // auto end = std::chrono::high_resolution_clock::now();
    // std::cout<<"capture time: ";
    // std::cout<<std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()<<std::endl;
    // 9-27 ms
    // 12.7 ms in average vs 11.1 ms in RUST with yap's implementation
    return true;
}

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
        long long time_stamp = mat_queue.front().second;
        cv::imshow("frame", frame);
        // std::cout<<"frame size: "<<frame.size()<<std::endl;
        cv::waitKey(1);
        mat_queue.pop();
        frame_count++;
        lock.unlock();
    }
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
