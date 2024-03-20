#ifndef BB_UTILS_H
#define BB_UTILS_H

#include <opencv2/opencv.hpp>
#include <Windows.h>

// timestamp data structure
typedef long long BBTimeStamp;

// find widow handle
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

// copilot generated code
cv::Mat bb_capture(HWND hwnd) {
    HDC hdcScreen = GetDC(NULL);
    HDC hdcWindow = GetDC(hwnd);

    RECT windowsRect;
    GetWindowRect(hwnd, &windowsRect);

    int width = windowsRect.right - windowsRect.left;
    int height = windowsRect.bottom - windowsRect.top;
    HDC hdcMemDC = CreateCompatibleDC(hdcWindow); 
    HBITMAP hbmScreen = CreateCompatibleBitmap(hdcWindow, width, height); 
    SelectObject(hdcMemDC, hbmScreen);

    BitBlt(hdcMemDC, 0, 0, width, height, hdcWindow, 0, 0, SRCCOPY);

    BITMAPINFOHEADER bmi = {0};
    bmi.biSize = sizeof(BITMAPINFOHEADER);
    bmi.biPlanes = 1;
    bmi.biBitCount = 32;
    bmi.biWidth = width;
    bmi.biHeight = -height;
    bmi.biCompression = BI_RGB;

    cv::Mat mat(height, width, CV_8UC4);
    GetDIBits(hdcWindow, hbmScreen, 0, height, mat.data, (BITMAPINFO*)&bmi, DIB_RGB_COLORS);

    DeleteObject(hbmScreen);
    DeleteDC(hdcMemDC);
    ReleaseDC(NULL, hdcScreen);
    ReleaseDC(hwnd, hdcWindow);

    return mat;
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




#endif
// BB_UTILS_H