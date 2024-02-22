def get_active_window_info():
    import win32gui
    # 获取当前活动窗口的句柄
    hwnd = win32gui.GetForegroundWindow()

    # 获取窗口标题
    window_title = win32gui.GetWindowText(hwnd)

    # 获取窗口类名
    class_name = win32gui.GetClassName(hwnd)

    return window_title, class_name

def curr_active_title_class():
    import time
    while True:    
        # 调用函数获取当前活动窗口的信息
        title, class_name = get_active_window_info()
        # 打印窗口标题和类名
        print("Window Title:", title)
        print("Class Name:", class_name)
        time.sleep(0.1)


if __name__ == "__main__":
    # winapi_post_message()
    curr_active_title_class()