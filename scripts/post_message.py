def winapi_post_message():
    import win32gui
    import win32con

    # 通过类名和标题查找窗口句柄
    hwnd = win32gui.FindWindow("UnityWndClass", "原神")


    if hwnd != 0:
        print("找到窗口句柄：", hwnd)
        
        win32gui.SendMessage(hwnd, win32con.WM_ACTIVATE, win32con.WA_ACTIVE, 0)

        win32gui.SendMessage(hwnd, win32con.VK_SPACE , win32con.VK_SPACE, 0)
        win32gui.SendMessage(hwnd, win32con.WM_KEYUP, win32con.VK_SPACE, 0)

if __name__ == "__main__":
    winapi_post_message()
'''
不设置焦点的话，均无效
设置焦点后，仅第一次有效，后面的都无效
'''