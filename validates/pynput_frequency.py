def thread_ver():
    # 测试pynput的频率
    from pynput.keyboard import Listener as KeyboardListener
    from pynput.mouse import Listener as MouseListener

    def on_press(key):
        print(key)

    def on_release(key):
        print(key)

    def on_click(x, y, button, pressed):
        print(x, y, button, pressed)

    import time

    last_time = time.time()
    time_steps = []

    def on_move(x, y):
        nonlocal last_time
        now = time.time()
        time_steps.append(now - last_time)
        last_time = now

        # print(x, y)
    def on_scroll(x, y, dx, dy):
        print(x, y, dx, dy)

        

    with KeyboardListener(on_press=on_press, on_release=on_release) as keyboard_listener:
        with MouseListener(on_click=on_click, on_move=on_move, on_scroll=on_scroll) as mouse_listener:
            
            # 每隔10s 打印频率
            # 还不错，能到90Hz
            while True:
                now = time.time()
                time_steps.append(now - last_time)
                last_time = now
                time.sleep(10)
                print("10s frequency: ", 1.0 / (sum(time_steps) / len(time_steps)))
                time_steps = []


            keyboard_listener.join()
            mouse_listener.join()

def syn_ver():
    from pynput import mouse

    last_x, last_y = 0, 0
    # The event listener will be running in this block
    while True:
        with mouse.Events() as events:
            # Block at most one second
            event = events.get(1.0)


            if event is None:
                print('You did not interact with the mouse within one second')
            else:
                if last_x == 0 and last_y == 0:
                    last_x, last_y = event.x, event.y
                    continue
                dx, dy = event.x - last_x, event.y - last_y
                last_x, last_y = event.x, event.y
                print('Pointer moved by {0}'.format((dx, dy)))
                # print('Received event {}'.format(event))



if __name__ == "__main__":
    # thread_ver()
    syn_ver()

# 频率够用，但是鼠标位于屏幕边界时不够稳定，放弃使用python