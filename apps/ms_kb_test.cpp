// 用于测试interception的键鼠捕获 以及 发送

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <windows.h>
#include "interception.h"

enum SCANCODE {
    SCANCODE_W = 0x11,
    SCANCODE_A = 0x1E,
    SCANCODE_S = 0x1F,
    SCANCODE_D = 0x20,
    SCANCODE_SPACE = 0x39,
    SCANCODE_ESC = 0x01,
    SCANCODE_1 = 0x02,
    SCANCODE_2 = 0x03
};

void countdown(int seconds) {
    for (int i = seconds; i >= 0; i--) {
        std::cout << "Countdown: " << i << " seconds" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << "Countdown complete!" << std::endl;
}

void test_listen()
{
    InterceptionContext ctx = interception_create_context();
    InterceptionDevice device;
    InterceptionStroke stroke;

    // 捕获鼠标移动以及键盘按键
    interception_set_filter(ctx, interception_is_mouse, 
                                INTERCEPTION_FILTER_MOUSE_ALL);
    interception_set_filter(ctx, interception_is_keyboard, 
                                INTERCEPTION_FILTER_KEY_DOWN|INTERCEPTION_FILTER_KEY_UP|INTERCEPTION_FILTER_KEY_E0);

    while (interception_receive(ctx, device = interception_wait(ctx), &stroke, 1) > 0) {
        if (interception_is_mouse(device)) {
            // mouse 尝试输出dx dy
            InterceptionMouseStroke &mstroke = reinterpret_cast<InterceptionMouseStroke &>(stroke);
            
            // 直接输出全部的信息
            std::cout << "Got mouse stroke: ";
            std::cout << "state : " << mstroke.state << ", flags : " << mstroke.flags << ", rolling : " << mstroke.rolling << ", x : " << mstroke.x << ", y : " << mstroke.y << ", information : " << mstroke.information << std::endl;


            interception_send(ctx, device, &stroke, 1);
        }
        
        if (interception_is_keyboard(device)) {
            // keyboard
            InterceptionKeyStroke &kstroke = reinterpret_cast<InterceptionKeyStroke &>(stroke);
            std::cout << "Got keyboard stroke: ";
            std::cout << "code : " << kstroke.code << ", state : " << kstroke.state << std::endl;
            interception_send(ctx, device, &stroke, 1);
        }
    }

    interception_destroy_context(ctx);
}

void test_send() {
    // 用于测试发送
    InterceptionContext ctx = interception_create_context();
    InterceptionDevice device;
    InterceptionStroke stroke;


    // 测试输入wasd、空格、esc、12
    device = INTERCEPTION_KEYBOARD(1);
    countdown(2);
    InterceptionKeyStroke kstroke;
    kstroke.code = SCANCODE_W;
    kstroke.state = INTERCEPTION_KEY_DOWN;
    // 需要类型转换
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&kstroke), 1);
    countdown(2);
    kstroke.code = SCANCODE_W;
    kstroke.state = INTERCEPTION_KEY_UP;
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&kstroke), 1);

    countdown(2);
    kstroke.code = SCANCODE_A;
    kstroke.state = INTERCEPTION_KEY_DOWN;
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&kstroke), 1);
    kstroke.code = SCANCODE_A;
    kstroke.state = INTERCEPTION_KEY_UP;
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&kstroke), 1);

    countdown(2);
    kstroke.code = SCANCODE_S;
    kstroke.state = INTERCEPTION_KEY_DOWN;
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&kstroke), 1);
    kstroke.code = SCANCODE_S;
    kstroke.state = INTERCEPTION_KEY_UP;
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&kstroke), 1);

    countdown(2);
    kstroke.code = SCANCODE_D;
    kstroke.state = INTERCEPTION_KEY_DOWN;
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&kstroke), 1);
    kstroke.code = SCANCODE_D;
    kstroke.state = INTERCEPTION_KEY_UP;
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&kstroke), 1);

    countdown(2);
    kstroke.code = SCANCODE_SPACE;
    kstroke.state = INTERCEPTION_KEY_DOWN;
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&kstroke), 1);
    kstroke.code = SCANCODE_SPACE;
    kstroke.state = INTERCEPTION_KEY_UP;
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&kstroke), 1);

    countdown(2);
    kstroke.code = SCANCODE_ESC;
    kstroke.state = INTERCEPTION_KEY_DOWN;
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&kstroke), 1);
    kstroke.code = SCANCODE_ESC;
    kstroke.state = INTERCEPTION_KEY_UP;
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&kstroke), 1);    

    countdown(2);
    kstroke.code = SCANCODE_1;
    kstroke.state = INTERCEPTION_KEY_DOWN;
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&kstroke), 1);
    kstroke.code = SCANCODE_1;
    kstroke.state = INTERCEPTION_KEY_UP;
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&kstroke), 1);

    countdown(2);
    kstroke.code = SCANCODE_2;
    kstroke.state = INTERCEPTION_KEY_DOWN;
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&kstroke), 1);
    kstroke.code = SCANCODE_2;
    kstroke.state = INTERCEPTION_KEY_UP;
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&kstroke), 1);

    // 测试输入鼠标移动
    device = INTERCEPTION_MOUSE(1);
    countdown(2);
    InterceptionMouseStroke mstroke;
    mstroke.state = INTERCEPTION_MOUSE_MOVE_RELATIVE;
    mstroke.flags = 0;
    mstroke.rolling = 0;
    mstroke.x = 10;
    mstroke.y = 10;
    mstroke.information = 0;
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&mstroke), 1);

    countdown(2);
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&mstroke), 1);
}

int main() {
    test_listen();
    // test_send();
    return 0;
}