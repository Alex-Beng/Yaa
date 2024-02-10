// 用于测试interception的键鼠捕获 以及 发送

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <windows.h>
#include "interception.h"
#include "ms_kb_utils.h"



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

    if (false) {        
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
        
    }

    // 测试输入鼠标移动
    device = INTERCEPTION_MOUSE(1);
    // countdown(2);
    InterceptionMouseStroke mstroke;
    mstroke.state = INTERCEPTION_MOUSE_MOVE_RELATIVE;
    mstroke.flags = 0;
    mstroke.rolling = 0;
    mstroke.x = 100;
    mstroke.y = 0;
    mstroke.information = 0;
    interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&mstroke), 1);

    countdown(2);
    auto send_n = 5;
    for (auto i=0; i<send_n; i++) {
        mstroke.state = INTERCEPTION_MOUSE_MOVE_RELATIVE;
        mstroke.flags = 0;
        mstroke.rolling = 0;
        mstroke.x = -100/send_n;
        mstroke.y = 0;
        mstroke.information = 0;
        interception_send(ctx, device, reinterpret_cast<InterceptionStroke *>(&mstroke), 1);
    }
}

int main() {
    // test_listen();
    test_send();
    return 0;
}