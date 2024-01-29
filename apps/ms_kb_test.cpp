// 用于测试interception的键鼠捕获 以及 发送

#include <iostream>
#include <vector>
#include <string>
#include <thread>
#include <windows.h>
#include "interception.h"

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

    // 测试输入wasd、空格、esc
    
}

int main() {
    test_listen();
    // test_send();
    return 0;
}