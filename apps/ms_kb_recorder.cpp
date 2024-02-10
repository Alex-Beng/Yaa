// 用于录制键盘鼠标 -> jsonl文件
// 以及重放，用于测试开环误差

#include "ms_kb_utils.h"

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <windows.h>
#include "interception.h"
#include "nlohmann/json.hpp"

void record() {
    // 摁F6开始录制
    std::cout << "Press F6 to start recording...\n" << std::endl;
    wait_untill_press(SCANCODE_F6);

    // 花上2^16打个lut
    // 表示按键状态，以去除键盘重复的
    bool key_state[1<<16] = {false};

    // 先存到内存里，然后再写入文件
    std::vector<ABEvent> events;
    InterceptionContext ctx = interception_create_context();
    InterceptionDevice device;
    InterceptionStroke stroke;


    std::cout << "Start recording..." << std::endl;
    countdown(3);
    std::cout << "Recording..." << std::endl;
    
    interception_set_filter(ctx, 
                            interception_is_mouse, 
                            INTERCEPTION_FILTER_MOUSE_ALL);
    interception_set_filter(ctx, 
                            interception_is_keyboard, 
                            INTERCEPTION_FILTER_KEY_DOWN|INTERCEPTION_FILTER_KEY_UP|INTERCEPTION_FILTER_KEY_E0);
    
    // 使用相对时间戳
    auto start_timestamp = std::chrono::high_resolution_clock::now();
    while (interception_receive(ctx, device = interception_wait(ctx), &stroke, 1) > 0) {
        ABEvent event;
        
        // 我直接取ns!
        auto curr_time = std::chrono::high_resolution_clock::now();
        auto curr_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(curr_time - start_timestamp).count();
        event.timestamp = curr_timestamp;
        if (interception_is_mouse(device)) {
            // 直接send，避免可能存在的性能问题
            interception_send(ctx, device, &stroke, 1);
            InterceptionMouseStroke &mstroke = reinterpret_cast<InterceptionMouseStroke &>(stroke);

            // 输出全部的信息
            // std::cout << "Got mouse stroke: ";
            // std::cout << "state : " << mstroke.state << ", flags : " << mstroke.flags << ", rolling : " << mstroke.rolling << ", x : " << mstroke.x << ", y : " << mstroke.y << ", information : " << mstroke.information << std::endl;

            event.type = EVENT_TYPE_MOUSE;
            event.mouse.dx = mstroke.x;
            event.mouse.dy = mstroke.y;
            event.mouse.event_type = mstroke.state;
        }

        if (interception_is_keyboard(device)) {
            interception_send(ctx, device, &stroke, 1);
            InterceptionKeyStroke &kstroke = reinterpret_cast<InterceptionKeyStroke &>(stroke);
            // 摁F6结束录制
            if (kstroke.code == SCANCODE_F6) {
                std::cout << "Stop recording..." << std::endl;
                break;
            }
            if (key_state[kstroke.code] == true && kstroke.state == INTERCEPTION_KEY_DOWN) {
                // 如果已经按下了，就不要再记录了
                continue;
            }
            key_state[kstroke.code] = kstroke.state == INTERCEPTION_KEY_DOWN;


            // 输出全部的信息
            // std::cout << "Got keyboard stroke: ";
            // std::cout << "code : " << kstroke.code << ", state : " << kstroke.state << ", information : " << kstroke.information << std::endl;

            event.type = EVENT_TYPE_KEYBOARD;
            event.keyboard.scancode = kstroke.code;
            event.keyboard.event_type = kstroke.state;
        } 

        events.push_back(event);
    }
    interception_destroy_context(ctx);
    std::cout << "Saving to file..." << std::endl;
    mskbevts2jsonl(events, "data.jsonl");

}

void process_event_time(std::vector<ABEvent>& events) {
    for (int i = events.size() - 1; i >= 1; i--) {
        events[i].timestamp -= events[i - 1].timestamp;
    }
    events[0].timestamp = 0;
    
}

void replay() {
    std::vector<ABEvent> events;
    jsonl2mskbevts(events, "data.jsonl");
    process_event_time(events);

    std::cout << "Press F6 to start replaying...\n" << std::endl;
    wait_untill_press(SCANCODE_F6);
    
    countdown(3);

    InterceptionContext ctx = interception_create_context();
    auto mouse_device = INTERCEPTION_MOUSE(1);
    auto keyboard_device = INTERCEPTION_KEYBOARD(1);
    InterceptionKeyStroke kstroke;
    InterceptionMouseStroke mstroke;

    for (auto& event : events) {
        // std::this_thread::sleep_until(std::chrono::system_clock::from_time_t(event.timestamp));
        // 直接sleep
        std::cout << "Sleep for: " << event.timestamp / 1e9 << " seconds" << std::endl;
        std::this_thread::sleep_for(std::chrono::nanoseconds(event.timestamp));
        if (event.type == EVENT_TYPE_KEYBOARD) {
            kstroke.code = event.keyboard.scancode;
            kstroke.state = event.keyboard.event_type;
            // interception_send(ctx, device, &kstroke, 1);
            // 需要类型转换
            interception_send(ctx, keyboard_device, reinterpret_cast<InterceptionStroke *>(&kstroke), 1);
        } else if (event.type == EVENT_TYPE_MOUSE) {
            
            mstroke.state = INTERCEPTION_MOUSE_MOVE_RELATIVE;
            mstroke.flags = 0;
            mstroke.rolling = 0;

            mstroke.x = event.mouse.dx;
            mstroke.y = event.mouse.dy;
            mstroke.state = event.mouse.event_type;
            
            // interception_send(ctx, device, &mstroke, 1);
            interception_send(ctx, mouse_device, reinterpret_cast<InterceptionStroke *>(&mstroke), 1);
        }
    }
    interception_destroy_context(ctx);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        record();
        std::cout << "Usage: ms_kb_recorder [record|replay]" << std::endl;
        return 0;
    }
    std::string mode = argv[1];
    if (mode == "record") {
        record();
    }
    else if (mode == "replay") {
        replay();
    }
    else {
        std::cout << "Usage: ms_kb_recorder [record|replay]" << std::endl;
        return -1; 
    }
    return 0;
}
