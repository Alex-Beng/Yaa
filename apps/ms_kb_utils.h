#ifndef MS_KB_UTILS_H
#define MS_KB_UTILS_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <thread>
#include <chrono>
#include <windows.h>
#include "interception.h"
#include "nlohmann/json.hpp"

// defines
enum SCANCODE {
    // 先把能用到的键盘按键都列出来
    SCANCODE_Q = 0x10,
    SCANCODE_W = 0x11,
    SCANCODE_E = 0x12,
    SCANCODE_R = 0x13,
    SCANCODE_T = 0x14,
    SCANCODE_Y = 0x15,
    SCANCODE_I = 0x17,
    SCANCODE_O = 0x18,
    SCANCODE_P = 0x19,

    SCANCODE_A = 0x1E,
    SCANCODE_S = 0x1F,
    SCANCODE_D = 0x20,
    SCANCODE_F = 0x21,
    SCANCODE_G = 0x22,
    SCANCODE_H = 0x23,
    SCANCODE_J = 0x24,
    SCANCODE_L = 0x26,

    SCANCODE_Z = 0x2C,
    SCANCODE_X = 0x2D,
    SCANCODE_C = 0x2E,
    SCANCODE_V = 0x2F,// V not used, just for backup
    SCANCODE_B = 0x30,
    SCANCODE_N = 0x31,// N not used, just for backup
    SCANCODE_M = 0x32,
    
    SCANCODE_ESC = 0x01,
    SCANCODE_1 = 0x02,
    SCANCODE_2 = 0x03,
    SCANCODE_3 = 0x04,
    SCANCODE_4 = 0x05,
    SCANCODE_SPACE = 0x39,
    SCANCODE_LSHIFT = 0x2A,
    SCANCODE_LTAB = 0x0F,
    SCANCODE_LALT = 0x38,
    SCANCODE_LCTRL = 0x1D,
    
    SCANCODE_F1 = 0x3B,
    SCANCODE_F2 = 0x3C,
    SCANCODE_F3 = 0x3D,
    SCANCODE_F4 = 0x3E,
    SCANCODE_F5 = 0x3F,
    SCANCODE_F6 = 0x40,
};

void countdown(int seconds) {
    for (int i = seconds; i >= 0; i--) {
        std::cout << "Countdown: " << i << " seconds" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    std::cout << "Countdown complete!" << std::endl;
}

enum EVENT_TYPE {
    EVENT_TYPE_MOUSE,
    EVENT_TYPE_KEYBOARD
};

enum MOUSE_EVENT_TYEP {
    MOUSE_EVENT_RELATIVE = INTERCEPTION_MOUSE_MOVE_RELATIVE,
    MOUSE_EVENT_LEFT_DOWN = INTERCEPTION_MOUSE_LEFT_BUTTON_DOWN,
    MOUSE_EVENT_LEFT_UP = INTERCEPTION_MOUSE_LEFT_BUTTON_UP,
    MOUSE_EVENT_RIGHT_DOWN = INTERCEPTION_MOUSE_RIGHT_BUTTON_DOWN,
    MOUSE_EVENT_RIGHT_UP = INTERCEPTION_MOUSE_RIGHT_BUTTON_UP,
    MOUSE_EVENT_MIDDLE_DOWN = INTERCEPTION_MOUSE_MIDDLE_BUTTON_DOWN,
    MOUSE_EVENT_MIDDLE_UP = INTERCEPTION_MOUSE_MIDDLE_BUTTON_UP,
};

enum KEYBOARD_EVENT_TYPE {
    KEYBOARD_EVENT_DOWN = INTERCEPTION_KEY_DOWN,
    KEYBOARD_EVENT_UP = INTERCEPTION_KEY_UP,
};

struct ABMouseEvent {
    short event_type;
    int dx;
    int dy;
};

struct ABKeyboardEvent {
    short event_type;
    int scancode;
};

struct ABEvent {
    EVENT_TYPE type;
    long long  timestamp;
    union {
        ABMouseEvent mouse;
        ABKeyboardEvent keyboard;
    };
};

// functions

bool mskbevts2jsonl(std::vector<ABEvent>& events, std::string path) {
    std::ofstream file(path);
    if (!file.is_open()) {
        std::cout << "Failed to open file." << std::endl;
        return false;
    }

    for (auto& event : events) {
        nlohmann::json json;
        json["timestamp"] = event.timestamp;
        if (event.type == EVENT_TYPE_MOUSE) {
            json["type"] = "mouse";
            json["dx"] = event.mouse.dx;
            json["dy"] = event.mouse.dy;
            json["event_type"] = event.mouse.event_type;
        } else if (event.type == EVENT_TYPE_KEYBOARD) {
            json["type"] = "keyboard";
            json["scancode"] = event.keyboard.scancode;
            json["event_type"] = event.keyboard.event_type;
        }

        file << json.dump() << std::endl;
    }

    file.close();
    return true;
}

bool jsonl2mskbevts(std::vector<ABEvent>& events, std::string path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cout << "Failed to open file." << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        nlohmann::json json = nlohmann::json::parse(line);
        ABEvent event;
        event.timestamp = json["timestamp"];
        std::string type = json["type"];
        if (type == "mouse") {
            event.type = EVENT_TYPE_MOUSE;
            event.mouse.dx = json["dx"];
            event.mouse.dy = json["dy"];
            event.mouse.event_type = json["event_type"];
        } else if (type == "keyboard") {
            event.type = EVENT_TYPE_KEYBOARD;
            event.keyboard.scancode = json["scancode"];
            event.keyboard.event_type = json["event_type"];
        }
        events.push_back(event);
    }
    return true;
}

void wait_untill_press(SCANCODE scancode) {
    InterceptionContext ctx = interception_create_context();
    InterceptionDevice device;
    InterceptionStroke stroke;
    interception_set_filter(ctx, interception_is_keyboard, INTERCEPTION_FILTER_KEY_DOWN);
    while (interception_receive(ctx, device = interception_wait(ctx), &stroke, 1) > 0) {
        // 还是得发回去先
        interception_send(ctx, device, &stroke, 1);
        if (interception_is_keyboard(device)) {
            InterceptionKeyStroke &kstroke = reinterpret_cast<InterceptionKeyStroke &>(stroke);
            if (kstroke.code == scancode) {
                interception_destroy_context(ctx);
                return;
            }
        }
    }
}



#endif
