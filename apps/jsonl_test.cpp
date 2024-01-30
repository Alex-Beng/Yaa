// 用于json line文件的读写测试
// generate by copilot chat
#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

void read() {
    std::fstream file("data.jsonl");
    if (!file.is_open()) {
        std::cout << "Failed to open file." << std::endl;
        return;
    }
    nlohmann::json j_read;
    std::string line;
    while (std::getline(file, line)) {
        j_read = nlohmann::json::parse(line);
        std::cout << j_read.dump() << std::endl;
    }
    file.close();
}

void write() {
    
}

int main() {
    std::ofstream file("data.jsonl");
    if (!file.is_open()) {
        std::cout << "Failed to open file." << std::endl;
        return 1;
    }

    nlohmann::json json1 = {
        {"name", "John"},
        {"age", 30}
    };
    nlohmann::json json2 = {
        {"name", "Jane"},
        {"age", 25}
    };

    file << json1.dump() << std::endl;
    file << json2.dump() << std::endl;

    file.close();
    read();
    return 0;
}