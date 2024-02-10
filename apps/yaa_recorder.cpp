// 用于同时录制屏幕和键鼠
// 三个线程。
// 屏幕录制使用生产者消费者模型
// 键鼠录制使用简单的先存储再处理模型

#include <iostream>
#include <thread>
