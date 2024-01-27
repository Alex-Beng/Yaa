# 基于rdp+新开窗口遥操作

rdp relative mouse过于难用，放弃基于rdp的遥操作！


# 直接使用cpp采集键鼠的输入事件+bitblt截屏

同时因为pynput没有原始的relative鼠标输入，且在边缘出现类似的问题，放弃python！

两个线程，分别生产键鼠事件和截屏。

通过队列提交给主线程，主线程控制采集频率。



