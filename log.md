# 遥操作计划

~~rdp + 固定鼠标->相对鼠标映射。~~ rdp relative mouse太难用了
~~rdp启动+其他远控软件启动~~，似乎不能截屏，至少parsec不能。先放弃，因为远控软件不可控。
直接截屏+捕获鼠标键盘消息进行录制。不开yap。
（手动F使用其他键映射过去，有interception

pynput无法直接捕获鼠标dx dy，只能捕获绝对位置。相对位置通过差分，在屏幕边缘效果很差。
放弃python进行捕获
cpp的interception可以捕获dx dy 左右键 up down 以及 键盘的 up down（需要测试）




在遥端跑genshin和yap，以实现自动拾取和自动tp。
遥端TODO：自动化的tp？

主端进行截屏和post 键盘鼠标message到rdp窗口。

# rdp 遥操作配置log

https://github.com/Night-stars-1/Auto_Simulated_Universe_Docs/blob/docs/docs/guide/bs.md


使用多用户rdp至同一台电脑，
实现yap在遥端运行，
Yaa在主端采集图像和操作数据。

rdp还需要relative鼠标输入。
https://github.com/xyzlancehe/rdp_relative_mouse


# ALOHA & Mobile ALOHA read note

## 采集频率

30Hz，遥操作。

# VPT read note
## 采集频率



VPT：20Hz 采集和

## 动作空间 


gym里面：discrete and continuous。

binary action：各种按键
连续：鼠标移动，离散到角度的bins




# 观测空间

VPT 是 渲染640x360原始像素，炒鸡离谱的16 : 9。输入到模型时候，resize -> 128x128，同时渲染一个鼠标指针以模拟人类操作。




压缩后遥操作试试。

# 删除空操作

