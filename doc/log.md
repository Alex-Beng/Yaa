# ~~遥操作计划~~ 人类示范采集计划

~~rdp + 固定鼠标->相对鼠标映射。~~ rdp relative mouse太难用了
~~rdp启动+其他远控软件启动~~，似乎不能截屏，至少parsec不能。先放弃，因为远控软件不可控。
直接截屏+捕获鼠标键盘消息进行录制。不开yap。
（手动F使用其他键映射过去，有interception

pynput无法直接捕获鼠标dx dy，只能捕获绝对位置。相对位置通过差分，在屏幕边缘效果很差。
放弃python进行捕获
cpp的interception可以捕获dx dy 左右键 up down 以及 键盘的 up down~~（需要测试）~~ 测试见apps/ms_kb_test，均可捕获。
捕获频率最高在125Hz左右，可以接受。
TODO: 需要验证

测试interception的dx dy和键盘输入。
对比：autohotkey使用windows的GetAsyncKeyState和GetRawInputData来捕获键盘和鼠标输入。
~~TODO：去除多余的键盘摁下事件。在录制or回放时候实现？录制！~~
~~TODO：对比开环误差。~~ 纯dx dy误差较小，联合误差相当大，-> 说明需要learning based！学， 给我狠狠地学！
猜测误差主要来源于replay？从驱动层replay相当于bottom up？既要经过驱动又要经过OS？

以完成基于interception的键盘鼠标录制。
录制输出为jsonl。
与VPT相同。
录制内容为dx dy scancode + state + 时间戳。

高精度时 in msvc_chron.h 
"using high_resolution_clock = steady_clock;"
NMD 就是别名
三个clock，high_resolution_clock, system_clock, steady_clock
system_clock 单位是100ns，剩下两个一样

基于bitblt进行屏幕录制。
目标：达到50Hz。
录制输出为视频+对应帧的时间戳，考虑同样jsonl格式。
直接保存jpg，1600x900的单帧在311 KB左右，一段60s，50Hz的jpg格式录制结果在0.9GB左右，不能接受。
既然都有损了，还是直接上视频了。
格式与OBS录屏保持一致，.mkv + H.264


经过对比，H264压缩率较好，使用！

输出为三通道+alpha通道视频+对应时间戳。


有没有抽象的必要？
抽象个utils吧，看着难受。


## traning data resample

重新采样到hdf5，抄act的record sim episodes。

使用同样的4:3 x 3?

实际感兴趣的键盘事件（用来确定状态）

W A S D Q E
X space shift T Z
1 2 3 4
鼠标左键 滚轮 dx dy

state dim = 19

目前采集中约10min只出现了一次在0.05s内键盘click的情况，
因此考虑直接不响应click事件，只响应up down事件。
对于鼠标也是仅有一次click事件，因此也不响应click事件，只响应up down事件。

keyboard event: scancode + 0/1. 0 for down, 1 for up
mouse event: state = 0/1/2/4/8/16/32/1024

|state|event|
|---|---|
|0| move, dx, dy|
|1| left down|
|2| left up|
|4| right down|
|8| right up|
|16| middle down|
|32| middle up|
|1024| wheel, rolling, <0 for down, >0 for up, K*120 for rolling|

~~逆天，recorder写错了，wheel的rolling没有被记录。~~ fixed

~~是否需要考虑归一化rolling dx dy？~~ done in training & inference，记录action & state 的mean 和std

目前（名椎滩好感）没被用到的action dims及对应的按键：
|||
|---|---|
|6|X|
|9|T|
|10|Z|
|11|1|
|12|2|
|13|3|
|14|4|
|16|MRo|


## 推理相关

0. 需要测试推理频率
1. 需要把实际环境封装成env
2. 在windows上使用gpu推理/量化，if needed

GPU推理，1650上940MB显存，95%使用率。推理频率在25Hz左右。

CPU推理，5800H，推理用时0.4s，推理频率在2.5Hz左右。

TODO: CPU+onnx；CPU+onnx+量化。

思考：是否应该按训练时的L1 loss进行作为推理时的连续动作-> 离散动作的阈值参考？

## explain TODOs in code

### 解释 dx dy 的量级为什么这么小，可能是计算数据集mean, std时padding的问题

本来就这么小，dx dy中有大量的接近0的数据，因此mean和std都很小。

![mdx](../pics/Mdx.png)
![mdy](../pics/Mdy.png)

padding的问题，已经解决，被copilot坑了。
[-pad_len:] 补成了 [pad_len:]，修改后用新方法验证一致。

### 研究 norm 对于示教中全零数据的影响

主要是std会变成0，导致推理时候除0错误。所以act原实现进行了clip






## SOME RESULTS in intercption test

重放用时而言，重放完毕用时大概是录制用时的两倍。
在将事件之间记录的延时减半之后，依旧在录制用时的两倍左右，说明事件之间手动的sleep并不是关键路径，
而可能是interception库的发送事件的效率问题。
TODO：对比OS的事件发送效率。which is the same with AutoHotKey.
优先级不高，问题不在重放上面。有反馈保底（？）

此外，对于鼠标，一次移动x，和分n次移动-x//n，偏差很大。
SOLUTION：将Windows的鼠标设置中的“提高指针精确度”关闭，可以显著降低偏差。


截屏采集结果频率在20Hz左右，可以接受。
直接降频推理得了，反正VPT也是20Hz。
键鼠采集结果频率在80Hz左右，可以接受。

![video_diff_timing](../pics/video_diff_timing.png)
![ms_kb_diff_timing](../pics/ms_kb_diff_timing.png)

## RESULTS in act training

![act_train](../pics/train_loss.png)

有人傻逼地训练时把samp traj设成了false，相当于只用前面100步的chunk进行训练。


TODOs: 

- [ ] 删除录制开始时的空操作帧。
- [ ] 动作添加sigmoid，变为二分类，更好预测键盘离散动作。
- [ ] 对于“冷门”动作添加置信度，添加计算loss的权重。
- [ ] 或者增加“冷门“动作的采样概率。
- [ ] 量化+onnx部署。

Thank to @[XizoB](https://github.com/XizoB)

## 环境依赖

cpp的

interception -> 自动安装

nlohmann/json -> 自动安装

OpenCV -> 手动安装

H.264需要**手动下载**codec，这个与opencv版本相关。

python的

目前与act一致。见[act install](https://github.com/tonyzhaozh/act?tab=readme-ov-file#installation)。



## 一些大声BB

~~在遥端跑genshin和yap，以实现自动拾取和自动tp。~~
~~遥端TODO：自动化的tp？~~

~~主端进行截屏和post 键盘鼠标message到rdp窗口。~~

通过录制进行演示，rdp帧率实在是不行。
截屏和键鼠捕获频率均能达到50Hz，可以接受。

实测截屏只能到20Hz。



# ALOHA & Mobile ALOHA read note

## 采集频率

30Hz，遥操作。

# VPT read note

## 模型

IDM+BC。网络结构类似，IDM因为可以是非因果的，添加了“3D”卷积，即在 x y t 上进行卷积

## 采集频率

VPT：20Hz 采集和推理。

## 动作空间 

gym里面：discrete and continuous。

binary action：各种按键
连续：鼠标移动，离散到角度的bins

实际上进行了factored action space和Joint Hierarchical Action Space的实验。
认为：Joint Hierarchical Action Space可以更好地建模人类操作，
实际上表现较差，（所以大概这就是放附录的原因）

## 观测空间

VPT 是 渲染640x360原始像素，炒鸡离谱的16 : 9。输入到模型时候，resize -> 128x128，同时渲染一个鼠标指针以模拟人类操作。

~~压缩后遥操作试试。~~

## 删除空操作


## 模型容量与loss的关系

IDM更小的容量带来更好的loss，进行 BC finetune 时相反。

## RL fine tune

没看懂，大概是BC的网络使用不同的loss计算，然后更新网络参数。