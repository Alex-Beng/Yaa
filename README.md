<div align="center">

# Yaa
Yet Another Artificial GI-player

又一个原神AI
</div>

# 介绍

基于~~目前大热的Diffusion Policy~~、ACT等~~强化学习~~/模仿学习算法，实现原神的基于动作生成的AI。

# TODOs

- [ ] Behavior Cloning
  - [ ] ACT
  - [ ] Deffusion Policy
  - [ ] VPT ?
- [ ] RL finetune


- [x] 定义输入和输出，observation space and action space actually
    - [x] in: 图像+感知状态
    - [ ] out: 动作
      - [x] factored action space
      - [ ] Joint Hierarchical Action Space
- [ ] 图像
  - [ ] ~~主画面+EQ+小地图~~
  - [x] 主画面+alpha通道
- [ ] 状态。（先不做）
  - [x] 键鼠状态
  - [ ] 体力条（dl？）
  - [ ] 人物血量
- [x] 基于~~遥操作~~录制的的人类示范
    - [ ] ~~基于asyncio的窗口遥操作~~
        - [x] 窗口捕获
        - [x] 获取键鼠
        - [x] ~~内插~~/采样/重放到20Hz，类似act的record sim episodes
        - [x] 训练数据可视化，类似act的visualize episodes
        - [ ] ~~转发键鼠~~
- [ ] 训练
    - [ ] all end 2 end / 拆分
      - [x] 首先尝试稻妻好感
      - [ ] 尝试固定c位+每日的e2e？
    - [ ] 任务s
        - [ ] 稻妻两个固定位置好感
        - [ ] 进行深渊的打
        - [ ] 特定配对的自动战斗 in 大世界
        - [ ] 从特定锚点出发拾取狗粮
        - [ ] 进行地脉的打
- [ ] 评估方法？
  - [ ] ~~action L1 loss 作为离散动作的阈值~~ 添加激活函数，变为二分类
- [ ] 推理部署
  - [x] 在数据上进行推理
  - [ ] 在真实环境进行推理
  - [ ] onnx部署+测试推理速度，需要达到20Hz


TODOs more important now:

- [x] 重新设计action head。添加sigmoid，变为分类，更好预测键盘离散动作。
- [ ] 对于“冷门”动作添加置信度，添加计算loss的权重。
- [ ] 或者增加“冷门“动作的采样概率。
- [ ] 删除录制开始时的空操作帧。
- [ ] 真实环境中的量化+onnx部署
- [ ] 测试去掉CVAE encoder模型
- [ ] 测试CNNMLP模型
- [x] 去掉ImageNet的norm？make it to -0.5~0.5
- [ ] backbone 不给梯度的选项
- [x] 键盘和鼠标动作使用不同loss，需要引入权重超参？
- [ ] click动作实际上比想象中常见，处理方法？
- [ ] fix 推理的 action 处理
- [ ] 单相机输入 tries（xjb弄了下，应该是norm的问题
- [ ] 鼠标变为离散动作 tries，类似VPT
  - [ ] L1 loss + BCE loss -> 负对数似然，同时也能添加loss的权重


Thank to @[XizoB](https://github.com/XizoB) for the first few TODOs.


# 仓库组成

- ``apps/`` cpp写的apps，每个cpp文件对应一个应用
  - ``bb_utils.h`` bitblt截屏的utils，copy from [cvat](https://github.com/GengGode/cvAutoTrack).
  - ``ms_kb_utils.h`` 使用interception库进行键鼠录制的utils
  - ``bitblt_test.cpp`` 基于生产者消费者模型的bitblt截屏测试
  - ``jsonl_test.cpp`` jsonl文件的读写测试
  - ``ms_kb_recorder.cpp`` 键盘+鼠标的录制测试
  - ``ms_kb_test.cpp`` 键盘+鼠标的捕获测试
  - ``yaa_recorder.cpp`` 进行人类示教的录制程序
- ``cmake/`` 第三方库的recipes
- ``detr/`` 模型的定义，modified from act, which is also modified from detr.
- ``doc/`` 亿点文档
- ``pics/`` 图片
- ``validates/`` 初期的python验证脚本，基本都没用
- ``act_imitate_learning.py`` ACT网络的模仿学习
- ``act_infer.py`` ACT网络的推理测试
- ``constants.py`` 在python脚本中共享的常量
- ``gi_env.py`` GI环境的类gym.env封装
- ``policy.py`` ACT policy的定义
- ``print_dataset_stats.py`` 打印示教数据集的normalize stats
- ``print_hdf5.py`` 打印重采样录制数据集的结构和维度数据
- ``record.ps1`` windows下的录制脚本
- ``resample_record.py`` 重采样录制到指定频率
- ``test.sh`` 固定参数运行 act_infer.py 的脚本
- ``train.sh`` 固定参数运行 act_imitate_learning.py 的脚本
- ``utils.py`` 数据集定义和处理的utils，以及其他一些utils
- ``vis.sh`` 简化参数运行 visualize_hdf5.py 的脚本
- ``visualize_hdf5.py`` 可视化重采样后的录制数据集

# 使用


分为录制示教和训练两个阶段（还有推理，但是还没做）。
名椎滩好感原始录制数据和使用colab进行训练notebook见[这里](https://drive.google.com/drive/folders/1m2RxUXDbJZ8_RCGmZfGicHqoe-YuQYEP?usp=drive_link)。

工作流为：录制 -> 重采样到20Hz -> 训练 -> 推理

## 录制示教

基于CPP。

- Windows 10
- MSVC 我也不知道需要什么版本
- CMake 3.14 单纯是凑个π，我也不知道需要什么版本

CPP的依赖库通过CMake下载，所以不需要额外的依赖。

运行时需要手动下载opencv编码视频依赖的dll，见运行时opencv的提示，有网页链接。


编译：
```shell
mkdir build
cd build
cmake ..
MSBuild.exe ALL_BUILD.vcxproj /p:Configuration=Release
```

运行：
```shell
./bin/release/yaa_recorder.exe --help
```

录制将会输出到：
```cpp
// output_path / task_name / 
//  {episode_id}.mp4 
//  {episode_id}_alpha.mp4 
//  {episode_id}_mskb.jsonl
//  {episode_id}_video.json
```

|名字|内容|
|---|---|
|{episode_id}.mp4|主画面|
|{episode_id}_alpha.mp4|alpha通道|
|{episode_id}_mskb.jsonl|键鼠事件|
|{episode_id}_video.json|视频帧时间戳|


## 训练

基于Python。

环境配置与act一致。见[act install](https://github.com/tonyzhaozh/act?tab=readme-ov-file#installation)。

重采样录制数据集到20Hz：
```shell
python resample_record.py --help
```

训练：
```shell
python act_imitate_learning.py --help
```
