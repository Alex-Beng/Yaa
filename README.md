<div align="center">

# Yaa
Yet Another Artificial GI-player

又一个原神AI
</div>

# 介绍

基于强化学习/模仿学习算法，实现原神的基于动作生成的AI。

目前处于开发阶段，开发计划见issue。
大致计划是先模仿学习，然后通过强化学习进行微调。

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
- ``imitate_learning.py`` BC训练，目前可用act/mlp
- ``infer.py`` BC推理，目前可用act/mlp
- ``config.py`` 训练和推理的配置
- ``constants.py`` py脚本中共享的常量
- ``gi_env.py`` GI环境的类gym.env封装
- ``policy.py`` ACT/MLP BC policy的定义
- ``print_dataset_stats.py`` 打印示教数据集的normalize stats
- ``print_hdf5.py`` 打印重采样录制数据集的结构和维度数据
- ``shell_scripts/`` 一些shell脚本
  - ``record.ps1`` windows下的录制脚本
  - ``vis.sh`` 简化参数运行 visualize_hdf5.py 的脚本
- ``utils.py`` 数据集定义和处理的utils，以及其他一些utils
- ``resample_record.py`` 重采样录制到指定频率
- ``visualize_hdf5.py`` 可视化重采样后的录制数据集

# BC使用


分为录制示教、训练和推理三个阶段。
目前推理还在使用pytorch，没有做任何部署和量化。

可使用notebook + colab的方式进行训练和推理：
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

## 推理

基于Python。

推理：
```shell
python infer.py --help
```
