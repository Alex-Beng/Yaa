<div align="center">

# Yaa
Yet Another Artificial GI-player

又一个原神AI
</div>

# 介绍

基于~~目前大热的Diffusion Policy~~、ACT等强化学习/模仿学习算法，实现原神的基于动作生成的AI。

# TODOs


- [x] 定义输入和输出
    - [x] in: 图像+感知状态
    - [x] out: 动作
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
- [ ] 推理部署
  - [ ] onnx部署+测试推理速度，需要达到20Hz

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
- ``visualize_hdf5.py`` 可视化重采样后的录制数据集