<div align="center">

# Yaa
Yet Another Artificial GI-player

又一个原神AI
</div>

# 介绍

基于目前大热的Diffusion Policy、ACT等强化学习/模仿学习算法，实现原神的基于动作生成的AI。

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