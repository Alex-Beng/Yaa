# Scancode

键盘扫描码，用于定义键盘的按键。用于键盘的驱动程序中。

interception使用的是set 1。[参考手册](https://www.marjorie.de/ps2/scancode-set1.htm)

# CROSS ENTROPY IMPLEMENTATIONS，交叉熵的实现


Implementation A: `torch.nn.functional.binary_cross_entropy` (seetorch.nn.BCELoss): the input values to this function have already had a sigmoid applied, e.g.
二元交叉熵，输入已经经过了sigmoid函数。

Implementation B:`torch.nn.functional.binary_cross_entropy_with_logits`(see torch.nn.BCEWithLogitsLoss): "this loss combines a Sigmoid layer and the BCELoss in one
single class. This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability."
这个loss结合了sigmoid和BCELoss，更稳定。使用了log-sum-exp技巧。


Implementation C:torch.nn.functional.nll_loss(see torch.nn.NLLLoss) : "the negative log likelihood loss. It is useful to train a classification problem with C classes. [...] The input given through a forward call is expected to contain log-probabilities of each class."
负对数似然损失。用于训练C类分类问题。输入是每个类的对数概率。


Implementation D: torch.nn.functional.cross_entropy(see torch.nn.CrossEntropyLoss): "this criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class. It is useful when training a classification problem with C classes. [...] The input is expected to contain raw, unnormalized scores for each class."
这个损失结合了LogSoftmax和NLLLoss。用于训练C类分类问题。输入是每个类的未归一化的分数。



引用：
1. [pytorch lossfunc cheetsheets](https://github.com/rasbt/stat479-deep-learning-ss19/blob/master/other/pytorch-lossfunc-cheatsheet.md)
2. [Connections: Log Likelihood, Cross-Entropy, KL Divergence, Logistic Regression, and Neural Networks](https://zhuanlan.zhihu.com/p/136169338)