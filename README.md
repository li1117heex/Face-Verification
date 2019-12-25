# Face-Verification

## 想法

- 预计会采用几大神经网络中的一种，如Alexnet等等
- 可以对flatten层的结果上进行其他模型如SVM，或者单纯判断距离，不知道有没有效果
- 可以对输入的图片进行opencv定位剪裁，别人很多都这么干，还有dlib
- 可以对输入的图片加色差，灰度，噪声，翻转等等，更match测试集，也增加训练数据
- 代码工作和利用Keras等进行的模型测试可以一起来

## 12/25

- 神经网络剩余：conv，maxpooling一部分，各种函数，优化器

## 连接

[排名](https://paperswithcode.com/sota/face-verification-on-labeled-faces-in-the)

[中文](https://zhuanlan.zhihu.com/p/76541084)

[keras中文文档](https://keras.io/zh/)