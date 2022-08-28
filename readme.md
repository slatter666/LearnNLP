## 手把手带你学NLP

### 1.需要学习的先导课程

这里放的都是官方地址，可以自行YouTube或者Bilibili寻找资源

- [吴恩达的深度学习](https://www.coursera.org/specializations/deep-learning) 
- [吴恩达的机器学习](https://www.coursera.org/specializations/deep-learning)
- [CS224n: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/)
- [龙良曲的Pytorch教程](https://study.163.com/course/introduction.htm?courseId=1208894818&_trace_c_p_k2_=bb6975ba72e04750a94f5ffd834d3c6b/)

当然在基础理论这一块有人觉得李宏毅讲得也不错，之前看过李宏毅讲transformer确实也很不错，可以自己搜索相关资源观看。不过个人比较推荐吴恩达，因为这边是英文讲课而且吴恩达这个英文相对来说并不会让人感觉特别费劲，是到CS224N的一个过渡，CS224N的全英文授课信息量就比较大了，如果英语不太好会有点难受。李宏毅当你在英文这边没听懂那么可以去那边补知识

### 2.Encoder实践

- 实现一个最基础的RNN模型来对名字分类，判断其所属国家
- 直接用Pytorch内置RNN模型来对名字分类，判断其所属国家
- 用Pytorch内置LSTM模型对名字分类，判断其所属国家
- 实现一个比较简单的基于词向量平均模型对IMDB数据集分类
- 用双向LSTM对IMDB数据集分类（将最终的hidden作为分类网络的输入）
- 用双向LSTM对IMDB数据集分类（将所有hidden作为分类网络的输入，介绍工具pytorch-lightning）

### 3.Encoder-Decoder实践

- 实现从阿拉伯数字到汉字的转换

暂时先写到这儿吧，先把前面的部分录完再说，精力有限估计一周两更或三更



