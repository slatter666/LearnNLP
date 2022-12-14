## Encoder模型

### 1. Implement a basic RNN model to classify which country a name belongs to(More specific)

**实现一个最基础的RNN模型用于名字所属国家的分类**
代码相对于原视频做了如下更改（可以直接Ctrl+F搜索“更改”找到相应的更改位置）：

- 视频中删除掉了torch.save没保存模型参数，因为一直没用过这个函数都是用工具自动保存的，后面加上了保存模型参数的代码
- 后来自己测试了一遍，发现最后一行预测类别的categorys写成了category，已经改成了categorys[predict]
- 为了方便各位直接使用训练好的模型参数进行训练而不是每次都等待一段时间，在evaluating阶段加入了导入模型参数的操作，在训练完一次有了模型参数之后注释掉train(epochs)，然后直接运行就可以进行预测了
- 输出best accuracy的时候拼写错误已更改
- 第一行打印hyperparameters拼写错误已更改
- accuracy写到循环中无法正常记录最大值已更改

## 2. Implement a basic RNN model to classify which country a name belongs to(More Concise)
**用Embedding实现一个RNN模型用于名字所属国家的分类**
代码相对于原视频做了如下更改：

- accuracy写到循环中无法正常记录最大值已更改

## 3. Implement a LSTM model to classify which country a name belongs to
**用LSTM模型用于名字所属分类，采用batch训练，数据集拆分为训练集、验证集、测试集**
代码相对于原视频做了如下更改：
- 暂无

## 4. Implement a model based on average of word vector to classify IMDB dataset
**用词向量平均做IMDB数据集的分类**
代码相对于原视频做了如下更改：
- 修改了learning rate值为0.01

## 5. Implement a bidirectional LSTM model to classify IMDB dataset
**用双向LSTM做IMDB数据集的分类**
代码相对于原视频做了如下更改：
- 暂无

## 6. Classify IMDB dataset with pytorch-lightning
**用双向LSTM做IMDB数据集的分类（使用pytorch-lightning）**
代码相对于原视频做了如下更改：