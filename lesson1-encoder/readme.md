## Encoder模型

### 1. Implement a basic RNN model to classify which country a name belongs to(More specific)

**实现一个最基础的RNN模型用于名字所属国家的分类**
代码相对于原视频做了如下更改（可以直接Ctrl+F搜索“更改”找到相应的更改位置）：

- 视频中删除掉了torch.save没保存模型参数，因为当时忘了用torch是这么保存的（用其他工具导致混淆了），后面加上了保存模型参数的代码
- 后来自己测试了一遍，发现最后一行预测类别的categorys写成了category，已经改成了categorys[predict]
- 为了方便各位直接使用训练好的模型参数进行训练而不是每次都等待一段时间，在evaluating阶段加入了导入模型参数的操作，在训练完一次有了模型参数之后注释掉train(epochs)，然后直接运行就可以进行预测了

## 2. Implement a basic RNN model to classify which country a name belongs to(More specific)