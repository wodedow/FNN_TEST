# FNN_TEST
测试环节只需要前向传播，不涉及参数的更新

网络结构的创建与初始化
```c/c++
  int L = 3;  //神经网络的层数
  int m[] = { img_size, 300,50, classes };  //每一层的结点数：m[1:]
  float accuracy = 0;

  FNN Net;
  Init_Network_FNN(Net, L, m);
```
读取训练程序保存的权重与偏置数据
* `read_weight_arrays` : 读取权重 `w` 的数据
* `read_bias` : 读取偏置 `b`
***
通过全局变量 `numbers` 来控制测试集的数目，最后输出准确率
