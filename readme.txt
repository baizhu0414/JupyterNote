这里存放Jupyter代码存放位置：
c.NotebookApp.notebook_dir = 'D:\PythonPlace\Jupyter'

Test01-基础语法
 - 数据类型：集合{}，元组()，列表[]，字典{key:val}。
 - 类型转换，in，遍历，一维数组取一段(去除首尾元素)，数组列表复制。
 
Test02-NumPy
 - 数组元素类型变换，数组reshape，生成等间隔数组，随机数组(np.random)。
 - 二维数组访问，深度复制，numpy多维数组拼接(axis)。
 
Test03-Pytorch基本语法
 - torch多维数组拼接(dim)，访问单个元素和一行，原地操作数据，Tensor数组和Numpy数组转化。
 - 小数更适合张量Tensor计算。
 
Test04-Pandas数据预处理
 - os库文件访问，pandas读表格csv，填充表格空数据NaN（求平均），根据列的值拆分列，DataFrame->Numpy->tensor(PyTorch)。
 - 删除NaN最多列（DataFrame.isna,sum,max,Series.to_dict，del原地删除，drop可选择原地或复制一份）。
 - DataFrame,Series:count统计列的缺失值，dropna（删除NaN较多列）， Series.idxmax返回val最大的key。
 
Test05-线代操作
 - 矩阵按位乘法(A*A,A**2)，数据类型torch中type(A)，行列求和(keepdims)，平均值(mean)，元素数，广播（除法等运算）。
 - 累计求和(i行前的数字和)，一维向量点积(dot内积)，矩阵*向量Ax(mv、Test09-matmul)，矩阵乘法mm，向量的L2范数(norm)。
 - 向量绝对值abs，len，多维数组sum，多维数组L2范数(norm)。
 
Test06-微积分(matplotlib)
 - 略，见下一文件。
Test07-自动微分
 - 梯度(反向传播)，清空x的梯度(grad.zero_)，向量转标量求反向传播，固定一个公式为常数(detach)，设置请求梯度。
 - matplotlib绘图，绘制反向传播图像（限制是Numpy数组比较方便，不支持Tensor。）
 
Test08-概率
 - torch.distributions，投硬币模拟，正态分布函数绘制。
 - help查函数，dir查模块，？查函数。

Test09-线性神经网络
 - 正态分布(normal)，线性回归方程y=Xw+b+ε，矩阵乘法(matmul)，生成器(迭代器yield)，返回下标对应子序列(list[idxs])。
 - 优化算法(sgd基于小批量（mini-batch）的随机梯度下降)。
 - 深度学习框架(nn)，数据处理工具(data模块构造PyTorch数据迭代器DataLoader)。
 - 交叉熵公式(极大似然函数)，softmax
 - transforms.ToTensor图像格式转换，torchvision.datasets.FashionMNIST加载数据集，_占位符，axes.flatten数组压缩(展平)。
 - enumerate函数用于获取循环索引和对应的值；zip函数用于将一维数组元素对应起来。
 - 迭代器访问数据集[iterator，next(iter(data.DataLoader(data, batch_size)))]，DataLoader自身也可作为迭代器访问。
 - 转换列表插入(trans.insert)。
 - 格式化小数(:f, :.2f, f-string)。
 
Test09_2-softmax手动实现
 - 二维数组访问( y_hat[[0, 1], y] )。
 - 获取矩阵每行最大值的下标( y_hat.argmax(axis=1) )。
 - '=='比较像等( y_hat.type(y)必须同类型，值相同才True )，bool.sum求和也需要调用( cmp.type(y.dtype) )。
 - 初始化0数组( [0.0]*n )，列表推导式( [expression for item in iterable]生成新列表 )
 
 
