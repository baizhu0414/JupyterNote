这里存放Jupyter代码存放位置：
c.NotebookApp.notebook_dir = 'D:\PythonPlace\Jupyter'

Test01-基础语法
 - 数据类型：集合{}，元组(tuple)，列表[list]，字典{key:val}。
 - 类型转换，in，遍历，一维数组取一段(去除首尾元素)，数组列表复制。
 
Test02-NumPy
 - 数组元素类型变换，数组reshape，生成等间隔数组，随机数组(np.random；rand,randn)，数组复制和倍数(numpy和python列表不同)。
 - 二维数组访问，深度复制，numpy多维数组拼接(axis)。
 - list*2乘法区别(Python& NumPy)。
 
Test03-Pytorch基本语法
 - torch多维数组拼接(dim)，访问单个元素和一行，原地操作数据，Tensor数组和Numpy数组转化。
 - 小数更适合张量Tensor计算。
 
Test04-Pandas数据预处理
 - os库文件访问，pandas读表格csv，填充表格空数据NaN（求平均），根据列的值拆分列，DataFrame->Numpy->tensor(PyTorch)。
 - 删除NaN最多列（DataFrame.isna,sum,max,Series.to_dict，del原地删除，drop可选择原地或复制一份）。
 - DataFrame,Series:count统计列的缺失值，dropna（删除NaN较多列）， Series.idxmax返回val最大的key。
 
Test05-线代操作torch
 - 矩阵按位乘法(A*A,A**2)，数据类型torch中type(A)，行列求和(keepdims)，平均值(mean)，元素数，广播（除法等运算）。
 - 累计求和(i行前的数字和)，一维向量点积(1D-dot内积[sum(a*b)]!)，矩阵*向量Ax(mv、Test09-matmul)，矩阵乘法(mm,Test09_2-matmul)，向量的L2范数(norm:标量张量)。
 - 向量绝对值abs，len，多维数组sum，多维数组L2范数(norm)。
 
Test06-微积分(matplotlib)
 - d2l函数定义：use_svg_display, set_figsize, set_axes, plot.
Test07-自动微分
 - 梯度(grad反向传播)，清空x的梯度(grad.zero_)，向量转标量求反向传播，固定一个公式为常数(detach)，设置请求梯度。
 - matplotlib绘图，绘制反向传播图像（限制是Numpy数组比较方便，不支持Tensor。）
 
Test08-概率
 - torch.distributions，投硬币模拟，正态分布函数绘制( X~N(μ, δ^2) )及生成数据模拟。
 - 生成序列 np.linspace(-2, 2, 100), np.arange(-2, 2, 0.1)功能类似。
 - help和 ？查函数，dir查模块。

Test09-线性神经网络(均方误差nn.MSELoss)
 - 正态分布( normal(w, b, shape) )，线性回归方程y=Xw+b+ε，矩阵乘法(matmul)，
 - 打乱数据(shuffle)，生成器(迭代器yield)，返回下标对应子序列(list[idxs])。
 - 优化算法(sgd基于小批量（mini-batch）的随机梯度下降)。
 - 深度学习框架(nn)，数据处理工具(data模块构造PyTorch数据迭代器DataLoader)。
 - 交叉熵公式(极大似然函数)，softmax[实现见9-2]
 - transforms.ToTensor图像格式转换，torchvision.datasets.FashionMNIST加载数据集，_占位符，axes.flatten数组压缩(展平)。
 - enumerate函数用于获取循环索引和对应的值；zip函数用于将一维数组元素对应起来；
 - 绘制2D单通道‘图像’[ax.imshow(img.asnumpy())，Pytorch张量->NumPy数组，Matplotlib 等库使用 NumPy 数组来绘制图像]。
 - 绘制线条[ax.plot(x, y, fmt); x,y为横纵坐标序列]
 - 迭代器访问数据集[iterator，next(iter(data.DataLoader(data, batch_size)))]，DataLoader自身也可作为迭代器访问。
 - 转换列表插入(trans.insert)。
 - 格式化小数(:f, :.2f, f-string)。
 - data_iter，load_array，get_fashion_mnist_labels, show_images, load_data_fashion_mnist.
 - synthetic_data+load_array，squared_loss(均方损失/最小二乘法)，synthetic_data。
 
Test09_2-softmax手动实现
 - 二维数组访问( y_hat[[0, 1], y] )。
 - 获取矩阵每行最大值的下标( y_hat.argmax(axis=1) )。
 - '=='比较像等( y_hat.type(y)必须同类型，值相同才True )，bool.sum求和也需要调用( cmp.type(y.dtype)转换为float列表 )。
 - 初始化0数组( [0.0]*n )，列表推导式( [expression for item in iterable]生成新列表 )
 - 元组相加会组合起来(a,b)+(c,)=(a,b,c)
 - Accumulator,Animator,cross_entropy，evaluate_accuracy，train_epoch_ch3，train_ch3，predict_ch3.
 - 生成二维空数组( x= [[] for _ in range(n)] )
 
Test09_3-softmax简单实现
 - 略，直接调用了系统的函数(nn.Linear等)。
 
Test10_感知机
 - ReLU函数，Sigmoid函数，tanh函数.
 - 参数初始化torch.randn，torch.zeros，torch.zeros_like；矩阵乘法X @ W1+b1；数据类型type(net)。
 
Test11_模型选择、过拟合
 - np.power(numpy.ndarray, ndarray)，ndarray转Tensor( torch.tensor ).
 - 数组集体赋值( true_w[0:4] = np.array([5, 4.3, -2.1, 1.2]) ).
 - evaluate_loss.
 
Test12_权重衰减、暂退法
 - Torch.Tensor：w.pow(2) 等价于 w**2 或 torch.square(w).
 - L2范式[l= loss+ lambd*l2_penalty(w)]，torch.optim.SGD(weight_decay).
 - relu+dropout层(Test09-2 train_ch3)，np.random.uniform生成随机的均匀分布的矩阵.
 
 