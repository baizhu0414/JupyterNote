这里存放Jupyter代码存放位置：
c.NotebookApp.notebook_dir = 'D:\PythonPlace\Jupyter'

Test01-基础语法
 - 数据类型：集合{}，元组()，列表[]，字典{key:val}
 - 类型转换，in，遍历，一维数组取一段(去除首尾元素)，数组列表复制
 
Test02-NumPy
 - 数组元素类型变换，数组reshape，生成等间隔数组，随即数组
 - 二维数组访问，深度复制，numpy多维数组拼接(axis)。
 
Test03-Pytorch基本语法
 - torch多维数组拼接(dim)，访问单个元素和一行，原地操作数据，Tensor数组和Numpy数组转化
 - 小数更适合张量Tensor计算。
 
Test04-Pandas数据预处理
 - os库文件访问，pandas读表格csv，填充表格空数据NaN（求平均），根据列的值拆分列，DataFrame->Numpy->tensor(PyTorch)
 - 删除NaN最多列（DataFrame.isna,sum,max,Series.to_dict，del原地删除，drop可选择原地或复制一份）
 - DataFrame,Series:count统计列的缺失值，dropna（删除NaN较多列）， Series.idxmax返回val最大的key。
 
Test05-线代操作
 - 矩阵按位乘法(A*A,A**2)，数据类型torch中type(A)，行列求和(keepdims)，平均值(mean)，元素数，广播（除法等运算）
 - 累计求和(i行前的数字和)，一维向量点积(dot内积)，矩阵*向量Ax(mv、Test09-matmul)，矩阵乘法mm，向量的L2范数(norm)
 - 向量绝对值abs，len，多维数组sum，多维数组L2范数(norm)。
 
Test06-微积分(matplotlib)
 - 略，见下一文件
Test07-自动微分
 - 梯度(反向传播)，清空x的梯度(grad.zero_)，向量转标量求反向传播，固定一个公式为常数(detach)，设置请求梯度。
 - matplotlib绘图，绘制反向传播图像。
 
Test08-概率
 - torch.distributions，投硬币模拟，正态分布函数绘制。
 - help查函数，dir查模块，？查函数

Test09-线性神经网络
 - 正态分布(normal)，线性回归方程y=Xw+b+ε，矩阵乘法(matmul)，生成器(迭代器yield)，返回下标对应子序列(list[idxs])
 - 优化算法(sgd基于小批量（mini-batch）的随机梯度下降)