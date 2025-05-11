"""
    Pytorch : Use GPU to speed up training
"""

import torch

'''
    tensor : just like ndarray in numpy, but can use GPU to calculate
'''


x1 = torch.zeros((3, 3), dtype=int)
# print(x1)

x2 = x1.numpy()
# print(x2)
# 转换为 numpy, 也可以从 numpy 转换来 : x2 = torch.from_numpy(x1)

# rand 和 randn 的区别 : 一个输出 (0,1) 的 均匀分布, 一个输出 (0,1) 的 标准正态分布, 二者语法上没有区别
'''
    torch.rand(*size,  out=None, dtype=None, layout=torch.strided,  device=None, requires_grad=False)
    torch.randn(*size,  out=None, dtype=None, layout=torch.strided,  device=None, requires_grad=False)
'''
x3 = torch.rand(2, 3, 4, 5)
# print(x3)
# print(x3.dtype)

x4 = torch.tensor([2, 4, 3])
# print(x4)
# print(x4.dtype)

x5 = x1.new_ones(2, 3, 4)
# print(x5)
# print(x1)

x6 = torch.rand(5, 3)

x7 = torch.cat((x1, x6), dim=0)
# print(x7)
# print(x7.shape)

x8 = torch.rand(4,5,6)
# print(x8)
# print(x8[1,2,3])
# print(x8[2,:,3:5])  # 这种切片操作 3:5 是左闭右开的


# 证明在 CPU 上运算速度比 GPU 运行速度慢
'''
import time

size = 10000
a_cpu = torch.rand((size, size))
b_cpu = torch.rand((size, size))

# 在 CPU 上计算
start_cpu = time.time()
result_cpu = torch.mm(a_cpu, b_cpu) 
end_cpu = time.time()
print(f"CPU 计算时间: {end_cpu - start_cpu:.6f} seconds")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
a_gpu = a_cpu.to(device)
b_gpu = b_cpu.to(device)

# 在 GPU 上计算
start_gpu = time.time()
result_gpu = torch.mm(a_gpu, b_gpu)
torch.cuda.synchronize()  # 确保所有 GPU 操作完成
end_gpu = time.time()
print(f"GPU 计算时间: {end_gpu - start_gpu:.6f} seconds")
'''

'''
    how to use GPU ?
    see the program (comment) above
    use 'to' method' : x.to("cuda")
'''
# 当然, 也可以直接将变量设置在 GPU 上 ( GPU 的 VRAM 上)
x9 = torch.rand((3, 4), device="cuda")

# print(torch.cuda.get_device_name())


'''
    Autograd : 自动微分引擎
'''

x10 = torch.randn((3,5), requires_grad=True)
y1 = 2
# print(x10 + y1)

x11 = torch.tensor([2,4,1], dtype=float, requires_grad=True)
y = x11 * x11 + 7 * x11
y.retain_grad()
z = y.sum()
z.retain_grad()
z.backward()
print(x11.grad)
# print(y.grad) 这是默认不允许的, y 是非叶子张量, 除非上面声明了 y.retain_grad()
print(z.grad)