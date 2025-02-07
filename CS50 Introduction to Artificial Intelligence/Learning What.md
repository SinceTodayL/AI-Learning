## Recording Learning What

### Project: pagerank

$ Core Formula: $

$$
PR(p) = \frac{1-d}{N} + d \sum_{i \in link(p)} \frac{PR(i)}{NumLinks(i)}
$$

首先是那个 $iterate\_pagerank$ 函数，我刚开始想用拓扑排序的想法，但是实际上是不对的，拓扑排序适用于有向无环图，但是很显然网页的链接并不一定满足无环的条件，导致拓扑排序有可能压根没有起点

后来问了DeepSeek，估计也是标准写法，就是先不管计算 $PR(p)$ 时依赖的 $PR(i)$ 是不是准确的，就一直按照公式计算，直到与前一次的差别 $TotalDiff$ 小于一个设定的阈值 $Threshold$ ，就认为已经收敛，将答案作为正确值

这种做法我以前从来没见过，虽然现在不能完全证明合理性，但是想象一下倒也合理

至于 $sample\_pagerank$ 函数，基于 $Markov Chain$ ，得出转移概率分布，然后不断模拟（利用Python的 `cur_page = random.choices(list(corpus.keys()), weights = transition.values(), k = 1)[0]` 函数），利用 $Monte Carlo Method$ 的思想得出概率分布，当然，核心在于函数是否能做到接近真随机

### Project: knights

这个项目就是一堆逻辑表达式：$And, Or, Not, Implication(蕴含式)$ ，已经给了一个自动判断命题是否能成立的程序，这个程序的写法是我要学习的，里面涉及到 Python 里面很多类的写法，因此让 DeepSeek 帮我写了一个 `Class Usage in Python.py `文件，里面都是面向对象编程范式的语法

比如 @classmethod 和 @staticmethod 的区别，前者可以接受 cls 参数，然后访问所在类中的成员和方法，适用于工厂模式，但是后者写法上和普通函数一致，不接受 self, cls 参数，但是就和C++的static成员函数一样，不能访问类中成员或非static的方法

至于什么是**工厂模式**，其重要思想在于**开闭原则**：对扩展开放，对修改关闭

@property 修饰在一个函数前面，代表这个函数被当作属性一样访问，不用最后加上 ()，而 @propertyname.setter 默认为该property的修改函数，当赋值操作（如 `property = new_value`）执行时，该函数会被自动调用

属性前面加一条下划线 (`_property`) 代表变量受保护，两条下划线(`__property`) 代表变量私有

Python同样支持运算符的重载，只是和C++中显式的对运算符操作重新写一个函数不同，Python通过重写 `__add__, __mul__` 等类似具有特殊含义名称的函数，如:

```Python
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def __add__(self, other):  // 这就是对加法运算的重载方法
        return Circle(self.radius + other.radius)
```

Python的析构函数名称是 `__del__`

### Project: minesweeper

在Python中很多对列表的操作都是原地修改，如 对list的remove、对dict的del，在迭代过程中，如果修改了该列表，就有可能会对后续的迭代造成影响，因此可以对迭代对象的副本进行操作，避免干扰迭代过程：

```python
data = {
    "temp1": 123,
    "temp2": 456,
    "data": 789,
    "temp3": 101112
}
# 复制所有键列表，再进行删除
for key in list(data.keys()):
    if key.startswith("temp"):
        del data[key]
print("删除temp开头的键后：", data)  
# 输出：{'data': 789}

```

而C++中少见这样支持自定义原地修改的函数，大部分都要自己手搓

这个项目中还有一个我第一次写没想到的，和前面pagerank的写法有点类似，就是利用while循环，不断地对知识库进行更新修改，得到新知识，直到不能更新为止，这里的代码看起来有相当高的时空复杂度，但也是得到正确答案的一种方法

### Project: heredity

看起来是一个遗传概率计算问题，实际上是贝叶斯定理运用的典型

在项目中，已知一个人是否患有某疾病，以及其亲属的表现型，已知该疾病等位基因在人群中各基因型的频率以及相应基因型的发病率，要求根据已知信息推断其他所有人的基因型和表现型的 $probability$ $distribution$ 

刚开始看到题中给的框架代码让我求 $joint$ $probability$ ，最后还要 $normalize$ ，看解法才反应过来这是要用贝叶斯，不是高中生物题:cry:

在 Uncertainty 这个 lecture 里面，从概率论的基本知识开始，详细解释了贝叶斯网络、蒙特卡罗方法、(隐)马尔可夫链，其中，贝叶斯网络可以较为精确的求出在给定 evidence 下事件的  $probability$ $distribution$，但是当节点和状态数过大时，计算量会很大；因此，我们可以利用每种状态下的 $transition$ $model$，通过Python中的随机函数去模拟这一过程，来近似的得到每一个事件的概率，也就是蒙特卡罗方法；马尔可夫链是一种根据 $conditional$ $probability$ 进行相应条件转移的方法，并且基于一个简化假设：当前状态只依赖于先前的有限个状态

lecture里面有一个说法让我印象深刻：条件概率正比于联合概率，即
$$
\exist \alpha \in constant, P(A| Condition) = \alpha P(A, Condition)
$$
感觉怪怪的，只能作为一种感性的理解，但是仔细想想，Condition 如果是一个已知的 evidence，其在计算时确实可以被当作一个常数，在某些较为简单的概率计算场景下，可以直接计算出每一种可能情况的 $joint$ $probability$ ，然后 $normalize$ ，就可以得到相应的条件概率分布，利用正比这一说法也很好理解，但是如果较为复杂，贝叶斯网络节点较多的时候，用这种方式理解恐怕有点不妥

本题其实就是利用了这种想法，求每一种可能状态的 $joint$ $probability$ ，最后 $normalize$ ，就得到了在已有一些 evidence 的情况下对应的条件分布，还需进一步理解，实际上这应该也是贝叶斯问题的核心思想
