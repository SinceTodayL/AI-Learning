## Recording Learning What



### Project: pagerank

$ Core Formula: $
$$
PR(p) = \frac{1-d}{N} + d \sum_{i \in link(p)} \frac{PR(i)}{NumLinks(i)}
$$
首先是那个 $iterate\_pagerank$ 函数，我刚开始想用拓扑排序的想法，但是实际上是不对的，拓扑排序适用于有向无环图，但是很显然网页的链接并不一定满足无环的条件，导致拓扑排序有可能压根没有起点

后来问了DeepSeek，估计也是标准写法，就是先不管计算 $PR(p)$ 时依赖的 $PR(i)$ 是不是准确的，就一直按照公式计算，直到与前一次的差别 $TotalDiff$ 小于一个设定的阈值 $Threshold$ ，就认为已经收敛，将答案作为正确值

这种做法我以前从来没见过，虽然现在不能完全证明合理性，但是想象一下倒也合理

至于 $sample\_pagerank$ 函数，基于 $Markov Chain$ ，得出转移概率分布，然后不断模拟（利用Python的`cur_page = random.choices(list(corpus.keys()), weights = transition.values(), k = 1)[0]` 函数），利用 $Monte Carlo Method$ 的思想得出概率分布，当然，核心在于函数是否能做到接近真随机



### Project: knights

这个项目就是一堆逻辑表达式：$And, Or, Not, Implication(蕴含式)$ ，已经给了一个自动判断命题是否能成立的程序，这个程序的写法是我要学习的，里面涉及到 Python 里面很多类的写法，因此让 DeepSeek 帮我写了一个 `Class Usage in Python.py `文件，里面都是面向对象编程范式的语法
