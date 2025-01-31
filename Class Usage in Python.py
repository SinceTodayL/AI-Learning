# -*- coding: utf-8 -*-
"""
Python 类（class）的完整示例文件
覆盖以下知识点：
1. 定义类
2. 构造函数 __init__
3. 实例属性和实例方法
4. 类属性和类方法
5. 静态方法
6. 继承和方法重写
7. 特殊方法（魔术方法）
8. 访问控制（私有属性和方法）
9. 属性装饰器 @property
10. 抽象基类
11. 枚举类
12. 类的序列化（pickle）
13. 类的迭代器
14. 类的上下文管理器
15. 类的描述符
16. 类的元类
17. 类的装饰器
"""

import pickle
from abc import ABC, abstractmethod
from enum import Enum

# ==============================================
# 1. 定义类
# ==============================================
class MyClass:
    """这是一个简单的类示例"""
    pass


# ==============================================
# 2. 构造函数 __init__
# ==============================================
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age


# ==============================================
# 3. 实例属性和实例方法
# ==============================================
class Student:
    def __init__(self, name, age):
        self.name = name  # 实例属性
        self.age = age

    def introduce(self):  # 实例方法
        print(f"Hi, I'm {self.name}, {self.age} years old.")


# ==============================================
# 4. 类属性和类方法
# ==============================================
class Counter:
    count = 0  # 类属性

    def __init__(self):
        Counter.count += 1  # 修改类属性

    @classmethod
    def get_count(cls):  # 类方法
        return cls.count


# ==============================================
# 5. 静态方法
# ==============================================
class MathUtils:
    @staticmethod
    def add(a, b):  # 静态方法
        return a + b


# ==============================================
# 6. 继承和方法重写
# ==============================================
class Animal:
    def speak(self):
        print("Animal speaks")

class Dog(Animal):  # 继承
    def speak(self):  # 方法重写
        print("Dog barks")


# ==============================================
# 7. 特殊方法（魔术方法）
# ==============================================
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):  # 定义字符串表示
        return f"Point({self.x}, {self.y})"

    def __add__(self, other):  # 定义加法行为
        return Point(self.x + other.x, self.y + other.y)


# ==============================================
# 8. 访问控制（私有属性和方法）
# ==============================================
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # 私有属性

    def __display_balance(self):  # 私有方法
        print(f"Balance: {self.__balance}")

    def show_balance(self):
        self.__display_balance()


# ==============================================
# 9. 属性装饰器 @property
# ==============================================
class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):  # 获取属性
        return self._radius

    @radius.setter
    def radius(self, value):  # 设置属性
        if value > 0:
            self._radius = value
        else:
            raise ValueError("Radius must be positive")


# ==============================================
# 10. 抽象基类
# ==============================================
class Shape(ABC):  # 抽象基类
    @abstractmethod
    def area(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):  # 实现抽象方法
        return self.width * self.height


# ==============================================
# 11. 枚举类
# ==============================================
class Color(Enum):  # 枚举类
    RED = 1
    GREEN = 2
    BLUE = 3


# ==============================================
# 12. 类的序列化（pickle）
# ==============================================
class Data:
    def __init__(self, value):
        self.value = value

    def save(self, filename):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        with open(filename, "rb") as file:
            return pickle.load(file)


# ==============================================
# 13. 类的迭代器
# ==============================================
class CountDown:
    def __init__(self, start):
        self.start = start

    def __iter__(self):  # 定义迭代器
        return self

    def __next__(self):  # 定义下一个值
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start


# ==============================================
# 14. 类的上下文管理器
# ==============================================
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode

    def __enter__(self):  # 进入上下文
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):  # 退出上下文
        self.file.close()


# ==============================================
# 15. 类的描述符
# ==============================================
class Descriptor:
    def __get__(self, instance, owner):
        return instance._value

    def __set__(self, instance, value):
        instance._value = value

class MyClassWithDescriptor:
    attr = Descriptor()  # 描述符属性

    def __init__(self, value):
        self._value = value


# ==============================================
# 16. 类的元类
# ==============================================
class Meta(type):  # 自定义元类
    def __new__(cls, name, bases, dct):
        print(f"Creating class {name}")
        return super().__new__(cls, name, bases, dct)

class MyMetaClass(metaclass=Meta):  # 使用元类
    pass


# ==============================================
# 17. 类的装饰器
# ==============================================
def class_decorator(cls):  # 类装饰器
    cls.new_attr = "new value"
    return cls

@class_decorator
class DecoratedClass:
    pass


# ==============================================
# 测试代码
# ==============================================
if __name__ == "__main__":
    # 1. 定义类
    obj = MyClass()
    print(f"1. 定义类: {obj}")

    # 2. 构造函数 __init__
    person = Person("Alice", 25)
    print(f"2. 构造函数: {person.name}, {person.age}")

    # 3. 实例属性和实例方法
    student = Student("Bob", 20)
    student.introduce()

    # 4. 类属性和类方法
    c1 = Counter()
    c2 = Counter()
    print(f"4. 类属性: {Counter.get_count()}")

    # 5. 静态方法
    print(f"5. 静态方法: {MathUtils.add(10, 20)}")

    # 6. 继承和方法重写
    dog = Dog()
    dog.speak()

    # 7. 特殊方法（魔术方法）
    p1 = Point(1, 2)
    p2 = Point(3, 4)
    print(f"7. 特殊方法: {p1 + p2}")

    # 8. 访问控制（私有属性和方法）
    account = BankAccount(1000)
    account.show_balance()

    # 9. 属性装饰器 @property
    circle = Circle(5)
    print(f"9. 属性装饰器: {circle.radius}")
    circle.radius = 10
    print(f"修改后半径: {circle.radius}")

    # 10. 抽象基类
    rect = Rectangle(10, 20)
    print(f"10. 抽象基类: {rect.area()}")

    # 11. 枚举类
    print(f"11. 枚举类: {Color.RED}")

    # 12. 类的序列化（pickle）
    data = Data(42)
    data.save("data.pkl")
    loaded_data = Data.load("data.pkl")
    print(f"12. 类的序列化: {loaded_data.value}")

    # 13. 类的迭代器
    print("13. 类的迭代器:")
    for num in CountDown(5):
        print(num)

    # 14. 类的上下文管理器
    with FileManager("test.txt", "w") as file:
        file.write("Hello, World!")
    print("14. 类的上下文管理器: 文件已写入")

    # 15. 类的描述符
    obj = MyClassWithDescriptor(100)
    print(f"15. 类的描述符: {obj.attr}")

    # 16. 类的元类
    print("16. 类的元类: 查看控制台输出")

    # 17. 类的装饰器
    print(f"17. 类的装饰器: {DecoratedClass.new_attr}")