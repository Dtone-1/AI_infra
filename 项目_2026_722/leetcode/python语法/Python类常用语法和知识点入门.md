# Python 类的常用语法和知识点入门

## 1. 类是什么？

类可以理解为“模板”或“设计图”，对象是根据这个模板创建出来的具体实例。

例如：

```python
class Student:
    pass
```

这里 `Student` 是一个类。类本身只是定义规则，还没有真正创建具体学生。

创建对象时写：

```python
s = Student()
```

这里 `s` 就是根据 `Student` 类创建出来的对象，也叫实例。

可以简单记成：

```text
类 = 模板 / 设计图
对象 = 根据类创建出来的具体东西
实例化 = 用类创建对象的过程
```

---

## 2. 最简单的类定义

```python
class Student:
    pass
```

解释：

```text
class：定义类的关键字
Student：类名，通常首字母大写
pass：占位语句，表示这个类暂时什么也不做
```

使用：

```python
s1 = Student()
s2 = Student()

print(s1)
print(s2)
```

`s1` 和 `s2` 都是 `Student` 类创建出来的对象，但它们是两个不同对象。

---

## 3. `__init__`：对象初始化方法

类最常见的写法是定义 `__init__` 方法。

```python
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
```

使用：

```python
s = Student("张三", 20)

print(s.name)
print(s.age)
```

输出：

```text
张三
20
```

解释：

```text
__init__：创建对象时自动执行的方法
self：当前对象自己
name、age：创建对象时传入的参数
self.name：当前对象保存的 name 属性
self.age：当前对象保存的 age 属性
```

当你写：

```python
s = Student("张三", 20)
```

大致等价于：

```text
创建一个 Student 对象
把 "张三" 传给 name
把 20 传给 age
执行 __init__
把 name 和 age 保存到对象内部
```

---

## 4. `self` 是什么？

`self` 表示当前这个对象自己。

例如：

```python
class Student:
    def __init__(self, name):
        self.name = name
```

创建两个对象：

```python
s1 = Student("张三")
s2 = Student("李四")

print(s1.name)
print(s2.name)
```

输出：

```text
张三
李四
```

这里：

```text
创建 s1 时，self 表示 s1
创建 s2 时，self 表示 s2
```

所以：

```python
self.name = name
```

不是给类统一保存一个 name，而是给“当前对象”保存一个 name。

---

## 5. 实例属性

实例属性是每个对象自己拥有的数据。

```python
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
```

这里的：

```python
self.name
self.age
```

就是实例属性。

不同对象的实例属性可以不同：

```python
s1 = Student("张三", 20)
s2 = Student("李四", 22)

print(s1.name, s1.age)
print(s2.name, s2.age)
```

输出：

```text
张三 20
李四 22
```

记忆：

```text
实例属性 = 对象自己保存的数据
```

---

## 6. 实例方法

类里面定义的函数通常叫“方法”。

```python
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        print(f"我叫 {self.name}，今年 {self.age} 岁")
```

使用：

```python
s = Student("张三", 20)
s.introduce()
```

输出：

```text
我叫 张三，今年 20 岁
```

解释：

```text
introduce 是实例方法
s.introduce() 表示让 s 这个对象执行 introduce 方法
方法内部可以通过 self.name、self.age 访问当前对象的数据
```

---

## 7. 实例化对象和调用方法的区别

```python
s = Student("张三", 20)
```

这是实例化对象，意思是创建一个 `Student` 对象。

```python
s.introduce()
```

这是调用对象的方法，意思是让已经创建好的对象执行某个动作。

对应关系：

```text
Student(...)      创建对象
s.introduce()     使用对象的方法
```

类比到 nano-vLLM：

```python
llm = LLM(path, tensor_parallel_size=1)
outputs = llm.generate(prompts, sampling_params)
```

可以理解为：

```text
LLM(...)        创建一个推理引擎对象
llm.generate()  调用这个对象的生成方法
```

---

## 8. 类属性

类属性是属于类本身、所有对象共享的数据。

```python
class Student:
    school = "某某大学"

    def __init__(self, name):
        self.name = name
```

使用：

```python
s1 = Student("张三")
s2 = Student("李四")

print(s1.school)
print(s2.school)
print(Student.school)
```

输出：

```text
某某大学
某某大学
某某大学
```

解释：

```text
school 是类属性
所有 Student 对象共享这个属性
```

区别：

```text
实例属性：每个对象各自一份，如 self.name
类属性：类统一拥有一份，如 Student.school
```

---

## 9. 类属性的典型例子：计数器

```python
from itertools import count

class Sequence:
    counter = count()

    def __init__(self):
        self.seq_id = next(Sequence.counter)
```

使用：

```python
s1 = Sequence()
s2 = Sequence()
s3 = Sequence()

print(s1.seq_id)
print(s2.seq_id)
print(s3.seq_id)
```

输出：

```text
0
1
2
```

解释：

```text
Sequence.counter 是类属性，所有 Sequence 对象共享同一个计数器
next(Sequence.counter) 每次取出下一个编号
self.seq_id 是实例属性，每个对象保存自己的编号
```

类比：

```text
Sequence.counter = 排号机
next(Sequence.counter) = 取下一张号码
self.seq_id = 当前对象拿到的号码
```

---

## 10. `@property`：把方法当属性用

有些值不想直接保存，而是想动态计算，可以用 `@property`。

```python
class Sequence:
    def __init__(self, prompt_tokens, total_tokens):
        self.num_prompt_tokens = prompt_tokens
        self.num_tokens = total_tokens

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens
```

使用：

```python
s = Sequence(prompt_tokens=5, total_tokens=8)

print(s.num_tokens)
print(s.num_completion_tokens)
```

输出：

```text
8
3
```

注意：

```python
s.num_completion_tokens
```

不是普通实例属性，而是 `@property` 修饰的方法。访问它时，Python 会自动执行：

```python
return self.num_tokens - self.num_prompt_tokens
```

区别：

```text
self.num_tokens：真实保存的数据
@property num_completion_tokens：动态计算出来的结果
```

---

## 11. 特殊方法：`__len__`

`__len__` 可以让对象支持 `len(obj)`。

```python
class Sequence:
    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.num_tokens = len(token_ids)

    def __len__(self):
        return self.num_tokens
```

使用：

```python
s = Sequence([10, 20, 30])
print(len(s))
```

输出：

```text
3
```

解释：

```text
len(s) 本质上会调用 s.__len__()
```

---

## 12. 特殊方法：`__getitem__`

`__getitem__` 可以让对象支持下标访问。

```python
class Sequence:
    def __init__(self, token_ids):
        self.token_ids = token_ids
        self.num_tokens = len(token_ids)

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]
```

使用：

```python
s = Sequence([10, 20, 30, 40])

print(s[0])
print(s[-1])
print(s[1:3])
```

输出：

```text
10
40
[20, 30]
```

解释：

```text
s[0]   本质上调用 s.__getitem__(0)
s[-1]  本质上调用 s.__getitem__(-1)
s[1:3] 本质上调用 s.__getitem__(slice(1, 3))
```

这个写法可以让自定义对象表现得像列表。

---

## 13. 特殊方法：`__repr__`

`__repr__` 控制对象打印时的显示形式，主要用于调试。

```python
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def __repr__(self):
        return f"Student(name={self.name!r}, age={self.age!r})"
```

使用：

```python
s = Student("张三", 20)
print(s)
```

输出：

```text
Student(name='张三', age=20)
```

如果不写 `__repr__`，打印对象可能是：

```text
<__main__.Student object at 0x...>
```

不方便看对象内部数据。

---

## 14. 继承

继承表示一个类可以复用另一个类的能力。

```python
class Animal:
    def eat(self):
        print("正在吃东西")

class Dog(Animal):
    def bark(self):
        print("汪汪叫")
```

使用：

```python
dog = Dog()
dog.eat()
dog.bark()
```

输出：

```text
正在吃东西
汪汪叫
```

解释：

```text
Dog 继承 Animal
Dog 自己没有写 eat，但可以使用 Animal 的 eat 方法
```

nano-vLLM 里类似：

```python
class LLM(LLMEngine):
    pass
```

意思是：

```text
LLM 继承 LLMEngine
LLM 自己不新增逻辑，但拥有 LLMEngine 的能力
```

---

## 15. `super()`：调用父类方法

如果子类想在自己的初始化中复用父类初始化，可以用 `super()`。

```python
class Animal:
    def __init__(self, name):
        self.name = name

class Dog(Animal):
    def __init__(self, name, color):
        super().__init__(name)
        self.color = color
```

使用：

```python
dog = Dog("小黑", "黑色")

print(dog.name)
print(dog.color)
```

输出：

```text
小黑
黑色
```

解释：

```text
super().__init__(name) 调用父类 Animal 的初始化逻辑
self.color = color 是 Dog 自己新增的属性
```

---

## 16. `*args` 和 `**kwargs`

### 16.1 `*args`

`*args` 接收多余的位置参数，打包成元组。

```python
def func(*args):
    print(args)

func(1, 2, 3)
```

输出：

```text
(1, 2, 3)
```

### 16.2 `**kwargs`

`**kwargs` 接收多余的关键字参数，打包成字典。

```python
def func(**kwargs):
    print(kwargs)

func(a=1, b=2)
```

输出：

```text
{'a': 1, 'b': 2}
```

在类初始化中常见：

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        self.model = model
        self.kwargs = kwargs
```

调用：

```python
engine = LLMEngine("Qwen", tensor_parallel_size=1, enforce_eager=True)
```

此时：

```python
model = "Qwen"
kwargs = {
    "tensor_parallel_size": 1,
    "enforce_eager": True,
}
```

---

## 17. 类型注解

Python 类和函数中经常看到类型注解。

```python
class Sequence:
    def __init__(self, token_ids: list[int]):
        self.token_ids = token_ids

    def __len__(self) -> int:
        return len(self.token_ids)
```

解释：

```text
token_ids: list[int] 表示 token_ids 应该是整数列表
-> int 表示这个函数应该返回 int
```

注意：Python 默认不会强制检查类型注解，它主要给编辑器、类型检查器和读代码的人看。

常见类型注解：

```python
name: str
age: int
score: float
is_finished: bool
token_ids: list[int]
outputs: list[dict]
```

---

## 18. `dataclass`：快速定义数据类

如果一个类主要用于保存数据，可以用 `dataclass` 简化。

```python
from dataclasses import dataclass

@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
```

使用：

```python
params = SamplingParams(temperature=0.6, max_tokens=256)
print(params)
```

输出类似：

```text
SamplingParams(temperature=0.6, max_tokens=256, ignore_eos=False)
```

`dataclass` 会自动生成：

```text
__init__
__repr__
__eq__
```

常见写法：

```python
@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
```

`slots=True` 表示：

```text
限制对象只能拥有声明过的字段
减少内存开销
防止误添加不存在的属性
```

---

## 19. `__post_init__`：dataclass 初始化后检查

```python
from dataclasses import dataclass

@dataclass
class SamplingParams:
    temperature: float = 1.0

    def __post_init__(self):
        assert self.temperature > 0, "temperature must be positive"
```

使用：

```python
params = SamplingParams(temperature=0.6)
```

正常。

```python
params = SamplingParams(temperature=0)
```

报错：

```text
AssertionError: temperature must be positive
```

解释：

```text
__post_init__ 会在 dataclass 自动生成的 __init__ 执行之后自动调用
常用于参数合法性检查、类型转换、计算派生字段
```

---

## 20. 一个完整例子：模拟大模型请求 Sequence

下面用一个简化版 `Sequence` 类，把前面的知识点串起来。

```python
from itertools import count
from dataclasses import dataclass
from enum import Enum, auto
from copy import copy


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params: SamplingParams):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

        if self.num_completion_tokens >= self.max_tokens:
            self.status = SequenceStatus.FINISHED

    def __repr__(self):
        return (
            f"Sequence(seq_id={self.seq_id}, "
            f"status={self.status.name}, "
            f"num_tokens={self.num_tokens}, "
            f"num_prompt_tokens={self.num_prompt_tokens}, "
            f"num_completion_tokens={self.num_completion_tokens})"
        )
```

使用这个类：

```python
params = SamplingParams(temperature=0.6, max_tokens=3)

seq = Sequence(token_ids=[101, 102, 103], sampling_params=params)

print(seq)
print(len(seq))
print(seq[0])
print(seq.prompt_token_ids)
print(seq.num_completion_tokens)
print(seq.is_finished)

seq.status = SequenceStatus.RUNNING
seq.append_token(201)
seq.append_token(202)
seq.append_token(203)

print(seq)
print(seq.token_ids)
print(seq.num_completion_tokens)
print(seq.is_finished)
```

可能输出：

```text
Sequence(seq_id=0, status=WAITING, num_tokens=3, num_prompt_tokens=3, num_completion_tokens=0)
3
101
[101, 102, 103]
0
False
Sequence(seq_id=0, status=FINISHED, num_tokens=6, num_prompt_tokens=3, num_completion_tokens=3)
[101, 102, 103, 201, 202, 203]
3
True
```

---

## 21. 这个完整例子里每个语法点对应什么？

```python
class Sequence:
```

定义类。

```python
block_size = 256
counter = count()
```

类属性，所有对象共享。

```python
def __init__(self, token_ids, sampling_params):
```

初始化方法，创建对象时自动执行。

```python
self.seq_id = next(Sequence.counter)
```

从类共享计数器中取一个唯一编号，保存到当前对象。

```python
self.token_ids = copy(token_ids)
```

复制传入的 token 列表，避免外部修改影响对象内部状态。

```python
def __len__(self):
```

让对象支持 `len(seq)`。

```python
def __getitem__(self, key):
```

让对象支持 `seq[0]`、`seq[-1]`、`seq[1:3]`。

```python
@property
```

让方法像属性一样访问，例如 `seq.num_completion_tokens`。

```python
def append_token(self, token_id):
```

普通实例方法，表示给当前序列追加一个新 token。

```python
def __repr__(self):
```

控制打印对象时的显示形式。

---

## 22. 最核心总结

Python 类的核心知识点可以压缩成：

```text
class 定义类
对象 = 类名(...) 实例化出来的结果
__init__ 创建对象时自动执行
self 表示当前对象自己
self.xxx 是实例属性
类名.xxx 是类属性
对象.方法() 调用实例方法
@property 可以把方法伪装成属性访问
__len__ 让对象支持 len(obj)
__getitem__ 让对象支持 obj[index]
__repr__ 控制对象打印形式
继承可以复用父类能力
super() 可以调用父类方法
@dataclass 可以快速创建数据容器类
```

读源码时重点判断：

```text
这是类属性，还是实例属性？
这是普通方法，还是 property？
这是在创建对象，还是在调用对象方法？
这个值是存起来的，还是动态计算出来的？
```

只要能分清这几件事，读 Python 类源码会容易很多。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-leetcode|模块-leetcode]]

%% 项目关联导航：结束 %%
