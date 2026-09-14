# Python 基本数据结构学习笔记：列表、元组、字典、集合

## 0. 总览：四种结构先建立整体印象

Python 中最常用的四种基础数据结构是：

| 数据结构 | 英文 | 基本形式 | 是否有序 | 是否可修改 | 是否允许重复 | 主要用途 |
|---|---|---|---|---|---|---|
| 列表 | `list` | `[1, 2, 3]` | 有序 | 可修改 | 允许重复 | 保存一串数据，按顺序访问、增删改 |
| 元组 | `tuple` | `(1, 2, 3)` | 有序 | 不可修改 | 允许重复 | 保存固定不变的一组数据 |
| 字典 | `dict` | `{"name": "Tom"}` | 有序 | 可修改 | key 不允许重复 | 用 key 查 value，保存映射关系 |
| 集合 | `set` | `{1, 2, 3}` | 无序 | 可修改 | 不允许重复 | 去重、判断是否存在、集合运算 |

一句话记忆：

```text
list  ：一串可修改的数据
 tuple：一串不可修改的数据
 dict ：key-value 映射表
 set  ：不重复元素集合
```

---

## 1. 列表 list

### 1.1 列表是什么

列表是 Python 中最常用的序列结构，用来保存一组有顺序的数据。

```python
nums = [10, 20, 30, 40]
```

它的特点是：

```text
1. 有顺序
2. 可以通过下标访问
3. 可以修改、添加、删除元素
4. 允许重复元素
```

例如：

```python
nums = [10, 20, 20, 30]
print(nums)
```

输出：

```text
[10, 20, 20, 30]
```

列表允许重复，所以两个 `20` 可以同时存在。

---

### 1.2 创建列表

```python
empty_list = []
nums = [1, 2, 3]
names = ["Tom", "Jerry", "Alice"]
mixed = [1, "hello", True, 3.14]
```

虽然列表可以混合不同类型，但实际写代码时，通常更推荐同一个列表中保存同一类数据，例如：

```python
token_ids = [101, 102, 103, 104]
prompts = ["你好", "介绍一下大模型推理"]
```

---

### 1.3 访问列表元素

列表使用下标访问，下标从 `0` 开始。

```python
nums = [10, 20, 30, 40]

print(nums[0])   # 10
print(nums[1])   # 20
print(nums[-1])  # 40，最后一个元素
```

注意：

```text
nums[0]  表示第 1 个元素
nums[1]  表示第 2 个元素
nums[-1] 表示最后一个元素
```

---

### 1.4 列表切片

切片用于取出列表的一部分。

```python
nums = [10, 20, 30, 40, 50]

print(nums[1:4])   # [20, 30, 40]
print(nums[:3])    # [10, 20, 30]
print(nums[2:])    # [30, 40, 50]
print(nums[:])     # 复制整个列表
```

切片规则：

```text
list[start:end]
包含 start，不包含 end
```

例如：

```python
nums[1:4]
```

取的是下标 `1、2、3`，不包括下标 `4`。

---

### 1.5 修改列表元素

列表是可变对象，可以直接修改。

```python
nums = [10, 20, 30]
nums[1] = 99
print(nums)
```

输出：

```text
[10, 99, 30]
```

---

### 1.6 添加元素

#### 1.6.1 `append()`：尾部添加一个元素

```python
nums = [1, 2, 3]
nums.append(4)
print(nums)
```

输出：

```text
[1, 2, 3, 4]
```

#### 1.6.2 `extend()`：一次添加多个元素

```python
nums = [1, 2]
nums.extend([3, 4, 5])
print(nums)
```

输出：

```text
[1, 2, 3, 4, 5]
```

#### 1.6.3 `insert()`：指定位置插入

```python
nums = [1, 2, 4]
nums.insert(2, 3)
print(nums)
```

输出：

```text
[1, 2, 3, 4]
```

---

### 1.7 删除元素

#### 1.7.1 `pop()`：弹出元素

```python
nums = [10, 20, 30]
last = nums.pop()

print(last)
print(nums)
```

输出：

```text
30
[10, 20]
```

也可以指定下标：

```python
nums = [10, 20, 30]
x = nums.pop(0)
print(x)     # 10
print(nums)  # [20, 30]
```

#### 1.7.2 `remove()`：按值删除

```python
nums = [10, 20, 30, 20]
nums.remove(20)
print(nums)
```

输出：

```text
[10, 30, 20]
```

`remove()` 只删除第一个匹配的元素。

---

### 1.8 遍历列表

```python
nums = [10, 20, 30]

for num in nums:
    print(num)
```

输出：

```text
10
20
30
```

如果同时需要下标和值：

```python
names = ["Tom", "Jerry", "Alice"]

for index, name in enumerate(names):
    print(index, name)
```

输出：

```text
0 Tom
1 Jerry
2 Alice
```

---

### 1.9 列表推导式

列表推导式用于快速生成列表。

基本语法：

```python
[表达式 for 临时变量 in 可迭代对象 if 条件]
```

例子：生成平方列表。

```python
nums = [1, 2, 3, 4]
squares = [x * x for x in nums]
print(squares)
```

输出：

```text
[1, 4, 9, 16]
```

加条件：只保留偶数平方。

```python
nums = [1, 2, 3, 4, 5, 6]
even_squares = [x * x for x in nums if x % 2 == 0]
print(even_squares)
```

输出：

```text
[4, 16, 36]
```

---

### 1.10 列表使用场景

列表适合：

```text
1. 保存一组有顺序的数据
2. 需要按下标访问元素
3. 需要追加、修改、删除元素
4. 需要保存允许重复的数据
```

典型例子：

```python
prompts = [
    "introduce yourself",
    "list all prime numbers within 100",
]

token_ids = [151644, 872, 198, 100345]
outputs = []
```

在大模型推理代码中，`token_ids` 通常用列表保存，因为它是一个按顺序排列的 token id 序列。

---

## 2. 元组 tuple

### 2.1 元组是什么

元组和列表很像，都是有序序列，但元组创建后不能修改。

```python
point = (3, 4)
```

特点：

```text
1. 有顺序
2. 可以通过下标访问
3. 不可修改
4. 允许重复元素
```

---

### 2.2 创建元组

```python
t1 = (1, 2, 3)
t2 = ("Tom", 20)
t3 = ()
```

只有一个元素的元组要特别注意：

```python
a = (1)
b = (1,)

print(type(a))  # int
print(type(b))  # tuple
```

输出：

```text
<class 'int'>
<class 'tuple'>
```

所以单元素元组必须写逗号：

```python
single = (1,)
```

---

### 2.3 访问元组元素

```python
point = (3, 4)

print(point[0])  # 3
print(point[1])  # 4
```

元组也支持切片：

```python
nums = (10, 20, 30, 40)
print(nums[1:3])
```

输出：

```text
(20, 30)
```

---

### 2.4 元组不可修改

```python
point = (3, 4)
point[0] = 10
```

这会报错：

```text
TypeError: 'tuple' object does not support item assignment
```

这就是元组和列表最重要的区别。

---

### 2.5 元组解包

元组经常用于一次返回多个值，或者一次赋值多个变量。

```python
point = (3, 4)
x, y = point

print(x)
print(y)
```

输出：

```text
3
4
```

函数返回多个值时，本质上也常常是返回元组：

```python
def get_state():
    return 10, 5, 3

num_tokens, num_prompt_tokens, num_cached_tokens = get_state()

print(num_tokens)
print(num_prompt_tokens)
print(num_cached_tokens)
```

输出：

```text
10
5
3
```

---

### 2.6 元组使用场景

元组适合：

```text
1. 一组数据创建后不希望被修改
2. 表示固定结构的数据，例如坐标、状态
3. 函数返回多个值
4. 用作字典的 key，前提是元组内部元素也可哈希
```

例子：

```python
position = (10, 20)
state = (num_tokens, num_prompt_tokens, block_table)
```

在源码中常见：

```python
return (
    self.num_tokens,
    self.num_prompt_tokens,
    self.num_cached_tokens,
)
```

这表示返回一个固定结构的状态元组。

---

## 3. 字典 dict

### 3.1 字典是什么

字典是 key-value 映射结构。

```python
student = {
    "name": "Tom",
    "age": 20,
    "major": "AI Infra",
}
```

结构是：

```python
{
    key1: value1,
    key2: value2,
}
```

特点：

```text
1. 通过 key 查 value
2. key 不能重复
3. value 可以重复
4. 字典可以修改
5. Python 3.7+ 中字典保持插入顺序
```

---

### 3.2 创建字典

```python
empty_dict = {}

config = {
    "model": "Qwen3-0.6B",
    "max_model_len": 2048,
    "tensor_parallel_size": 1,
}
```

注意：空的 `{}` 是字典，不是集合。

```python
x = {}
print(type(x))
```

输出：

```text
<class 'dict'>
```

空集合要写：

```python
s = set()
```

---

### 3.3 读取字典元素

```python
config = {
    "model": "Qwen3-0.6B",
    "max_model_len": 2048,
}

print(config["model"])
print(config["max_model_len"])
```

输出：

```text
Qwen3-0.6B
2048
```

如果 key 不存在，直接用 `[]` 会报错：

```python
print(config["temperature"])
```

可能报：

```text
KeyError: 'temperature'
```

安全读取可以用 `get()`：

```python
temperature = config.get("temperature", 1.0)
print(temperature)
```

输出：

```text
1.0
```

---

### 3.4 修改和新增字典元素

```python
config = {
    "model": "Qwen3-0.6B",
    "max_model_len": 2048,
}

config["max_model_len"] = 4096
config["tensor_parallel_size"] = 1

print(config)
```

输出：

```text
{'model': 'Qwen3-0.6B', 'max_model_len': 4096, 'tensor_parallel_size': 1}
```

同样的语法：

```python
config[key] = value
```

如果 key 已存在，就是修改；如果 key 不存在，就是新增。

---

### 3.5 删除字典元素

```python
config = {
    "model": "Qwen3-0.6B",
    "temperature": 0.6,
}

del config["temperature"]
print(config)
```

输出：

```text
{'model': 'Qwen3-0.6B'}
```

---

### 3.6 遍历字典

遍历 key：

```python
for key in config:
    print(key)
```

遍历 value：

```python
for value in config.values():
    print(value)
```

遍历 key-value：

```python
for key, value in config.items():
    print(key, value)
```

例子：

```python
config = {
    "model": "Qwen3-0.6B",
    "max_model_len": 2048,
    "tensor_parallel_size": 1,
}

for key, value in config.items():
    print(key, "=", value)
```

输出：

```text
model = Qwen3-0.6B
max_model_len = 2048
tensor_parallel_size = 1
```

---

### 3.7 字典推导式

字典推导式用于快速生成字典。

基本语法：

```python
{key表达式: value表达式 for 临时变量 in 可迭代对象 if 条件}
```

例子 1：数字到平方。

```python
nums = [1, 2, 3, 4]
square_dict = {x: x * x for x in nums}
print(square_dict)
```

输出：

```text
{1: 1, 2: 4, 3: 9, 4: 16}
```

例子 2：单词到长度。

```python
words = ["apple", "banana", "cat"]
length_dict = {word: len(word) for word in words}
print(length_dict)
```

输出：

```text
{'apple': 5, 'banana': 6, 'cat': 3}
```

例子 3：过滤合法配置项。

```python
kwargs = {
    "model": "Qwen3-0.6B",
    "max_model_len": 2048,
    "unknown_param": 123,
}

config_fields = {"model", "max_model_len", "tensor_parallel_size"}

valid_kwargs = {
    key: value
    for key, value in kwargs.items()
    if key in config_fields
}

print(valid_kwargs)
```

输出：

```text
{'model': 'Qwen3-0.6B', 'max_model_len': 2048}
```

---

### 3.8 字典使用场景

字典适合：

```text
1. 用名字查数据
2. 保存配置项
3. 用 id 查对象
4. 保存结构化信息
5. 表示映射关系
```

典型例子：

```python
output = {
    "text": "模型生成的内容",
    "token_ids": [101, 102, 103],
}

print(output["text"])
```

在模型推理代码中常见：

```python
outputs = [
    {"text": "hello", "token_ids": [1, 2, 3]},
    {"text": "world", "token_ids": [4, 5, 6]},
]
```

---

## 4. 集合 set

### 4.1 集合是什么

集合是保存不重复元素的数据结构。

```python
s = {1, 2, 3}
```

特点：

```text
1. 不允许重复
2. 无序
3. 可以快速判断某个元素是否存在
4. 支持交集、并集、差集等集合运算
```

---

### 4.2 创建集合

```python
s1 = {1, 2, 3}
s2 = set([1, 2, 2, 3])
empty_set = set()
```

注意：

```python
empty = {}
```

这是空字典，不是空集合。

空集合必须写：

```python
empty_set = set()
```

---

### 4.3 集合自动去重

```python
nums = [1, 2, 2, 3, 3, 3]
unique_nums = set(nums)
print(unique_nums)
```

输出类似：

```text
{1, 2, 3}
```

集合无序，所以打印顺序不保证固定。

---

### 4.4 添加和删除元素

```python
s = {1, 2, 3}

s.add(4)
print(s)

s.remove(2)
print(s)
```

输出类似：

```text
{1, 2, 3, 4}
{1, 3, 4}
```

`remove()` 删除不存在的元素会报错。

```python
s.remove(999)
```

安全删除可以用 `discard()`：

```python
s.discard(999)
```

即使元素不存在，也不会报错。

---

### 4.5 判断元素是否存在

```python
config_fields = {"model", "max_model_len", "tensor_parallel_size"}

print("model" in config_fields)
print("temperature" in config_fields)
```

输出：

```text
True
False
```

集合适合做快速存在性判断。

---

### 4.6 集合运算

```python
a = {1, 2, 3}
b = {3, 4, 5}
```

并集：

```python
print(a | b)
```

输出：

```text
{1, 2, 3, 4, 5}
```

交集：

```python
print(a & b)
```

输出：

```text
{3}
```

差集：

```python
print(a - b)
```

输出：

```text
{1, 2}
```

对称差集：

```python
print(a ^ b)
```

输出：

```text
{1, 2, 4, 5}
```

---

### 4.7 集合推导式

集合推导式用于快速生成集合。

基本语法：

```python
{表达式 for 临时变量 in 可迭代对象 if 条件}
```

例子：生成平方集合。

```python
nums = [1, 2, 2, 3]
squares = {x * x for x in nums}
print(squares)
```

输出：

```text
{1, 4, 9}
```

例子：提取字段名集合。

```python
fields = ["model", "max_model_len", "model"]
config_fields = {field for field in fields}
print(config_fields)
```

输出类似：

```text
{'model', 'max_model_len'}
```

---

### 4.8 集合使用场景

集合适合：

```text
1. 去重
2. 快速判断元素是否存在
3. 保存不重复的字段名、id、状态
4. 做交集、并集、差集运算
```

典型例子：

```python
config_fields = {"model", "max_model_len", "tensor_parallel_size"}

if "model" in config_fields:
    print("model 是合法字段")
```

在源码中常见：

```python
config_fields = {field.name for field in fields(Config)}
```

含义是：取出 `Config` 这个 dataclass 里的所有字段名，组成一个不重复集合。

---

## 5. 四种结构对比总结

| 结构 | 是否有序 | 是否可变 | 是否允许重复 | 访问方式 | 典型用途 |
|---|---|---|---|---|---|
| `list` | 有序 | 可变 | 允许 | 下标 | 顺序数据、token 列表、任务列表 |
| `tuple` | 有序 | 不可变 | 允许 | 下标 | 固定结构、函数多返回值、状态快照 |
| `dict` | 有序 | 可变 | key 不重复 | key | 配置、映射、结构化结果 |
| `set` | 无序 | 可变 | 不允许 | 不能用下标 | 去重、存在性判断、集合运算 |

---

## 6. 怎么选择使用哪一个

### 6.1 需要保存一串有顺序、会变化的数据

用 `list`。

```python
token_ids = [101, 102, 103]
token_ids.append(104)
```

---

### 6.2 需要保存一组固定不变的数据

用 `tuple`。

```python
position = (10, 20)
state = (num_tokens, num_prompt_tokens, block_table)
```

---

### 6.3 需要用名字查数据

用 `dict`。

```python
config = {
    "model": "Qwen3-0.6B",
    "temperature": 0.6,
}

print(config["model"])
```

---

### 6.4 需要去重或判断是否存在

用 `set`。

```python
valid_fields = {"model", "max_tokens", "temperature"}

if "model" in valid_fields:
    print("合法字段")
```

---

## 7. 常见易错点

### 7.1 空集合不能写 `{}`

```python
x = {}
print(type(x))
```

输出：

```text
<class 'dict'>
```

正确空集合：

```python
s = set()
```

---

### 7.2 单元素元组必须加逗号

```python
a = (1)
b = (1,)

print(type(a))  # int
print(type(b))  # tuple
```

---

### 7.3 字典 key 不能重复

```python
d = {
    "name": "Tom",
    "name": "Jerry",
}

print(d)
```

输出：

```text
{'name': 'Jerry'}
```

后面的值会覆盖前面的值。

---

### 7.4 集合不能用下标访问

```python
s = {1, 2, 3}
print(s[0])
```

会报错，因为集合无序，不支持下标。

---

### 7.5 列表直接赋值不是复制

```python
a = [1, 2, 3]
b = a

b.append(4)
print(a)
```

输出：

```text
[1, 2, 3, 4]
```

因为 `a` 和 `b` 指向同一个列表。

如果想复制：

```python
b = a.copy()
# 或
b = a[:]
```

---

## 8. 一个综合例子

假设我们要保存一次大模型请求的信息：

```python
prompt = "介绍一下大模型推理"
token_ids = [101, 205, 333, 444]
allowed_params = {"temperature", "max_tokens", "ignore_eos"}
request_info = {
    "prompt": prompt,
    "token_ids": token_ids,
    "sampling_params": {
        "temperature": 0.6,
        "max_tokens": 256,
        "ignore_eos": False,
    },
}
state = (len(token_ids), 0)
```

这里分别用到了：

```text
prompt：字符串
 token_ids：list，保存有顺序的 token id
 allowed_params：set，保存不重复的合法参数名
 request_info：dict，用 key-value 保存结构化信息
 state：tuple，保存固定结构的状态快照
```

打印：

```python
print(request_info["prompt"])
print(request_info["token_ids"][-1])
print("temperature" in allowed_params)
print(state[0])
```

输出：

```text
介绍一下大模型推理
444
True
4
```

---

## 9. 最终记忆版

```text
list  = []        有序、可变、可重复，适合保存一串数据
 tuple = ()        有序、不可变、可重复，适合保存固定结构
 dict  = {k: v}    key-value 映射，适合通过名字查值
 set   = {x, y}    无序、不重复，适合去重和判断存在
```

选择口诀：

```text
要顺序、要修改：list
要顺序、不修改：tuple
要根据名字查值：dict
要去重、判断存在：set
```


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-leetcode|模块-leetcode]]

%% 项目关联导航：结束 %%
