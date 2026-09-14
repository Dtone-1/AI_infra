# Python `for` 循环与 `if` 条件判断语法详解

本文档面向 Python 初学者和算法面试刷题场景，系统整理 Python 中 `for` 循环和 `if` 条件判断的常见语法、使用方式、易错点和面试代码习惯。

---

## 目录

1. `for` 循环的基本作用
2. `for` 循环的基本语法
3. `for` 遍历列表
4. `for` 遍历字符串
5. `for` 搭配 `range()`
6. `range()` 的完整用法
7. `for` 搭配 `enumerate()`
8. `for` 遍历字典
9. `for` 搭配 `zip()`
10. 嵌套 `for` 循环
11. `break`
12. `continue`
13. `pass`
14. `for...else`
15. `if` 的基本作用
16. `if` 基本语法
17. `if...else`
18. `if...elif...else`
19. 比较运算符
20. 逻辑运算符
21. 成员判断：`in` / `not in`
22. 身份判断：`is` / `is not`
23. Python 中的真假值判断
24. `if` 的嵌套
25. 条件表达式 / 三元表达式
26. `for` 和 `if` 结合使用
27. 列表推导式中的 `for` 和 `if`
28. 字典推导式中的 `for` 和 `if`
29. 集合推导式中的 `for` 和 `if`
30. 面试代码中的常见写法
31. 常见易错点总结

---

# 1. `for` 循环的基本作用

`for` 循环用于**遍历一个可迭代对象**。

可迭代对象包括：

- 列表 `list`
- 字符串 `str`
- 元组 `tuple`
- 字典 `dict`
- 集合 `set`
- `range()` 对象
- 文件对象
- 其他实现了迭代协议的对象

简单理解：

> `for` 循环就是“从一组数据中，一个一个取出元素，然后执行某段代码”。

---

# 2. `for` 循环的基本语法

```python
for 变量 in 可迭代对象:
    循环体
```

例如：

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

解释：

```python
for num in nums:
```

意思是：

> 从 `nums` 里面依次取出每个元素，赋值给变量 `num`。

第一次循环：

```python
num = 10
```

第二次循环：

```python
num = 20
```

第三次循环：

```python
num = 30
```

---

# 3. `for` 遍历列表

列表是算法题中最常见的数据结构。

```python
nums = [1, 2, 3, 4, 5]

for num in nums:
    print(num)
```

输出：

```text
1
2
3
4
5
```

如果想求和：

```python
nums = [1, 2, 3, 4, 5]

total = 0

for num in nums:
    total += num

print(total)
```

输出：

```text
15
```

这里：

```python
total += num
```

等价于：

```python
total = total + num
```

---

# 4. `for` 遍历字符串

字符串也可以被遍历。

```python
s = "abc"

for ch in s:
    print(ch)
```

输出：

```text
a
b
c
```

常见场景：统计字符出现次数。

```python
s = "banana"

count = {}

for ch in s:
    count[ch] = count.get(ch, 0) + 1

print(count)
```

输出：

```python
{'b': 1, 'a': 3, 'n': 2}
```

---

# 5. `for` 搭配 `range()`

如果你想控制循环次数，通常使用 `range()`。

```python
for i in range(5):
    print(i)
```

输出：

```text
0
1
2
3
4
```

注意：

```python
range(5)
```

生成的是：

```text
0, 1, 2, 3, 4
```

不包括 `5`。

---

# 6. `range()` 的完整用法

## 6.1 `range(stop)`

```python
for i in range(5):
    print(i)
```

表示从 `0` 到 `4`。

```text
0
1
2
3
4
```

---

## 6.2 `range(start, stop)`

```python
for i in range(2, 6):
    print(i)
```

输出：

```text
2
3
4
5
```

含义：

```python
range(2, 6)
```

表示：

> 从 `2` 开始，到 `6` 之前结束。

包括 `2`，不包括 `6`。

---

## 6.3 `range(start, stop, step)`

```python
for i in range(1, 10, 2):
    print(i)
```

输出：

```text
1
3
5
7
9
```

第三个参数 `2` 表示步长为 `2`。

---

## 6.4 倒序遍历

```python
for i in range(5, 0, -1):
    print(i)
```

输出：

```text
5
4
3
2
1
```

注意：

```python
range(5, 0, -1)
```

包括 `5`，不包括 `0`。

如果想倒序遍历数组下标：

```python
nums = [10, 20, 30, 40]

for i in range(len(nums) - 1, -1, -1):
    print(i, nums[i])
```

输出：

```text
3 40
2 30
1 20
0 10
```

---

# 7. `for` 搭配 `enumerate()`

如果你既需要元素，又需要下标，推荐使用 `enumerate()`。

不推荐写法：

```python
nums = [10, 20, 30]

for i in range(len(nums)):
    print(i, nums[i])
```

推荐写法：

```python
nums = [10, 20, 30]

for i, num in enumerate(nums):
    print(i, num)
```

输出：

```text
0 10
1 20
2 30
```

解释：

```python
for i, num in enumerate(nums):
```

每次循环会同时拿到：

- `i`：当前下标
- `num`：当前元素

---

## 7.1 `enumerate()` 指定起始下标

```python
names = ["Tom", "Jerry", "Alice"]

for i, name in enumerate(names, start=1):
    print(i, name)
```

输出：

```text
1 Tom
2 Jerry
3 Alice
```

---

# 8. `for` 遍历字典

字典是键值对结构。

```python
student = {
    "name": "Tom",
    "age": 18,
    "score": 95
}
```

---

## 8.1 默认遍历字典的 key

```python
for key in student:
    print(key)
```

输出：

```text
name
age
score
```

等价于：

```python
for key in student.keys():
    print(key)
```

---

## 8.2 遍历 value

```python
for value in student.values():
    print(value)
```

输出：

```text
Tom
18
95
```

---

## 8.3 同时遍历 key 和 value

推荐写法：

```python
for key, value in student.items():
    print(key, value)
```

输出：

```text
name Tom
age 18
score 95
```

算法题中非常常用：

```python
count = {
    "a": 3,
    "b": 2,
    "c": 1
}

for ch, freq in count.items():
    print(ch, freq)
```

---

# 9. `for` 搭配 `zip()`

`zip()` 可以把多个序列“并排”遍历。

```python
names = ["Tom", "Jerry", "Alice"]
scores = [90, 85, 95]

for name, score in zip(names, scores):
    print(name, score)
```

输出：

```text
Tom 90
Jerry 85
Alice 95
```

如果两个列表长度不同，`zip()` 会以较短的为准。

```python
a = [1, 2, 3]
b = ["x", "y"]

for x, y in zip(a, b):
    print(x, y)
```

输出：

```text
1 x
2 y
```

`3` 不会被遍历到。

---

# 10. 嵌套 `for` 循环

`for` 循环里面还可以写 `for` 循环。

```python
for i in range(3):
    for j in range(2):
        print(i, j)
```

输出：

```text
0 0
0 1
1 0
1 1
2 0
2 1
```

解释：

外层循环每执行一次，内层循环会完整执行一遍。

---

## 10.1 遍历二维数组

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

for row in matrix:
    for num in row:
        print(num)
```

输出：

```text
1
2
3
4
5
6
7
8
9
```

如果需要下标：

```python
matrix = [
    [1, 2, 3],
    [4, 5, 6]
]

m = len(matrix)
n = len(matrix[0])

for i in range(m):
    for j in range(n):
        print(i, j, matrix[i][j])
```

输出：

```text
0 0 1
0 1 2
0 2 3
1 0 4
1 1 5
1 2 6
```

---

# 11. `break`

`break` 用于**结束当前最近的一层循环**。

```python
for i in range(5):
    if i == 3:
        break
    print(i)
```

输出：

```text
0
1
2
```

当 `i == 3` 时，执行 `break`，整个 `for` 循环结束。

---

## 11.1 `break` 在嵌套循环中只结束最近的一层循环

```python
for i in range(3):
    for j in range(3):
        if j == 1:
            break
        print(i, j)
```

输出：

```text
0 0
1 0
2 0
```

解释：

`break` 位于内层循环中，所以它只结束内层的：

```python
for j in range(3):
```

不会结束外层的：

```python
for i in range(3):
```

---

# 12. `continue`

`continue` 用于**跳过当前这一轮循环，直接进入下一轮循环**。

```python
for i in range(5):
    if i == 2:
        continue
    print(i)
```

输出：

```text
0
1
3
4
```

当 `i == 2` 时，执行 `continue`，跳过本轮后面的 `print(i)`，直接进入下一轮循环。

---

## 12.1 `continue` 在嵌套循环中只作用于最近的一层循环

```python
for i in range(2):
    for j in range(3):
        if j == 1:
            continue
        print(i, j)
```

输出：

```text
0 0
0 2
1 0
1 2
```

解释：

`continue` 只跳过当前这轮内层循环，不影响外层循环。

---

# 13. `pass`

`pass` 表示“什么也不做”。

它常用于占位。

```python
for i in range(5):
    pass
```

这段代码可以运行，但什么都不会发生。

---

## 13.1 `pass` 用在 `if` 中

```python
x = 10

if x > 0:
    pass
else:
    print("x <= 0")
```

当你暂时还没想好某个分支要写什么时，可以先写 `pass`，防止语法报错。

---

## 13.2 `pass`、`continue`、`break` 的区别

```python
for i in range(5):
    if i == 2:
        pass
    print(i)
```

输出：

```text
0
1
2
3
4
```

`pass` 什么都不做，所以 `print(i)` 仍然会执行。

```python
for i in range(5):
    if i == 2:
        continue
    print(i)
```

输出：

```text
0
1
3
4
```

`continue` 会跳过当前这一轮循环。

```python
for i in range(5):
    if i == 2:
        break
    print(i)
```

输出：

```text
0
1
```

`break` 会直接结束循环。

---

# 14. `for...else`

Python 中 `for` 后面可以接 `else`。

语法：

```python
for 变量 in 可迭代对象:
    循环体
else:
    循环正常结束后执行的代码
```

注意：

> `for...else` 中的 `else` 会在循环没有被 `break` 打断时执行。

---

## 14.1 没有 `break`，执行 `else`

```python
for i in range(3):
    print(i)
else:
    print("循环正常结束")
```

输出：

```text
0
1
2
循环正常结束
```

---

## 14.2 有 `break`，不执行 `else`

```python
for i in range(5):
    if i == 2:
        break
    print(i)
else:
    print("循环正常结束")
```

输出：

```text
0
1
```

因为循环被 `break` 打断了，所以 `else` 不执行。

---

## 14.3 `for...else` 常用于查找

```python
nums = [1, 3, 5, 7]
target = 4

for num in nums:
    if num == target:
        print("找到了")
        break
else:
    print("没找到")
```

输出：

```text
没找到
```

如果 `target = 5`：

```python
nums = [1, 3, 5, 7]
target = 5

for num in nums:
    if num == target:
        print("找到了")
        break
else:
    print("没找到")
```

输出：

```text
找到了
```

---

# 15. `if` 的基本作用

`if` 用于条件判断。

简单理解：

> 如果某个条件成立，就执行某段代码；否则不执行，或者执行另一段代码。

---

# 16. `if` 基本语法

```python
if 条件:
    条件成立时执行的代码
```

例如：

```python
age = 20

if age >= 18:
    print("成年人")
```

输出：

```text
成年人
```

如果条件不成立：

```python
age = 16

if age >= 18:
    print("成年人")
```

这段代码没有输出，因为 `age >= 18` 不成立。

---

# 17. `if...else`

语法：

```python
if 条件:
    条件成立时执行
else:
    条件不成立时执行
```

例如：

```python
age = 16

if age >= 18:
    print("成年人")
else:
    print("未成年人")
```

输出：

```text
未成年人
```

---

# 18. `if...elif...else`

如果有多个条件，可以使用 `elif`。

```python
score = 85

if score >= 90:
    print("优秀")
elif score >= 80:
    print("良好")
elif score >= 60:
    print("及格")
else:
    print("不及格")
```

输出：

```text
良好
```

解释：

Python 会从上到下依次判断：

1. `score >= 90`，不成立；
2. `score >= 80`，成立，于是执行 `print("良好")`；
3. 后面的 `elif` 和 `else` 不再判断。

---

## 18.1 `elif` 的顺序很重要

错误示例：

```python
score = 95

if score >= 60:
    print("及格")
elif score >= 80:
    print("良好")
elif score >= 90:
    print("优秀")
```

输出：

```text
及格
```

虽然 `95` 也满足 `score >= 90`，但是程序先遇到 `score >= 60`，这个条件已经成立，所以后面不会继续判断。

正确写法：

```python
score = 95

if score >= 90:
    print("优秀")
elif score >= 80:
    print("良好")
elif score >= 60:
    print("及格")
else:
    print("不及格")
```

---

# 19. 比较运算符

`if` 后面的条件通常由比较运算符组成。

| 运算符 | 含义 | 示例 |
|---|---|---|
| `==` | 等于 | `x == 3` |
| `!=` | 不等于 | `x != 3` |
| `>` | 大于 | `x > 3` |
| `<` | 小于 | `x < 3` |
| `>=` | 大于等于 | `x >= 3` |
| `<=` | 小于等于 | `x <= 3` |

示例：

```python
x = 10

if x > 5:
    print("x 大于 5")

if x == 10:
    print("x 等于 10")

if x != 0:
    print("x 不等于 0")
```

输出：

```text
x 大于 5
x 等于 10
x 不等于 0
```

---

## 19.1 链式比较

Python 支持链式比较。

```python
x = 5

if 1 <= x <= 10:
    print("x 在 1 到 10 之间")
```

输出：

```text
x 在 1 到 10 之间
```

等价于：

```python
if x >= 1 and x <= 10:
    print("x 在 1 到 10 之间")
```

更推荐链式比较，因为更清晰。

---

# 20. 逻辑运算符

多个条件可以用逻辑运算符连接。

| 运算符 | 含义 |
|---|---|
| `and` | 并且 |
| `or` | 或者 |
| `not` | 取反 |

---

## 20.1 `and`

两个条件都成立，整体才成立。

```python
age = 20
has_ticket = True

if age >= 18 and has_ticket:
    print("可以入场")
```

输出：

```text
可以入场
```

---

## 20.2 `or`

只要有一个条件成立，整体就成立。

```python
is_vip = False
has_coupon = True

if is_vip or has_coupon:
    print("可以优惠")
```

输出：

```text
可以优惠
```

---

## 20.3 `not`

`not` 用于取反。

```python
is_empty = False

if not is_empty:
    print("不为空")
```

输出：

```text
不为空
```

常见写法：

```python
nums = []

if not nums:
    print("列表为空")
```

输出：

```text
列表为空
```

---

# 21. 成员判断：`in` / `not in`

`in` 用于判断一个元素是否在某个容器中。

```python
nums = [1, 2, 3]

if 2 in nums:
    print("2 在 nums 中")
```

输出：

```text
2 在 nums 中
```

---

## 21.1 `not in`

```python
nums = [1, 2, 3]

if 5 not in nums:
    print("5 不在 nums 中")
```

输出：

```text
5 不在 nums 中
```

---

## 21.2 面试中常配合集合使用

```python
seen = set()

nums = [1, 2, 3, 2]

for num in nums:
    if num in seen:
        print("有重复元素")
        break

    seen.add(num)
```

输出：

```text
有重复元素
```

注意：

- `x in list`：时间复杂度是 `O(n)`
- `x in set`：平均时间复杂度是 `O(1)`
- `key in dict`：平均时间复杂度是 `O(1)`

所以算法题中经常用 `set` 或 `dict` 做快速查找。

---

# 22. 身份判断：`is` / `is not`

`is` 判断两个变量是否指向同一个对象。

最常见用途是判断 `None`。

推荐写法：

```python
x = None

if x is None:
    print("x 是 None")
```

输出：

```text
x 是 None
```

不推荐写法：

```python
if x == None:
    print("x 是 None")
```

虽然很多时候也能运行，但 Python 规范中更推荐：

```python
if x is None:
```

判断不是 `None`：

```python
if x is not None:
    print("x 不是 None")
```

---

# 23. Python 中的真假值判断

在 Python 中，很多对象可以直接放在 `if` 后面判断真假。

---

## 23.1 以下值通常被认为是假

```python
False
None
0
0.0
""
[]
{}
set()
tuple()
```

示例：

```python
nums = []

if not nums:
    print("nums 是空列表")
```

输出：

```text
nums 是空列表
```

---

## 23.2 非空容器通常为真

```python
nums = [1, 2, 3]

if nums:
    print("nums 非空")
```

输出：

```text
nums 非空
```

---

## 23.3 常见推荐写法

判断列表为空：

```python
if not nums:
    return []
```

判断字符串为空：

```python
if not s:
    return ""
```

判断字典为空：

```python
if not count:
    print("字典为空")
```

判断链表节点为空：

```python
if not head:
    return None
```

或者更严谨：

```python
if head is None:
    return None
```

---

# 24. `if` 的嵌套

`if` 里面可以继续写 `if`。

```python
age = 20
has_ticket = True

if age >= 18:
    if has_ticket:
        print("可以入场")
    else:
        print("没有票")
else:
    print("未成年人")
```

输出：

```text
可以入场
```

不过如果条件不复杂，通常可以用 `and` 简化：

```python
if age >= 18 and has_ticket:
    print("可以入场")
```

---

# 25. 条件表达式 / 三元表达式

Python 支持一种简短的条件表达式。

语法：

```python
值1 if 条件 else 值2
```

含义：

> 如果条件成立，结果是值1；否则结果是值2。

示例：

```python
age = 20

status = "成年人" if age >= 18 else "未成年人"

print(status)
```

输出：

```text
成年人
```

---

## 25.1 求两个数的较大值

```python
a = 10
b = 20

max_val = a if a > b else b

print(max_val)
```

输出：

```text
20
```

---

## 25.2 不要过度使用三元表达式

不推荐：

```python
result = "A" if score >= 90 else "B" if score >= 80 else "C" if score >= 60 else "D"
```

虽然能运行，但可读性差。

推荐：

```python
if score >= 90:
    result = "A"
elif score >= 80:
    result = "B"
elif score >= 60:
    result = "C"
else:
    result = "D"
```

面试代码中，清晰比炫技更重要。

---

# 26. `for` 和 `if` 结合使用

实际代码中，`for` 和 `if` 经常一起使用。

---

## 26.1 找出所有偶数

```python
nums = [1, 2, 3, 4, 5, 6]

evens = []

for num in nums:
    if num % 2 == 0:
        evens.append(num)

print(evens)
```

输出：

```python
[2, 4, 6]
```

---

## 26.2 找最大值

```python
nums = [3, 1, 8, 2, 5]

max_val = nums[0]

for num in nums:
    if num > max_val:
        max_val = num

print(max_val)
```

输出：

```text
8
```

---

## 26.3 查找目标值

```python
nums = [1, 3, 5, 7]
target = 5

for i, num in enumerate(nums):
    if num == target:
        print("找到了，下标是", i)
        break
```

输出：

```text
找到了，下标是 2
```

---

# 27. 列表推导式中的 `for` 和 `if`

列表推导式可以用一行代码生成列表。

---

## 27.1 基本列表推导式

普通写法：

```python
nums = [1, 2, 3, 4]

squares = []

for num in nums:
    squares.append(num * num)

print(squares)
```

输出：

```python
[1, 4, 9, 16]
```

列表推导式写法：

```python
nums = [1, 2, 3, 4]

squares = [num * num for num in nums]

print(squares)
```

输出：

```python
[1, 4, 9, 16]
```

---

## 27.2 带 `if` 过滤条件的列表推导式

普通写法：

```python
nums = [1, 2, 3, 4, 5, 6]

evens = []

for num in nums:
    if num % 2 == 0:
        evens.append(num)
```

列表推导式写法：

```python
nums = [1, 2, 3, 4, 5, 6]

evens = [num for num in nums if num % 2 == 0]

print(evens)
```

输出：

```python
[2, 4, 6]
```

语法结构：

```python
[表达式 for 变量 in 可迭代对象 if 条件]
```

---

## 27.3 带 `if...else` 的列表推导式

如果要对每个元素做二选一转换，语法是：

```python
[值1 if 条件 else 值2 for 变量 in 可迭代对象]
```

例如，把偶数保留，奇数变成 `0`：

```python
nums = [1, 2, 3, 4, 5]

res = [num if num % 2 == 0 else 0 for num in nums]

print(res)
```

输出：

```python
[0, 2, 0, 4, 0]
```

注意这两种写法位置不同：

过滤：

```python
[num for num in nums if num % 2 == 0]
```

二选一转换：

```python
[num if num % 2 == 0 else 0 for num in nums]
```

---

# 28. 字典推导式中的 `for` 和 `if`

字典推导式用于快速创建字典。

```python
nums = [1, 2, 3]

square_map = {num: num * num for num in nums}

print(square_map)
```

输出：

```python
{1: 1, 2: 4, 3: 9}
```

带 `if` 过滤：

```python
nums = [1, 2, 3, 4]

square_map = {num: num * num for num in nums if num % 2 == 0}

print(square_map)
```

输出：

```python
{2: 4, 4: 16}
```

---

# 29. 集合推导式中的 `for` 和 `if`

集合推导式用于创建集合，自动去重。

```python
nums = [1, 2, 2, 3, 3, 4]

res = {num for num in nums if num % 2 == 0}

print(res)
```

输出可能是：

```python
{2, 4}
```

集合本身无序，所以输出顺序不一定固定。

---

# 30. 面试代码中的常见写法

## 30.1 遍历数组

```python
for num in nums:
    ...
```

适合只需要元素。

---

## 30.2 遍历数组下标和元素

```python
for i, num in enumerate(nums):
    ...
```

适合既需要下标又需要元素。

---

## 30.3 通过下标遍历

```python
for i in range(len(nums)):
    ...
```

适合需要访问相邻元素，例如：

```python
for i in range(1, len(nums)):
    if nums[i] == nums[i - 1]:
        print("相邻重复")
```

---

## 30.4 双指针

```python
left = 0
right = len(nums) - 1

while left < right:
    if nums[left] + nums[right] == target:
        return [left, right]
    elif nums[left] + nums[right] < target:
        left += 1
    else:
        right -= 1
```

虽然这里使用的是 `while`，但通常和 `for`、`if` 一起出现在算法题中。

---

## 30.5 哈希表查找

```python
index_map = {}

for i, num in enumerate(nums):
    need = target - num

    if need in index_map:
        return [index_map[need], i]

    index_map[num] = i
```

这是两数之和的经典写法。

---

## 30.6 计数

```python
count = {}

for num in nums:
    count[num] = count.get(num, 0) + 1
```

或者：

```python
from collections import defaultdict

count = defaultdict(int)

for num in nums:
    count[num] += 1
```

---

## 30.7 分组

```python
from collections import defaultdict

groups = defaultdict(list)

for s in strs:
    key = ''.join(sorted(s))
    groups[key].append(s)

ans = list(groups.values())
```

这是字母异位词分组的经典写法。

---

# 31. 常见易错点总结

## 31.1 `if` 后面必须有冒号

错误：

```python
if x > 0
    print(x)
```

正确：

```python
if x > 0:
    print(x)
```

---

## 31.2 `for` 后面也必须有冒号

错误：

```python
for num in nums
    print(num)
```

正确：

```python
for num in nums:
    print(num)
```

---

## 31.3 Python 用缩进表示代码块

错误：

```python
if x > 0:
print(x)
```

正确：

```python
if x > 0:
    print(x)
```

---

## 31.4 `=` 和 `==` 不一样

`=` 是赋值。

```python
x = 10
```

`==` 是判断是否相等。

```python
if x == 10:
    print("x 等于 10")
```

不要写成：

```python
if x = 10:
    print("错误")
```

---

## 31.5 不要把 `break` 和 `continue` 理解成跳出 `if`

错误理解：

```text
break / continue 是跳出 if
```

正确理解：

```text
break / continue 作用于离它最近的 for 或 while 循环，和 if 嵌套几层没有关系。
```

例如：

```python
for i in range(5):
    if i == 2:
        continue
    print(i)
```

`continue` 是跳过当前这一轮 `for` 循环，不是“跳出 if”。

---

## 31.6 遍历列表时不要随便删除元素

容易出错：

```python
nums = [1, 2, 3, 4]

for num in nums:
    if num % 2 == 0:
        nums.remove(num)

print(nums)
```

这种写法可能跳过元素。

更推荐创建新列表：

```python
nums = [1, 2, 3, 4]

res = []

for num in nums:
    if num % 2 != 0:
        res.append(num)

print(res)
```

或者使用列表推导式：

```python
nums = [1, 2, 3, 4]

res = [num for num in nums if num % 2 != 0]

print(res)
```

---

## 31.7 判断空列表推荐用 `if not nums`

推荐：

```python
if not nums:
    return []
```

不推荐：

```python
if len(nums) == 0:
    return []
```

两者都能运行，但前者更 Pythonic。

---

## 31.8 判断 `None` 推荐用 `is None`

推荐：

```python
if x is None:
    print("x 是 None")
```

不推荐：

```python
if x == None:
    print("x 是 None")
```

---

## 31.9 `range(n)` 不包含 `n`

```python
for i in range(3):
    print(i)
```

输出：

```text
0
1
2
```

不会输出 `3`。

---

## 31.10 `elif` 顺序会影响结果

错误写法：

```python
score = 95

if score >= 60:
    print("及格")
elif score >= 90:
    print("优秀")
```

输出：

```text
及格
```

正确写法：

```python
score = 95

if score >= 90:
    print("优秀")
elif score >= 60:
    print("及格")
```

输出：

```text
优秀
```

---

# 32. 最后总结

## `for` 循环核心

```python
for 变量 in 可迭代对象:
    循环体
```

常见搭配：

```python
for num in nums:
    ...
```

```python
for i in range(len(nums)):
    ...
```

```python
for i, num in enumerate(nums):
    ...
```

```python
for key, value in dict.items():
    ...
```

---

## `if` 判断核心

```python
if 条件:
    条件成立时执行
elif 其他条件:
    其他条件成立时执行
else:
    以上条件都不成立时执行
```

常见条件：

```python
if x > 0:
```

```python
if x in seen:
```

```python
if not nums:
```

```python
if node is None:
```

```python
if left < right and nums[left] == nums[right]:
```

---

## 面试中最推荐的习惯

1. 遍历元素用 `for num in nums`
2. 需要下标用 `for i, num in enumerate(nums)`
3. 需要次数用 `for i in range(n)`
4. 判断空列表用 `if not nums`
5. 判断 `None` 用 `is None`
6. 查找元素优先用 `set` / `dict`
7. 不要把 `break` / `continue` 理解成跳出 `if`
8. 写嵌套循环时一定想清楚每一层循环执行多少次
9. 写 `if...elif...else` 时注意条件顺序
10. 代码清晰比一行写完更重要



%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-leetcode|模块-leetcode]]

%% 项目关联导航：结束 %%
