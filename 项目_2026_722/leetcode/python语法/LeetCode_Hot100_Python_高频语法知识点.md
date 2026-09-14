# LeetCode Hot100 高频 Python 语法知识点详解

本文档面向正在刷 LeetCode Hot100、准备面试手撕代码的 Python 初学者。很多题解中会出现 `defaultdict`、`deque`、`Counter`、`heapq`、`set`、`enumerate`、`lambda` 等写法。它们不是“奇怪关键字”，而是 Python 算法题中常用的数据结构工具和语法习惯。

阅读建议：

1. 先理解每个知识点“解决什么问题”；
2. 再看它在 Hot100 中对应哪些题型；
3. 最后背模板，而不是死记单个语法。

---

## 目录

1. `from typing import List, Optional`
2. `class Solution` 和 `self`
3. `dict`：哈希表
4. `dict.get()`
5. `set`：集合
6. `defaultdict(int)`：计数字典
7. `defaultdict(list)`：分组字典
8. `Counter`：计数器
9. `deque`：双端队列
10. `heapq`：堆 / 优先队列
11. `enumerate()`：同时拿下标和元素
12. `range()`：下标循环
13. `zip()`：并行遍历
14. `sort()` / `sorted()`
15. `key=lambda x: ...`
16. `float("inf")`
17. `None`、`is None`
18. `in` / `not in`
19. `list` 作为栈
20. 切片 `[:]`、`[::-1]`
21. 字符串、`join()`、`split()`
22. `ord()` / `chr()`
23. `//`、`%`、`divmod()`
24. `nonlocal`
25. `lru_cache`
26. `OrderedDict`
27. 链表 `ListNode`
28. 二叉树 `TreeNode`
29. 二维数组初始化
30. Hot100 常见模板总结

---


# 1. `from typing import List, Optional`

## 1.1 它是什么？

LeetCode 代码里经常看到：

```python
from typing import List, Optional
```

这是 Python 的类型注解工具。

```python
List[int]
```

表示“整数列表”。

```python
Optional[TreeNode]
```

表示“可能是 `TreeNode`，也可能是 `None`”。

---

## 1.2 示例

```python
from typing import List

class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        pass
```

含义：

- `nums: List[int]`：`nums` 是整数列表；
- `target: int`：`target` 是整数；
- `-> List[int]`：返回值是整数列表。

---

## 1.3 Hot100 场景

几乎所有 LeetCode 官方函数签名都会用到类型注解，比如：

```python
def threeSum(self, nums: List[int]) -> List[List[int]]:
```

含义是：

> 输入是整数列表，输出是二维整数列表。

比如三数之和返回：

```python
[[-1, -1, 2], [-1, 0, 1]]
```

---

## 1.4 面试必须写吗？

不一定。手撕代码时你可以写：

```python
def two_sum(nums, target):
    pass
```

但你必须看得懂 LeetCode 的类型注解。


# 2. `class Solution` 和 `self`

## 2.1 `class Solution` 是什么？

LeetCode 通常要求你把答案写在类里：

```python
class Solution:
    def twoSum(self, nums, target):
        pass
```

你主要关注函数内部逻辑即可。

---

## 2.2 `self` 是什么？

类里面的方法，第一个参数通常是 `self`：

```python
def twoSum(self, nums, target):
```

`self` 表示当前对象本身。LeetCode 会自动创建 `Solution` 对象并调用你的方法。

你一般不用手动传 `self`。

---

## 2.3 可以用 `self` 保存答案吗？

可以：

```python
class Solution:
    def maxDepth(self, root):
        self.ans = 0
```

但面试中更推荐局部变量或 `nonlocal`，因为逻辑更清楚。

---

## 2.4 常见写法

```python
class Solution:
    def maxDepth(self, root):
        if not root:
            return 0

        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)

        return max(left, right) + 1
```

这里：

```python
self.maxDepth(...)
```

表示调用当前类里的方法。


# 3. `dict`：哈希表

## 3.1 它是什么？

`dict` 是字典，也叫哈希表，用来存储键值对：

```python
d = {
    "name": "Tom",
    "age": 18
}
```

核心作用：

> 通过 key 快速找到 value。

---

## 3.2 基本操作

```python
d = {}

d["a"] = 1
d["b"] = 2

print(d["a"])  # 1
```

判断 key 是否存在：

```python
if "a" in d:
    print("存在")
```

遍历字典：

```python
for key, value in d.items():
    print(key, value)
```

---

## 3.3 Hot100 高频原因

哈希表能把很多查找从 `O(n)` 优化到平均 `O(1)`。

典型题：

- 两数之和；
- 和为 K 的子数组；
- 字母异位词分组；
- 最长连续序列；
- LRU 缓存；
- 复制带随机指针的链表。

---

## 3.4 两数之和模板

```python
def twoSum(nums, target):
    index_map = {}

    for i, num in enumerate(nums):
        need = target - num

        if need in index_map:
            return [index_map[need], i]

        index_map[num] = i

    return []
```

解释：

```python
index_map[num] = i
```

表示记录：

> 数字 `num` 出现的位置是 `i`。

```python
if need in index_map:
```

表示：

> 如果之前出现过我现在需要的数，就找到答案。


# 4. `dict.get()`

## 4.1 它是什么？

`get()` 是字典取值方法。

语法：

```python
d.get(key, default_value)
```

意思是：

> 如果 key 存在，返回对应 value；如果 key 不存在，返回默认值。

---

## 4.2 示例

```python
count = {}

print(count.get("a", 0))
```

输出：

```python
0
```

因为 `"a"` 不存在，所以返回默认值 `0`。

---

## 4.3 用于计数

```python
count = {}

for ch in "banana":
    count[ch] = count.get(ch, 0) + 1

print(count)
```

输出：

```python
{'b': 1, 'a': 3, 'n': 2}
```

这句：

```python
count[ch] = count.get(ch, 0) + 1
```

等价于：

```python
if ch not in count:
    count[ch] = 0

count[ch] += 1
```

---

## 4.4 Hot100 场景

- 字符计数；
- 数字频率统计；
- 前缀和次数统计；
- 滑动窗口维护字符数量。


# 5. `set`：集合

## 5.1 它是什么？

`set` 是集合，特点：

1. 元素不重复；
2. 查找很快；
3. 无序。

创建：

```python
seen = set()
```

加入元素：

```python
seen.add(x)
```

判断元素是否存在：

```python
if x in seen:
    ...
```

---

## 5.2 示例：判断是否有重复

```python
def containsDuplicate(nums):
    seen = set()

    for num in nums:
        if num in seen:
            return True

        seen.add(num)

    return False
```

---

## 5.3 为什么不用 list？

如果写：

```python
seen = []

if num in seen:
    ...
```

`num in seen` 是 `O(n)`。

如果用：

```python
seen = set()
```

`num in seen` 平均是 `O(1)`。

---

## 5.4 Hot100 场景

- 无重复字符的最长子串；
- 最长连续序列；
- 环形链表；
- 单词搜索 visited；
- 岛屿类题目 visited。


# 6. `defaultdict(int)`：计数字典

## 6.1 它是什么？

使用前导入：

```python
from collections import defaultdict
```

创建：

```python
count = defaultdict(int)
```

意思是：

> 当访问不存在的 key 时，默认值自动是 `0`。

---

## 6.2 为什么是 `int`？

因为：

```python
int()
```

结果是：

```python
0
```

所以：

```python
defaultdict(int)
```

表示默认值是 `0`。

---

## 6.3 示例

```python
from collections import defaultdict

count = defaultdict(int)

nums = [1, 2, 1, 3, 2, 1]

for num in nums:
    count[num] += 1

print(count)
```

输出类似：

```python
defaultdict(<class 'int'>, {1: 3, 2: 2, 3: 1})
```

可以理解成普通字典：

```python
{1: 3, 2: 2, 3: 1}
```

---

## 6.4 Hot100 典型：和为 K 的子数组

```python
from collections import defaultdict

def subarraySum(nums, k):
    prefix_count = defaultdict(int)
    prefix_count[0] = 1

    prefix_sum = 0
    ans = 0

    for num in nums:
        prefix_sum += num
        ans += prefix_count[prefix_sum - k]
        prefix_count[prefix_sum] += 1

    return ans
```

这里：

```python
prefix_count = defaultdict(int)
```

表示：

> 统计每个前缀和出现过几次。

如果某个前缀和没出现过，访问它时自动返回 `0`，不会报错。


# 7. `defaultdict(list)`：分组字典

## 7.1 它是什么？

```python
from collections import defaultdict

groups = defaultdict(list)
```

意思是：

> 当访问不存在的 key 时，自动创建一个空列表 `[]`。

---

## 7.2 示例

```python
from collections import defaultdict

groups = defaultdict(list)

groups["a"].append("apple")
groups["a"].append("ant")
groups["b"].append("banana")

print(groups)
```

输出类似：

```python
defaultdict(<class 'list'>, {'a': ['apple', 'ant'], 'b': ['banana']})
```

---

## 7.3 Hot100 典型：字母异位词分组

```python
from collections import defaultdict

def groupAnagrams(strs):
    groups = defaultdict(list)

    for s in strs:
        key = ''.join(sorted(s))
        groups[key].append(s)

    return list(groups.values())
```

解释：

```python
key = ''.join(sorted(s))
```

把字符串排序后作为分组依据。

比如：

```python
"eat" -> "aet"
"tea" -> "aet"
"ate" -> "aet"
```

所以：

```python
groups["aet"] = ["eat", "tea", "ate"]
```

---

## 7.4 关键语句

```python
groups[key].append(s)
```

意思是：

> 找到 `key` 对应的列表，把 `s` 放进去。

如果 `key` 不存在，`defaultdict(list)` 会自动先创建：

```python
groups[key] = []
```


# 8. `Counter`：计数器

## 8.1 它是什么？

`Counter` 是专门用来计数的工具。

导入：

```python
from collections import Counter
```

---

## 8.2 统计列表

```python
from collections import Counter

nums = [1, 2, 1, 3, 2, 1]

count = Counter(nums)

print(count)
```

输出：

```python
Counter({1: 3, 2: 2, 3: 1})
```

---

## 8.3 统计字符串

```python
from collections import Counter

s = "banana"

count = Counter(s)

print(count)
```

输出：

```python
Counter({'a': 3, 'n': 2, 'b': 1})
```

---

## 8.4 访问不存在的 key

```python
count["x"]
```

如果 `"x"` 不存在，返回 `0`，不会报错。

---

## 8.5 Hot100 场景

- 有效的字母异位词；
- 最小覆盖子串；
- 前 K 个高频元素；
- 滑动窗口字符计数；
- 字符串排列相关题。

---

## 8.6 判断异位词

```python
from collections import Counter

def isAnagram(s, t):
    return Counter(s) == Counter(t)
```


# 9. `deque`：双端队列

## 9.1 它是什么？

导入：

```python
from collections import deque
```

创建：

```python
q = deque()
```

`deque` 是双端队列，支持从两边高效加入和删除。

---

## 9.2 常用操作

```python
q.append(x)       # 从右边加入
q.popleft()       # 从左边取出
q.appendleft(x)   # 从左边加入
q.pop()           # 从右边取出
```

---

## 9.3 为什么 BFS 要用 deque？

普通列表：

```python
q.pop(0)
```

时间复杂度是 `O(n)`。

`deque`：

```python
q.popleft()
```

时间复杂度是 `O(1)`。

所以 BFS 中推荐：

```python
q = deque()
```

---

## 9.4 BFS 模板

```python
from collections import deque

def bfs(start):
    q = deque([start])
    visited = set([start])

    while q:
        node = q.popleft()

        for nxt in graph[node]:
            if nxt not in visited:
                visited.add(nxt)
                q.append(nxt)
```

---

## 9.5 Hot100 场景

- 二叉树层序遍历；
- 腐烂的橘子；
- 岛屿数量；
- 图的最短路径；
- 滑动窗口最大值。


# 10. `heapq`：堆 / 优先队列

## 10.1 它是什么？

`heapq` 是 Python 的堆工具。

导入：

```python
import heapq
```

Python 默认是小根堆：

> 每次弹出最小值。

---

## 10.2 基本操作

```python
import heapq

heap = []

heapq.heappush(heap, 3)
heapq.heappush(heap, 1)
heapq.heappush(heap, 2)

print(heapq.heappop(heap))
```

输出：

```python
1
```

---

## 10.3 常用函数

```python
heapq.heappush(heap, x)  # 入堆，O(log n)
heapq.heappop(heap)      # 出堆，O(log n)
heapq.heapify(nums)      # 原地建堆，O(n)
```

---

## 10.4 大根堆怎么写？

Python 没有直接的大根堆，通常用负数模拟。

```python
import heapq

heap = []

for x in nums:
    heapq.heappush(heap, -x)

max_val = -heapq.heappop(heap)
```

---

## 10.5 Hot100 场景

- 前 K 个高频元素；
- 合并 K 个升序链表；
- 数据流的中位数；
- 任务调度器。

---

## 10.6 前 K 个高频元素

```python
from collections import Counter
import heapq

def topKFrequent(nums, k):
    count = Counter(nums)
    heap = []

    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))

        if len(heap) > k:
            heapq.heappop(heap)

    return [num for freq, num in heap]
```

堆中保存：

```python
(freq, num)
```

Python 会先按 `freq` 排序。


# 11. `enumerate()`：同时拿下标和元素

## 11.1 基本用法

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

---

## 11.2 为什么推荐？

不推荐：

```python
for i in range(len(nums)):
    num = nums[i]
```

推荐：

```python
for i, num in enumerate(nums):
```

后者更清晰。

---

## 11.3 Hot100 典型：两数之和

```python
def twoSum(nums, target):
    index_map = {}

    for i, num in enumerate(nums):
        need = target - num

        if need in index_map:
            return [index_map[need], i]

        index_map[num] = i

    return []
```


# 12. `range()`：下标循环

## 12.1 `range(n)`

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

注意不包括 `5`。

---

## 12.2 `range(start, stop)`

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

---

## 12.3 `range(start, stop, step)`

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

---

## 12.4 倒序遍历

```python
for i in range(len(nums) - 1, -1, -1):
    print(nums[i])
```

---

## 12.5 Hot100 场景

- 动态规划；
- 二分查找；
- 矩阵遍历；
- 回溯枚举；
- 三数之和外层循环。


# 13. `zip()`：并行遍历

## 13.1 基本用法

```python
a = [1, 2, 3]
b = ["x", "y", "z"]

for num, ch in zip(a, b):
    print(num, ch)
```

输出：

```text
1 x
2 y
3 z
```

---

## 13.2 长度不同怎么办？

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

`zip()` 会以较短的序列为准。

---

## 13.3 Hot100 场景

- 同时遍历两个字符串；
- 比较两个数组；
- 构造映射关系；
- 矩阵转置相关写法。

矩阵转置例子：

```python
matrix = [
    [1, 2],
    [3, 4]
]

transposed = list(zip(*matrix))
print(transposed)
```

输出：

```python
[(1, 3), (2, 4)]
```


# 14. `sort()` / `sorted()`

## 14.1 `sort()`

```python
nums = [3, 1, 2]
nums.sort()
print(nums)
```

输出：

```python
[1, 2, 3]
```

`sort()` 会原地修改原列表。

---

## 14.2 `sorted()`

```python
nums = [3, 1, 2]

arr = sorted(nums)

print(arr)
print(nums)
```

输出：

```python
[1, 2, 3]
[3, 1, 2]
```

`sorted()` 返回一个新列表，不修改原列表。

---

## 14.3 Hot100 场景

- 三数之和；
- 合并区间；
- 字母异位词分组；
- 最大数；
- 贪心类问题。

---

## 14.4 三数之和为什么先排序？

排序后可以使用双指针。

```python
nums.sort()

for i in range(len(nums) - 2):
    left = i + 1
    right = len(nums) - 1
```

通过移动 `left` 和 `right` 控制总和大小。


# 15. `key=lambda x: ...`

## 15.1 它是什么？

排序时可以指定排序依据：

```python
intervals.sort(key=lambda x: x[0])
```

意思是：

> 按照每个区间的第 0 个元素排序。

---

## 15.2 按第二个元素排序

```python
pairs = [[1, 3], [2, 2], [4, 1]]

pairs.sort(key=lambda x: x[1])

print(pairs)
```

输出：

```python
[[4, 1], [2, 2], [1, 3]]
```

---

## 15.3 多关键字排序

先按第一个升序，再按第二个降序：

```python
points.sort(key=lambda x: (x[0], -x[1]))
```

---

## 15.4 Hot100 场景

- 合并区间；
- 根据身高重建队列；
- 前 K 个高频元素；
- 任务调度器；
- 贪心排序题。


# 16. `float("inf")`

## 16.1 它是什么？

```python
float("inf")
```

表示正无穷。

```python
float("-inf")
```

表示负无穷。

---

## 16.2 初始化最小值

```python
min_val = float("inf")

for num in nums:
    if num < min_val:
        min_val = num
```

---

## 16.3 初始化最大值

```python
max_val = float("-inf")

for num in nums:
    if num > max_val:
        max_val = num
```

---

## 16.4 Hot100 场景

- 最大子数组和；
- 买卖股票的最佳时机；
- 二叉树最大路径和；
- 动态规划最值初始化。


# 17. `None`、`is None`

## 17.1 `None` 是什么？

`None` 表示空值。

链表和树中经常看到：

```python
if head is None:
    return None
```

---

## 17.2 推荐写法

判断为空：

```python
if node is None:
    return
```

判断不为空：

```python
if node is not None:
    ...
```

---

## 17.3 为什么不用 `== None`？

虽然很多时候也能运行，但更推荐：

```python
is None
```

因为 `None` 是单例对象，用 `is` 判断更规范。

---

## 17.4 Hot100 场景

链表：

```python
while cur is not None:
    cur = cur.next
```

二叉树：

```python
if root is None:
    return 0
```


# 18. `in` / `not in`

## 18.1 基本用法

```python
if x in nums:
    print("存在")
```

```python
if x not in seen:
    seen.add(x)
```

---

## 18.2 不同容器复杂度

| 写法 | 平均复杂度 |
|---|---|
| `x in list` | `O(n)` |
| `x in set` | `O(1)` |
| `key in dict` | `O(1)` |
| `ch in string` | `O(n)` |

---

## 18.3 Hot100 高频写法

```python
seen = set()

for num in nums:
    if num in seen:
        return True
    seen.add(num)
```

两数之和：

```python
if need in index_map:
    return [index_map[need], i]
```


# 19. `list` 作为栈

## 19.1 栈是什么？

栈的特点：

```text
后进先出
```

Python 中通常用列表实现栈。

---

## 19.2 基本模板

```python
stack = []

stack.append(x)  # 入栈
x = stack.pop()  # 出栈
```

---

## 19.3 查看栈顶

```python
stack[-1]
```

---

## 19.4 有效括号示例

```python
def isValid(s):
    stack = []
    pairs = {
        ')': '(',
        ']': '[',
        '}': '{'
    }

    for ch in s:
        if ch in pairs:
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
        else:
            stack.append(ch)

    return not stack
```

---

## 19.5 Hot100 场景

- 有效括号；
- 字符串解码；
- 每日温度；
- 接雨水；
- 柱状图最大矩形。


# 20. 切片 `[:]`、`[::-1]`

## 20.1 基本语法

```python
nums[start:end:step]
```

---

## 20.2 常见写法

```python
nums[:]       # 复制整个列表
nums[1:3]     # 取下标 1 到 2
nums[::-1]    # 反转
nums[:3]      # 前三个
nums[3:]      # 从下标 3 到最后
```

---

## 20.3 示例

```python
nums = [1, 2, 3, 4, 5]

print(nums[1:4])
print(nums[::-1])
```

输出：

```python
[2, 3, 4]
[5, 4, 3, 2, 1]
```

---

## 20.4 回溯中为什么用 `path[:]`

```python
ans.append(path[:])
```

表示复制当前路径。

不能写：

```python
ans.append(path)
```

因为这样保存的是同一个列表引用，后面 `path` 变化，答案里的内容也会变。


# 21. 字符串、`join()`、`split()`

## 21.1 字符串可遍历

```python
s = "abc"

for ch in s:
    print(ch)
```

---

## 21.2 字符串不可变

不能直接写：

```python
s[0] = "x"
```

如果要修改字符串，可以转列表：

```python
arr = list(s)
arr[0] = "x"
s = ''.join(arr)
```

---

## 21.3 `join()`

```python
arr = ["a", "b", "c"]
s = ''.join(arr)
print(s)
```

输出：

```python
"abc"
```

字母异位词分组常见：

```python
key = ''.join(sorted(s))
```

---

## 21.4 `split()`

```python
s = "hello world"
words = s.split()
```

输出：

```python
["hello", "world"]
```

指定分隔符：

```python
s = "a,b,c"
arr = s.split(",")
```

输出：

```python
["a", "b", "c"]
```


# 22. `ord()` / `chr()`

## 22.1 `ord()`

返回字符编码：

```python
ord("a")
```

结果：

```python
97
```

---

## 22.2 `chr()`

把编码转成字符：

```python
chr(97)
```

结果：

```python
"a"
```

---

## 22.3 字符计数数组

```python
count = [0] * 26

for ch in s:
    idx = ord(ch) - ord('a')
    count[idx] += 1
```

---

## 22.4 Hot100 场景

- 字母异位词；
- 最小覆盖子串；
- 滑动窗口字符串题；
- 字符频率统计。


# 23. `//`、`%`、`divmod()`

## 23.1 `//` 整数除法

```python
5 // 2
```

结果：

```python
2
```

二分查找中：

```python
mid = left + (right - left) // 2
```

不要写：

```python
mid = (left + right) / 2
```

因为 `/` 得到浮点数。

---

## 23.2 `%` 取余

```python
5 % 2
```

结果：

```python
1
```

常用于判断奇偶：

```python
if num % 2 == 0:
    print("偶数")
```

---

## 23.3 `divmod()`

```python
q, r = divmod(5, 2)
```

得到：

```python
q = 2
r = 1
```

等价于：

```python
q = 5 // 2
r = 5 % 2
```


# 24. `nonlocal`

## 24.1 它是什么？

当内部函数想修改外层函数的变量时，需要用 `nonlocal`。

---

## 24.2 示例

```python
def outer():
    ans = 0

    def inner():
        nonlocal ans
        ans += 1

    inner()
    return ans
```

---

## 24.3 Hot100 典型：二叉树最大路径和

```python
def maxPathSum(root):
    ans = float("-inf")

    def dfs(node):
        nonlocal ans

        if not node:
            return 0

        left = max(dfs(node.left), 0)
        right = max(dfs(node.right), 0)

        ans = max(ans, node.val + left + right)

        return node.val + max(left, right)

    dfs(root)
    return ans
```

这里：

```python
nonlocal ans
```

表示：

> 我要修改外层函数里的 `ans`。


# 25. `lru_cache`

## 25.1 它是什么？

`lru_cache` 是记忆化缓存工具，常用于递归 DP。

导入：

```python
from functools import lru_cache
```

使用：

```python
@lru_cache(None)
def dfs(i):
    ...
```

---

## 25.2 斐波那契示例

```python
from functools import lru_cache

@lru_cache(None)
def fib(n):
    if n <= 1:
        return n

    return fib(n - 1) + fib(n - 2)
```

没有缓存时会重复计算；有缓存后，每个 `n` 只计算一次。

---

## 25.3 Hot100 场景

- 爬楼梯；
- 打家劫舍；
- 单词拆分；
- 编辑距离；
- 正则表达式匹配；
- 分割回文串。

---

## 25.4 注意

被缓存的参数必须可哈希。

可以：

```python
dfs(i, j)
```

不可以直接：

```python
dfs(path)
```

如果 `path` 是列表，因为列表不可哈希。


# 26. `OrderedDict`

## 26.1 它是什么？

`OrderedDict` 是有顺序的字典。

导入：

```python
from collections import OrderedDict
```

LRU 缓存题中经常看到。

---

## 26.2 常用方法

```python
od.move_to_end(key)
od.popitem(last=False)
```

`move_to_end(key)`：

> 把 key 移到末尾，表示最近使用。

`popitem(last=False)`：

> 弹出最前面的元素，通常表示最久未使用。

---

## 26.3 LRU Cache 示例

```python
from collections import OrderedDict

class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key: int) -> int:
        if key not in self.cache:
            return -1

        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache.move_to_end(key)

        self.cache[key] = value

        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
```

---

## 26.4 面试注意

有些面试官会要求你手写“双向链表 + 哈希表”，不能直接用 `OrderedDict`。  
但是你要能看懂这种题解。


# 27. 链表 `ListNode`

## 27.1 LeetCode 链表节点

通常定义如下：

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
```

节点有：

```python
val   # 当前值
next  # 下一个节点
```

---

## 27.2 遍历链表

```python
cur = head

while cur:
    print(cur.val)
    cur = cur.next
```

---

## 27.3 虚拟头节点 `dummy`

链表题常用：

```python
dummy = ListNode(0)
cur = dummy
```

最后返回：

```python
return dummy.next
```

---

## 27.4 Hot100 场景

- 两数相加；
- 删除链表倒数第 N 个节点；
- 合并两个有序链表；
- 合并 K 个升序链表；
- 反转链表；
- 环形链表。


# 28. 二叉树 `TreeNode`

## 28.1 LeetCode 二叉树节点

通常定义如下：

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

---

## 28.2 DFS 模板

```python
def dfs(root):
    if not root:
        return

    dfs(root.left)
    dfs(root.right)
```

---

## 28.3 求最大深度

```python
def maxDepth(root):
    if not root:
        return 0

    left = maxDepth(root.left)
    right = maxDepth(root.right)

    return max(left, right) + 1
```

---

## 28.4 层序遍历

```python
from collections import deque

def levelOrder(root):
    if not root:
        return []

    ans = []
    q = deque([root])

    while q:
        level = []

        for _ in range(len(q)):
            node = q.popleft()
            level.append(node.val)

            if node.left:
                q.append(node.left)

            if node.right:
                q.append(node.right)

        ans.append(level)

    return ans
```


# 29. 二维数组初始化

## 29.1 正确写法

```python
dp = [[0] * n for _ in range(m)]
```

表示创建 `m` 行 `n` 列的二维数组。

---

## 29.2 错误写法

```python
dp = [[0] * n] * m
```

这个写法会导致每一行其实是同一个列表引用。

示例：

```python
dp = [[0] * 3] * 2

dp[0][0] = 1

print(dp)
```

输出：

```python
[[1, 0, 0], [1, 0, 0]]
```

你只改了第一行，第二行也跟着变了。

---

## 29.3 正确示例

```python
dp = [[0] * 3 for _ in range(2)]

dp[0][0] = 1

print(dp)
```

输出：

```python
[[1, 0, 0], [0, 0, 0]]
```

---

## 29.4 Hot100 场景

- 不同路径；
- 最小路径和；
- 编辑距离；
- 最长公共子序列；
- 岛屿数量 visited 数组。


# 30. Hot100 常见模板总结

## 30.1 哈希表模板

```python
index_map = {}

for i, num in enumerate(nums):
    need = target - num

    if need in index_map:
        return [index_map[need], i]

    index_map[num] = i
```

常见题：

- 两数之和；
- 和为 K 的子数组；
- 最长连续序列。

---

## 30.2 计数模板

```python
from collections import defaultdict

count = defaultdict(int)

for x in nums:
    count[x] += 1
```

常见题：

- 前 K 个高频元素；
- 字符计数；
- 滑动窗口。

---

## 30.3 分组模板

```python
from collections import defaultdict

groups = defaultdict(list)

for s in strs:
    key = ''.join(sorted(s))
    groups[key].append(s)

return list(groups.values())
```

常见题：

- 字母异位词分组。

---

## 30.4 BFS 队列模板

```python
from collections import deque

q = deque([start])
visited = set([start])

while q:
    node = q.popleft()

    for nxt in graph[node]:
        if nxt not in visited:
            visited.add(nxt)
            q.append(nxt)
```

常见题：

- 层序遍历；
- 腐烂的橘子；
- 图最短路径。

---

## 30.5 栈模板

```python
stack = []

for x in nums:
    stack.append(x)

    while stack and 条件:
        stack.pop()
```

常见题：

- 有效括号；
- 每日温度；
- 接雨水；
- 字符串解码。

---

## 30.6 堆模板

```python
import heapq

heap = []

for x in nums:
    heapq.heappush(heap, x)

while heap:
    x = heapq.heappop(heap)
```

常见题：

- 前 K 个高频元素；
- 合并 K 个升序链表；
- 数据流中位数。

---

## 30.7 回溯模板

```python
ans = []
path = []

def backtrack(start):
    if 满足条件:
        ans.append(path[:])
        return

    for i in range(start, len(nums)):
        path.append(nums[i])
        backtrack(i + 1)
        path.pop()
```

常见题：

- 子集；
- 全排列；
- 组合总和；
- 括号生成。

---

## 30.8 滑动窗口模板

```python
left = 0
window = {}

for right, ch in enumerate(s):
    window[ch] = window.get(ch, 0) + 1

    while 不满足条件:
        left_ch = s[left]
        window[left_ch] -= 1
        left += 1

    更新答案
```

常见题：

- 无重复字符的最长子串；
- 最小覆盖子串；
- 找到字符串中所有字母异位词。

---

# 31. 优先掌握清单

如果你现在刚开始刷 Hot100，建议按下面顺序掌握：

| 优先级 | 知识点 | 原因 |
|---|---|---|
| 1 | `dict` | 哈希表题核心 |
| 2 | `set` | 快速查找、去重 |
| 3 | `defaultdict(int)` | 计数、前缀和 |
| 4 | `defaultdict(list)` | 分组 |
| 5 | `Counter` | 快速计数 |
| 6 | `deque` | BFS、队列 |
| 7 | `heapq` | Top K、优先队列 |
| 8 | `enumerate` | 下标和元素 |
| 9 | `sort` / `sorted` | 排序 + 双指针 |
| 10 | `lambda` | 自定义排序 |
| 11 | `path[:]` | 回溯复制路径 |
| 12 | `nonlocal` | 树形 DFS 中维护答案 |

---

# 32. 最后总结

Hot100 中这些“陌生语法”本质上对应的是常见数据结构：

| Python 写法 | 本质用途 |
|---|---|
| `dict` | 哈希表 |
| `set` | 去重 + 快速查找 |
| `defaultdict(int)` | 计数 |
| `defaultdict(list)` | 分组 |
| `Counter` | 频率统计 |
| `deque` | 队列 / BFS |
| `heapq` | 堆 / 优先队列 |
| `list.append/pop` | 栈 |
| `enumerate` | 下标 + 元素 |
| `lambda` | 排序规则 |
| `path[:]` | 复制当前结果 |
| `lru_cache` | 记忆化搜索 |
| `OrderedDict` | LRU 缓存 |

你刷题时不要孤立地背语法，而要把它和题型绑定：

- 看到“频率”想到 `Counter` / `defaultdict(int)`；
- 看到“分组”想到 `defaultdict(list)`；
- 看到“BFS”想到 `deque`；
- 看到“Top K”想到 `heapq`；
- 看到“快速查找”想到 `set` / `dict`；
- 看到“回溯保存路径”想到 `path[:]`；
- 看到“递归重复计算”想到 `lru_cache`。


%% 项目关联导航：开始 %%
## 项目关联导航

- 所属模块：[[outputs/项目整理/模块-leetcode|模块-leetcode]]

%% 项目关联导航：结束 %%
