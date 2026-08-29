# Session 2 — Introduction to Python Libraries: NumPy & Pandas

**NumPy Fundamentals · Pandas for Data Handling · Pandas for Data Cleaning · Matplotlib & Seaborn**

| | |
|---|---|
| **Notebook** | [session-02-numpy-pandas.ipynb](../notebooks/session-02-numpy-pandas.ipynb) |
| **Previous** | [Session 1 — Python Refresher](session-01-python-refresher.md) |
| **Next** | [Session 3 — Visualisation & Preprocessing](session-03-eda-preprocessing.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **Session 1 gave you Python. This session gives you the two libraries that turn Python into a data tool.** Everything from Session 3 onwards is built on them.
>
> Every topic uses only what came before it. Work through them in order.

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Say why NumPy exists and when a list is not enough
2. Create arrays five different ways and know the shape of each
3. Index and slice arrays in one and two dimensions
4. Explain the difference between a copy and a view — and why it bites people
5. Reshape an array and know why some reshapes are impossible
6. Load a CSV, explore it, and describe what is in it
7. Read a correlation table without over-claiming
8. Plot straight from a DataFrame
9. **Clean a messy dataset: empty cells, wrong formats, wrong data, duplicates**
10. Build proper figures with Matplotlib, and statistical plots with Seaborn

---

## The twenty-three topics

**Part A — NumPy Fundamentals**

| # | Topic | | # | Topic |
|---|---|---|---|---|
| 1 | [NumPy Intro](#1-numpy-intro) | | 6 | [Copy vs View](#6-numpy-copy-vs-view) |
| 2 | [Creating Arrays](#2-numpy-creating-arrays) | | 7 | [Array Shape](#7-numpy-array-shape) |
| 3 | [Array Indexing](#3-numpy-array-indexing) | | 8 | [Array Reshape](#8-numpy-array-reshape) |
| 4 | [Array Slicing](#4-numpy-array-slicing) | | 9 | [Array Iterating](#9-numpy-array-iterating) |
| 5 | [Data Types](#5-numpy-data-types) | | | |

**Part B — Pandas for Data Handling**

| # | Topic | | # | Topic |
|---|---|---|---|---|
| 10 | [Pandas Introduction](#10-pandas-introduction) | | 14 | [Analyzing Data](#14-pandas-analyzing-data) |
| 11 | [Pandas Series](#11-pandas-series) | | 15 | [Correlations](#15-pandas-correlations) |
| 12 | [Pandas DataFrames](#12-pandas-dataframes) | | 16 | [Plotting](#16-pandas-plotting) |
| 13 | [Read CSV](#13-pandas-read-csv) | | | |

**Part C — Pandas for Data Cleaning**

| # | Topic | | # | Topic |
|---|---|---|---|---|
| 17 | [Cleaning Data](#17-cleaning-data) | | 20 | [Cleaning Wrong Data](#20-cleaning-wrong-data) |
| 18 | [Cleaning Empty Cells](#18-cleaning-empty-cells) | | 21 | [Removing Duplicates](#21-removing-duplicates) |
| 19 | [Cleaning Wrong Format](#19-cleaning-wrong-format) | | | |

**Part D — Python Libraries for Data Visualization**

| # | Topic | | # | Topic |
|---|---|---|---|---|
| 22 | [Matplotlib](#22-matplotlib) | | 23 | [Seaborn](#23-seaborn) |

**Six checkpoint problems** sit between the topics, each using only what you have learned so far:

| After topic | Problem |
|---|---|
| 4 | [⭐ Temperature analyser](#-checkpoint-problem-1--temperature-analyser) |
| 9 | [⭐ Multiplication grid](#-checkpoint-problem-2--multiplication-grid) |
| 12 | [⭐ Report card builder](#-checkpoint-problem-3--report-card-builder) |
| 16 | [⭐ Sales dashboard](#-checkpoint-problem-4--sales-dashboard) |
| 21 | [⭐ The full cleaning pipeline](#-checkpoint-problem-5--the-full-cleaning-pipeline) |
| 23 | [⭐ The one-page data story](#-checkpoint-problem-6--the-one-page-data-story) |

**Every topic has the same shape:**

```text
📘 Examples      3-4 short examples of the new idea
🌍 Scenarios     3 examples from real situations
✏️ Tasks         5 scenario-based tasks, with solutions
❓ MCQs          5 questions, with answers and why
```

---

# Part A — NumPy Fundamentals

# 1. NumPy Intro

**NumPy is Python's library for working with numbers in bulk.** Its one big idea is the **array**: a grid of values, all the same type.

🧠 **Analogy: a Python list is a shopping bag; a NumPy array is an egg carton.** The bag holds anything — apples, a book, a phone — so to count the apples you must look at every item and check what it is. The carton holds only eggs, in fixed slots. **Because everything is the same and evenly spaced, you can work on the whole carton at once.**

## Why not just use a list?

```python
prices = [100, 200, 300]
prices * 2            # [100, 200, 300, 100, 200, 300]  <- REPEATS the list
```

**That is almost never what you wanted.** With NumPy:

```python
import numpy as np

prices = np.array([100, 200, 300])
prices * 2            # array([200, 400, 600])   <- doubles each PRICE
```

> **This is called *vectorisation*: one operation applied to every element at once.** It is why NumPy exists, and why it is typically 10 to 100 times faster than a Python loop on the same data.

## Installing and importing

```bash
pip install numpy
```

```python
import numpy as np       # np is the universal convention - always use it
print(np.__version__)
```

## 📘 Examples

**Example 1 — your first array**

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
print(a)             # [1 2 3 4 5]
print(type(a))       # <class 'numpy.ndarray'>
```

**Notice there are no commas in the printed output.** That is how you can tell an array from a list at a glance.

**Example 2 — maths on the whole array at once**

```python
a = np.array([1, 2, 3, 4, 5])

print(a + 10)        # [11 12 13 14 15]
print(a * 2)         # [ 2  4  6  8 10]
print(a ** 2)        # [ 1  4  9 16 25]
```

**No loop anywhere.** Compare that with Session 1, where doubling a list needed a `for` loop and an `append`.

**Example 3 — two arrays together**

```python
maths = np.array([78, 92, 45])
physics = np.array([80, 88, 50])

print(maths + physics)          # [158 180  95]  - element by element
print((maths + physics) / 2)    # [79. 90. 47.5] - the averages
```

**Example 4 — the built-in summaries**

```python
marks = np.array([78, 92, 45, 88, 61])

print(marks.sum())      # 364
print(marks.mean())     # 72.8
print(marks.max())      # 92
print(marks.min())      # 45
print(marks.std())      # 17.53...  the spread
```

## 🌍 Scenarios

**Scenario 1 — applying a bonus to every salary**

```python
salaries = np.array([32000, 45000, 51000, 28000])
after_raise = salaries * 1.10          # a 10% raise for everyone
print(after_raise.astype(int))         # [35200 49500 56100 30800]
```

**Scenario 2 — converting a whole column of temperatures**

```python
celsius = np.array([0, 25, 37, 100])
fahrenheit = celsius * 9 / 5 + 32
print(fahrenheit)                      # [ 32.  77.  98.6 212. ]
```

**One line for the whole column.** In Session 1 this needed a loop.

**Scenario 3 — a class result summary**

```python
marks = np.array([78, 92, 45, 88, 61, 73, 39, 95])

print(f"Students: {marks.size}")
print(f"Average:  {marks.mean():.2f}")
print(f"Highest:  {marks.max()}")
print(f"Passed:   {(marks >= 40).sum()}")     # True counts as 1
```

**`(marks >= 40).sum()` is worth pausing on.** The comparison gives an array of `True`/`False`, and summing it counts the `True`s. **You will use this constantly.**

## ✏️ Tasks

1. A shop has prices `[120, 85, 260, 45]`. Create an array and print every price with 18% GST added, rounded to 2 decimals.
2. Two students' marks in four subjects are `[78, 92, 45, 88]` and `[80, 85, 60, 90]`. Print the difference in each subject and who did better overall.
3. Daily rainfall for a week is `[0, 12, 5, 0, 22, 8, 0]`. Print the total, the average, and how many days had no rain.
4. Convert the distances `[5, 12, 3, 8]` kilometres into metres, then into miles (1 km = 0.621371 miles), to 2 decimals.
5. Given marks `[45, 78, 92, 33, 61]`, print how many passed (≥ 40) and the pass percentage.

<details><summary>Solutions</summary>

```python
import numpy as np

prices = np.array([120, 85, 260, 45])                                  # 1
print(np.round(prices * 1.18, 2))          # [141.6  100.3  306.8   53.1]

a = np.array([78, 92, 45, 88])                                         # 2
b = np.array([80, 85, 60, 90])
print("difference:", a - b)                # [-2  7 -15 -2]
print("A total:", a.sum(), " B total:", b.sum())
print("better:", "B" if b.sum() > a.sum() else "A")

rain = np.array([0, 12, 5, 0, 22, 8, 0])                               # 3
print(f"total {rain.sum()} mm, average {rain.mean():.2f} mm")
print("dry days:", (rain == 0).sum())      # 3

km = np.array([5, 12, 3, 8])                                           # 4
print("metres:", km * 1000)
print("miles: ", np.round(km * 0.621371, 2))

marks = np.array([45, 78, 92, 33, 61])                                 # 5
passed = (marks >= 40).sum()
print(f"{passed} of {marks.size} passed = {passed / marks.size:.0%}")
# The comparison gives True/False; summing it COUNTS the Trues.
```
</details>

## ❓ MCQs

**Q1.** What does `[1, 2, 3] * 2` give for a Python **list**?
- (a) `[2, 4, 6]`  (b) `[1, 2, 3, 1, 2, 3]`  (c) An error  (d) `[1, 2, 3]`

**Q2.** What does `np.array([1, 2, 3]) * 2` give?
- (a) `[2, 4, 6]`  (b) `[1, 2, 3, 1, 2, 3]`  (c) An error  (d) `6`

**Q3.** What is *vectorisation*?
- (a) Drawing vectors  (b) One operation applied to every element at once, with no loop  (c) Sorting  (d) Converting to a list

**Q4.** What must be true of everything inside a NumPy array?
- (a) Nothing  (b) It must all be the same type  (c) It must be numbers  (d) It must be sorted

**Q5.** What does `(marks >= 40).sum()` count?
- (a) The total marks  (b) How many marks are 40 or above  (c) The average  (d) Nothing

<details><summary>Answers</summary>

**A1 — (b).** For a list, `*` **repeats**. This is almost never what you wanted with data.

**A2 — (a) `[2, 4, 6]`.** For an array, `*` applies to every element.

**A3 — (b).** It is why NumPy exists, and typically 10–100× faster than a Python loop.

**A4 — (b) All the same type.** That uniformity is what makes the speed possible — the egg carton, not the shopping bag.

**A5 — (b).** The comparison produces `True`/`False`, and `True` counts as 1.
</details>

---

# 2. NumPy Creating Arrays

**There are five ways to make an array, and you will use all of them.**

| Way | Use it when |
|---|---|
| `np.array([...])` | You have the values already |
| `np.zeros(n)` / `np.ones(n)` | You need a placeholder to fill in |
| `np.arange(start, stop, step)` | You need a counted sequence |
| `np.linspace(start, stop, n)` | You need `n` evenly spaced values |
| `np.random.*` | You need test data |

> ⚠️ **`arange` excludes the stop, `linspace` includes it.** `np.arange(0, 10, 2)` gives `[0 2 4 6 8]`; `np.linspace(0, 10, 5)` gives `[0. 2.5 5. 7.5 10.]`. **`arange` counts by step; `linspace` counts by how many you want.**

## Dimensions

```python
np.array(42)                          # 0-D  - a single value
np.array([1, 2, 3])                   # 1-D  - a row
np.array([[1, 2, 3], [4, 5, 6]])      # 2-D  - a table
np.array([[[1, 2], [3, 4]]])          # 3-D  - a stack of tables
```

**Count the opening square brackets** — that is the number of dimensions.

## 📘 Examples

**Example 1 — from values you already have**

```python
import numpy as np

a = np.array([1, 2, 3, 4, 5])
b = np.array([[1, 2, 3], [4, 5, 6]])       # 2-D: two rows, three columns
print(a)
print(b)
```

**Example 2 — placeholders**

```python
print(np.zeros(5))            # [0. 0. 0. 0. 0.]
print(np.ones(3))             # [1. 1. 1.]
print(np.zeros((2, 3)))       # a 2x3 grid of zeros - note the DOUBLE brackets
print(np.full(4, 7))          # [7 7 7 7]
```

> ⚠️ **The shape goes in as a tuple:** `np.zeros((2, 3))`, not `np.zeros(2, 3)`.

**Example 3 — sequences**

```python
print(np.arange(5))              # [0 1 2 3 4]        - like range()
print(np.arange(1, 11))          # [ 1  2 ... 10]
print(np.arange(0, 10, 2))       # [0 2 4 6 8]        - step of 2
print(np.linspace(0, 1, 5))      # [0. 0.25 0.5 0.75 1.] - FIVE values, ends included
```

**Example 4 — random test data**

```python
rng = np.random.default_rng(42)       # fix the seed so results repeat

print(rng.integers(1, 100, 5))        # 5 whole numbers from 1 to 99
print(rng.random(3).round(3))         # 3 decimals between 0 and 1
print(rng.normal(70, 10, 5).round(1)) # 5 values around a mean of 70
```

> **Always set a seed when you share results.** Without `default_rng(42)` your numbers differ every run, and nobody can reproduce your work.

## 🌍 Scenarios

**Scenario 1 — an empty attendance register to fill in later**

```python
students, days = 5, 20
register = np.zeros((students, days), dtype=int)
print(register.shape)          # (5, 20)
register[0, 0] = 1             # mark student 0 present on day 0
print(register[0, :5])         # [1 0 0 0 0]
```

**Scenario 2 — hourly time slots for a timetable**

```python
hours = np.arange(9, 18)       # 9am to 5pm
print(hours)                   # [ 9 10 11 12 13 14 15 16 17]
```

**Scenario 3 — generating marks to test your code before real data arrives**

```python
rng = np.random.default_rng(0)
fake_marks = rng.integers(35, 100, 30)
print(f"{fake_marks.size} students, average {fake_marks.mean():.1f}")
```

**Generating fake data to test with is a real professional habit**, not a shortcut — it lets you write and debug your analysis before anyone hands you the real file.

## ✏️ Tasks

1. Create an array of the even numbers from 2 to 20 using `arange`.
2. Create 7 evenly spaced values between 0 and 100 using `linspace`.
3. Create a 3×4 grid of zeros and print its shape.
4. Generate 10 random exam marks between 40 and 100 with a fixed seed, and print the average.
5. Create an array of the twelve month numbers, and a matching array of 12 zeros to hold monthly sales.

<details><summary>Solutions</summary>

```python
import numpy as np

print(np.arange(2, 21, 2))                     # 1  [2 4 6 ... 20]
print(np.linspace(0, 100, 7))                  # 2  seven values, both ends included

grid = np.zeros((3, 4))                        # 3
print(grid)
print(grid.shape)                              # (3, 4)

rng = np.random.default_rng(1)                 # 4
marks = rng.integers(40, 101, 10)
print(marks, f"average {marks.mean():.2f}")

months = np.arange(1, 13)                      # 5
sales = np.zeros(12)
print(months)
print(sales)
# The zeros array is a PLACEHOLDER - you fill it in as the data arrives.
```
</details>

## ❓ MCQs

**Q1.** What does `np.arange(0, 10, 2)` produce?
- (a) `[0 2 4 6 8 10]`  (b) `[0 2 4 6 8]`  (c) `[0 1 2 ... 9]`  (d) `[2 4 6 8 10]`

**Q2.** What does `np.linspace(0, 10, 5)` produce?
- (a) `[0 5 10]`  (b) `[0. 2.5 5. 7.5 10.]`  (c) `[0 1 2 3 4]`  (d) Five random values

**Q3.** How do you create a 2×3 grid of zeros?
- (a) `np.zeros(2, 3)`  (b) `np.zeros((2, 3))`  (c) `np.zeros[2, 3]`  (d) `np.zero(2, 3)`

**Q4.** How many dimensions does `np.array([[1, 2], [3, 4]])` have?
- (a) 1  (b) 2  (c) 4  (d) 0

**Q5.** Why set a random seed?
- (a) It is faster  (b) So your results can be reproduced by someone else  (c) It makes numbers more random  (d) It is required

<details><summary>Answers</summary>

**A1 — (b).** **`arange` excludes the stop**, exactly like `range()` in Session 1.

**A2 — (b).** **`linspace` includes both ends** and gives you exactly the count you asked for.

**A3 — (b).** The shape goes in as a **tuple** — note the double brackets.

**A4 — (b) 2.** Count the opening square brackets.

**A5 — (b).** Without one, your numbers change every run and nobody can check your work.
</details>

---

# 3. NumPy Array Indexing

**Getting a single value out. Positions start at 0** — exactly like strings and lists in Session 1.

```python
a = np.array([10, 20, 30, 40, 50])
#              0   1   2   3   4
#             -5  -4  -3  -2  -1

a[0]       # 10   the first
a[2]       # 30
a[-1]      # 50   the last
```

## Two dimensions: `[row, column]`

```python
b = np.array([[1, 2, 3],
              [4, 5, 6]])

b[0, 0]     # 1    row 0, column 0
b[1, 2]     # 6    row 1, column 2
b[0, -1]    # 3    row 0, last column
```

> **Write `b[1, 2]`, not `b[1][2]`.** Both work, but the comma form is the NumPy way and it is faster — `b[1][2]` builds an intermediate row first.

## 📘 Examples

**Example 1 — one dimension**

```python
import numpy as np

marks = np.array([78, 92, 45, 88, 61])
print(marks[0])       # 78   first student
print(marks[-1])      # 61   last student
print(marks[2])       # 45
```

**Example 2 — two dimensions**

```python
# Three students, four subjects
grid = np.array([[78, 92, 45, 88],
                 [65, 70, 80, 75],
                 [90, 85, 95, 88]])

print(grid[0, 0])     # 78   student 0, subject 0
print(grid[2, 3])     # 88   student 2, subject 3
print(grid[1, -1])    # 75   student 1, last subject
```

**Example 3 — changing a value**

```python
marks = np.array([78, 92, 45, 88, 61])
marks[2] = 50                       # remark that paper
print(marks)                        # [78 92 50 88 61]
```

**Example 4 — fancy indexing: several positions at once**

```python
marks = np.array([78, 92, 45, 88, 61])
print(marks[[0, 2, 4]])             # [78 45 61]  - pick three positions
print(marks[marks > 70])            # [78 92 88]  - pick by CONDITION
```

**`marks[marks > 70]` is boolean indexing**, and it is one of the most useful things in NumPy. **Read it as: "the marks, where the marks are above 70".**

## 🌍 Scenarios

**Scenario 1 — reading one cell of a sales grid**

```python
# rows = shops, columns = months
sales = np.array([[120, 135, 150],
                  [200, 180, 210],
                  [ 90, 110, 105]])

print(f"Shop 2, month 3: {sales[1, 2]}")     # 210
print(f"Shop 3, month 1: {sales[2, 0]}")     # 90
```

**Scenario 2 — correcting a data entry error**

```python
temperatures = np.array([28, 31, 350, 29, 30])   # 350 is clearly wrong
temperatures[2] = 30                             # fix it
print(temperatures)                              # [28 31 30 29 30]
```

**Scenario 3 — finding who passed, without a loop**

```python
marks = np.array([45, 78, 92, 33, 61, 39])
print(marks[marks >= 40])            # [45 78 92 61]  - the passing marks
print((marks >= 40).sum())           # 4              - how many
```

## ✏️ Tasks

1. From `[15, 28, 33, 47, 52]`, print the first, last, and third value.
2. Create a 3×3 grid of the numbers 1–9 and print the middle value.
3. A shop's daily sales are `[450, 380, 620, 510, 295]`. Print the best day's figure and its position.
4. From temperatures `[28, 31, 35, 29, 42, 30]`, print only those above 30.
5. In `[12, 45, 7, 89, 23]`, replace any value below 10 with 10, then print the array.

<details><summary>Solutions</summary>

```python
import numpy as np

a = np.array([15, 28, 33, 47, 52])                                     # 1
print(a[0], a[-1], a[2])                    # 15 52 33

grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])                     # 2
print(grid[1, 1])                           # 5

sales = np.array([450, 380, 620, 510, 295])                            # 3
print("best:", sales.max(), "on day", sales.argmax() + 1)
# argmax() gives the POSITION of the largest value.

temps = np.array([28, 31, 35, 29, 42, 30])                             # 4
print(temps[temps > 30])                    # [31 35 42]

nums = np.array([12, 45, 7, 89, 23])                                   # 5
nums[nums < 10] = 10
print(nums)                                 # [12 45 10 89 23]
# You can ASSIGN through a boolean index too, not just read.
```
</details>

## ❓ MCQs

**Q1.** What does `np.array([10, 20, 30])[1]` return?
- (a) 10  (b) 20  (c) 30  (d) An error

**Q2.** How do you get row 1, column 2 of a 2-D array `b`?
- (a) `b(1, 2)`  (b) `b[1, 2]`  (c) `b[1; 2]`  (d) `b.get(1, 2)`

**Q3.** What does `a[-1]` return?
- (a) An error  (b) The last element  (c) The first element  (d) Nothing

**Q4.** What does `marks[marks > 70]` return?
- (a) `True`/`False`  (b) Only the marks above 70  (c) The count  (d) An error

**Q5.** What does `sales.argmax()` return?
- (a) The largest value  (b) The **position** of the largest value  (c) The smallest value  (d) The average

<details><summary>Answers</summary>

**A1 — (b) 20.** Positions start at 0.

**A2 — (b) `b[1, 2]`.** The comma form is the NumPy way; `b[1][2]` also works but builds an intermediate row.

**A3 — (b) The last element.** Same as strings and lists in Session 1.

**A4 — (b).** **Boolean indexing** — read it as "the marks, where the marks are above 70".

**A5 — (b) The position.** Use `.max()` for the value itself.
</details>

---

# 4. NumPy Array Slicing

**Getting a run of values. `array[start:stop:step]` — and the stop is excluded**, exactly as in Session 1.

```python
a = np.array([10, 20, 30, 40, 50, 60])

a[1:4]        # [20 30 40]   positions 1, 2, 3 - NOT 4
a[:3]         # [10 20 30]   from the beginning
a[3:]         # [40 50 60]   to the end
a[-2:]        # [50 60]      the last two
a[::2]        # [10 30 50]   every second one
a[::-1]       # [60 50 40 30 20 10]   reversed
```

## Slicing two dimensions

**`array[row_slice, column_slice]`** — a comma separates the two.

```python
b = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])

b[0, :]        # [1 2 3 4]        row 0, all columns
b[:, 1]        # [2 6 10]         all rows, column 1
b[0:2, 1:3]    # [[2 3], [6 7]]   a rectangular block
b[:, -1]       # [4 8 12]         the last column
```

> **`b[:, 1]` is the pattern you will use most.** The `:` means "every row", so it reads as *"give me column 1 for every row"* — which is exactly what selecting a column means.

## 📘 Examples

**Example 1 — one dimension**

```python
import numpy as np

a = np.array([10, 20, 30, 40, 50, 60])
print(a[1:4])       # [20 30 40]
print(a[:3])        # [10 20 30]
print(a[-3:])       # [40 50 60]
print(a[::2])       # [10 30 50]
```

**Example 2 — reversing**

```python
a = np.array([1, 2, 3, 4, 5])
print(a[::-1])      # [5 4 3 2 1]
```

**Example 3 — rows and columns**

```python
grid = np.array([[78, 92, 45],
                 [65, 70, 80],
                 [90, 85, 95]])

print(grid[0, :])       # [78 92 45]   student 0's marks
print(grid[:, 0])       # [78 65 90]   everyone's subject-0 marks
print(grid[:2, :2])     # top-left 2x2 block
```

**Example 4 — the averages that fall out of it**

```python
grid = np.array([[78, 92, 45],
                 [65, 70, 80],
                 [90, 85, 95]])

print(grid.mean(axis=1).round(1))   # [71.7 71.7 90. ]  each STUDENT's average
print(grid.mean(axis=0).round(1))   # [77.7 82.3 73.3]  each SUBJECT's average
```

> **`axis=1` goes across the row; `axis=0` goes down the column.** A useful way to remember: **`axis=0` is the direction the row numbers increase.**

## 🌍 Scenarios

**Scenario 1 — the last week of a month's sales**

```python
daily_sales = np.arange(1, 31) * 100      # 30 days
print(daily_sales[-7:])                    # the last seven days
print(f"last week total: {daily_sales[-7:].sum()}")
```

**Scenario 2 — pulling one subject's column out of a mark sheet**

```python
# rows = students, columns = maths, physics, chemistry
marks = np.array([[78, 92, 45],
                  [65, 70, 80],
                  [90, 85, 95],
                  [55, 60, 70]])

physics = marks[:, 1]
print(physics)                     # [92 70 85 60]
print(f"physics average: {physics.mean():.2f}")
```

**Scenario 3 — every second reading, to thin out a sensor log**

```python
readings = np.array([20.1, 20.3, 20.2, 20.8, 21.0, 21.2, 21.1, 20.9])
print(readings[::2])               # [20.1 20.2 21.  21.1]
print(f"kept {readings[::2].size} of {readings.size} readings")
```

## ✏️ Tasks

1. From `np.arange(1, 21)`, print the first five, the last five, and every third value.
2. Print the numbers 1 to 10 in reverse using slicing.
3. Build a 4×4 grid of the numbers 1–16 and print the second column, the third row, and the bottom-right 2×2 block.
4. A week of temperatures is `[28, 31, 35, 29, 42, 30, 27]`. Print the weekday values only (the first five).
5. From a 3×4 mark sheet, print each student's average and each subject's average.

<details><summary>Solutions</summary>

```python
import numpy as np

a = np.arange(1, 21)                                                   # 1
print(a[:5])          # [1 2 3 4 5]
print(a[-5:])         # [16 17 18 19 20]
print(a[::3])         # [ 1  4  7 10 13 16 19]

print(np.arange(1, 11)[::-1])                                          # 2

grid = np.arange(1, 17).reshape(4, 4)                                  # 3
print(grid)
print("column 1:", grid[:, 1])          # [ 2  6 10 14]
print("row 2:   ", grid[2, :])          # [ 9 10 11 12]
print("block:\n", grid[2:, 2:])         # bottom-right 2x2

temps = np.array([28, 31, 35, 29, 42, 30, 27])                         # 4
print("weekdays:", temps[:5])

marks = np.array([[78, 92, 45, 88],                                    # 5
                  [65, 70, 80, 75],
                  [90, 85, 95, 88]])
print("per student:", marks.mean(axis=1).round(2))
print("per subject:", marks.mean(axis=0).round(2))
# axis=1 goes ACROSS the row; axis=0 goes DOWN the column.
```
</details>

## ❓ MCQs

**Q1.** What does `np.array([10,20,30,40,50])[1:4]` return?
- (a) `[20 30 40 50]`  (b) `[20 30 40]`  (c) `[10 20 30]`  (d) `[20 30]`

**Q2.** How do you select the whole of column 2 from a 2-D array `b`?
- (a) `b[2]`  (b) `b[:, 2]`  (c) `b[2, :]`  (d) `b[, 2]`

**Q3.** What does `a[::-1]` do?
- (a) Deletes the array  (b) Reverses it  (c) Returns the last element  (d) An error

**Q4.** What does `grid.mean(axis=1)` compute for a students×subjects grid?
- (a) Each subject's average  (b) Each student's average  (c) The overall average  (d) An error

**Q5.** In `a[1:4]`, is position 4 included?
- (a) Yes  (b) No — the stop is excluded  (c) Only for 2-D  (d) Only for floats

<details><summary>Answers</summary>

**A1 — (b) `[20 30 40]`.** Positions 1, 2 and 3.

**A2 — (b) `b[:, 2]`.** `b[2, :]` gives you row 2 instead.

**A3 — (b) Reverses it.** A step of −1 walks backwards.

**A4 — (b) Each student's average.** `axis=1` goes across the row.

**A5 — (b) No.** The same rule as string and list slicing in Session 1.
</details>

---

## ⭐ Checkpoint Problem 1 — Temperature analyser

> **Uses only:** creating arrays, indexing, slicing. Topics 1–4.

**The problem.** You have a month of daily temperatures. Print the month's average, the hottest and coldest days (with their dates), the first week's average, and how many days were above 30°.

<details><summary>Solution</summary>

```python
import numpy as np

rng = np.random.default_rng(7)
temps = rng.integers(22, 40, 30)          # 30 days of temperatures
print("Temperatures:", temps)

print(f"\nMonth average : {temps.mean():.2f} C")
print(f"Hottest day   : {temps.max()} C on day {temps.argmax() + 1}")
print(f"Coldest day   : {temps.min()} C on day {temps.argmin() + 1}")
print(f"First week avg: {temps[:7].mean():.2f} C")
print(f"Last week avg : {temps[-7:].mean():.2f} C")
print(f"Days above 30 : {(temps > 30).sum()} of {temps.size}")
```

**Three ideas doing the work here:**

- `argmax()` gives the **position**; `+ 1` turns a 0-based position into a day number.
- `temps[:7]` and `temps[-7:]` slice the first and last week.
- `(temps > 30).sum()` counts without a loop.
</details>

**Make it harder:**

1. Print the average for each of the four weeks, using slicing.
2. Print every day that was hotter than the month average.
3. Reshape the first 28 days into 4 weeks × 7 days and print each week's average with `axis=1`.

---

# 5. NumPy Data Types

**Every array has one `dtype`, and everything in it shares that type.** That is the egg-carton rule from Topic 1, made concrete.

| Code | Type | Example |
|---|---|---|
| `i` | integer | `int64` |
| `f` | float | `float64` |
| `b` | boolean | `bool` |
| `U` | text (unicode) | `<U5` |

```python
np.array([1, 2, 3]).dtype          # int64
np.array([1.0, 2.0]).dtype         # float64
np.array([True, False]).dtype      # bool
np.array(["a", "bb"]).dtype        # <U2   - text, up to 2 characters
```

## Setting the type yourself

```python
np.array([1, 2, 3], dtype=float)      # [1. 2. 3.]
np.array([1.9, 2.9], dtype=int)       # [1 2]   - CHOPPED, not rounded
```

## Converting an existing array

```python
a = np.array([1.9, 2.9, 3.9])
b = a.astype(int)
print(b)          # [1 2 3]   - astype() makes a NEW array
print(a)          # [1.9 2.9 3.9] - the original is untouched
```

> ⚠️ **Converting float to int chops the decimal, it does not round** — the same trap as `int()` in Session 1. Use `np.round()` first if you want rounding.

## The mixing rule

```python
np.array([1, 2, 3.5])            # [1.  2.  3.5]  -> ALL become float
np.array([1, 2, "three"])        # ['1' '2' 'three'] -> ALL become text
```

**Put one float among integers and everything becomes float. Put one string in and everything becomes text.** NumPy picks the type that can hold every value — and once your numbers are text, arithmetic stops working.

## 📘 Examples

**Example 1 — checking the type**

```python
import numpy as np

print(np.array([1, 2, 3]).dtype)         # int64
print(np.array([1.5, 2.5]).dtype)        # float64
print(np.array([True, False]).dtype)     # bool
print(np.array(["yes", "no"]).dtype)     # <U3
```

**Example 2 — asking for a type**

```python
a = np.array([1, 2, 3], dtype=float)
print(a)            # [1. 2. 3.]
print(a.dtype)      # float64
```

**Example 3 — converting, and the chopping trap**

```python
prices = np.array([19.99, 5.49, 120.75])

print(prices.astype(int))            # [ 19   5 120]  <- CHOPPED
print(np.round(prices).astype(int))  # [ 20   5 121]  <- rounded properly
```

**Example 4 — the mixing rule in action**

```python
print(np.array([1, 2, 3.5]))         # [1.  2.  3.5]     one float -> all float
print(np.array([1, 2, "3"]))         # ['1' '2' '3']     one string -> all text

bad = np.array([1, 2, "3"])
# bad.sum()   -> these are TEXT now, so this does not do arithmetic
print(bad.astype(int).sum())         # 6  - convert back first
```

## 🌍 Scenarios

**Scenario 1 — memory matters on large data**

```python
big_float = np.ones(1_000_000, dtype="float64")
big_small = np.ones(1_000_000, dtype="int8")
print(f"float64: {big_float.nbytes / 1_000_000:.1f} MB")
print(f"int8   : {big_small.nbytes / 1_000_000:.1f} MB")
```

**Choosing the smallest type that fits your data can cut memory eightfold.** On a million rows that is the difference between fitting in memory and not.

**Scenario 2 — data arriving as text from a file**

```python
from_file = np.array(["25", "30", "45"])      # everything from a file is text
print(from_file.dtype)                        # <U2

ages = from_file.astype(int)
print(ages.mean())                            # 33.33...
```

**This is the same lesson as `input()` in Session 1.** Text in, convert before arithmetic.

**Scenario 3 — a flag column stored as 0 and 1**

```python
passed = np.array([1, 0, 1, 1, 0])
print(passed.astype(bool))                    # [True False True True False]
print(passed.sum(), "of", passed.size, "passed")
```

## ✏️ Tasks

1. Create an array of `[1, 2, 3]` as floats and print its dtype.
2. Convert `[9.7, 3.2, 8.8]` to integers. Then convert it again, rounding properly. Compare.
3. What is the dtype of `np.array([1, 2, "3"])`? Convert it so you can sum it.
4. Create an array of a million zeros as `int8` and as `float64`. Print the memory each uses.
5. Marks arrive from a file as `["78", "92", "45"]`. Convert them and print the average.

<details><summary>Solutions</summary>

```python
import numpy as np

a = np.array([1, 2, 3], dtype=float)                                   # 1
print(a, a.dtype)                       # [1. 2. 3.] float64

b = np.array([9.7, 3.2, 8.8])                                          # 2
print(b.astype(int))                    # [9 3 8]   <- chopped
print(np.round(b).astype(int))          # [10  3  9] <- rounded
# astype() CHOPS. Round first if you want rounding.

c = np.array([1, 2, "3"])                                              # 3
print(c.dtype)                          # <U21 (text)
print(c.astype(int).sum())              # 6

small = np.zeros(1_000_000, dtype="int8")                              # 4
big = np.zeros(1_000_000, dtype="float64")
print(f"int8:    {small.nbytes / 1_000_000:.1f} MB")
print(f"float64: {big.nbytes / 1_000_000:.1f} MB")   # 8x more

marks = np.array(["78", "92", "45"]).astype(int)                       # 5
print(f"average {marks.mean():.2f}")
```
</details>

## ❓ MCQs

**Q1.** What is the dtype of `np.array([1, 2, 3.5])`?
- (a) int64  (b) float64  (c) mixed  (d) An error

**Q2.** What does `np.array([1.9, 2.9]).astype(int)` give?
- (a) `[2 3]`  (b) `[1 2]`  (c) `[1.9 2.9]`  (d) An error

**Q3.** What is the dtype of `np.array([1, 2, "3"])`?
- (a) int64  (b) Text — everything is converted to string  (c) mixed  (d) float64

**Q4.** Why choose a smaller dtype like `int8`?
- (a) It is more accurate  (b) It uses far less memory on large arrays  (c) It is required  (d) It is slower

**Q5.** Data read from a file arrives as text. Before doing arithmetic you must…
- (a) Nothing  (b) Convert it with `.astype()`  (c) Sort it  (d) Reshape it

<details><summary>Answers</summary>

**A1 — (b) float64.** **One float makes everything float** — NumPy picks the type that holds every value.

**A2 — (b) `[1 2]`.** It **chops**. The same trap as `int()` in Session 1.

**A3 — (b) Text.** And arithmetic stops working until you convert back.

**A4 — (b).** `int8` uses an eighth of the memory of `float64`.

**A5 — (b).** The same lesson as `input()` in Session 1.
</details>

---

# 6. NumPy Copy vs View

**This topic exists because of one surprise that costs beginners hours.**

🧠 **Analogy: a photocopy versus a window.** A **photocopy** of a page is separate — scribble on it and the original is untouched. A **window** into a room is not a second room: rearrange the furniture "through the window" and the actual room changes. **A copy is a photocopy; a view is a window.**

```python
original = np.array([1, 2, 3, 4, 5])

c = original.copy()      # a COPY  - independent
v = original.view()      # a VIEW  - looks at the same data

original[0] = 99

print(original)   # [99  2  3  4  5]
print(c)          # [ 1  2  3  4  5]   <- unaffected
print(v)          # [99  2  3  4  5]   <- changed too
```

## ⚠️ The surprise: slicing gives you a VIEW

```python
a = np.array([1, 2, 3, 4, 5])
s = a[1:4]          # this looks like a new array. It is NOT.
s[0] = 555

print(a)            # [  1 555   3   4   5]   <- the ORIGINAL changed
```

> **This is where the hours go.** You slice out "just the bit I want", change it, and the original silently changes underneath you. **NumPy does this on purpose — copying data is expensive — but you must know it is happening.**
>
> **If you intend to modify a slice, `.copy()` it first.**

```python
s = a[1:4].copy()      # now it really is separate
```

> ⚠️ **Note: Pandas behaves differently here.** Slicing a DataFrame usually gives you a copy, and modifying it raises a `SettingWithCopyWarning`. Different library, different rule — do not carry this habit across.

## How to tell which you have

```python
print(c.base)     # None            -> it owns its data, it is a COPY
print(v.base)     # the array       -> it borrows, it is a VIEW
```

**`.base` is `None` for a copy and points at the original for a view.**

## 📘 Examples

**Example 1 — the basic difference**

```python
import numpy as np

original = np.array([10, 20, 30])
c = original.copy()
v = original.view()

original[0] = 999
print("original:", original)     # [999  20  30]
print("copy:    ", c)            # [ 10  20  30]
print("view:    ", v)            # [999  20  30]
```

**Example 2 — changing through the view changes the original**

```python
original = np.array([10, 20, 30])
v = original.view()
v[2] = 777
print(original)      # [ 10  20 777]   <- changed from the other side
```

**Example 3 — the slicing trap**

```python
data = np.array([1, 2, 3, 4, 5, 6])
chunk = data[2:5]           # a VIEW, not a new array
chunk[:] = 0
print(data)                 # [1 2 0 0 0 6]   <- the original was edited
```

**Example 4 — and the fix**

```python
data = np.array([1, 2, 3, 4, 5, 6])
chunk = data[2:5].copy()    # now genuinely separate
chunk[:] = 0
print(data)                 # [1 2 3 4 5 6]   <- untouched
print(chunk)                # [0 0 0]
```

## 🌍 Scenarios

**Scenario 1 — keeping the raw data safe**

```python
raw = np.array([28, 31, 35, 29, 42])
cleaned = raw.copy()               # ALWAYS copy before cleaning
cleaned[cleaned > 40] = 40         # cap the outlier

print("raw:    ", raw)             # [28 31 35 29 42]  - still original
print("cleaned:", cleaned)         # [28 31 35 29 40]
```

**Keeping an untouched raw copy is a professional habit.** When a result looks wrong, you need something to compare against.

**Scenario 2 — the bug this causes in real code**

```python
readings = np.array([20.1, 20.3, 99.9, 20.8, 21.0])

morning = readings[:3]         # "let me just look at the morning"
morning[2] = 20.5              # "and fix that bad reading"

print(readings)                # [20.1 20.3 20.5 20.8 21. ]
# The original changed. Sometimes that is what you wanted -
# but if you were only inspecting, you have quietly altered your data.
```

**Scenario 3 — deliberately using a view to save memory**

```python
big = np.arange(1_000_000)
window = big[500_000:500_010]     # a view - no data is copied at all
window[0] = -1
print(big[500_000])               # -1
print(f"view nbytes: {window.nbytes}  (it borrows, it does not own)")
```

**On large arrays a view is free and a copy is not.** That is why NumPy defaults to views.

## ✏️ Tasks

1. Make an array, take a copy and a view, change the original, and print all three.
2. Slice an array and change the slice. Show that the original changed.
3. Repeat task 2 with `.copy()` and show that the original is now safe.
4. Use `.base` to tell a copy from a view.
5. You are asked to cap all temperatures above 40 at 40, but must keep the raw data. Write it safely.

<details><summary>Solutions</summary>

```python
import numpy as np

original = np.array([1, 2, 3, 4, 5])                                   # 1
c, v = original.copy(), original.view()
original[0] = 99
print("original", original)   # [99  2  3  4  5]
print("copy    ", c)          # [ 1  2  3  4  5]
print("view    ", v)          # [99  2  3  4  5]

a = np.array([1, 2, 3, 4, 5])                                          # 2
s = a[1:4]
s[0] = 555
print(a)                      # [  1 555   3   4   5]  <- ORIGINAL changed
# A SLICE IS A VIEW. This is where beginners lose hours.

a = np.array([1, 2, 3, 4, 5])                                          # 3
s = a[1:4].copy()
s[0] = 555
print(a)                      # [1 2 3 4 5]  <- safe

print(c.base)                 # None      -> a COPY, it owns its data     # 4
print(v.base)                 # the array -> a VIEW, it borrows

raw = np.array([28, 31, 35, 29, 42, 45])                               # 5
capped = raw.copy()
capped[capped > 40] = 40
print("raw:   ", raw)         # unchanged - you can always go back
print("capped:", capped)
```
</details>

## ❓ MCQs

**Q1.** What does `array.copy()` give you?
- (a) A window onto the same data  (b) An independent array with its own data  (c) A reference  (d) A list

**Q2.** You slice an array and modify the slice. What happens to the original?
- (a) Nothing  (b) It changes too — a slice is a view  (c) An error  (d) It is deleted

**Q3.** What does `.base` return for a copy?
- (a) The original array  (b) `None`  (c) `True`  (d) The shape

**Q4.** Why does NumPy return views rather than copies by default?
- (a) Views are more accurate  (b) Copying large data is expensive  (c) It is a bug  (d) Views are easier to type

**Q5.** You want to modify a slice without touching the original. You should…
- (a) Just modify it  (b) Add `.copy()` to the slice  (c) Use `.view()`  (d) Convert to a list

<details><summary>Answers</summary>

**A1 — (b) An independent array.** The photocopy.

**A2 — (b) It changes too.** **This is the single most surprising thing in NumPy for beginners.**

**A3 — (b) `None`.** A copy owns its data; a view's `.base` points at the array it borrows from.

**A4 — (b).** On a million-element array a view is free and a copy is not.

**A5 — (b) `.copy()`.** Note Pandas behaves differently — slicing a DataFrame usually copies.
</details>

---

# 7. NumPy Array Shape

**The shape tells you how many rows, columns and layers an array has.** It is a tuple.

```python
a = np.array([1, 2, 3, 4])
print(a.shape)         # (4,)     four values in one row

b = np.array([[1, 2, 3], [4, 5, 6]])
print(b.shape)         # (2, 3)   two rows, three columns
```

| Attribute | Tells you | Example on `b` |
|---|---|---|
| `.shape` | Size in each dimension | `(2, 3)` |
| `.ndim` | How many dimensions | `2` |
| `.size` | Total number of elements | `6` |
| `.dtype` | The type of the values | `int64` |

> **`(4,)` with a trailing comma is a one-element tuple** — it means one dimension of length 4. `(4, 1)` would be four rows of one column, which is a different thing.

## Why shape matters

**Almost every error you will hit in Session 5 onwards is a shape error.**

```python
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])

# a + b
# -> ValueError: operands could not be broadcast together
#    with shapes (3,) (2,2)
#    Uncomment it to see the real error.
print("a:", a.shape, "  b:", b.shape)     # (3,)   (2, 2)   <- they do not match
```

**When something fails, print the shapes first.** It solves the problem most of the time.

## 📘 Examples

**Example 1 — the four attributes**

```python
import numpy as np

grid = np.array([[1, 2, 3, 4],
                 [5, 6, 7, 8],
                 [9, 10, 11, 12]])

print("shape:", grid.shape)     # (3, 4)
print("ndim: ", grid.ndim)      # 2
print("size: ", grid.size)      # 12
print("dtype:", grid.dtype)     # int64
```

**Example 2 — one dimension versus two**

```python
a = np.array([1, 2, 3])
b = np.array([[1, 2, 3]])

print(a.shape)      # (3,)     one dimension
print(b.shape)      # (1, 3)   TWO dimensions: one row, three columns
print(a.ndim, b.ndim)          # 1 2
```

**They print almost identically but behave differently.** Count the brackets.

**Example 3 — shape after slicing**

```python
grid = np.arange(12).reshape(3, 4)

print(grid[0].shape)        # (4,)    one row -> 1-D
print(grid[:, 0].shape)     # (3,)    one column -> also 1-D
print(grid[0:1].shape)      # (1, 4)  a slice of one row -> still 2-D
```

**Indexing drops a dimension; slicing keeps it.** That difference causes a lot of confusion — and a lot of shape errors.

**Example 4 — reading a shape error**

```python
a = np.array([1, 2, 3])
b = np.array([[1, 2], [3, 4]])

print("a:", a.shape, " b:", b.shape)     # a: (3,)  b: (2, 2)
# 3 does not match 2, so a + b cannot work.
# PRINTING THE SHAPES tells you why, immediately.
```

## 🌍 Scenarios

**Scenario 1 — checking a dataset loaded correctly**

```python
rng = np.random.default_rng(0)
data = rng.integers(0, 100, (100, 5))       # 100 records, 5 fields

print(f"{data.shape[0]} rows, {data.shape[1]} columns")
print(f"{data.size} values in total")
```

**`.shape[0]` is rows and `.shape[1]` is columns.** This is the first thing you check on any new dataset.

**Scenario 2 — a colour image is a 3-D array**

```python
image = np.zeros((480, 640, 3), dtype=np.uint8)   # height, width, RGB
print("shape:", image.shape)
print(f"{image.shape[0]} x {image.shape[1]} pixels, {image.shape[2]} colour channels")
print(f"{image.size:,} values in total")
```

**Scenario 3 — diagnosing a real error**

```python
students = np.array([[78, 92], [65, 70], [90, 85]])     # 3 students, 2 subjects
weights = np.array([0.6, 0.4])                          # subject weights

print(students.shape, weights.shape)      # (3, 2) (2,)
print(students @ weights)                 # [83.6 67.  88. ]  weighted totals
# The LAST dimension of students (2) matches weights (2), so this works.
```

## ✏️ Tasks

1. Create a 3×4 array and print its shape, ndim, size and dtype.
2. What are the shapes of `np.array([1,2,3])` and `np.array([[1,2,3]])`? Explain the difference.
3. Take a 4×5 array. Print the shape of one row, one column, and a one-row slice.
4. Create an array representing a 720×1280 colour image. How many values does it hold?
5. Two arrays fail to add. Print both shapes and explain in one sentence why.

<details><summary>Solutions</summary>

```python
import numpy as np

g = np.arange(12).reshape(3, 4)                                        # 1
print(g.shape, g.ndim, g.size, g.dtype)      # (3, 4) 2 12 int64

a, b = np.array([1, 2, 3]), np.array([[1, 2, 3]])                      # 2
print(a.shape, b.shape)                      # (3,) (1, 3)
# a is ONE dimension of length 3. b is TWO dimensions: one row of three.
# Count the opening brackets.

h = np.arange(20).reshape(4, 5)                                        # 3
print(h[0].shape)        # (5,)    indexing DROPS a dimension
print(h[:, 0].shape)     # (4,)
print(h[0:1].shape)      # (1, 5)  slicing KEEPS it

img = np.zeros((720, 1280, 3), dtype=np.uint8)                         # 4
print(img.shape, f"{img.size:,} values")     # 2,764,800

x, y = np.array([1, 2, 3]), np.array([[1, 2], [3, 4]])                 # 5
print("x:", x.shape, " y:", y.shape)         # (3,) (2, 2)
# The last dimensions are 3 and 2 - they do not match, so NumPy cannot
# line the arrays up element by element.
```
</details>

## ❓ MCQs

**Q1.** What is the shape of `np.array([[1,2,3],[4,5,6]])`?
- (a) `(3, 2)`  (b) `(2, 3)`  (c) `(6,)`  (d) `(2,)`

**Q2.** What does `.size` return?
- (a) The number of rows  (b) The total number of elements  (c) The memory used  (d) The dtype

**Q3.** What is the difference between shape `(3,)` and `(1, 3)`?
- (a) None  (b) `(3,)` is 1-D; `(1, 3)` is 2-D with one row  (c) `(3,)` is bigger  (d) `(1,3)` is invalid

**Q4.** `grid[0]` versus `grid[0:1]` on a 2-D array:
- (a) Identical  (b) Indexing drops a dimension; slicing keeps it  (c) Both drop it  (d) Both keep it

**Q5.** Your code fails with a broadcast error. What should you print first?
- (a) The values  (b) The shapes of both arrays  (c) The dtype  (d) The size

<details><summary>Answers</summary>

**A1 — (b) `(2, 3)`.** Rows first, then columns.

**A2 — (b) The total number of elements** — 2 × 3 = 6 here.

**A3 — (b).** They print almost identically and behave differently.

**A4 — (b).** A frequent source of shape errors.

**A5 — (b) The shapes.** **It solves the problem most of the time.**
</details>

---

# 8. NumPy Array Reshape

**Reshaping rearranges the same values into a different grid.** Nothing is added or removed.

🧠 **Analogy: rearranging 12 chairs.** You can put them in 3 rows of 4, or 4 rows of 3, or 2 rows of 6, or one long row of 12. **You cannot make 5 rows of 3 — that needs 15 chairs, and you only have 12.**

```python
a = np.arange(12)          # [0 1 2 ... 11]

a.reshape(3, 4)            # 3 rows, 4 columns
a.reshape(4, 3)            # 4 rows, 3 columns
a.reshape(2, 6)            # 2 rows, 6 columns
a.reshape(2, 3, 2)         # 3-D: 2 layers of 3x2
```

> ⚠️ **The numbers must multiply to the size.** `a.reshape(5, 3)` on 12 values raises `ValueError: cannot reshape array of size 12 into shape (5,3)`.

## The `-1` shortcut

**Put `-1` in one position and NumPy works it out for you.**

```python
a = np.arange(12)

a.reshape(3, -1)      # "3 rows, you figure out the columns"  -> (3, 4)
a.reshape(-1, 2)      # "2 columns, you figure out the rows"  -> (6, 2)
a.reshape(-1)         # flatten to one dimension              -> (12,)
```

**`reshape(-1)` is the standard way to flatten an array.**

## 📘 Examples

**Example 1 — the same values, different grids**

```python
import numpy as np

a = np.arange(12)
print(a.reshape(3, 4))
print(a.reshape(4, 3))
print(a.reshape(2, 6))
```

**Example 2 — the impossible reshape**

```python
a = np.arange(12)
# a.reshape(5, 3)
# -> ValueError: cannot reshape array of size 12 into shape (5,3)
#    5 x 3 = 15, and you only have 12 values.
print(a.size)          # 12
```

**Example 3 — the `-1` shortcut**

```python
a = np.arange(12)
print(a.reshape(3, -1).shape)     # (3, 4)
print(a.reshape(-1, 6).shape)     # (2, 6)
print(a.reshape(-1).shape)        # (12,)   flattened
```

**Example 4 — reshape returns a view**

```python
a = np.arange(6)
b = a.reshape(2, 3)
b[0, 0] = 99
print(a)            # [99  1  2  3  4  5]   <- the original changed
```

**Reshape gives you a view, not a copy** — Topic 6's rule applies here too.

## 🌍 Scenarios

**Scenario 1 — a month of readings into weeks**

```python
readings = np.arange(1, 29)              # 28 days
weeks = readings.reshape(4, 7)           # 4 weeks of 7 days
print(weeks)
print("weekly totals:", weeks.sum(axis=1))    # [ 28  77 126 175]
```

**Reshaping turned a flat log into something you can summarise per week** — with the `axis` idea from Topic 4.

**Scenario 2 — flattening an image to feed a model**

```python
image = np.arange(12).reshape(3, 4)     # a tiny 3x4 "image"
flat = image.reshape(-1)
print(image.shape, "->", flat.shape)    # (3, 4) -> (12,)
```

**Many models want one long row per sample.** Flattening is how you get there — and you will do exactly this in Session 9.

**Scenario 3 — one sample for a prediction**

```python
one_student = np.array([78, 92, 45, 88])
print(one_student.shape)                       # (4,)
print(one_student.reshape(1, -1).shape)        # (1, 4)
# Models expect a 2-D "table of samples", even for a single row.
# reshape(1, -1) means "one row, however many columns that needs".
```

> **You will meet this exact line in Session 5** when predicting for one person. It is here so it is not a surprise.

## ✏️ Tasks

1. Reshape `np.arange(24)` into 4×6, then 6×4, then 2×3×4.
2. Try to reshape 10 values into 3×4. What error do you get, and why?
3. Use `-1` to reshape `np.arange(20)` into 5 rows, then into 4 columns.
4. Take 28 daily readings, reshape into 4 weeks, and print each week's average.
5. Flatten a 3×4 grid into one dimension two different ways.

<details><summary>Solutions</summary>

```python
import numpy as np

a = np.arange(24)                                                      # 1
print(a.reshape(4, 6).shape, a.reshape(6, 4).shape, a.reshape(2, 3, 4).shape)

b = np.arange(10)                                                      # 2
# b.reshape(3, 4)
# -> ValueError: cannot reshape array of size 10 into shape (3,4)
print(b.size, "values, but 3 x 4 needs", 3 * 4)
# The numbers must MULTIPLY to the size. 12 chairs cannot make 5 rows of 3.

c = np.arange(20)                                                      # 3
print(c.reshape(5, -1).shape)       # (5, 4)
print(c.reshape(-1, 4).shape)       # (5, 4)

rng = np.random.default_rng(3)                                         # 4
readings = rng.integers(20, 40, 28)
weeks = readings.reshape(4, 7)
print("weekly averages:", weeks.mean(axis=1).round(2))

g = np.arange(12).reshape(3, 4)                                        # 5
print(g.reshape(-1))        # [0 1 2 ... 11]
print(g.flatten())          # same values, but flatten() makes a COPY
# reshape(-1) gives a VIEW; flatten() gives a COPY. Topic 6's rule again.
```
</details>

## ❓ MCQs

**Q1.** Can you reshape 12 values into a 5×3 grid?
- (a) Yes  (b) No — 5×3 is 15, and you have 12  (c) Only with `-1`  (d) Only for floats

**Q2.** What does `a.reshape(3, -1)` mean?
- (a) 3 rows and −1 columns  (b) 3 rows, and NumPy works out the columns  (c) An error  (d) Remove 1 row

**Q3.** What does `a.reshape(-1)` do?
- (a) Reverses it  (b) Flattens it to one dimension  (c) Deletes it  (d) Transposes it

**Q4.** Does `reshape` return a copy or a view?
- (a) A copy  (b) A view — changing it changes the original  (c) Neither  (d) It depends on the dtype

**Q5.** Why does `reshape(1, -1)` matter for Session 5?
- (a) It is faster  (b) Models expect a 2-D table of samples, even for one row  (c) It saves memory  (d) It does not

<details><summary>Answers</summary>

**A1 — (b) No.** The numbers must multiply to the size. Twelve chairs cannot make five rows of three.

**A2 — (b).** The `-1` means "work this one out for me".

**A3 — (b) Flattens it.** The standard idiom. Note `flatten()` does the same but returns a copy.

**A4 — (b) A view.** Topic 6's rule applies here too.

**A5 — (b).** Predicting for a single person needs a one-row table, not a bare list.
</details>

---

# 9. NumPy Array Iterating

**You *can* loop over an array. Usually you should not.**

```python
a = np.array([1, 2, 3])

for x in a:            # works
    print(x)
```

## But for 2-D, a plain loop gives you rows, not values

```python
b = np.array([[1, 2], [3, 4]])

for row in b:
    print(row)         # [1 2] then [3 4]   -> ROWS, not numbers
```

**To reach the individual values you need a nested loop, or `np.nditer`:**

```python
for row in b:
    for value in row:
        print(value)         # 1 2 3 4

for value in np.nditer(b):   # the NumPy way, any number of dimensions
    print(value)             # 1 2 3 4
```

## `np.ndenumerate` — when you need the position too

```python
for index, value in np.ndenumerate(b):
    print(index, value)      # (0, 0) 1 ... (1, 1) 4
```

**This is the array version of `enumerate()` from Session 1.**

## ⚠️ The important part: prefer vectorisation

```python
a = np.array([1, 2, 3, 4, 5])

doubled = []                 # the SLOW way
for x in a:
    doubled.append(x * 2)

doubled = a * 2              # the NumPy way - faster and shorter
```

> **If you find yourself writing a loop over a NumPy array, stop and ask whether an operation on the whole array would do it.** Nine times out of ten it will, and it will be far faster. **That is the entire point of the library.**

| Instead of looping to… | Write |
|---|---|
| Double every value | `a * 2` |
| Add two arrays | `a + b` |
| Count values above 40 | `(a > 40).sum()` |
| Keep values above 40 | `a[a > 40]` |
| Replace values below 0 | `a[a < 0] = 0` |
| Total, average, largest | `a.sum()`, `a.mean()`, `a.max()` |

## 📘 Examples

**Example 1 — looping in one dimension**

```python
import numpy as np

marks = np.array([78, 92, 45])
for m in marks:
    print(f"Mark: {m}")
```

**Example 2 — 2-D gives you rows**

```python
grid = np.array([[1, 2, 3], [4, 5, 6]])

for row in grid:
    print("row:", row)          # [1 2 3] then [4 5 6]
```

**Example 3 — reaching every value**

```python
grid = np.array([[1, 2], [3, 4]])

for v in np.nditer(grid):
    print(v, end=" ")           # 1 2 3 4
print()

for idx, v in np.ndenumerate(grid):
    print(idx, v)               # (0, 0) 1 ...
```

**Example 4 — the same job, two ways**

```python
marks = np.array([78, 92, 45, 88, 61])

passed = 0                      # loop version
for m in marks:
    if m >= 40:
        passed += 1

passed = (marks >= 40).sum()    # NumPy version - one line, much faster
print(passed)                   # 5
```

## 🌍 Scenarios

**Scenario 1 — printing a formatted report (a fair use of a loop)**

```python
names = np.array(["Arun", "Priya", "Ravi"])
marks = np.array([78, 92, 45])

for name, mark in zip(names, marks):
    status = "PASS" if mark >= 40 else "FAIL"
    print(f"{name:<8}{mark:>4}  {status}")
```

**Printing is a genuine reason to loop** — you need one line per student. Calculating is not.

**Scenario 2 — the loop you should not write**

```python
prices = np.array([120, 85, 260, 45])

with_tax = []                          # DON'T
for p in prices:
    with_tax.append(p * 1.18)

with_tax = prices * 1.18               # DO
print(np.round(with_tax, 2))
```

**Scenario 3 — walking a grid with positions**

```python
sales = np.array([[120, 135], [200, 180], [90, 110]])

for (shop, month), value in np.ndenumerate(sales):
    if value < 100:
        print(f"Shop {shop + 1}, month {month + 1}: only {value} - investigate")
```

**Finding *where* something is genuinely needs the index**, and `ndenumerate` gives it to you.

## ✏️ Tasks

1. Loop over `[10, 20, 30]` and print each value doubled. Then do it without a loop.
2. Loop over a 2×3 grid and print each row, then each individual value.
3. Use `np.ndenumerate` to print the position and value of every element in a 3×3 grid.
4. Count how many marks in `[45, 78, 92, 33, 61]` are above 60 — first with a loop, then without.
5. Print a formatted pass/fail table for five students. Explain why a loop is right here.

<details><summary>Solutions</summary>

```python
import numpy as np

a = np.array([10, 20, 30])                                             # 1
for x in a:
    print(x * 2, end=" ")
print()
print(a * 2)                     # the NumPy way - shorter and far faster

grid = np.array([[1, 2, 3], [4, 5, 6]])                                # 2
for row in grid:
    print("row:", row)
for v in np.nditer(grid):
    print(v, end=" ")
print()

g = np.arange(1, 10).reshape(3, 3)                                     # 3
for idx, v in np.ndenumerate(g):
    print(idx, v)

marks = np.array([45, 78, 92, 33, 61])                                 # 4
count = 0
for m in marks:
    if m > 60:
        count += 1
print(count)                     # 3
print((marks > 60).sum())        # 3 - one line, no loop

names = np.array(["Arun", "Priya", "Ravi", "Meera", "Sam"])            # 5
scores = np.array([78, 92, 45, 33, 61])
for n, s in zip(names, scores):
    print(f"{n:<8}{s:>4}  {'PASS' if s >= 40 else 'FAIL'}")
# A loop is RIGHT here because you need one PRINTED LINE per student.
# It would be wrong for calculating - use vectorised operations for that.
```
</details>

## ❓ MCQs

**Q1.** Looping over a 2-D array with a plain `for` gives you…
- (a) Individual values  (b) Rows  (c) Columns  (d) An error

**Q2.** What does `np.nditer(b)` give you?
- (a) Rows  (b) Every individual value, whatever the dimensions  (c) The shape  (d) Indexes only

**Q3.** What does `np.ndenumerate` add over `np.nditer`?
- (a) Speed  (b) The position of each value as well as the value  (c) Sorting  (d) Nothing

**Q4.** Which is the better way to double every value?
- (a) A `for` loop with `append`  (b) `a * 2`  (c) `np.nditer`  (d) A while loop

**Q5.** When is a loop over an array genuinely appropriate?
- (a) Never  (b) When you need one printed line per element  (c) For arithmetic  (d) For counting

<details><summary>Answers</summary>

**A1 — (b) Rows.** You need a nested loop or `nditer` to reach the values.

**A2 — (b).** It works for any number of dimensions.

**A3 — (b) The position.** The array version of `enumerate()` from Session 1.

**A4 — (b) `a * 2`.** **If you are looping over an array to calculate, stop and look for the vectorised version.**

**A5 — (b) Printing.** Formatting one line per item is a real reason to loop; arithmetic is not.
</details>

---

## ⭐ Checkpoint Problem 2 — Multiplication grid

> **Uses only:** creating arrays, reshape, iterating, slicing. Topics 1–9.

**The problem.** Build a 10×10 multiplication table as a NumPy array — **without writing a nested loop to fill it** — then print it neatly, and print the diagonal (the square numbers).

**Hint:** `np.arange(1, 11).reshape(-1, 1)` gives you a column, and a column times a row gives you a grid.

<details><summary>Solution</summary>

```python
import numpy as np

rows = np.arange(1, 11).reshape(-1, 1)     # a COLUMN: shape (10, 1)
cols = np.arange(1, 11)                    # a ROW:    shape (10,)

table = rows * cols                        # (10,1) x (10,) -> (10,10)
print("shape:", table.shape)

# Print it with a header
print("    " + "".join(f"{c:>5}" for c in cols))
print("    " + "-" * 50)
for i, row in enumerate(table, start=1):
    print(f"{i:>3}|" + "".join(f"{v:>5}" for v in row))

print("\nDiagonal (the square numbers):", np.diag(table))
print("Row 7:", table[6])
print("Column 7:", table[:, 6])
```

**The whole table is built in one line: `rows * cols`.**

NumPy stretches the `(10, 1)` column and the `(10,)` row against each other to fill a `(10, 10)` grid. That stretching is called **broadcasting**, and it is why no loop is needed. The loop that remains is only for **printing** — exactly the distinction from Topic 9.
</details>

**Make it harder:**

1. Print only the even products, using boolean indexing.
2. Print the sum of each row and each column using `axis`.
3. Build a 12×12 table and print just the bottom-right 4×4 block with slicing.

---

# Part B — Pandas for Data Handling

# 10. Pandas Introduction

**NumPy handles grids of numbers. Pandas handles *tables* — with column names, mixed types, and missing values.** That is what real data looks like.

🧠 **Analogy: a spreadsheet you drive with code.** Pandas is Excel without the mouse: named columns, filters, sorting, grouping and charts — all reproducible, all on a million rows, all in a script someone else can re-run.

| | NumPy | Pandas |
|---|---|---|
| Holds | A grid of one type | A table of mixed types |
| Columns have names | ❌ | ✅ |
| Handles missing values | Awkwardly | ✅ Built in |
| Reads CSV / Excel | ❌ | ✅ |
| Best for | Maths on numbers | Real datasets |

> **Pandas is built on top of NumPy.** Every Pandas column is a NumPy array underneath, which is why everything you learned in Topics 1–9 still applies.

## The two objects

| Object | Is | Think of it as |
|---|---|---|
| **Series** | One column | A single labelled list |
| **DataFrame** | A whole table | A dictionary of Series |

## Installing and importing

```bash
pip install pandas
```

```python
import pandas as pd        # pd is the universal convention
print(pd.__version__)
```

## 📘 Examples

**Example 1 — a table in three lines**

```python
import pandas as pd

df = pd.DataFrame({
    "name": ["Arun", "Priya", "Ravi"],
    "marks": [78, 92, 45],
})
print(df)
```

```text
    name  marks
0   Arun     78
1  Priya     92
2   Ravi     45
```

**The `0 1 2` on the left is the *index*** — Pandas puts one on every table automatically.

**Example 2 — mixed types, which NumPy cannot do**

```python
df = pd.DataFrame({
    "name": ["Arun", "Priya"],      # text
    "marks": [78, 92],              # whole numbers
    "fee_paid": [True, False],      # booleans
    "cgpa": [8.7, 9.1],             # decimals
})
print(df.dtypes)
```

**Each column has its own type.** A NumPy array would force them all into one.

**Example 3 — named columns**

```python
df = pd.DataFrame({"name": ["Arun", "Priya"], "marks": [78, 92]})

print(df["marks"])              # by NAME, not by position
print(df["marks"].mean())       # 85.0
```

**No more remembering that column 3 is the marks.**

**Example 4 — Pandas is NumPy underneath**

```python
import numpy as np

df = pd.DataFrame({"marks": [78, 92, 45]})
print(type(df["marks"].values))       # <class 'numpy.ndarray'>
print(df["marks"].values * 2)         # [156 184  90]
```

## 🌍 Scenarios

**Scenario 1 — a class register**

```python
register = pd.DataFrame({
    "roll": [1, 2, 3, 4],
    "name": ["Arun", "Priya", "Ravi", "Meera"],
    "attendance": [18, 20, 12, 19],
})
register["percent"] = (register["attendance"] / 20 * 100).round(1)
print(register)
```

**Adding a calculated column is one line** — and it applies to every row at once, the vectorisation idea from Topic 1.

**Scenario 2 — a shop's stock list**

```python
stock = pd.DataFrame({
    "item": ["tea", "coffee", "samosa"],
    "price": [15, 25, 20],
    "quantity": [40, 25, 60],
})
stock["value"] = stock["price"] * stock["quantity"]
print(stock)
print(f"Total stock value: {stock['value'].sum()}")
```

**Scenario 3 — why not just use a dictionary?**

```python
# Plain Python
students = [{"name": "Arun", "marks": 78}, {"name": "Priya", "marks": 92}]
average = sum(s["marks"] for s in students) / len(students)

# Pandas
df = pd.DataFrame(students)
average = df["marks"].mean()
```

**Both work on three rows. On 100,000 rows, with filtering, grouping and sorting, the Pandas version stays one line each and the Python version becomes a program.**

## ✏️ Tasks

1. Build a DataFrame of four cities with their populations, and print it.
2. Build a DataFrame with a text, a numeric and a boolean column. Print `dtypes`.
3. Build a stock table and add a `value` column that is price × quantity.
4. Build a register of five students with attendance out of 25, and add a percentage column.
5. In one sentence each, give two things Pandas does that NumPy cannot.

<details><summary>Solutions</summary>

```python
import pandas as pd

cities = pd.DataFrame({                                                # 1
    "city": ["Kochi", "Delhi", "Mumbai", "Chennai"],
    "population": [677381, 16787941, 12442373, 4646732],
})
print(cities)

mixed = pd.DataFrame({                                                 # 2
    "name": ["Arun", "Priya"], "marks": [78, 92], "fee_paid": [True, False]})
print(mixed.dtypes)

stock = pd.DataFrame({"item": ["tea", "coffee"], "price": [15, 25],    # 3
                      "quantity": [40, 25]})
stock["value"] = stock["price"] * stock["quantity"]
print(stock)

reg = pd.DataFrame({                                                   # 4
    "name": ["Arun", "Priya", "Ravi", "Meera", "Sam"],
    "attendance": [22, 25, 14, 20, 18]})
reg["percent"] = (reg["attendance"] / 25 * 100).round(1)
print(reg)

# 5 - Pandas gives each column its own NAME and its own TYPE, so one table
#     can hold text, numbers and booleans together.
#     Pandas reads and writes CSV and Excel files directly, and handles
#     missing values as a first-class idea rather than an awkward NaN.
```
</details>

## ❓ MCQs

**Q1.** What is a Pandas DataFrame best described as?
- (a) A list  (b) A table with named columns and mixed types  (c) A NumPy array  (d) A dictionary

**Q2.** What is a Pandas Series?
- (a) A whole table  (b) One column  (c) A row  (d) An index

**Q3.** What library is Pandas built on top of?
- (a) Matplotlib  (b) NumPy  (c) scikit-learn  (d) None

**Q4.** What is the `0 1 2` down the left of a printed DataFrame?
- (a) The first column  (b) The index  (c) An error  (d) Row totals

**Q5.** Which can hold a text column and a numeric column together?
- (a) A NumPy array  (b) A Pandas DataFrame  (c) Both  (d) Neither

<details><summary>Answers</summary>

**A1 — (b).** Named columns, mixed types, built-in missing-value handling.

**A2 — (b) One column.** A DataFrame is essentially a dictionary of Series.

**A3 — (b) NumPy.** Which is why Topics 1–9 still apply.

**A4 — (b) The index.** Pandas adds one automatically.

**A5 — (b) A DataFrame.** A NumPy array would force everything into one type — the mixing rule from Topic 5.
</details>

---

# 11. Pandas Series

**A Series is one column: values plus an index.**

```python
import pandas as pd

s = pd.Series([10, 20, 30])
print(s)
```

```text
0    10
1    20
2    30
dtype: int64
```

## Giving it your own index

```python
s = pd.Series([78, 92, 45], index=["Arun", "Priya", "Ravi"], name="marks")
print(s["Priya"])         # 92   - look up by LABEL
print(s.iloc[1])          # 92   - look up by POSITION
```

> **`.loc` uses labels, `.iloc` uses positions.** Remember it as **i**loc = **i**nteger position. This distinction runs through all of Pandas.

## From a dictionary

```python
s = pd.Series({"maths": 92, "physics": 78, "chemistry": 85})
print(s)
```

**The keys become the index automatically.**

## What a Series can do

```python
marks = pd.Series([78, 92, 45, 88, 61])

marks.mean()          # 72.8
marks.max()           # 92
marks.sum()           # 364
marks.describe()      # count, mean, std, min, quartiles, max — all at once
marks.sort_values()   # sorted
marks[marks > 70]     # boolean filter, exactly like NumPy in Topic 3
marks * 2             # vectorised, exactly like NumPy in Topic 1
```

## 📘 Examples

**Example 1 — a plain Series**

```python
import pandas as pd

s = pd.Series([10, 20, 30, 40])
print(s)
print(s[0], s.sum(), s.mean())
```

**Example 2 — a labelled Series**

```python
marks = pd.Series([78, 92, 45], index=["Arun", "Priya", "Ravi"])
print(marks["Priya"])          # 92
print(marks.idxmax())          # Priya   - WHO scored highest
print(marks.idxmin())          # Ravi
```

**`idxmax()` gives you the label, not the position** — which is usually what you actually wanted.

**Example 3 — from a dictionary**

```python
subjects = pd.Series({"maths": 92, "physics": 78, "chemistry": 85})
print(subjects)
print(f"best subject: {subjects.idxmax()} ({subjects.max()})")
```

**Example 4 — filtering and describing**

```python
marks = pd.Series([78, 92, 45, 88, 61, 33])

print(marks[marks >= 40])         # only the passes
print(f"passed: {(marks >= 40).sum()} of {marks.size}")
print(marks.describe())
```

## 🌍 Scenarios

**Scenario 1 — monthly sales, labelled by month**

```python
sales = pd.Series([120, 135, 150, 90, 200],
                  index=["Jan", "Feb", "Mar", "Apr", "May"])

print(f"best month : {sales.idxmax()} ({sales.max()})")
print(f"worst month: {sales.idxmin()} ({sales.min()})")
print(f"total      : {sales.sum()}")
print(f"months above average: {list(sales[sales > sales.mean()].index)}")
```

**Scenario 2 — counting categories**

```python
grades = pd.Series(["A", "B", "A", "C", "B", "A", "F"])
print(grades.value_counts())
```

```text
A    3
B    2
C    1
F    1
```

**`value_counts()` is one of the most useful methods in Pandas.** It answers "how many of each?" in one call.

**Scenario 3 — a temperature log with dates**

```python
temps = pd.Series([28, 31, 35, 29, 42],
                  index=pd.date_range("2026-03-01", periods=5))
print(temps)
print(f"hottest: {temps.idxmax().date()} at {temps.max()}C")
```

## ✏️ Tasks

1. Create a Series of five daily sales figures and print the total and average.
2. Create a Series of marks indexed by student name. Print who scored highest.
3. Create a Series from a dictionary of three subjects and their marks. Print the weakest subject.
4. From a Series of grades `["A","B","A","C","B","A"]`, print how many of each.
5. From a Series of monthly sales, print only the months that beat the average.

<details><summary>Solutions</summary>

```python
import pandas as pd

sales = pd.Series([450, 380, 620, 510, 295])                           # 1
print(f"total {sales.sum()}, average {sales.mean():.2f}")

marks = pd.Series([78, 92, 45], index=["Arun", "Priya", "Ravi"])       # 2
print(f"highest: {marks.idxmax()} with {marks.max()}")
# idxmax() gives the LABEL. max() gives the value.

subjects = pd.Series({"maths": 92, "physics": 78, "chemistry": 85})    # 3
print(f"weakest: {subjects.idxmin()} ({subjects.min()})")

grades = pd.Series(["A", "B", "A", "C", "B", "A"])                     # 4
print(grades.value_counts())

monthly = pd.Series([120, 135, 150, 90, 200],                          # 5
                    index=["Jan", "Feb", "Mar", "Apr", "May"])
print(monthly[monthly > monthly.mean()])
```
</details>

## ❓ MCQs

**Q1.** A Pandas Series is…
- (a) A whole table  (b) One column, with an index  (c) A row  (d) A dictionary

**Q2.** What is the difference between `.loc` and `.iloc`?
- (a) None  (b) `.loc` uses labels, `.iloc` uses integer positions  (c) `.iloc` uses labels  (d) `.loc` is faster

**Q3.** What does `marks.idxmax()` return?
- (a) The highest mark  (b) The label of the highest mark  (c) The position  (d) The average

**Q4.** What does `value_counts()` do?
- (a) Sums the values  (b) Counts how many times each value appears  (c) Counts the rows  (d) Counts nulls

**Q5.** Creating a Series from a dictionary makes the keys…
- (a) The values  (b) The index  (c) The name  (d) Ignored

<details><summary>Answers</summary>

**A1 — (b).** A DataFrame is a collection of Series sharing one index.

**A2 — (b).** **i**loc = **i**nteger position. This distinction runs through all of Pandas.

**A3 — (b) The label** — usually what you actually wanted. `.max()` gives the value.

**A4 — (b).** One of the most useful methods in Pandas.

**A5 — (b) The index.**
</details>

---

# 12. Pandas DataFrames

**A DataFrame is the whole table.** Think of it as a dictionary whose values are Series, all sharing one index.

## Creating one

```python
import pandas as pd

# From a dictionary of lists - the usual way, one list per COLUMN
df = pd.DataFrame({
    "name": ["Arun", "Priya", "Ravi"],
    "marks": [78, 92, 45],
    "city": ["Kochi", "Delhi", "Kochi"],
})
print(df)

# From a list of dictionaries - one dict per ROW
from_rows = pd.DataFrame([
    {"name": "Arun", "marks": 78},
    {"name": "Priya", "marks": 92},
])
print(from_rows)
```

**Both give you a table.** Use the first when your data is column-shaped (as a CSV is), and the second when it arrives one record at a time (as an API gives you).

## Selecting

```python
df["marks"]                 # one column  -> a Series
df[["name", "marks"]]       # several columns -> a DataFrame (note DOUBLE brackets)

df.loc[0]                   # row by LABEL
df.iloc[1]                  # row by POSITION
df.loc[0, "name"]           # one cell
df.iloc[0:2]                # first two rows
```

> ⚠️ **`df["marks"]` gives a Series; `df[["marks"]]` gives a one-column DataFrame.** The double brackets are a *list* of column names.

## Filtering

```python
df[df["marks"] > 70]                             # one condition
df[(df["marks"] > 70) & (df["city"] == "Kochi")] # AND
df[(df["marks"] > 90) | (df["marks"] < 50)]      # OR
```

> ⚠️ **Use `&` and `|`, not `and` and `or`** — and **wrap each condition in brackets**. Python's `and` works on single true/false values; a whole column needs the element-wise `&`. Forgetting the brackets gives a confusing error because `&` binds tighter than `>`.

## Adding, changing, removing columns

```python
# illustrative: a syntax reference. Each line acts on a `df` of your own.
df["passed"] = df["marks"] >= 40             # a new column from a condition
df["marks"] = df["marks"] + 5                # moderate everyone
df = df.drop(columns=["city"])               # remove a column
df = df.rename(columns={"marks": "score"})   # rename one
```

## Sorting and grouping

```python
df.sort_values("marks")                        # low to high
df.sort_values("marks", ascending=False)       # high to low
df.groupby("city")["marks"].mean()             # average per city
```

> **`groupby` is the single most useful skill in Pandas.** Read it as: *"split the table by city, then take the mean of marks in each part."*

## 📘 Examples

**Example 1 — build and inspect**

```python
import pandas as pd

df = pd.DataFrame({
    "name": ["Arun", "Priya", "Ravi", "Meera"],
    "marks": [78, 92, 45, 88],
    "city": ["Kochi", "Delhi", "Kochi", "Delhi"],
})
print(df)
print(df.shape)              # (4, 3)
print(df.columns.tolist())   # ['name', 'marks', 'city']
```

**Example 2 — selecting**

```python
print(df["marks"])                # a Series
print(df[["name", "marks"]])      # a DataFrame - double brackets
print(df.loc[0])                  # the first row
print(df.loc[0, "name"])          # Arun - one cell
```

**Example 3 — filtering**

```python
print(df[df["marks"] > 70])
print(df[(df["marks"] > 70) & (df["city"] == "Kochi")])
```

**Example 4 — a new column, sorting, grouping**

```python
df["grade"] = ["B", "A", "F", "B"]
print(df.sort_values("marks", ascending=False))
print(df.groupby("city")["marks"].mean())
```

```text
city
Delhi    90.0
Kochi    61.5
```

## 🌍 Scenarios

**Scenario 1 — a shop's sales by branch**

```python
sales = pd.DataFrame({
    "branch": ["North", "South", "North", "South", "East"],
    "month": ["Jan", "Jan", "Feb", "Feb", "Jan"],
    "amount": [12000, 15000, 13500, 14200, 9800],
})

print(sales.groupby("branch")["amount"].sum().sort_values(ascending=False))
print(f"\nBest single month: {sales.loc[sales['amount'].idxmax(), 'branch']}")
```

**Scenario 2 — finding students who need support**

```python
students = pd.DataFrame({
    "name": ["Arun", "Priya", "Ravi", "Meera", "Sam"],
    "marks": [78, 92, 33, 88, 38],
    "attendance": [80, 95, 55, 90, 72],
})

at_risk = students[(students["marks"] < 40) | (students["attendance"] < 60)]
print(at_risk[["name", "marks", "attendance"]])
```

**Scenario 3 — a summary table for a report**

```python
summary = students.agg({"marks": ["mean", "min", "max"],
                        "attendance": ["mean", "min", "max"]}).round(2)
print(summary)
```

**`agg` asks several questions of several columns at once** — exactly what a report needs.

## ✏️ Tasks

1. Build a DataFrame of five students with name, marks and city. Print its shape and columns.
2. Select just the name and marks columns, then just the third row.
3. Filter for students scoring above 60 **and** living in a particular city.
4. Add a `passed` column, then sort the table by marks, highest first.
5. Group by city and print the average marks and the number of students in each.

<details><summary>Solutions</summary>

```python
import pandas as pd

df = pd.DataFrame({                                                    # 1
    "name": ["Arun", "Priya", "Ravi", "Meera", "Sam"],
    "marks": [78, 92, 45, 88, 61],
    "city": ["Kochi", "Delhi", "Kochi", "Delhi", "Kochi"],
})
print(df.shape, df.columns.tolist())

print(df[["name", "marks"]])                                           # 2
print(df.iloc[2])

print(df[(df["marks"] > 60) & (df["city"] == "Kochi")])                # 3
# & not `and`, and EACH condition in its own brackets.

df["passed"] = df["marks"] >= 40                                       # 4
print(df.sort_values("marks", ascending=False))

print(df.groupby("city")["marks"].agg(["mean", "count"]).round(2))     # 5
# groupby = "split the table by city, then summarise each part".
```
</details>

## ❓ MCQs

**Q1.** What does `df["marks"]` return?
- (a) A DataFrame  (b) A Series  (c) A list  (d) An array

**Q2.** What does `df[["name", "marks"]]` return?
- (a) A Series  (b) A DataFrame with two columns  (c) An error  (d) A list

**Q3.** Which combines two filter conditions correctly?
- (a) `df[a > 1 and b < 2]`  (b) `df[(a > 1) & (b < 2)]`  (c) `df[a > 1 && b < 2]`  (d) `df[a > 1, b < 2]`

**Q4.** What does `df.groupby("city")["marks"].mean()` do?
- (a) Sorts by city  (b) Splits the table by city and averages marks in each part  (c) Filters by city  (d) Counts cities

**Q5.** `.loc` versus `.iloc` on a DataFrame:
- (a) Identical  (b) `.loc` uses labels, `.iloc` uses integer positions  (c) `.loc` is for columns only  (d) `.iloc` is deprecated

<details><summary>Answers</summary>

**A1 — (b) A Series.** One column.

**A2 — (b) A DataFrame.** The inner brackets are a **list** of column names.

**A3 — (b).** **`&` not `and`, and each condition in brackets** — `&` binds tighter than `>`.

**A4 — (b).** Split, then summarise each part. The most useful skill in Pandas.

**A5 — (b).** The same rule as for a Series in Topic 11.
</details>

---

## ⭐ Checkpoint Problem 3 — Report card builder

> **Uses only:** Series, DataFrames, filtering, grouping. Topics 10–12. **No CSV yet.**

**The problem.** Build a DataFrame of six students with marks in three subjects. Add a total, an average and a grade column, then print a class summary: the topper, the class average, how many passed, and the average per subject.

<details><summary>Solution</summary>

```python
import pandas as pd

df = pd.DataFrame({
    "name":      ["Arun", "Priya", "Ravi", "Meera", "Sam", "Divya"],
    "maths":     [78, 92, 33, 88, 61, 45],
    "physics":   [82, 88, 41, 79, 55, 60],
    "chemistry": [75, 95, 38, 91, 68, 52],
})

subjects = ["maths", "physics", "chemistry"]

df["total"] = df[subjects].sum(axis=1)              # axis=1 = across the row
df["average"] = df[subjects].mean(axis=1).round(2)

def grade_for(avg):
    if avg >= 90: return "A"
    if avg >= 75: return "B"
    if avg >= 60: return "C"
    if avg >= 40: return "D"
    return "F"

df["grade"] = df["average"].apply(grade_for)        # apply runs it per row

print(df.to_string(index=False))

print("\n--- CLASS SUMMARY ---")
print(f"Topper        : {df.loc[df['average'].idxmax(), 'name']} "
      f"({df['average'].max()})")
print(f"Class average : {df['average'].mean():.2f}")
print(f"Passed        : {(df['average'] >= 40).sum()} of {len(df)}")
print("\nAverage per subject:")
print(df[subjects].mean().round(2).to_string())
print("\nGrade distribution:")
print(df["grade"].value_counts().to_string())
```

**Three things doing the work:**

- **`axis=1`** sums *across* each row — the same axis idea as Topic 4.
- **`.apply(grade_for)`** runs your Session 1 function on every value. It is the escape hatch for when there is no built-in Pandas method.
- **`.idxmax()`** returns the *index label* of the top average, which `.loc` then turns into a name.
</details>

**Make it harder:**

1. Add a `rank` column using `df["average"].rank(ascending=False)`.
2. Print only the students who failed any single subject.
3. Add a `city` column and print the average per city with `groupby`.

---

# 13. Pandas Read CSV

**Almost all real data arrives as a CSV file.** One function reads it.

```python
# illustrative: a syntax reference, not runnable as written.
import pandas as pd

df = pd.read_csv("data.csv")                    # a local file
df = pd.read_csv("https://example.com/d.csv")   # straight from a URL
```

**Reading from a URL means nothing to download** — which is why every example in this course does it that way. Your code then runs identically on your laptop and in Colab.

## The first four things to do, every time

```python
df.head()        # the first 5 rows  - does it look right?
df.shape         # (rows, columns)   - did it all load?
df.info()        # types and missing values
df.describe()    # numeric summaries
```

> **Run these four before anything else, on every dataset, for the rest of your career.** They take ten seconds and catch most problems immediately.

## Useful options

```python
# illustrative: a syntax reference, not runnable as written.
pd.read_csv(path, nrows=1000)              # just the first 1000 rows - fast peek
pd.read_csv(path, usecols=["age", "income"])  # only the columns you need
pd.read_csv(path, na_values=["?", "N/A", "-"])  # treat these as missing
pd.read_csv(path, sep=";")                 # for semicolon-separated files
pd.read_csv(path, index_col=0)             # use the first column as the index
```

## Saving

```python
df.to_csv("cleaned.csv", index=False)      # index=False stops an extra column
```

> ⚠️ **Always pass `index=False` when saving**, unless your index is meaningful. Otherwise every save adds a nameless `Unnamed: 0` column, and after three rounds you have three of them.

## 📘 Examples

**Example 1 — load and look**

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

df = pd.read_csv(BASE + "loan_data_10k.csv")
print(df.shape)          # (10000, 14)
print(df.head())
```

**Example 2 — what `info()` tells you**

```python
df.info()
# For each column: the name, how many NON-NULL values, and the dtype.
# If a column shows fewer non-nulls than the row count, it has gaps.
# If a number column shows dtype 'object', it has text in it somewhere.
```

**Example 3 — reading only what you need**

```python
small = pd.read_csv(BASE + "loan_data_10k.csv",
                    usecols=["person_age", "person_income", "loan_status"],
                    nrows=100)
print(small.shape)       # (100, 3)
```

**On a large file this is the difference between waiting and not waiting.**

**Example 4 — a quick peek before committing**

```python
peek = pd.read_csv(BASE + "loan_data_10k.csv", nrows=5)
print(peek.columns.tolist())
# Now you know the column names WITHOUT loading ten thousand rows.
```

## 🌍 Scenarios

**Scenario 1 — the standard first look**

```python
df = pd.read_csv(BASE + "loan_data_10k.csv")

print("SHAPE:", df.shape)
print("\nCOLUMNS:", df.columns.tolist())
print("\nMISSING VALUES:")
print(df.isna().sum()[df.isna().sum() > 0])
print("\nDUPLICATES:", df.duplicated().sum())
```

**Four questions, ten seconds, every new dataset.**

**Scenario 2 — a file that uses `?` for missing**

```python
adult = pd.read_csv(BASE + "classification/adult.csv", na_values=["?"])
print(adult.isna().sum()[adult.isna().sum() > 0])
```

**Without `na_values`, those `?` marks stay as text** and quietly turn a numeric column into an object column.

**Scenario 3 — saving your cleaned work**

```python
clean = df[df["person_age"] < 100]         # drop the impossible ages
clean.to_csv("loans_clean.csv", index=False)
print(f"saved {len(clean)} of {len(df)} rows")
```

## ✏️ Tasks

1. Load the loan dataset from the URL and print its shape and first five rows.
2. Run `info()` on it. Which columns have missing values, and which are text?
3. Load only three columns and only the first 200 rows.
4. Load `classification/adult.csv` treating `?` as missing, and count the gaps.
5. Filter a dataset, save it with `index=False`, reload it, and confirm the shape.

<details><summary>Solutions</summary>

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

df = pd.read_csv(BASE + "loan_data_10k.csv")                           # 1
print(df.shape)                    # (10000, 14)
print(df.head())

df.info()                                                              # 2
print("\nmissing:\n", df.isna().sum()[df.isna().sum() > 0])
print("\ntext columns:", df.select_dtypes("object").columns.tolist())

small = pd.read_csv(BASE + "loan_data_10k.csv",                        # 3
                    usecols=["person_age", "person_income", "loan_status"],
                    nrows=200)
print(small.shape)                 # (200, 3)

adult = pd.read_csv(BASE + "classification/adult.csv", na_values=["?"])  # 4
print(adult.isna().sum()[adult.isna().sum() > 0])
# Without na_values those "?" stay as TEXT and quietly poison the column.

clean = df[df["person_age"] < 100]                                     # 5
clean.to_csv("loans_clean.csv", index=False)
print(pd.read_csv("loans_clean.csv").shape)
# index=False stops Pandas adding an "Unnamed: 0" column on every save.
```
</details>

## ❓ MCQs

**Q1.** Which function reads a CSV file?
- (a) `pd.open_csv()`  (b) `pd.read_csv()`  (c) `pd.load_csv()`  (d) `pd.csv()`

**Q2.** What are the four things to run on every new dataset?
- (a) `head`, `shape`, `info`, `describe`  (b) `plot`, `corr`, `sort`, `group`  (c) `sum`, `mean`, `max`, `min`  (d) `dropna`, `fillna`, `drop`, `rename`

**Q3.** Why pass `index=False` to `to_csv()`?
- (a) It is faster  (b) Otherwise every save adds an extra unnamed index column  (c) It is required  (d) To sort the file

**Q4.** A file uses `?` for missing values. What should you do?
- (a) Nothing  (b) Pass `na_values=["?"]` to `read_csv`  (c) Delete those rows by hand  (d) Use `dropna()`

**Q5.** In `info()`, a numeric-looking column shows dtype `object`. This means…
- (a) It is fine  (b) There is text somewhere in it  (c) It is empty  (d) It is the index

<details><summary>Answers</summary>

**A1 — (b) `pd.read_csv()`.** It reads local paths and URLs alike.

**A2 — (a).** Ten seconds, and they catch most problems immediately.

**A3 — (b).** After three saves you would have three `Unnamed` columns.

**A4 — (b).** Otherwise the `?` stays as text and turns the column into `object`.

**A5 — (b).** Something non-numeric is hiding in it — often exactly those `?` marks.
</details>

---

# 14. Pandas Analyzing Data

**You have loaded the file. Now: what is actually in it?**

🧠 **Analogy: meeting someone new.** You do not start with their deepest secrets. You start with the basics — name, where they are from, what they do — and build from there. **`head`, `info` and `describe` are those first questions.**

| Method | Answers |
|---|---|
| `df.head(n)` / `df.tail(n)` | What do the first/last rows look like? |
| `df.shape` | How big is it? |
| `df.info()` | What types, and where are the gaps? |
| `df.describe()` | What is the numeric spread? |
| `df.isna().sum()` | How many missing per column? |
| `df["col"].value_counts()` | How many of each category? |
| `df["col"].unique()` / `.nunique()` | What values exist / how many distinct? |
| `df.duplicated().sum()` | Any repeated rows? |

## Reading `describe()`

```text
       person_age  person_income
count    10000.00       10000.00     <- how many non-missing
mean        27.70       72289.64     <- the average
std          6.04       58462.43     <- the spread
min         20.00        8000.00
25%         24.00       41621.75     <- a quarter are below this
50%         26.00       60954.00     <- the MEDIAN
75%         30.00       87421.50
max        144.00     2448661.00     <- look at this
```

**Two things jump out of that table, and both matter.**

1. **`person_age` max is 144.** Nobody is 144. That is a data error, and you now know to look for it — Topic 20 deals with it.
2. **`person_income` mean is 72,290 but the median is 60,954.** The mean is dragged well above the middle by a few very large incomes, up to 2.4 million.

> **When the mean is much higher than the median, the data is skewed by large values.** Compare them on every numeric column — it is the fastest outlier detector you have.

## 📘 Examples

**Example 1 — the first look**

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

df = pd.read_csv(BASE + "loan_data_10k.csv")
print(df.shape)
print(df.head(3))
print(df.dtypes)
```

**Example 2 — numeric summaries**

```python
print(df[["person_age", "person_income", "credit_score"]].describe().round(2))
```

**Example 3 — categories**

```python
print(df["loan_intent"].value_counts())
print(f"\ndistinct intents: {df['loan_intent'].nunique()}")
print(f"target balance:\n{df['loan_status'].value_counts(normalize=True).round(3)}")
```

**`value_counts(normalize=True)` gives proportions instead of counts** — which is how you check whether your target is balanced. **Do this on every classification problem** (Session 5 explains why).

**Example 4 — mean versus median, the skew check**

```python
for col in ["person_age", "person_income", "credit_score"]:
    mean, median = df[col].mean(), df[col].median()
    flag = "  <- SKEWED" if mean > median * 1.15 else ""
    print(f"{col:<16} mean {mean:>12,.1f}   median {median:>10,.1f}{flag}")
```

## 🌍 Scenarios

**Scenario 1 — the ten-second dataset report**

```python
def quick_report(df):
    print(f"Rows: {len(df):,}   Columns: {len(df.columns)}")
    print(f"Duplicates: {df.duplicated().sum()}")
    missing = df.isna().sum()
    print(f"Columns with gaps: {(missing > 0).sum()}")
    if (missing > 0).any():
        print(missing[missing > 0].to_string())
    print(f"Numeric: {len(df.select_dtypes('number').columns)}  "
          f"Text: {len(df.select_dtypes('object').columns)}")

quick_report(df)
```

**Write this once and reuse it on every dataset for the rest of the course.**

**Scenario 2 — grouping to answer a real question**

```python
print(df.groupby("loan_intent")["loan_amnt"].agg(["count", "mean"]).round(0)
        .sort_values("mean", ascending=False))
```

*"Which kind of loan is largest on average?"* — one line.

**Scenario 3 — spotting the impossible value**

```python
print("age range:", df["person_age"].min(), "to", df["person_age"].max())
print("rows over 100:", (df["person_age"] > 100).sum())
print(df[df["person_age"] > 100][["person_age", "person_income"]])
```

**One row claims an age of 144.** You found it in three lines — and Topic 20 covers what to do about it.

## ✏️ Tasks

1. Load the loan data and print shape, dtypes, and the first three rows.
2. Run `describe()` and identify one column with a suspicious maximum.
3. Print the value counts of `loan_intent`, and the proportion for `loan_status`.
4. For every numeric column, print mean and median and flag any where the mean is much higher.
5. Group by `person_education` and print the average income for each level, sorted.

<details><summary>Solutions</summary>

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "loan_data_10k.csv")

print(df.shape); print(df.dtypes); print(df.head(3))                   # 1

print(df.describe().round(2).to_string())                              # 2
print("\nperson_age max is 144 - nobody is 144. That is a DATA ERROR.")

print(df["loan_intent"].value_counts())                                # 3
print(df["loan_status"].value_counts(normalize=True).round(3))
# normalize=True gives PROPORTIONS - this is how you check target balance.

for col in df.select_dtypes("number").columns:                         # 4
    mean, median = df[col].mean(), df[col].median()
    flag = "  <- SKEWED" if median and mean > median * 1.15 else ""
    print(f"{col:<28} mean {mean:>12,.1f}  median {median:>10,.1f}{flag}")

print(df.groupby("person_education")["person_income"]                  # 5
        .mean().round(0).sort_values(ascending=False))
```
</details>

## ❓ MCQs

**Q1.** Which method shows the dtypes and the count of non-null values?
- (a) `describe()`  (b) `info()`  (c) `head()`  (d) `shape`

**Q2.** In `describe()`, what does the `50%` row show?
- (a) The mean  (b) The median  (c) Half the rows  (d) The standard deviation

**Q3.** The mean income is 72,290 and the median is 60,954. This tells you…
- (a) Nothing  (b) The data is skewed by some very large values  (c) There is an error  (d) Half the rows are missing

**Q4.** What does `value_counts(normalize=True)` give you?
- (a) Counts  (b) Proportions  (c) Sorted values  (d) Unique values

**Q5.** `person_age` has a maximum of 144. You should…
- (a) Ignore it  (b) Treat it as a data error and investigate  (c) Delete the column  (d) Round it

<details><summary>Answers</summary>

**A1 — (b) `info()`.**

**A2 — (b) The median** — the middle value.

**A3 — (b) Skewed.** **Comparing mean and median is the fastest outlier detector you have.**

**A4 — (b) Proportions.** How you check whether a classification target is balanced.

**A5 — (b).** Topic 20 covers exactly this.
</details>

---

# 15. Pandas Correlations

**A correlation says how strongly two numeric columns move together.**

```python
df.corr(numeric_only=True)
```

| Value | Meaning |
|---|---|
| **+1.0** | Perfect: one goes up, the other goes up exactly in step |
| **+0.5** | Moderate positive |
| **0.0** | No linear relationship |
| **−0.5** | Moderate negative |
| **−1.0** | Perfect: one goes up, the other goes down |

**Rough reading guide:** below 0.3 is weak, 0.3–0.7 is moderate, above 0.7 is strong.

## ⚠️ Two warnings, and both matter

> **1. Correlation is not causation.** Ice-cream sales correlate with drownings. Ice cream does not cause drowning — hot weather causes both. **A correlation tells you two things move together. It never tells you why.**

> **2. Correlation only measures *straight-line* relationships.** Two columns can be perfectly related and still score near zero if the relationship is curved. **Always plot it as well** — which is Topic 16.

## 📘 Examples

**Example 1 — the correlation table**

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "loan_data_10k.csv")

print(df[["person_age", "person_income", "loan_amnt",
          "credit_score", "loan_status"]].corr().round(3))
```

**Example 2 — the useful view: what relates to my target?**

```python
target = df.corr(numeric_only=True)["loan_status"].sort_values(ascending=False)
print(target.round(3))
```

```text
loan_status                   1.000
loan_percent_income           0.405
loan_int_rate                 0.363
loan_amnt                     0.042
credit_score                 -0.002
person_income                -0.210
```

**Read that.** `loan_percent_income` at **0.405** is the strongest signal — the bigger the loan relative to income, the more likely this outcome. `credit_score` at **−0.002** is essentially unrelated on its own.

> **"Unrelated on its own" is not "useless".** A column can have near-zero correlation and still matter in combination with others — Session 6 measures exactly this. **Never drop a column on correlation alone.**

**Example 3 — a correlation that means nothing**

```python
import numpy as np
rng = np.random.default_rng(0)
noise = pd.DataFrame({"a": rng.normal(0, 1, 30), "b": rng.normal(0, 1, 30)})
print(f"correlation of two random columns: {noise['a'].corr(noise['b']):.3f}")
```

**Two completely unrelated columns rarely correlate at exactly zero.** On small samples you will see 0.2 or 0.3 by pure chance. **Be sceptical of a moderate correlation on a small dataset.**

**Example 4 — the curve that correlation misses**

```python
x = pd.Series(np.linspace(-5, 5, 100))
y = x ** 2                                  # a PERFECT relationship
print(f"correlation: {x.corr(y):.3f}")      # about 0.000
```

**`y` is completely determined by `x`, and the correlation is zero.** The relationship is a U-shape, and correlation only sees straight lines. **This is why you plot.**

## 🌍 Scenarios

**Scenario 1 — which advertising channel moves sales?**

```python
ads = pd.read_csv(BASE + "regression/advertising.csv")
print(ads.corr()["Sales"].sort_values(ascending=False).round(3))
```

**TV correlates most strongly with sales.** That is a starting point for Session 5, not a conclusion — it does not prove the adverts *caused* the sales.

**Scenario 2 — finding columns that duplicate each other**

```python
c = df.corr(numeric_only=True).abs()
pairs = c.unstack().sort_values(ascending=False)
pairs = pairs[(pairs < 1.0) & (pairs > 0.7)]
print(pairs.drop_duplicates())
```

**Two columns correlating above about 0.9 are telling you nearly the same thing.** You usually keep one.

**Scenario 3 — the honest write-up**

```text
loan_percent_income correlates 0.405 with loan_status - the strongest
single relationship in the dataset.

What this does NOT tell us:
  - whether a high loan-to-income ratio CAUSES the outcome
  - whether the relationship is a straight line (we must plot it)
  - whether it still holds once other columns are accounted for
```

**Writing down what a number does not tell you is a professional habit**, and it is what Session 8 is built on.

## ✏️ Tasks

1. Print the correlation table for four numeric columns of the loan dataset.
2. Print every column's correlation with `loan_status`, sorted. Which is strongest?
3. Generate two random columns of 30 values and check their correlation. Repeat a few times.
4. Create `x` and `y = x**2` and show that the correlation is near zero despite a perfect relationship.
5. Find any pair of columns correlating above 0.7 and say what you would do about it.

<details><summary>Solutions</summary>

```python
import pandas as pd, numpy as np
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "loan_data_10k.csv")

print(df[["person_age", "person_income", "loan_amnt", "credit_score"]]  # 1
        .corr().round(3))

print(df.corr(numeric_only=True)["loan_status"]                        # 2
        .sort_values(ascending=False).round(3))
# loan_percent_income at 0.405 is strongest.

for seed in range(4):                                                  # 3
    rng = np.random.default_rng(seed)
    a, b = pd.Series(rng.normal(0, 1, 30)), pd.Series(rng.normal(0, 1, 30))
    print(f"seed {seed}: {a.corr(b):+.3f}")
# Unrelated columns rarely land at exactly 0. Be sceptical of a moderate
# correlation on a small sample.

x = pd.Series(np.linspace(-5, 5, 100)); y = x ** 2                     # 4
print(f"correlation: {x.corr(y):.3f}")     # ~0.000, for a PERFECT relationship
# Correlation only sees STRAIGHT LINES. This is why you plot.

c = df.corr(numeric_only=True).abs().unstack().sort_values(ascending=False)  # 5
print(c[(c < 1.0) & (c > 0.7)].drop_duplicates())
# Two columns above ~0.9 say nearly the same thing - usually keep one.
```
</details>

## ❓ MCQs

**Q1.** A correlation of −0.8 means…
- (a) No relationship  (b) A strong relationship where one rises as the other falls  (c) An error  (d) 80% accuracy

**Q2.** Ice-cream sales correlate with drownings. This proves…
- (a) Ice cream causes drowning  (b) Nothing about cause — hot weather likely drives both  (c) Drowning causes ice-cream sales  (d) The data is wrong

**Q3.** `x` and `y = x**2` have a correlation near 0. Why?
- (a) They are unrelated  (b) Correlation only measures straight-line relationships  (c) A bug  (d) The sample is too small

**Q4.** `credit_score` correlates −0.002 with the target. You should…
- (a) Drop it immediately  (b) Keep it — a column can matter in combination with others  (c) Square it  (d) Ignore the target

**Q5.** Two columns correlate at 0.95. This usually means…
- (a) Both are essential  (b) They say nearly the same thing, so you usually keep one  (c) One is the target  (d) An error

<details><summary>Answers</summary>

**A1 — (b).** The sign gives the direction; the size gives the strength.

**A2 — (b).** **Correlation is not causation.** A correlation never tells you why.

**A3 — (b).** A U-shaped relationship is invisible to correlation. **Always plot as well.**

**A4 — (b).** **Never drop a column on correlation alone** — Session 6 measures exactly this.

**A5 — (b).** They are near-duplicates.
</details>

---

# 16. Pandas Plotting

**Pandas can plot straight from a DataFrame.** No separate library call needed for a quick look.

```python
# illustrative: a syntax reference, not runnable as written.
import matplotlib.pyplot as plt

df["column"].plot(kind="hist")
plt.show()
```

| `kind=` | Shows | Use it for |
|---|---|---|
| `"hist"` | The shape of one column | Is it skewed? Where is the bulk? |
| `"box"` | Median, quartiles, outliers | Spotting extreme values |
| `"scatter"` | Two columns against each other | Is there a relationship? |
| `"line"` | Values in order | Change over time |
| `"bar"` | One value per category | Comparing groups |
| `"pie"` | Proportions of a whole | Only with very few categories |

> **Plot before you model.** A chart shows you in one second what a table of numbers takes minutes to reveal — and it catches the curved relationships that Topic 15's correlations miss entirely.

## 📘 Examples

**Example 1 — a histogram: what shape is this column?**

```python
import pandas as pd
import matplotlib.pyplot as plt
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "loan_data_10k.csv")

df["person_income"].plot(kind="hist", bins=50, title="Income distribution")
plt.xlabel("income")
plt.show()
```

**A long tail to the right** — most people earn modestly and a few earn enormously. That is the skew you spotted numerically in Topic 14, now visible.

**Example 2 — a scatter: do two columns relate?**

```python
df.plot(kind="scatter", x="person_income", y="loan_amnt", alpha=0.2,
        title="Income vs loan amount")
plt.show()
```

> **`alpha=0.2` makes points semi-transparent.** With ten thousand points a solid scatter is a black blob; transparency shows you where they pile up.

**Example 3 — a bar chart of categories**

```python
df["loan_intent"].value_counts().plot(kind="bar", title="Why people borrow")
plt.ylabel("applications")
plt.tight_layout()
plt.show()
```

**Example 4 — a box plot to find outliers**

```python
df["person_age"].plot(kind="box", title="Age")
plt.show()
# The dots far above the box are outliers - including that age of 144.
```

## 🌍 Scenarios

**Scenario 1 — comparing groups**

```python
df.groupby("loan_intent")["loan_amnt"].mean().sort_values().plot(
    kind="barh", title="Average loan by purpose")
plt.xlabel("average amount")
plt.tight_layout()
plt.show()
```

**`groupby` then `.plot()` is a two-line answer to a real question.**

**Scenario 2 — several columns at once**

```python
df[["person_age", "credit_score"]].plot(
    kind="hist", bins=40, alpha=0.6, subplots=True, figsize=(7, 5))
plt.tight_layout()
plt.show()
```

**Scenario 3 — a trend over time**

```python
sales = pd.Series([120, 135, 150, 90, 200, 185],
                  index=["Jan", "Feb", "Mar", "Apr", "May", "Jun"])
sales.plot(kind="line", marker="o", title="Monthly sales")
plt.ylabel("sales")
plt.grid(alpha=0.3)
plt.show()
```

## ✏️ Tasks

1. Plot a histogram of `person_income` with 50 bins. Describe its shape in one sentence.
2. Plot `credit_score` against `loan_amnt` as a scatter with `alpha=0.2`.
3. Plot the value counts of `loan_intent` as a bar chart.
4. Plot a box plot of `person_age` and say what the dots above the box mean.
5. Group by `person_education`, take the mean income, and plot it as a horizontal bar chart.

<details><summary>Solutions</summary>

```python
import pandas as pd
import matplotlib.pyplot as plt
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
df = pd.read_csv(BASE + "loan_data_10k.csv")

df["person_income"].plot(kind="hist", bins=50, title="Income")         # 1
plt.xlabel("income"); plt.show()
# Strongly right-skewed: most incomes are modest, a few are enormous.

df.plot(kind="scatter", x="credit_score", y="loan_amnt", alpha=.2)     # 2
plt.show()

df["loan_intent"].value_counts().plot(kind="bar")                      # 3
plt.tight_layout(); plt.show()

df["person_age"].plot(kind="box", title="Age")                         # 4
plt.show()
# The dots above the box are OUTLIERS - values far from the bulk.
# One of them is the impossible age of 144.

df.groupby("person_education")["person_income"].mean().sort_values().plot(  # 5
    kind="barh", title="Average income by education")
plt.tight_layout(); plt.show()
```
</details>

## ❓ MCQs

**Q1.** Which plot shows the shape of a single numeric column?
- (a) scatter  (b) hist  (c) bar  (d) pie

**Q2.** Which plot shows the relationship between two numeric columns?
- (a) hist  (b) scatter  (c) bar  (d) box

**Q3.** Why use `alpha=0.2` on a scatter of 10,000 points?
- (a) It is faster  (b) Transparency reveals where points pile up instead of a solid blob  (c) It changes the colour  (d) It sorts them

**Q4.** What do the dots above a box plot's whisker represent?
- (a) The mean  (b) Outliers  (c) Missing values  (d) The median

**Q5.** Why plot as well as computing correlations?
- (a) Charts look nicer  (b) Correlation misses curved relationships that a plot shows instantly  (c) It is faster  (d) Correlations are unreliable

<details><summary>Answers</summary>

**A1 — (b) hist.**

**A2 — (b) scatter.**

**A3 — (b).** With ten thousand solid points you see a black blob and learn nothing.

**A4 — (b) Outliers** — including that age of 144.

**A5 — (b).** Topic 15's `y = x**2` example correlates at zero despite a perfect relationship.
</details>

---

## ⭐ Checkpoint Problem 4 — Sales dashboard

> **Uses only:** read_csv, analysing, correlations, plotting. Topics 13–16.

**The problem.** Load the advertising dataset and produce a one-screen summary: its shape, a numeric description, which channel correlates most with sales, and a three-panel chart showing each channel against sales.

<details><summary>Solution</summary>

```python
import pandas as pd
import matplotlib.pyplot as plt
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

ads = pd.read_csv(BASE + "regression/advertising.csv")

print("SHAPE:", ads.shape)
print("\nCOLUMNS:", ads.columns.tolist())
print("\nMISSING:", ads.isna().sum().sum())
print("\n", ads.describe().round(2).to_string())

print("\nCORRELATION WITH SALES:")
corr = ads.corr()["Sales"].drop("Sales").sort_values(ascending=False)
print(corr.round(3).to_string())
print(f"\nStrongest channel: {corr.idxmax()} at {corr.max():.3f}")

# One panel per channel, so you can SEE each relationship
channels = ["TV", "Radio", "Newspaper"]
fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, ch in zip(axes, channels):
    ax.scatter(ads[ch], ads["Sales"], alpha=.6, edgecolor="k", linewidth=.3)
    ax.set_title(f"{ch}  (r = {ads[ch].corr(ads['Sales']):.3f})")
    ax.set_xlabel(f"{ch} spend")
    ax.grid(alpha=.3)
axes[0].set_ylabel("Sales")
plt.tight_layout()
plt.show()
```

**What the three panels tell you that the correlation table cannot:**

- **TV** shows a clear upward band — but it *fans out*, so high spend gives less predictable returns.
- **Radio** shows a looser upward trend.
- **Newspaper** is close to a shapeless cloud, which matches its weak correlation.

**Seeing the fan shape in the TV panel is the whole reason to plot.** No single correlation number contains it.
</details>

**Make it harder:**

1. Add a `Total_Spend` column and plot it against Sales.
2. Print the correlation of `Total_Spend` with Sales. Is it stronger than TV alone?
3. Draw a heatmap of the correlation matrix with `plt.imshow(ads.corr())`.

---

# Part C — Pandas for Data Cleaning

# 17. Cleaning Data

**Real data is messy. Cleaning it is most of the job.**

🧠 **Analogy: cooking with vegetables straight from the field.** They arrive with soil on them, a few are bruised, one is the wrong vegetable entirely, and two are the same onion counted twice. **You wash, sort and discard before you cook. Nobody skips this step, and nobody puts it in the photograph.**

## The four kinds of mess

| Problem | Looks like | Topic |
|---|---|---|
| **Empty cells** | `NaN`, blanks, `None` | [18](#18-cleaning-empty-cells) |
| **Wrong format** | A date written as `20260307` | [19](#19-cleaning-wrong-format) |
| **Wrong data** | A workout of 450 minutes; an age of 144 | [20](#20-cleaning-wrong-data) |
| **Duplicates** | The same row entered twice | [21](#21-removing-duplicates) |

## The messy dataset we will fix

Everything in Topics 17–21 uses this small fitness log. **It is built in code rather than downloaded so you can see every defect at once:**

```python
import pandas as pd

def messy_log():
    """A deliberately messy fitness log - every defect is visible."""
    return pd.DataFrame({
        "Duration": [60, 60, 60, 45, 45, 60, 60, 450, 30, 60, 60, 60],
        "Date": ["2026/03/01", "2026/03/02", "2026/03/03", "2026/03/04",
                 "2026/03/05", "2026/03/06", "20260307", "2026/03/08",
                 None, "2026/03/10", "2026/03/11", "2026/03/11"],
        "Pulse": [110, 117, 103, 109, 117, 102, 110, 104, 109, 98, 103, 103],
        "Maxpulse": [130, 145, 135, 175, 148, 127, 136, 134, 133, 124, 147, 147],
        "Calories": [409.1, 479.0, 340.0, 282.4, 406.0, 300.0, 374.0, 253.3,
                     195.1, None, 329.3, 329.3],
    })

df = messy_log()
print(df)
```

**Four defects are hidden in there:**

| Row | Defect |
|---|---|
| 6 | `Date` is `20260307` — no slashes, wrong format |
| 7 | `Duration` is `450` minutes — seven and a half hours |
| 8, 9 | A missing `Date` and a missing `Calories` |
| 11 | An exact copy of row 10 |

## The rule that saves you

```python
# illustrative: a syntax reference, not runnable as written.
df = pd.read_csv("data.csv")
clean = df.copy()          # <- ALWAYS. Then clean `clean`, never `df`.
```

> **Keep the raw data untouched.** This is Topic 6's copy-versus-view lesson applied to your whole workflow: when a result looks wrong, you need something to compare against.

## 📘 Examples

**Example 1 — find the mess before fixing it**

```python
df = messy_log()

print("shape:", df.shape)
print("\nmissing values:")
print(df.isna().sum())
print("\nduplicate rows:", df.duplicated().sum())
print("\ndtypes:")
print(df.dtypes)
```

**`Date` has dtype `object`** — because it is text, not a date. That is your clue for Topic 19.

**Example 2 — the reusable inspection**

```python
def inspect(df, name="data"):
    print(f"--- {name}: {df.shape[0]} rows, {df.shape[1]} columns ---")
    missing = df.isna().sum()
    if missing.any():
        print("missing:", missing[missing > 0].to_dict())
    else:
        print("missing: none")
    print("duplicates:", df.duplicated().sum())

inspect(df, "raw fitness log")
```

**Example 3 — always work on a copy**

```python
raw = messy_log()
clean = raw.copy()

clean = clean.drop_duplicates()
print("raw   :", raw.shape)        # (12, 5)  - untouched
print("clean :", clean.shape)      # (11, 5)
```

**Example 4 — cleaning is a sequence, not one call**

```python
clean = (messy_log()
         .drop_duplicates()                                   # Topic 21
         .assign(Date=lambda d: pd.to_datetime(d["Date"],
                                               format="mixed", errors="coerce"))
         .dropna())                                           # Topic 18
print(clean.shape)
```

**Chaining reads top to bottom as a recipe.** You do not need to write it this way, but it makes the *order* of your cleaning explicit — and order matters.

## 🌍 Scenarios

**Scenario 1 — the first thing you do with any new file**

```python
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
real = pd.read_csv(BASE + "prepreprocessing/pre_data.csv")

inspect(real, "pre_data.csv")
print(real)
```

**Scenario 2 — a cleaning log, so you can defend your choices**

```python
raw = messy_log()
log = []

clean = raw.copy()
before = len(clean); clean = clean.drop_duplicates()
log.append(f"dropped {before - len(clean)} duplicate row(s)")

before = len(clean); clean = clean.dropna()
log.append(f"dropped {before - len(clean)} row(s) with gaps")

print("\n".join(log))
print(f"kept {len(clean)} of {len(raw)} rows")
```

**Write down what you removed and why.** Somebody will ask, and "I cleaned it" is not an answer.

**Scenario 3 — how much would you lose by dropping everything imperfect?**

```python
raw = messy_log()
print(f"all rows              : {len(raw)}")
print(f"after dropna()        : {len(raw.dropna())}")
print(f"after drop_duplicates : {len(raw.drop_duplicates())}")
print(f"after both            : {len(raw.dropna().drop_duplicates())}")
```

**Check this before you delete anything.** On a small dataset, dropping every imperfect row can cost you a third of your data.

## ✏️ Tasks

1. Build the messy log and print its shape, missing counts and duplicate count.
2. Write an `inspect()` function and run it on both the messy log and `pre_data.csv`.
3. Show that cleaning a copy leaves the original untouched.
4. Count how many rows survive `dropna()`, `drop_duplicates()`, and both together.
5. List the four kinds of mess and give one example of each from the fitness log.

<details><summary>Solutions</summary>

```python
import pandas as pd

def messy_log():
    return pd.DataFrame({
        "Duration": [60, 60, 60, 45, 45, 60, 60, 450, 30, 60, 60, 60],
        "Date": ["2026/03/01", "2026/03/02", "2026/03/03", "2026/03/04",
                 "2026/03/05", "2026/03/06", "20260307", "2026/03/08",
                 None, "2026/03/10", "2026/03/11", "2026/03/11"],
        "Pulse": [110, 117, 103, 109, 117, 102, 110, 104, 109, 98, 103, 103],
        "Maxpulse": [130, 145, 135, 175, 148, 127, 136, 134, 133, 124, 147, 147],
        "Calories": [409.1, 479.0, 340.0, 282.4, 406.0, 300.0, 374.0, 253.3,
                     195.1, None, 329.3, 329.3],
    })

df = messy_log()                                                       # 1
print(df.shape)
print(df.isna().sum().to_string())
print("duplicates:", df.duplicated().sum())

def inspect(d, name="data"):                                           # 2
    print(f"--- {name}: {d.shape[0]} rows, {d.shape[1]} cols ---")
    m = d.isna().sum()
    print("missing:", m[m > 0].to_dict() if m.any() else "none")
    print("duplicates:", d.duplicated().sum())

inspect(df, "messy log")
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
inspect(pd.read_csv(BASE + "prepreprocessing/pre_data.csv"), "pre_data.csv")

raw = messy_log(); clean = raw.copy().drop_duplicates()                # 3
print("raw:", raw.shape, " clean:", clean.shape)   # raw is untouched

print(len(raw), len(raw.dropna()), len(raw.drop_duplicates()),         # 4
      len(raw.dropna().drop_duplicates()))

# 5 - Empty cells : Date row 8, Calories row 9
#     Wrong format: Date row 6 is "20260307" - no slashes
#     Wrong data  : Duration row 7 is 450 minutes = 7.5 hours
#     Duplicates  : row 11 is an exact copy of row 10
```
</details>

## ❓ MCQs

**Q1.** What are the four kinds of messy data?
- (a) Big, small, fast, slow  (b) Empty cells, wrong format, wrong data, duplicates  (c) Text, numbers, dates, booleans  (d) Rows, columns, cells, indexes

**Q2.** Why work on `df.copy()` rather than the original?
- (a) It is faster  (b) So you keep the raw data to compare against when something looks wrong  (c) It is required  (d) It saves memory

**Q3.** A date column shows dtype `object`. This means…
- (a) It is a date  (b) It is being stored as text, not as a date  (c) It is empty  (d) It is the index

**Q4.** Why check how many rows survive `dropna()` before running it?
- (a) Curiosity  (b) On a small dataset you can lose a large share of your data  (c) It is faster  (d) You do not need to

**Q5.** Why keep a cleaning log?
- (a) It is required  (b) So you can say what you removed and why when someone asks  (c) It speeds things up  (d) It prevents errors

<details><summary>Answers</summary>

**A1 — (b).** Topics 18 to 21, one each.

**A2 — (b).** Topic 6's lesson applied to your whole workflow.

**A3 — (b) Text.** `object` on a column that should be numeric or a date is always a clue.

**A4 — (b).** On the twelve-row fitness log, `dropna()` costs you two rows out of twelve.

**A5 — (b).** **"I cleaned it" is not an answer.**
</details>

---

# 18. Cleaning Empty Cells

**Empty cells are the commonest defect, and you have two choices: remove them, or fill them.**

## Finding them

```python
df.isna()             # True/False for every cell
df.isna().sum()       # how many per column
df.isna().sum().sum() # the total
df.isna().any(axis=1) # which ROWS have any gap
```

## Option 1 — remove

```python
df.dropna()                         # drop any row with ANY gap
df.dropna(subset=["Calories"])      # only if Calories is missing
df.dropna(axis=1)                   # drop the COLUMN instead
df.dropna(thresh=4)                 # keep rows with at least 4 real values
```

> ⚠️ **`dropna()` returns a new DataFrame — it does not change the original.** You must assign the result: `df = df.dropna()`.

## Option 2 — fill

```python
# illustrative: a syntax reference, not runnable as written.
df["Calories"] = df["Calories"].fillna(0)                       # a fixed value
df["Calories"] = df["Calories"].fillna(df["Calories"].mean())   # the mean
df["Calories"] = df["Calories"].fillna(df["Calories"].median()) # the median
df["Country"] = df["Country"].fillna(df["Country"].mode()[0])   # most common
df["Calories"] = df["Calories"].ffill()                         # last known value
```

## Which filler?

| Use | When |
|---|---|
| **Mean** | Numeric, roughly symmetric, no big outliers |
| **Median** | Numeric **and skewed** — the safer default |
| **Mode** | Categorical text |
| **`ffill`** | Time series, where the last known value is a fair guess |
| **Drop** | Few gaps, plenty of data |

> **Prefer the median for numeric columns.** One extreme value drags the mean a long way — you saw exactly this in Topic 14, where income's mean sat far above its median. The median barely moves.

## 📘 Examples

**Example 1 — find the gaps**

```python
df = messy_log()
print(df.isna().sum())
print("\nrows with any gap:")
print(df[df.isna().any(axis=1)])
```

**Example 2 — dropping**

```python
print("before:", df.shape)              # (12, 5)
print("after :", df.dropna().shape)     # (10, 5)   two rows lost
```

**Example 3 — filling with mean versus median**

```python
print(f"mean   {df['Calories'].mean():.1f}")
print(f"median {df['Calories'].median():.1f}")

filled = df.copy()
filled["Calories"] = filled["Calories"].fillna(filled["Calories"].median())
print("gaps remaining:", filled["Calories"].isna().sum())     # 0
```

**Example 4 — the mode, for text**

```python
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
real = pd.read_csv(BASE + "prepreprocessing/pre_data.csv")

print("before:", real["Country"].isna().sum())
real["Country"] = real["Country"].fillna(real["Country"].mode()[0])
print("after :", real["Country"].isna().sum())
# .mode() returns a Series (there can be ties), so take [0].
```

## 🌍 Scenarios

**Scenario 1 — choosing per column**

```python
real = pd.read_csv(BASE + "prepreprocessing/pre_data.csv")
clean = real.copy()

clean["Country"] = clean["Country"].fillna(clean["Country"].mode()[0])  # text -> mode
clean["Age"] = clean["Age"].fillna(clean["Age"].median())               # numeric -> median
clean["Salary"] = clean["Salary"].fillna(clean["Salary"].median())

print(clean.isna().sum().sum(), "gaps remaining")
```

**Different columns need different treatment.** One blanket `fillna(0)` would put a salary of zero in a real person's row.

**Scenario 2 — why zero is usually wrong**

```python
c = df["Calories"]
print(f"real mean        : {c.mean():.1f}")
print(f"if filled with 0 : {c.fillna(0).mean():.1f}")
print(f"if filled median : {c.fillna(c.median()).mean():.1f}")
```

**Filling with zero drags the average down and invents a false fact** — it says the person burned no calories, which is not what "we do not know" means.

**Scenario 3 — deciding by how much is missing**

```python
missing_pct = (df.isna().sum() / len(df) * 100).round(1)
for col, pct in missing_pct.items():
    if pct == 0:      advice = "fine"
    elif pct < 5:     advice = "drop those rows"
    elif pct < 40:    advice = "fill it"
    else:             advice = "consider dropping the COLUMN"
    print(f"{col:<10}{pct:>6}%   {advice}")
```

**A rough but useful rule:** under 5% missing, drop the rows; up to about 40%, fill; above that, ask whether the column is worth keeping at all.

## ✏️ Tasks

1. Print how many empty cells each column of the fitness log has, and which rows they are in.
2. Drop all rows with any gap. How many rows do you lose?
3. Fill `Calories` with its median, then confirm no gaps remain.
4. On `pre_data.csv`, fill the text column with the mode and the numeric ones with the median.
5. Compare the column mean after filling with 0 and after filling with the median. Explain the difference.

<details><summary>Solutions</summary>

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

def messy_log():
    return pd.DataFrame({
        "Duration": [60, 60, 60, 45, 45, 60, 60, 450, 30, 60, 60, 60],
        "Date": ["2026/03/01", "2026/03/02", "2026/03/03", "2026/03/04",
                 "2026/03/05", "2026/03/06", "20260307", "2026/03/08",
                 None, "2026/03/10", "2026/03/11", "2026/03/11"],
        "Pulse": [110, 117, 103, 109, 117, 102, 110, 104, 109, 98, 103, 103],
        "Maxpulse": [130, 145, 135, 175, 148, 127, 136, 134, 133, 124, 147, 147],
        "Calories": [409.1, 479.0, 340.0, 282.4, 406.0, 300.0, 374.0, 253.3,
                     195.1, None, 329.3, 329.3],
    })

df = messy_log()
print(df.isna().sum().to_string())                                     # 1
print(df[df.isna().any(axis=1)])

print(len(df), "->", len(df.dropna()))     # 12 -> 10, two rows lost   # 2

f = df.copy()                                                          # 3
f["Calories"] = f["Calories"].fillna(f["Calories"].median())
print("gaps:", f["Calories"].isna().sum())      # 0

real = pd.read_csv(BASE + "prepreprocessing/pre_data.csv").copy()      # 4
real["Country"] = real["Country"].fillna(real["Country"].mode()[0])
real["Age"] = real["Age"].fillna(real["Age"].median())
real["Salary"] = real["Salary"].fillna(real["Salary"].median())
print("gaps remaining:", real.isna().sum().sum())     # 0

c = df["Calories"]                                                     # 5
print(f"real mean {c.mean():.1f} | zero-filled {c.fillna(0).mean():.1f} "
      f"| median-filled {c.fillna(c.median()).mean():.1f}")
# Zero-filling INVENTS A FALSE FACT - it claims no calories were burned,
# which is not the same as "we do not know". It drags the average down.
```
</details>

## ❓ MCQs

**Q1.** How do you count missing values per column?
- (a) `df.count()`  (b) `df.isna().sum()`  (c) `df.missing()`  (d) `df.null()`

**Q2.** Does `df.dropna()` change the original DataFrame?
- (a) Yes  (b) No — it returns a new one, so you must assign it  (c) Only for numbers  (d) Only with `axis=1`

**Q3.** For a skewed numeric column, the safer filler is…
- (a) The mean  (b) The median  (c) Zero  (d) The mode

**Q4.** Why is filling with 0 usually wrong?
- (a) It is slow  (b) It invents a false fact — "zero" is not "unknown"  (c) Pandas forbids it  (d) It causes errors

**Q5.** For a missing text category, use…
- (a) The mean  (b) The median  (c) The mode  (d) Zero

<details><summary>Answers</summary>

**A1 — (b) `df.isna().sum()`.**

**A2 — (b).** Assign the result: `df = df.dropna()`.

**A3 — (b) The median.** One extreme value drags the mean a long way — Topic 14's income column showed exactly that.

**A4 — (b).** Zero is a claim; missing is an absence of information.

**A5 — (c) The mode** — the most common value. Remember `.mode()[0]`.
</details>

---

# 19. Cleaning Wrong Format

**Data that is the right value in the wrong shape.** Dates are the usual culprit.

In the fitness log, most dates read `2026/03/01`. **Row 6 reads `20260307`** — same date, no slashes. Pandas cannot compare, sort or filter by date while the column is text.

## Converting to real dates

```python
df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")
```

| Argument | Does |
|---|---|
| `format="mixed"` | Work each value out individually, so several layouts parse |
| `errors="coerce"` | Turn anything unparseable into `NaT` instead of crashing |
| `errors="raise"` | Stop with an error (the default) |

> **`errors="coerce"` is the important one.** Without it, one bad value stops the whole conversion. With it, the bad values become `NaT` (Not a Time) — Pandas' `NaN` for dates — which you then handle as an empty cell using Topic 18.

## Why it is worth doing

```python
df["Date"].dt.day_name()      # 'Sunday'
df["Date"].dt.month           # 3
df[df["Date"] > "2026-03-05"] # real date filtering
df.sort_values("Date")        # sorts CHRONOLOGICALLY, not alphabetically
```

**None of that works on text.** Sorting text dates puts `2026/1/5` after `2026/11/2`.

## Numbers stored as text

```python
# illustrative: a syntax reference, not runnable as written.
df["price"] = pd.to_numeric(df["price"], errors="coerce")
```

**The same idea.** A price column containing `"1,200"` or `"N/A"` is text; `to_numeric` with `coerce` turns the bad ones into `NaN`.

## 📘 Examples

**Example 1 — spotting the problem**

```python
df = messy_log()
print(df["Date"].dtype)        # object  -> it is TEXT
print(df["Date"].tolist())     # you can see '20260307' has no slashes
```

**Example 2 — converting**

```python
clean = df.copy()
clean["Date"] = pd.to_datetime(clean["Date"], format="mixed", errors="coerce")

print(clean["Date"].dtype)             # datetime64[ns]
print(clean["Date"].tolist()[5:9])
print("unparseable:", clean["Date"].isna().sum())     # 1  (the None in row 8)
```

**`20260307` parsed correctly to 2026-03-07.** The only failure is the genuinely missing value.

**Example 3 — what you can now do**

```python
clean = clean.dropna(subset=["Date"])
print(clean["Date"].dt.day_name().tolist()[:4])
print(clean[clean["Date"] > "2026-03-08"][["Date", "Duration"]])
```

**Example 4 — numbers stored as text**

```python
prices = pd.Series(["1200", "850", "N/A", "2,400"])
print(pd.to_numeric(prices, errors="coerce"))
# [1200.0, 850.0, NaN, NaN]
# "N/A" and "2,400" both fail - the comma is not a number character.
# Strip it first: prices.str.replace(",", "")
```

## 🌍 Scenarios

**Scenario 1 — the full date clean**

```python
raw = messy_log()
clean = raw.copy()

clean["Date"] = pd.to_datetime(clean["Date"], format="mixed", errors="coerce")
bad = clean["Date"].isna().sum()
print(f"{bad} date(s) could not be parsed")

clean = clean.dropna(subset=["Date"])          # Topic 18 handles them
print(f"kept {len(clean)} of {len(raw)} rows")
print(clean["Date"].min(), "to", clean["Date"].max())
```

**Scenario 2 — inconsistent text categories**

```python
countries = pd.Series(["france", "FRANCE", " France ", "Spain", "spain"])
print(countries.nunique())                                    # 5  - wrong!

fixed = countries.str.strip().str.title()
print(fixed.nunique())                                        # 2  - correct
print(fixed.value_counts())
```

**`france`, `FRANCE` and ` France ` are the same country counted three times.** Stripping and normalising case is a wrong-format fix, and it changes your counts.

**Scenario 3 — a currency column**

```python
amounts = pd.Series(["Rs 1,200", "Rs 850", "Rs 2,400"])
numeric = (amounts.str.replace("Rs ", "", regex=False)
                  .str.replace(",", "", regex=False)
                  .astype(float))
print(numeric)
print(f"total: {numeric.sum():,.0f}")
```

## ✏️ Tasks

1. Print the dtype of the `Date` column and explain why it is `object`.
2. Convert it with `format="mixed"` and `errors="coerce"`. How many failed?
3. After converting, print the day name for each row and sort by date.
4. Convert `["1200", "850", "N/A"]` to numbers. What happens to `"N/A"`?
5. Normalise `["france", "FRANCE", " France "]` into one consistent value.

<details><summary>Solutions</summary>

```python
import pandas as pd

def messy_log():
    return pd.DataFrame({
        "Duration": [60, 60, 60, 45, 45, 60, 60, 450, 30, 60, 60, 60],
        "Date": ["2026/03/01", "2026/03/02", "2026/03/03", "2026/03/04",
                 "2026/03/05", "2026/03/06", "20260307", "2026/03/08",
                 None, "2026/03/10", "2026/03/11", "2026/03/11"],
        "Pulse": [110, 117, 103, 109, 117, 102, 110, 104, 109, 98, 103, 103],
        "Maxpulse": [130, 145, 135, 175, 148, 127, 136, 134, 133, 124, 147, 147],
        "Calories": [409.1, 479.0, 340.0, 282.4, 406.0, 300.0, 374.0, 253.3,
                     195.1, None, 329.3, 329.3],
    })

df = messy_log()
print(df["Date"].dtype)                                                # 1
# 'object' means TEXT. Pandas cannot sort, compare or filter by date
# while the column is text.

clean = df.copy()                                                      # 2
clean["Date"] = pd.to_datetime(clean["Date"], format="mixed", errors="coerce")
print(clean["Date"].dtype, "| failed:", clean["Date"].isna().sum())   # 1
# '20260307' PARSED FINE. The only failure is the genuinely missing value.

ok = clean.dropna(subset=["Date"]).sort_values("Date")                 # 3
print(ok["Date"].dt.day_name().tolist())

print(pd.to_numeric(pd.Series(["1200", "850", "N/A"]),                 # 4
                    errors="coerce"))
# "N/A" becomes NaN - errors="coerce" turns failures into missing values
# instead of crashing, so Topic 18 can then handle them.

c = pd.Series(["france", "FRANCE", " France "])                        # 5
print(c.str.strip().str.title().unique())      # ['France']
```
</details>

## ❓ MCQs

**Q1.** A date column has dtype `object`. What does that mean?
- (a) It is a date  (b) It is stored as text  (c) It is empty  (d) It is numeric

**Q2.** What does `errors="coerce"` do in `pd.to_datetime()`?
- (a) Raises an error  (b) Turns unparseable values into `NaT` instead of crashing  (c) Deletes them  (d) Ignores the column

**Q3.** Why convert text dates to real dates?
- (a) It looks nicer  (b) So you can sort, filter and compare chronologically  (c) It saves memory  (d) Pandas requires it

**Q4.** Sorting text dates alphabetically puts…
- (a) Them in the right order  (b) `2026/11/2` before `2026/1/5` — the wrong order  (c) Nothing  (d) An error

**Q5.** `"france"`, `"FRANCE"` and `" France "` in one column…
- (a) Are counted as one  (b) Are counted as three different values until you strip and normalise case  (c) Cause an error  (d) Are ignored

<details><summary>Answers</summary>

**A1 — (b) Text.** The clue that a format fix is needed.

**A2 — (b).** **The important argument.** Without it, one bad value stops the whole conversion.

**A3 — (b).** None of that works on text.

**A4 — (b).** Alphabetically, `11` sorts before `1` — a classic silent bug.

**A5 — (b).** A wrong-format fix that changes your counts.
</details>

---

# 20. Cleaning Wrong Data

**A value that is the right type and the right format, and still cannot be true.**

In the fitness log, `Duration` row 7 is **450 minutes** — seven and a half hours of exercise. Every other row is 30, 45 or 60. In the loan dataset, `person_age` reaches **144**.

> **The computer cannot spot these. Only you can, because it needs knowing what the numbers mean.** This is the topic where domain sense beats code.

## Finding them

```python
df["Duration"].describe()        # look at min and max
df["Duration"].unique()          # what values actually occur?
df[df["Duration"] > 120]         # anything above a sensible limit
df["Duration"].plot(kind="box")  # Topic 16 - outliers as dots
```

## Fixing them: three choices

```python
# 1. Replace with a sensible value
df.loc[df["Duration"] > 120, "Duration"] = 60

# 2. Replace with the median
df.loc[df["Duration"] > 120, "Duration"] = df["Duration"].median()

# 3. Remove the row entirely
df = df[df["Duration"] <= 120]
```

> ⚠️ **Use `df.loc[condition, "column"] = value` to change cells.** Writing `df[df["Duration"] > 120]["Duration"] = 60` looks reasonable and **silently does nothing** — it modifies a temporary copy. Pandas warns you with `SettingWithCopyWarning`. **Always use `.loc`.**

## Which choice?

| Do this | When |
|---|---|
| **Replace with a sensible value** | You know what it should have been (a typo you can reason about) |
| **Replace with the median** | You do not know, but you want to keep the rest of the row |
| **Remove the row** | The value is central to your analysis, and you have plenty of data |
| **Leave it and flag it** | You are not sure it is wrong |

> **Removing the row throws away every other column too.** With 10,000 rows that is fine. With 12, it is expensive.

## 📘 Examples

**Example 1 — spotting it**

```python
df = messy_log()
print(df["Duration"].describe())
print("\nvalues present:", sorted(df["Duration"].unique()))
# [30, 45, 60, 450]  -> 450 stands out immediately
```

**Example 2 — the three fixes**

```python
a = df.copy(); a.loc[a["Duration"] > 120, "Duration"] = 60
b = df.copy(); b.loc[b["Duration"] > 120, "Duration"] = b["Duration"].median()
c = df[df["Duration"] <= 120]

print("replace with 60    :", sorted(a["Duration"].unique()))
print("replace with median:", sorted(b["Duration"].unique()))
print("remove the row     :", c.shape, "vs", df.shape)
```

**Example 3 — the `.loc` trap**

```python
wrong = df.copy()
# wrong[wrong["Duration"] > 120]["Duration"] = 60     # does NOTHING
right = df.copy()
right.loc[right["Duration"] > 120, "Duration"] = 60   # works

print("still 450?", 450 in wrong["Duration"].values)  # True  - not changed
print("still 450?", 450 in right["Duration"].values)  # False - fixed
```

**Example 4 — on real data**

```python
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
loans = pd.read_csv(BASE + "loan_data_10k.csv")

print("age range:", loans["person_age"].min(), "to", loans["person_age"].max())
print("rows over 100:", (loans["person_age"] > 100).sum())

clean = loans[loans["person_age"] <= 100]
print(f"kept {len(clean):,} of {len(loans):,} rows")
```

**One impossible row out of ten thousand — dropping it costs nothing.**

## 🌍 Scenarios

**Scenario 1 — set sensible limits per column**

```python
limits = {
    "Duration": (10, 180),      # minutes: a workout is not 7 hours
    "Pulse": (40, 200),         # beats per minute
    "Maxpulse": (40, 220),
    "Calories": (20, 1500),
}

df = messy_log()
for col, (lo, hi) in limits.items():
    bad = ((df[col] < lo) | (df[col] > hi)).sum()
    if bad:
        print(f"{col}: {bad} value(s) outside {lo}-{hi}")
```

**Writing the limits down forces you to think about what each column means** — which is the actual work in this topic.

**Scenario 2 — clipping instead of deleting**

```python
clipped = df.copy()
clipped["Duration"] = clipped["Duration"].clip(10, 180)
print(sorted(clipped["Duration"].unique()))       # 450 becomes 180
```

**`clip()` caps values at a floor and ceiling in one call.** It keeps the row and limits the damage.

**Scenario 3 — flag rather than fix**

```python
flagged = df.copy()
flagged["suspect"] = flagged["Duration"] > 120
print(flagged[["Duration", "suspect"]])
print(f"\n{flagged['suspect'].sum()} row(s) flagged for review")
```

**When you are not sure a value is wrong, flag it and keep it.** Someone with domain knowledge can decide later, and you have not destroyed anything.

## ✏️ Tasks

1. Print `describe()` and `unique()` for `Duration` and identify the impossible value.
2. Fix it three ways — sensible value, median, remove — and compare the results.
3. Show that `df[cond]["col"] = value` fails and `df.loc[cond, "col"] = value` works.
4. On the loan dataset, find and remove the impossible ages. How many rows do you lose?
5. Write a limits dictionary for the fitness log and report every violation.

<details><summary>Solutions</summary>

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

def messy_log():
    return pd.DataFrame({
        "Duration": [60, 60, 60, 45, 45, 60, 60, 450, 30, 60, 60, 60],
        "Date": ["2026/03/01", "2026/03/02", "2026/03/03", "2026/03/04",
                 "2026/03/05", "2026/03/06", "20260307", "2026/03/08",
                 None, "2026/03/10", "2026/03/11", "2026/03/11"],
        "Pulse": [110, 117, 103, 109, 117, 102, 110, 104, 109, 98, 103, 103],
        "Maxpulse": [130, 145, 135, 175, 148, 127, 136, 134, 133, 124, 147, 147],
        "Calories": [409.1, 479.0, 340.0, 282.4, 406.0, 300.0, 374.0, 253.3,
                     195.1, None, 329.3, 329.3],
    })

df = messy_log()
print(df["Duration"].describe().to_string())                           # 1
print("values:", sorted(df["Duration"].unique()))
# 450 minutes is 7.5 hours. Every other row is 30, 45 or 60.

a = df.copy(); a.loc[a["Duration"] > 120, "Duration"] = 60             # 2
b = df.copy(); b.loc[b["Duration"] > 120, "Duration"] = b["Duration"].median()
c = df[df["Duration"] <= 120]
print(sorted(a.Duration.unique()), sorted(b.Duration.unique()), c.shape)

wrong, right = df.copy(), df.copy()                                    # 3
# wrong[wrong["Duration"] > 120]["Duration"] = 60     # modifies a COPY
right.loc[right["Duration"] > 120, "Duration"] = 60
print(450 in wrong["Duration"].values, 450 in right["Duration"].values)
# True, False. ALWAYS use .loc to assign.

loans = pd.read_csv(BASE + "loan_data_10k.csv")                        # 4
print("max age:", loans["person_age"].max())          # 144
clean = loans[loans["person_age"] <= 100]
print(f"lost {len(loans) - len(clean)} of {len(loans):,} rows")   # 1

limits = {"Duration": (10, 180), "Pulse": (40, 200),                   # 5
          "Maxpulse": (40, 220), "Calories": (20, 1500)}
for col, (lo, hi) in limits.items():
    bad = ((df[col] < lo) | (df[col] > hi)).sum()
    print(f"{col:<10}{bad} outside {lo}-{hi}")
```
</details>

## ❓ MCQs

**Q1.** A `Duration` of 450 minutes among values of 30, 45 and 60 is…
- (a) Missing data  (b) Wrong data — possible to store, impossible in reality  (c) Wrong format  (d) A duplicate

**Q2.** Which correctly changes cells matching a condition?
- (a) `df[df["a"] > 5]["a"] = 0`  (b) `df.loc[df["a"] > 5, "a"] = 0`  (c) `df["a"][df["a"] > 5] = 0`  (d) `df.set(...)`

**Q3.** What happens with `df[cond]["col"] = value`?
- (a) It works  (b) It silently modifies a temporary copy and does nothing  (c) It raises an error  (d) It deletes the column

**Q4.** What does `.clip(10, 180)` do?
- (a) Deletes values outside the range  (b) Caps values at 10 and 180, keeping the row  (c) Rounds them  (d) Counts them

**Q5.** Why can the computer not find wrong data on its own?
- (a) It is too slow  (b) Knowing a value is impossible requires knowing what it means  (c) Pandas lacks the feature  (d) It can

<details><summary>Answers</summary>

**A1 — (b) Wrong data.** The type and format are fine; the value cannot be true.

**A2 — (b) `.loc`.**

**A3 — (b).** It **silently does nothing**, with a `SettingWithCopyWarning`. **Always use `.loc`.**

**A4 — (b).** It keeps the row and limits the damage.

**A5 — (b).** **This is the topic where domain sense beats code.**
</details>

---

# 21. Removing Duplicates

**The same row, entered twice.** Usually a data-entry slip or a botched file merge.

## Finding them

```python
df.duplicated()              # True for every row that repeats an earlier one
df.duplicated().sum()        # how many
df[df.duplicated(keep=False)]  # show ALL copies, including the first
```

> **`duplicated()` marks the *second and later* copies, not the first.** So on two identical rows it returns one `True`, not two. Use `keep=False` when you want to see every copy.

## Removing them

```python
df = df.drop_duplicates()                              # keep the first
df = df.drop_duplicates(keep="last")                   # keep the last
df = df.drop_duplicates(subset=["Date"])               # same DATE = duplicate
df = df.drop_duplicates().reset_index(drop=True)       # tidy the index
```

> **After dropping rows the index has gaps** — 0, 1, 2, 4, 5. `reset_index(drop=True)` renumbers it. Without `drop=True` the old index is kept as a new column.

## ⚠️ Not every duplicate is a mistake

```python
sales = pd.DataFrame({
    "item": ["tea", "tea", "coffee"],
    "price": [15, 15, 25],
})
```

**Two people genuinely bought tea at 15.** Dropping that is deleting a real sale.

> **Before dropping duplicates, ask: could two rows legitimately be identical?** If yes, you need a column that distinguishes them — a timestamp, a transaction ID — and you should be de-duplicating on *that*, not on the whole row.

## 📘 Examples

**Example 1 — finding them**

```python
df = messy_log()
print("duplicates:", df.duplicated().sum())        # 1
print(df[df.duplicated(keep=False)])               # rows 10 AND 11
```

**Example 2 — removing them**

```python
print("before:", df.shape)                         # (12, 5)
clean = df.drop_duplicates()
print("after :", clean.shape)                      # (11, 5)
print("index:", clean.index.tolist())              # note the gap at 11
clean = clean.reset_index(drop=True)
print("index:", clean.index.tolist())              # 0..10, tidy
```

**Example 3 — duplicates on a subset of columns**

```python
by_date = df.drop_duplicates(subset=["Date"])
print("by whole row:", df.drop_duplicates().shape)   # (11, 5)
print("by Date only:", by_date.shape)               # (11, 5)
# Here they agree. On a bigger dataset they often will not - two DIFFERENT
# workouts on the same date would be dropped by the subset version.
```

**Example 4 — keeping the last instead of the first**

```python
records = pd.DataFrame({
    "student": ["Arun", "Priya", "Arun"],
    "marks": [65, 92, 78],           # Arun was remarked
})
print(records.drop_duplicates(subset=["student"], keep="last"))
# Keeps Arun's 78 - the CORRECTED mark, not the original.
```

**`keep="last"` is what you want when later rows are corrections.**

## 🌍 Scenarios

**Scenario 1 — a merged file with repeated rows**

```python
january = pd.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})
february = pd.DataFrame({"id": [3, 4, 5], "amount": [300, 400, 500]})

merged = pd.concat([january, february], ignore_index=True)
print("merged:", merged.shape)                     # (6, 2)
print("duplicates:", merged.duplicated().sum())    # 1  - id 3 is in both
clean = merged.drop_duplicates().reset_index(drop=True)
print("clean:", clean.shape)                       # (5, 2)
```

**Overlapping months is the classic way duplicates appear.**

**Scenario 2 — the duplicate that is not a mistake**

```python
sales = pd.DataFrame({
    "item": ["tea", "tea", "tea", "coffee"],
    "price": [15, 15, 15, 25],
})
print("rows:", len(sales))                            # 4
print("after drop_duplicates:", len(sales.drop_duplicates()))   # 2
print(f"\nRevenue before: {sales['price'].sum()}")    # 70
print(f"Revenue after : {sales.drop_duplicates()['price'].sum()}")  # 40
```

**You just deleted 30 rupees of real sales.** Three people bought tea; `drop_duplicates` cannot tell that from an error. **Add a transaction ID and the problem disappears.**

**Scenario 3 — de-duplicating on the right column**

```python
sales = pd.DataFrame({
    "txn_id": [101, 102, 103, 103],          # 103 was recorded twice
    "item": ["tea", "tea", "coffee", "coffee"],
    "price": [15, 15, 25, 25],
})
clean = sales.drop_duplicates(subset=["txn_id"])
print(clean)
print(f"revenue: {clean['price'].sum()}")     # 55 - correct
```

**Now the two tea sales survive and the genuinely duplicated transaction goes.** The right column made it unambiguous.

## ✏️ Tasks

1. Count the duplicates in the fitness log and print every copy, including the first.
2. Drop them, then reset the index and confirm it is renumbered.
3. Merge two overlapping DataFrames and remove the resulting duplicates.
4. Build a sales table where two identical rows are both genuine. Show what `drop_duplicates` costs you.
5. Fix task 4 by adding a transaction ID and de-duplicating on that instead.

<details><summary>Solutions</summary>

```python
import pandas as pd

def messy_log():
    return pd.DataFrame({
        "Duration": [60, 60, 60, 45, 45, 60, 60, 450, 30, 60, 60, 60],
        "Date": ["2026/03/01", "2026/03/02", "2026/03/03", "2026/03/04",
                 "2026/03/05", "2026/03/06", "20260307", "2026/03/08",
                 None, "2026/03/10", "2026/03/11", "2026/03/11"],
        "Pulse": [110, 117, 103, 109, 117, 102, 110, 104, 109, 98, 103, 103],
        "Maxpulse": [130, 145, 135, 175, 148, 127, 136, 134, 133, 124, 147, 147],
        "Calories": [409.1, 479.0, 340.0, 282.4, 406.0, 300.0, 374.0, 253.3,
                     195.1, None, 329.3, 329.3],
    })

df = messy_log()
print("duplicates:", df.duplicated().sum())                            # 1
print(df[df.duplicated(keep=False)])
# keep=False shows ALL copies. Plain duplicated() marks only the SECOND.

clean = df.drop_duplicates()                                           # 2
print(clean.index.tolist())            # has a gap
clean = clean.reset_index(drop=True)
print(clean.index.tolist())            # 0..10, tidy

jan = pd.DataFrame({"id": [1, 2, 3], "amount": [100, 200, 300]})       # 3
feb = pd.DataFrame({"id": [3, 4, 5], "amount": [300, 400, 500]})
merged = pd.concat([jan, feb], ignore_index=True)
print(merged.shape, "->", merged.drop_duplicates().shape)

sales = pd.DataFrame({"item": ["tea", "tea", "tea", "coffee"],         # 4
                      "price": [15, 15, 15, 25]})
print(f"real revenue {sales['price'].sum()}, "
      f"after dropping {sales.drop_duplicates()['price'].sum()}")
# 70 -> 40. You just deleted 30 rupees of REAL SALES. Three people
# bought tea; drop_duplicates cannot tell that from an error.

sales = pd.DataFrame({"txn_id": [101, 102, 103, 103],                  # 5
                      "item": ["tea", "tea", "coffee", "coffee"],
                      "price": [15, 15, 25, 25]})
clean = sales.drop_duplicates(subset=["txn_id"])
print(clean, f"\nrevenue {clean['price'].sum()}")    # 55 - correct
```
</details>

## ❓ MCQs

**Q1.** What does `df.duplicated()` mark?
- (a) All copies  (b) The second and later copies, not the first  (c) The first only  (d) Nothing

**Q2.** How do you see every copy including the first?
- (a) `df.duplicated()`  (b) `df.duplicated(keep=False)`  (c) `df.drop_duplicates()`  (d) `df.unique()`

**Q3.** Why call `reset_index(drop=True)` after dropping rows?
- (a) It is faster  (b) The index has gaps; this renumbers it  (c) It is required  (d) It sorts the data

**Q4.** Two rows show tea at 15. Dropping one…
- (a) Is always right  (b) May delete a real second sale  (c) Is required  (d) Does nothing

**Q5.** When is `keep="last"` the right choice?
- (a) Never  (b) When later rows are corrections of earlier ones  (c) Always  (d) Only for text

<details><summary>Answers</summary>

**A1 — (b).** Two identical rows give one `True`, not two.

**A2 — (b) `keep=False`.**

**A3 — (b).** Without `drop=True` the old index is kept as a new column.

**A4 — (b).** **`drop_duplicates` cannot tell a real repeat from an error.** Add a transaction ID.

**A5 — (b).** Keeping the corrected mark rather than the original.
</details>

---

## ⭐ Checkpoint Problem 5 — The full cleaning pipeline

> **Uses everything:** Topics 17–21, and the analysis skills from 13–16.

**The problem.** Take the messy fitness log and clean it completely — duplicates, wrong formats, wrong data and empty cells — printing a report of exactly what you changed at each step. **Then say what you would tell someone who asked whether your cleaning was reasonable.**

<details><summary>Solution</summary>

```python
import pandas as pd

def messy_log():
    return pd.DataFrame({
        "Duration": [60, 60, 60, 45, 45, 60, 60, 450, 30, 60, 60, 60],
        "Date": ["2026/03/01", "2026/03/02", "2026/03/03", "2026/03/04",
                 "2026/03/05", "2026/03/06", "20260307", "2026/03/08",
                 None, "2026/03/10", "2026/03/11", "2026/03/11"],
        "Pulse": [110, 117, 103, 109, 117, 102, 110, 104, 109, 98, 103, 103],
        "Maxpulse": [130, 145, 135, 175, 148, 127, 136, 134, 133, 124, 147, 147],
        "Calories": [409.1, 479.0, 340.0, 282.4, 406.0, 300.0, 374.0, 253.3,
                     195.1, None, 329.3, 329.3],
    })

raw = messy_log()
clean = raw.copy()                    # NEVER touch the raw data
log = []

# --- 1. Duplicates FIRST, so they cannot skew the medians we compute later
before = len(clean)
clean = clean.drop_duplicates().reset_index(drop=True)
log.append(f"dropped {before - len(clean)} duplicate row(s)")

# --- 2. Wrong format: text dates -> real dates
clean["Date"] = pd.to_datetime(clean["Date"], format="mixed", errors="coerce")
log.append(f"converted Date to datetime ({clean['Date'].isna().sum()} unparseable)")

# --- 3. Wrong data: an impossible workout length
bad = (clean["Duration"] > 120).sum()
clean.loc[clean["Duration"] > 120, "Duration"] = clean["Duration"].median()
log.append(f"replaced {bad} impossible Duration value(s) with the median")

# --- 4. Empty cells, column by column
n_cal = clean["Calories"].isna().sum()
clean["Calories"] = clean["Calories"].fillna(clean["Calories"].median())
log.append(f"filled {n_cal} missing Calories with the median")

before = len(clean)
clean = clean.dropna(subset=["Date"]).reset_index(drop=True)
log.append(f"dropped {before - len(clean)} row(s) with no usable Date")

# --- Report
print("CLEANING LOG")
for i, step in enumerate(log, 1):
    print(f"  {i}. {step}")

print(f"\nRows: {len(raw)} -> {len(clean)}")
print(f"Missing values remaining: {clean.isna().sum().sum()}")
print(f"Duplicates remaining    : {clean.duplicated().sum()}")
print(f"Duration range          : {clean['Duration'].min()}-{clean['Duration'].max()}")
print(f"\n{clean.to_string()}")
```

**Two decisions in that pipeline are worth defending, and both are about order:**

1. **Duplicates go first.** The duplicated row would otherwise be counted twice in every median you compute — so removing it first makes those medians honest.
2. **The bad `Duration` is replaced, not dropped.** That row has four other perfectly good columns. With only twelve rows, throwing them away is expensive; with ten thousand it would not matter.

**What you tell someone who asks whether this was reasonable:**

```text
I removed 1 duplicate row, replaced 1 impossible Duration (450 minutes)
with the median, filled 1 missing Calories value with the median, and
dropped 1 row whose date could not be recovered.

12 rows in, 10 rows out.

The two median substitutions are assumptions, not facts. If those two
rows matter to a conclusion, the conclusion is weak and should be
checked against the raw data.
```

**That last paragraph is the part that separates careful work from a script that ran.**
</details>

**Make it harder:**

1. Turn the pipeline into a function `clean_log(df)` that returns both the cleaned frame and the log.
2. Run it on `pre_data.csv` and adapt it to that file's columns.
3. Compare `Calories.mean()` before and after cleaning. Did your cleaning change the answer? By how much?

---

# Part D — Python Libraries for Data Visualization

# 22. Matplotlib

**Topic 16 showed you `df.plot()` — Pandas' shortcut for a quick look. Matplotlib is what sits underneath it**, and you reach for it directly whenever you need control: several panels, exact labels, a saved file.

🧠 **Analogy: a phone camera versus a proper camera.** `df.plot()` is the phone — point, shoot, good enough for a look. Matplotlib is the camera with dials: more to learn, and the only way to get exactly the picture you want. **You use both, for different jobs.**

## The one pattern to learn

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))    # a figure, and one set of axes
ax.plot([1, 2, 3], [2, 4, 9], marker="o")
ax.set_title("My chart")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()
```

| Object | Is |
|---|---|
| **`fig`** | The whole picture — the sheet of paper |
| **`ax`** | One set of axes on it — where things are actually drawn |

> **Use `fig, ax = plt.subplots()` every time, and draw on `ax`.** You will see older code that calls `plt.plot()` directly; it works, but it draws on a hidden "current" figure, and the moment you want two panels it falls apart. **The `ax` habit scales; the `plt` habit does not.**

## The plot types

```python
# illustrative: a syntax reference, not runnable as written.
ax.plot(x, y)          # a line
ax.scatter(x, y)       # points
ax.bar(labels, values) # vertical bars
ax.barh(labels, values)# horizontal bars
ax.hist(values, bins=30)  # a distribution
ax.boxplot(values)     # median, quartiles, outliers
ax.pie(values, labels=labels)
```

## Labelling — never optional

```python
ax.set_title("Income distribution")
ax.set_xlabel("annual income")
ax.set_ylabel("number of applicants")
ax.legend()                     # needs label= on each plot call
ax.grid(alpha=0.3)              # a faint grid helps reading values
```

> **An unlabelled chart is not a result, it is a decoration.** If a reader cannot tell what the axes are, the chart has told them nothing.

## Several panels

```python
# illustrative: a syntax reference, not runnable as written.
fig, axes = plt.subplots(1, 3, figsize=(13, 4))    # 1 row, 3 columns
axes[0].hist(a)
axes[1].scatter(x, y)
axes[2].bar(labels, values)
plt.tight_layout()      # stops the labels overlapping
plt.show()
```

**`plt.tight_layout()` is nearly always worth calling.** Without it, titles and axis labels collide.

## Saving

```python
fig.savefig("chart.png", dpi=150, bbox_inches="tight")
```

**`bbox_inches="tight"` trims the white border**, and `dpi=150` makes it sharp enough for a report.

## 📘 Examples

**Example 1 — the basic figure**

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, np.sin(x), label="sin")
ax.plot(x, np.cos(x), label="cos")
ax.set_title("Two curves")
ax.set_xlabel("x")
ax.set_ylabel("value")
ax.legend()
ax.grid(alpha=0.3)
plt.show()
```

**Example 2 — a histogram with real data**

```python
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
loans = pd.read_csv(BASE + "loan_data_10k.csv")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(loans["person_income"], bins=50, color="steelblue", edgecolor="white")
ax.set_title("Income distribution")
ax.set_xlabel("annual income")
ax.set_ylabel("applicants")
plt.show()
```

**Example 3 — three panels side by side**

```python
fig, axes = plt.subplots(1, 3, figsize=(13, 4))

axes[0].hist(loans["person_age"], bins=30, color="steelblue")
axes[0].set_title("Age")

axes[1].hist(loans["credit_score"], bins=30, color="seagreen")
axes[1].set_title("Credit score")

axes[2].scatter(loans["person_income"], loans["loan_amnt"], alpha=0.1, s=6)
axes[2].set_title("Income vs loan amount")

for a in axes:
    a.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**Example 4 — annotating the thing you want noticed**

```python
fig, ax = plt.subplots(figsize=(7, 4))
ax.boxplot(loans["person_age"], vert=False)
ax.set_title("Age — note the outlier")
ax.set_xlabel("age")
ax.annotate("impossible: age 144",
            xy=(loans["person_age"].max(), 1),
            xytext=(110, 1.25),
            arrowprops=dict(arrowstyle="->", color="crimson"),
            color="crimson")
plt.show()
```

**An annotation is how you make a chart argue a point** rather than just display data.

## 🌍 Scenarios

**Scenario 1 — a chart for a report, saved to a file**

```python
fig, ax = plt.subplots(figsize=(8, 4.5))
counts = loans["loan_intent"].value_counts()
ax.barh(counts.index, counts.values, color="steelblue")
ax.set_title("Why people borrow")
ax.set_xlabel("number of applications")
ax.grid(alpha=0.3, axis="x")
plt.tight_layout()
fig.savefig("loan_intent.png", dpi=150, bbox_inches="tight")
print("saved loan_intent.png")
```

**Scenario 2 — comparing two groups on one axis**

```python
approved = loans[loans["loan_status"] == 1]["person_income"]
declined = loans[loans["loan_status"] == 0]["person_income"]

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(declined, bins=50, alpha=0.6, label="declined", range=(0, 200000))
ax.hist(approved, bins=50, alpha=0.6, label="approved", range=(0, 200000))
ax.set_title("Income by outcome")
ax.set_xlabel("income")
ax.legend()
plt.show()
```

**`alpha` and a shared `range` are what make two histograms comparable.** Without the shared range they use different bins and cannot be read against each other.

**Scenario 3 — a grid of distributions**

```python
cols = ["person_age", "person_income", "credit_score", "loan_amnt"]

fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.flat, cols):
    ax.hist(loans[col], bins=40, color="steelblue", edgecolor="white")
    ax.set_title(col)
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.show()
```

**`axes.flat` flattens the 2×2 grid into a single sequence** so you can `zip` it with your column list — the reshape idea from Topic 8, doing real work.

## ✏️ Tasks

1. Draw a line chart of `y = x**2` for x from −10 to 10, with a title and both axis labels.
2. Draw a histogram of `credit_score` with 40 bins, labelled properly.
3. Draw two panels side by side: a histogram of age and a scatter of income against loan amount.
4. Draw a horizontal bar chart of `loan_intent` counts and save it as a PNG at 150 dpi.
5. Draw a 2×2 grid of histograms for four numeric columns, using `axes.flat`.

<details><summary>Solutions</summary>

```python
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
loans = pd.read_csv(BASE + "loan_data_10k.csv")

x = np.linspace(-10, 10, 200)                                          # 1
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(x, x ** 2)
ax.set_title("y = x squared"); ax.set_xlabel("x"); ax.set_ylabel("y")
ax.grid(alpha=.3); plt.show()

fig, ax = plt.subplots(figsize=(7, 4))                                 # 2
ax.hist(loans["credit_score"], bins=40, color="steelblue", edgecolor="white")
ax.set_title("Credit score"); ax.set_xlabel("score"); ax.set_ylabel("applicants")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(11, 4))                        # 3
axes[0].hist(loans["person_age"], bins=30); axes[0].set_title("Age")
axes[1].scatter(loans["person_income"], loans["loan_amnt"], alpha=.1, s=6)
axes[1].set_title("Income vs loan")
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(8, 4.5))                               # 4
counts = loans["loan_intent"].value_counts()
ax.barh(counts.index, counts.values)
ax.set_title("Why people borrow"); ax.set_xlabel("applications")
plt.tight_layout()
fig.savefig("intent.png", dpi=150, bbox_inches="tight")

cols = ["person_age", "person_income", "credit_score", "loan_amnt"]    # 5
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.flat, cols):
    ax.hist(loans[col], bins=40); ax.set_title(col); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
# axes.flat turns the 2x2 grid into one sequence you can zip with a list.
```
</details>

## ❓ MCQs

**Q1.** What do `fig` and `ax` represent?
- (a) Two charts  (b) `fig` is the whole picture; `ax` is one set of axes on it  (c) `ax` is the figure  (d) Nothing

**Q2.** Why prefer `fig, ax = plt.subplots()` over bare `plt.plot()`?
- (a) It is faster  (b) It scales to several panels; the `plt` style draws on a hidden current figure  (c) It uses less memory  (d) `plt.plot` is deprecated

**Q3.** What does `plt.tight_layout()` do?
- (a) Shrinks the figure  (b) Stops titles and labels overlapping  (c) Saves the file  (d) Adds a grid

**Q4.** What does `bbox_inches="tight"` do in `savefig`?
- (a) Compresses the file  (b) Trims the surrounding white border  (c) Sets the dpi  (d) Adds a border

**Q5.** Two histograms are being compared. What must you set?
- (a) The colour  (b) A shared `range` (and use `alpha`), so both use comparable bins  (c) The dpi  (d) Nothing

<details><summary>Answers</summary>

**A1 — (b).** The sheet of paper, and one drawing area on it.

**A2 — (b).** **The `ax` habit scales; the `plt` habit does not.**

**A3 — (b).** Nearly always worth calling.

**A4 — (b).** Along with `dpi=150`, it makes a chart report-ready.

**A5 — (b).** Different bins cannot be read against each other.
</details>

---

# 23. Seaborn

**Seaborn is Matplotlib with the statistics built in and the styling already done.**

🧠 **Analogy: a kit versus raw ingredients.** Matplotlib hands you flour, eggs and an oven — anything is possible, and you do every step. Seaborn hands you a cake mix designed for exactly the cake you asked for. **Seaborn is built on Matplotlib, so anything Seaborn draws you can then adjust with Matplotlib.**

## The difference in one comparison

```python
# illustrative: a syntax reference, not runnable as written.
# Matplotlib: you compute the groups yourself
means = df.groupby("category")["value"].mean()
ax.bar(means.index, means.values)

# Seaborn: you name the columns and it does the grouping AND the error bars
sns.barplot(data=df, x="category", y="value")
```

**Seaborn understands DataFrames.** You pass `data=`, then column *names* — no manual grouping.

## The plots worth knowing

| Function | Shows |
|---|---|
| `sns.histplot(data, x=)` | A distribution, optionally with a smooth `kde=True` curve |
| `sns.boxplot(data, x=, y=)` | Distribution per category, with outliers |
| `sns.violinplot(data, x=, y=)` | Like a box plot, but showing the full shape |
| `sns.scatterplot(data, x=, y=, hue=)` | Two columns, coloured by a third |
| `sns.countplot(data, x=)` | How many of each category |
| `sns.barplot(data, x=, y=)` | Average per category, with a confidence bar |
| `sns.heatmap(matrix, annot=True)` | A correlation matrix as colour |
| `sns.pairplot(df, hue=)` | Every column against every other |
| `sns.regplot(data, x=, y=)` | A scatter with a fitted trend line |

## `hue` — the argument that earns its keep

```python
# illustrative: a syntax reference, not runnable as written.
sns.scatterplot(data=df, x="income", y="loan_amnt", hue="loan_status")
```

**One extra word splits the whole chart by a third column.** Doing that in Matplotlib means filtering and plotting each group by hand.

## Styling

```python
# illustrative: a syntax reference, not runnable as written.
sns.set_theme(style="whitegrid")     # do this once at the top of your notebook
```

**Every chart afterwards — including Matplotlib ones — picks up the theme.**

## 📘 Examples

**Example 1 — a distribution with a smooth curve**

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
loans = pd.read_csv(BASE + "loan_data_10k.csv")
sample = loans.sample(1000, random_state=0)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(7, 4))
sns.histplot(data=sample, x="person_income", bins=40, kde=True)
plt.title("Income distribution")
plt.show()
```

**`kde=True` adds a smoothed outline** — useful for seeing shape, but remember it is an estimate, not the data.

**Example 2 — `hue` splitting by a third column**

```python
plt.figure(figsize=(7, 5))
sns.scatterplot(data=sample, x="person_income", y="loan_amnt",
                hue="loan_status", alpha=0.6)
plt.title("Income vs loan amount, by outcome")
plt.show()
```

**Example 3 — a box plot per category**

```python
plt.figure(figsize=(9, 4.5))
sns.boxplot(data=sample, x="loan_intent", y="loan_amnt")
plt.title("Loan amount by purpose")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.show()
```

**Example 4 — a correlation heatmap**

```python
numeric = loans.select_dtypes("number")
plt.figure(figsize=(9, 7))
sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm",
            center=0, square=True, cbar_kws={"shrink": .7})
plt.title("Correlation matrix")
plt.tight_layout()
plt.show()
```

> **`center=0` matters on a correlation heatmap.** It puts zero at the middle of the colour scale, so positive and negative relationships are visually opposite rather than arbitrary shades. **This is Topic 15's table, made readable at a glance.**

## 🌍 Scenarios

**Scenario 1 — the plot that answers "what relates to my target?"**

```python
plt.figure(figsize=(6, 6))
corr = loans.select_dtypes("number").corr()[["loan_status"]].sort_values(
    "loan_status", ascending=False)
sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm", center=0)
plt.title("Correlation with loan_status")
plt.tight_layout()
plt.show()
```

**One column of a heatmap is often more useful than the whole square** — it answers a specific question rather than showing everything.

**Scenario 2 — every column against every other**

```python
cols = ["person_age", "person_income", "loan_amnt", "loan_status"]
sns.pairplot(loans[cols].sample(300, random_state=1), hue="loan_status",
             plot_kws={"alpha": 0.5, "s": 20})
plt.show()
```

> ⚠️ **`pairplot` is expensive.** It draws n² panels, so always `.sample()` first and pick a handful of columns. On ten thousand rows and fourteen columns it will hang.

**Scenario 3 — a scatter with a trend line**

```python
ads = pd.read_csv(BASE + "regression/advertising.csv")

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, ch in zip(axes, ["TV", "Radio", "Newspaper"]):
    sns.regplot(data=ads, x=ch, y="Sales", ax=ax,
                scatter_kws={"alpha": .5, "s": 25},
                line_kws={"color": "crimson"})
    ax.set_title(f"{ch}  (r = {ads[ch].corr(ads['Sales']):.3f})")
plt.tight_layout()
plt.show()
```

**`regplot` fits and draws the trend line for you** — and passing `ax=ax` is how you place a Seaborn plot into a Matplotlib grid. **That is the two libraries working together.**

## ✏️ Tasks

1. Draw a `histplot` of `credit_score` with `kde=True`, on the `whitegrid` theme.
2. Draw a `scatterplot` of income against loan amount, coloured by `loan_status` with `hue`.
3. Draw a `boxplot` of `loan_amnt` by `loan_intent`, with the x labels rotated.
4. Draw a correlation `heatmap` of the numeric columns with `annot=True` and `center=0`.
5. Draw a `pairplot` of four columns on a 300-row sample, coloured by the target.

<details><summary>Solutions</summary>

```python
import seaborn as sns, matplotlib.pyplot as plt, pandas as pd
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"
loans = pd.read_csv(BASE + "loan_data_10k.csv")
sample = loans.sample(1000, random_state=0)
sns.set_theme(style="whitegrid")

plt.figure(figsize=(7, 4))                                             # 1
sns.histplot(data=sample, x="credit_score", bins=40, kde=True)
plt.title("Credit score"); plt.show()

plt.figure(figsize=(7, 5))                                             # 2
sns.scatterplot(data=sample, x="person_income", y="loan_amnt",
                hue="loan_status", alpha=.6)
plt.show()

plt.figure(figsize=(9, 4.5))                                           # 3
sns.boxplot(data=sample, x="loan_intent", y="loan_amnt")
plt.xticks(rotation=30, ha="right"); plt.tight_layout(); plt.show()

plt.figure(figsize=(9, 7))                                             # 4
sns.heatmap(loans.select_dtypes("number").corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, square=True)
plt.tight_layout(); plt.show()
# center=0 puts zero at the middle of the colour scale, so positive and
# negative relationships look opposite instead of being arbitrary shades.

cols = ["person_age", "person_income", "loan_amnt", "loan_status"]     # 5
sns.pairplot(loans[cols].sample(300, random_state=1), hue="loan_status",
             plot_kws={"alpha": .5, "s": 20})
plt.show()
# ALWAYS sample first - pairplot draws n^2 panels and will hang on
# ten thousand rows.
```
</details>

## ❓ MCQs

**Q1.** What is Seaborn built on top of?
- (a) NumPy  (b) Matplotlib  (c) Pandas  (d) Nothing

**Q2.** What does `hue="loan_status"` do?
- (a) Sets one colour  (b) Splits and colours the chart by that column  (c) Sets transparency  (d) Sorts the data

**Q3.** Which Seaborn function shows a correlation matrix as colour?
- (a) `histplot`  (b) `heatmap`  (c) `boxplot`  (d) `countplot`

**Q4.** Why pass `center=0` to a correlation heatmap?
- (a) It centres the title  (b) It puts zero at the middle of the colour scale, so positive and negative look opposite  (c) It removes the legend  (d) It is required

**Q5.** Why should you `.sample()` before calling `pairplot`?
- (a) For accuracy  (b) It draws n² panels and will hang on a large dataset  (c) It is required  (d) To sort the data

<details><summary>Answers</summary>

**A1 — (b) Matplotlib.** Which is why you can adjust any Seaborn chart with Matplotlib afterwards.

**A2 — (b).** **One extra word splits the whole chart by a third column.**

**A3 — (b) `heatmap`.** Topic 15's table, made readable at a glance.

**A4 — (b).** Otherwise the colours are arbitrary shades rather than meaningful opposites.

**A5 — (b).** On ten thousand rows and fourteen columns it will hang.
</details>

---

## ⭐ Checkpoint Problem 6 — The one-page data story

> **Uses everything in this session:** load, clean, analyse, correlate and visualise.

**The problem.** Take the loan dataset and produce a single figure of four panels that tells someone, without you speaking, what is in this data. Print a short written summary alongside it.

<details><summary>Solution</summary>

```python
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
BASE = "https://raw.githubusercontent.com/tech4alltraining/aiml/refs/heads/main/datasets/"

# --- load and clean (Topics 13, 17-21)
raw = pd.read_csv(BASE + "loan_data_10k.csv")
df = raw.copy()
df = df.drop_duplicates()
df = df[df["person_age"] <= 100]              # the impossible age
df = df.dropna()

print(f"rows {len(raw):,} -> {len(df):,} after cleaning")

# --- the figure (Topics 22-23)
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

sns.histplot(data=df, x="person_income", bins=50, ax=axes[0, 0])
axes[0, 0].set_title("1. Income is strongly right-skewed")
axes[0, 0].set_xlim(0, 250000)

sns.countplot(data=df, y="loan_intent",
              order=df["loan_intent"].value_counts().index, ax=axes[0, 1])
axes[0, 1].set_title("2. Why people borrow")

sns.boxplot(data=df, x="loan_status", y="loan_percent_income", ax=axes[1, 0])
axes[1, 0].set_title("3. Loan-to-income differs by outcome")

corr = df.select_dtypes("number").corr()[["loan_status"]].sort_values(
    "loan_status", ascending=False)
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            ax=axes[1, 1], cbar=False)
axes[1, 1].set_title("4. What relates to the outcome")

plt.tight_layout()
plt.show()

# --- the written summary
print(f"""
WHAT THIS DATA SAYS

  Rows            : {len(df):,} after cleaning
  Target balance  : {df['loan_status'].mean():.1%} approved
  Income          : median {df['person_income'].median():,.0f},
                    mean {df['person_income'].mean():,.0f}  <- mean >> median, so SKEWED
  Strongest signal: loan_percent_income, r = {df['loan_percent_income'].corr(df['loan_status']):.3f}

WHAT IT DOES NOT SAY
  Correlation is not causation. Panel 3 shows the two groups differ;
  it does not show that a high loan-to-income ratio CAUSED the outcome.
""")
```

**Why these four panels and not four others:**

1. **A distribution** — shows the shape of the most important numeric column, and the skew you would otherwise only find by comparing mean and median.
2. **A category count** — shows what the rows actually are.
3. **A comparison** — puts the strongest predictor against the target, so the relationship is visible rather than just a number.
4. **A correlation column** — ranks everything else, so a reader knows where to look next.

**Every panel answers a question a reader would ask.** That is what makes it a story rather than four charts.
</details>

**Make it harder:**

1. Add a fifth panel showing income by education level.
2. Save the figure at 150 dpi and check it is readable at full size.
3. Write the same summary for a different dataset without changing the structure.

---

# ✅ Before you move on

**NumPy**

- [ ] I know why an array beats a list, and what vectorisation means
- [ ] I can create arrays with `array`, `zeros`, `arange`, `linspace` and random
- [ ] I can index and slice in one and two dimensions
- [ ] I know `arange` excludes the stop and `linspace` includes it
- [ ] I know an array has one dtype, and that one float makes everything float
- [ ] **I know a slice is a view, and I `.copy()` before modifying one**
- [ ] I check `.shape` first when something fails
- [ ] I can reshape, and I know why some reshapes are impossible
- [ ] **I reach for a vectorised operation before writing a loop**

**Pandas — handling**

- [ ] I know the difference between a Series and a DataFrame
- [ ] I use `.loc` for labels and `.iloc` for positions
- [ ] I filter with `&` and `|`, each condition in brackets
- [ ] I can `groupby` and summarise
- [ ] I run `head`, `shape`, `info`, `describe` on every new dataset
- [ ] I compare mean and median to spot skew
- [ ] I know correlation is not causation, and that it misses curves
- [ ] I plot before I model

**Pandas — cleaning**

- [ ] I always clean a **copy**, never the raw data
- [ ] I can find and handle empty cells, and I prefer the median for skewed columns
- [ ] I can convert text dates with `errors="coerce"`
- [ ] I can spot impossible values, and I use `.loc` to change them
- [ ] I know not every duplicate is a mistake
- [ ] **I keep a cleaning log and can defend every choice in it**

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-02-numpy-pandas.ipynb) | Every example above, runnable |
| [Exercises & assignments](../exercises-assignments.md) | Longer graded work |
| [Session 3 — Visualisation & Preprocessing](session-03-eda-preprocessing.md) | Where this becomes model-ready data |
