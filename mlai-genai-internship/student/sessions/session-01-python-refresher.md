# Session 1 — Python Refresher

**Variables · Data Types · Operators · Conditionals & Loops · Functions · Collections · Strings · Classes and Objects**

> **This session assumes you have never written a line of Python.** Nothing here needs maths beyond arithmetic.

| | |
|---|---|
| **Notebook** | [session-01-python-refresher.ipynb](../notebooks/session-01-python-refresher.ipynb) |
| **Next** | [Session 2 — NumPy & Pandas](README.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Create variables and print them readably with f-strings
2. Name the four basic data types and convert between them
3. Use arithmetic, comparison, logical and membership operators
4. Write `if` / `elif` / `else` that behaves correctly
5. Loop over a list, a dictionary and a range
6. Choose the right collection — list, tuple, set or dict — and justify it
7. Slice and clean strings
8. Write a function that **returns** a value
9. Define a **class**, create objects from it, and explain why that is useful
10. Read an error message and find the actual problem

---

## The eight topics

| # | Topic | The one thing to take away |
|---|---|---|
| 1 | [Variables and printing](#1-variables-and-printing) | `=` means *put this in this box* |
| 2 | [Data types](#2-data-types) | `"21"` and `21` are different things |
| 3 | [Operators](#3-operators) | `=` assigns, `==` compares |
| 4 | [Conditionals and loops](#4-conditionals-and-loops) | `elif`, never a second `if` |
| 5 | [Collections](#5-collections) | List or dict, nine times out of ten |
| 6 | [Strings](#6-strings) | Strings are immutable — methods return a **new** one |
| 7 | [Functions](#7-functions) | `print` shows, `return` hands back |
| 8 | [Classes and objects](#8-classes-and-objects) | A class is a *blueprint*; an object is a *thing built from it* |

---

# 1. Variables and printing

🧠 **Analogy: labelled boxes.** A variable is a labelled box you put a value in. Write `age = 21` and you have a box labelled `age` containing 21.

**The `=` sign is not "equals".** It means *put this value in this box* — an instruction, not a statement of fact. And the box keeps only the **last** thing you put in it.

## 📘 Examples

**Example 1 — creating and reading variables**

```python
name = "Asha"
age = 21
height = 5.4
is_student = True

print(name)
print(age)

score = 50
score = 87          # the box now holds 87; the 50 is gone
print(score)
```

**Example 2 — f-strings, the readable way to print**

```python
name, marks = "Asha", 87.456

print("Student", name, "scored", marks)      # clumsy
print(f"Student {name} scored {marks}")      # f-string
print(f"Student {name} scored {marks:.1f}")  # :.1f = one decimal place
print(f"{name} needs {100 - marks:.1f} more") # maths inside the braces
```

**Example 3 — aligning output**

```python
students = {"Asha": 87.4, "Ravi": 74.15, "Meera": 95.5}

for name, mark in students.items():
    print(f"{name:<10}{mark:>7.1f}")   # <10 left-align, >7 right-align
```

## ✏️ Practice

1. Print the text `Learning Python` on one line.
2. Create `college` holding your college name and print it.
3. With `subject = "Machine Learning"` and `hours = 40`, print `I am studying Machine Learning for 40 hours` using an f-string.
4. Create `price = 249.789` and print it to **2 decimal places**.
5. Set `a = 10`, then change `a` to `25`, then print `a`. What appears, and why?

<details><summary>Solutions</summary>

```python
print("Learning Python")                                    # 1

college = "NIT Calicut"; print(college)                     # 2

subject, hours = "Machine Learning", 40                     # 3
print(f"I am studying {subject} for {hours} hours")

price = 249.789; print(f"{price:.2f}")                      # 4

a = 10; a = 25; print(a)    # 5 -> 25. A variable holds only the LAST value.
```
</details>

## ❓ MCQs

**Q1.** What does `=` do in Python?
- (a) Tests whether two things are equal
- (b) Puts a value into a variable
- (c) Prints a value
- (d) Creates a constant

**Q2.** After `x = 5` then `x = 9`, what does `print(x)` show?
- (a) `5`  (b) `9`  (c) `59`  (d) An error

**Q3.** What is the difference between `print(age)` and `print("age")`?
- (a) Nothing
- (b) The first shows the value; the second shows the literal text
- (c) The first shows the text; the second shows the value
- (d) The second is invalid

**Q4.** What does the `f` do in `print(f"Hello {name}")`?
- (a) Formats the output as a float
- (b) Makes it a function
- (c) Lets `{ }` be replaced by a variable's value
- (d) Forces the text to lowercase

**Q5.** `price = 3.14159`. What does `print(f"{price:.2f}")` show?
- (a) `3.14159`  (b) `3.14`  (c) `3.1`  (d) `3`

<details><summary>Answers</summary>

**A1 — (b).** It is an instruction: *put this value in this box*. `==` is the one that tests equality.

**A2 — (b) `9`.** A variable holds only the last value assigned to it.

**A3 — (b).** Anything inside quotes is treated as literal text.

**A4 — (c).** Without the `f`, Python prints the braces literally: `Hello {name}`.

**A5 — (b) `3.14`.** `:.2f` formats to two decimal places.
</details>

## 🎯 Tasks

**Task 1 — Your ID card.** Create variables for your name, age, course, semester and CGPA, then print a tidy card using f-string alignment (`:<10`, `:>7`).

**Task 2 — Hospital patient record.** A hospital checks in a patient: name `John Doe`, age 30, blood group `O+`, weight 72.5 kg, new patient `True`. Create a variable for each, print each with its `type()`, then print a one-line summary. Re-measure the weight as 71.8 and print the summary again.

---

# 2. Data types

🧠 **Analogy: what kind of thing is in the box?** A box can hold text, a whole number, a decimal, or a yes/no. Python cares which, because you can add two numbers but not two yes/nos.

| Type | Called | Example | What it is |
|---|---|---|---|
| `str` | string | `"Asha"` | Text. **Always in quotes** |
| `int` | integer | `21` | A whole number |
| `float` | float | `5.4` | A decimal number |
| `bool` | boolean | `True` | Yes or no. **Capital** T and F |

> ⚠️ **`"21"` and `21` are different things.** The first is text that happens to look like a number.

## 📘 Examples

**Example 1 — checking the type**

```python
for value in ["Asha", 21, 5.4, True]:
    print(f"{str(value):<8} {type(value).__name__}")
```

**Example 2 — the text-vs-number trap**

```python
print("21" + "5")     # "215"  - text is GLUED
print(21 + 5)         # 26     - numbers are ADDED

try:
    print("21" + 5)   # TypeError - Python will not guess
except TypeError as error:
    print("ERROR:", error)
```

**Example 3 — converting**

```python
print(int("21") + 5)        # 26
print(float("3.14") * 2)    # 6.28
print("Age: " + str(21))    # "Age: 21"

print(int(7.9))             # 7  <- CHOPS the decimal, does NOT round
print(round(7.9))           # 8

print(int(float("12.5")))   # 12 - two steps; int("12.5") alone fails

age = int(input("Your age: "))   # input() ALWAYS returns a string
```

## ✏️ Practice

1. Create one variable of each type and print each with its `type()`.
2. What is `"7" + "3"`? What is `7 + 3`? Explain the difference in a comment.
3. Convert the text `"45"` to a number and add 5.
4. Convert `9.99` to an integer. What happened to the `.99`?
5. Try `int("12.5")`. Read the error, then fix it so it works.

<details><summary>Solutions</summary>

```python
for v in ["Asha", 21, 5.4, True]:                       # 1
    print(f"{str(v):<8} {type(v).__name__}")

print("7" + "3")   # "73" - text is GLUED together      # 2
print(7 + 3)       # 10   - numbers are ADDED

print(int("45") + 5)         # 3 -> 50

print(int(9.99))             # 4 -> 9. int() CHOPS, it does not round
print(round(9.99))           #      -> 10

print(int(float("12.5")))    # 5 -> 12. Convert to float FIRST
```
</details>

## ❓ MCQs

**Q1.** What is `type(True)`?
- (a) `str`  (b) `int`  (c) `bool`  (d) `float`

**Q2.** `int(7.9)` returns 7, not 8. Why?
- (a) It rounds to the nearest even number
- (b) It chops off the decimal part
- (c) It is a bug
- (d) It rounds down only for odd numbers

**Q3.** Why does `int("12.5")` fail when `int("12")` works?
- (a) `"12.5"` is too long
- (b) `int()` on a string expects a whole number in text form
- (c) Quotes are not allowed with decimals
- (d) It does not fail

**Q4.** You read a number with `input()` and adding 5 gives a `TypeError`. Why?
- (a) `input()` always returns a **string**
- (b) `input()` returns a float
- (c) You cannot add to user input
- (d) `+` does not work on numbers

**Q5.** Which of these is **not** valid Python?
- (a) `x = True`  (b) `x = "True"`  (c) `x = true`  (d) `x = 1`

<details><summary>Answers</summary>

**A1 — (c) `bool`.**

**A2 — (b).** `int()` truncates towards zero. Use `round(7.9)` if you want 8.

**A3 — (b).** `"12.5"` has a decimal point, so it is not a valid integer string. Use `int(float("12.5"))`.

**A4 — (a).** The commonest beginner error in Python. Convert first: `int(input(...))`.

**A5 — (c).** Python needs a **capital** `True`. Lowercase `true` raises a `NameError`.
</details>

## 🎯 Tasks

**Task 1 — The type detective.** For `values = [42, "42", 42.0, True, "hello", 0, "", None, [1, 2]]`, print each value, its type, and whether `bool(value)` is True or False. **Which four are 'falsy', and why do you think those four were chosen?**

**Task 2 — Safe converter.** For `inputs = ["45", "3.14", "hello", "-7", "12.5", "", "1e3"]`, print either the converted number and its type, or a friendly message saying why it failed. Use `try` / `except ValueError`. **Do not let the program crash.**

---

# 3. Operators

Operators are the symbols that *do* things.

| Group | Operators | They give back |
|---|---|---|
| **Arithmetic** | `+  -  *  /  //  %  **` | A number |
| **Comparison** | `==  !=  >  <  >=  <=` | `True` or `False` |
| **Logical** | `and  or  not` | `True` or `False` |
| **Membership** | `in`, `not in` | `True` or `False` |

> ⚠️ **`=` and `==` are completely different.** `=` puts a value in a box. `==` asks "are these the same?" Mixing them up is the most common typo in programming.

## 📘 Examples

**Example 1 — arithmetic, including the three people forget**

```python
a, b = 17, 5

print(a / b)    # 3.4   normal division ALWAYS gives a float
print(a // b)   # 3     floor division: the whole part only
print(a % b)    # 2     modulus: the REMAINDER
print(a ** 2)   # 289   power

print(10 % 2)   # 0 -> 10 is EVEN
print(11 % 2)   # 1 -> 11 is ODD
```

**Example 2 — comparison and logic**

```python
marks, attendance = 78, 65

print(marks == 78)                          # True
print(marks >= 75 and attendance >= 75)     # False - BOTH must hold
print(marks >= 75 or attendance >= 75)      # True  - at least one
print(not (marks >= 40))                    # False
```

**Example 3 — membership works on lists and text**

```python
subjects = ["Maths", "Physics", "Python"]

print("Python" in subjects)      # True
print("History" in subjects)     # False
print("y" in "Python")           # True  - works on strings too
```

## ✏️ Practice

1. Print `23 // 4` and `23 % 4`. Explain each in a comment.
2. Write an expression that is `True` when a number `n` is even.
3. With `age = 20` and `has_id = True`, write **one** expression true only when both hold.
4. With `fruits = ["apple", "banana"]`, check whether `"mango"` is in it.
5. Predict `print(5 == 5.0)` before running it. Were you right?

<details><summary>Solutions</summary>

```python
print(23 // 4)   # 5 - how many whole 4s fit into 23          # 1
print(23 % 4)    # 3 - what is left over

n = 10; print(n % 2 == 0)                                     # 2

age, has_id = 20, True; print(age >= 18 and has_id)           # 3

print("mango" in ["apple", "banana"])   # False               # 4

print(5 == 5.0)   # True - Python compares the VALUE, not the type   # 5
```
</details>

## ❓ MCQs

**Q1.** What is the value of `17 // 5`?
- (a) `3.4`  (b) `3`  (c) `2`  (d) `4`

**Q2.** What does `%` most commonly get used for?
- (a) Percentages
- (b) Testing divisibility, e.g. `n % 2 == 0` for even
- (c) String formatting only
- (d) Rounding

**Q3.** Which expression is `True` only when **both** conditions hold?
- (a) `a or b`  (b) `a and b`  (c) `not a`  (d) `a in b`

**Q4.** What is the difference between `=` and `==`?
- (a) None; they are interchangeable
- (b) `=` compares, `==` assigns
- (c) `=` assigns, `==` compares
- (d) `==` is only for numbers

**Q5.** What does `"y" in "Python"` return?
- (a) `True`  (b) `False`  (c) `1`  (d) An error — `in` needs a list

<details><summary>Answers</summary>

**A1 — (b) `3`.** Floor division keeps only the whole part. `17 / 5` would give `3.4`.

**A2 — (b).** `n % 2 == 0` is the standard even test.

**A3 — (b) `and`.** `or` needs only one side to hold.

**A4 — (c).** Confusing them is the most common typo in programming.

**A5 — (a) `True`.** `in` works on strings as well as lists — it checks whether one string appears inside another.
</details>

## 🎯 Tasks

**Task 1 — Simple calculator.** Store two numbers, print all seven arithmetic results labelled, say which is larger, say whether both are even using `and`, and handle division by zero without crashing. Test with `(17, 5)`, `(10, 0)` and `(-8, 3)`. **Does negative floor division do what you expected?** Look up why.

**Task 2 — Eligibility checker.** A scholarship needs CGPA ≥ 8.0, attendance ≥ 75%, no backlogs, and family income < 500000. Write **one** expression combining all four, print a clear message — then print **which specific conditions failed**. That last part is the difference between a program and a useful program.

---

# 4. Conditionals and loops

🧠 **Analogy: a recipe with decisions and repetition.** Conditionals are "if the sauce is too thick, add water". Loops are "stir for 20 minutes".

> ⚠️ **Indentation is not decoration in Python — it is the syntax.** The indented lines are the ones that belong to the `if` or the `for`. Get it wrong and the program does something different, **often with no error at all**.

## 📘 Examples

**Example 1 — why `elif` and not a second `if`**

```python
m = 95

g = ""
if m >= 90:
    g = "A"
if m >= 75:        # this ALSO runs, and overwrites the A
    g = "B"
print(g)           # "B"  <- WRONG

if m >= 90:
    g = "A"
elif m >= 75:      # only runs if the first was False
    g = "B"
print(g)           # "A"  <- correct
```

**Example 2 — the three loop patterns**

```python
marks = [78, 92, 65, 88]
subjects = ["Maths", "Physics", "Python", "Stats"]

for mark in marks:                        # values
    print(mark)

for i, mark in enumerate(marks):          # position AND value
    print(f"Subject {i + 1}: {mark}")

for sub, mark in zip(subjects, marks):    # two lists together
    print(f"{sub:<10} {mark}")

for i in range(1, 6):                     # a sequence of numbers
    print(i)
```

**Example 3 — building a result, and `while`**

```python
passed = []
for mark in marks:
    if mark >= 60:
        passed.append(mark)
print(passed)

passed_short = [m for m in marks if m >= 60]   # the same, in one line
print(passed_short)

countdown = 3
while countdown > 0:
    print(countdown)
    countdown -= 1        # MUST change, or it loops forever
print("go!")
```

## ✏️ Practice

1. Loop over `[12, 7, 30, 45, 8]` printing `even` or `odd` next to each.
2. Find the largest number in `[23, 45, 12, 67, 34]` **without** using `max()`.
3. Print 1 to 10, but print `Fizz` instead of any multiple of 3.
4. Given `scores = [85, 42, 91, 58, 77]`, count how many are 60 or above.
5. Use a list comprehension to build the squares of `[1, 2, 3, 4, 5]`.

<details><summary>Solutions</summary>

```python
for n in [12, 7, 30, 45, 8]:                                  # 1
    print(n, "even" if n % 2 == 0 else "odd")

nums = [23, 45, 12, 67, 34]                                   # 2
largest = nums[0]
for n in nums:
    if n > largest:
        largest = n
print(largest)

for i in range(1, 11):                                        # 3
    print("Fizz" if i % 3 == 0 else i)

scores = [85, 42, 91, 58, 77]                                 # 4
print(sum(1 for s in scores if s >= 60))

print([n ** 2 for n in [1, 2, 3, 4, 5]])                      # 5
```
</details>

## ❓ MCQs

**Q1.** Why use `elif` instead of a second `if`?
- (a) It is shorter to type
- (b) `elif` only runs when the earlier conditions were False
- (c) `if` cannot be used twice
- (d) There is no difference

**Q2.** What does `enumerate(marks)` give you?
- (a) The values only
- (b) The positions only
- (c) The position and the value as a pair
- (d) A sorted copy

**Q3.** What is wrong with this?
```python
n = 5
while n > 0:
    print(n)
```
- (a) `while` needs an `else`
- (b) `n` never changes, so it loops forever
- (c) `print` is not allowed in a loop
- (d) Nothing

**Q4.** In Python, what defines which lines belong to an `if`?
- (a) Curly braces  (b) Semicolons  (c) Indentation  (d) The `end` keyword

**Q5.** What does `range(1, 6)` produce?
- (a) 1, 2, 3, 4, 5, 6  (b) 1, 2, 3, 4, 5  (c) 0, 1, 2, 3, 4, 5  (d) 1 to 6 inclusive of 0

<details><summary>Answers</summary>

**A1 — (b).** Two separate `if`s both run, so a mark of 95 gets set to "A" then overwritten with "B". This is one of the commonest logic bugs beginners write.

**A2 — (c).** It saves you creating and updating a counter by hand.

**A3 — (b).** A `while` loop must contain something that eventually makes its condition False — here, `n -= 1`.

**A4 — (c) Indentation.** In most other languages braces do this job. Wrong indentation in Python often produces no error, just silently different behaviour.

**A5 — (b) 1, 2, 3, 4, 5.** `range` includes the start and **excludes** the stop.
</details>

## 🎯 Tasks

**Task 1 — Hospital triage.** Given five patients with `name`, `age`, `temp` and `pain`, assign each a priority: `CRITICAL` if temp ≥ 39 **or** pain ≥ 8; `URGENT` if temp ≥ 38 **or** pain ≥ 5; otherwise `ROUTINE`. Then: anyone under 12 or over 70 moves **up one level**. Print them grouped, count each category, and answer *"who needs to be seen first?"*

**Task 2 — Number games.** FizzBuzz to 50 · print all primes below 50 · print the 1–10 times table as an aligned grid · repeatedly sum a number's digits until one remains (`9875` → `29` → `11` → `2`) · reverse a number **without** converting it to a string.

---

# 5. Collections

🧠 **Analogy: four kinds of container.**

| Type | Analogy | Ordered? | Changeable? | Duplicates? | Written as |
|---|---|---|---|---|---|
| **list** | A shelf of books | Yes | Yes | Yes | `[1, 2, 3]` |
| **tuple** | A sealed box of coordinates | Yes | **No** | Yes | `(1, 2)` |
| **set** | A bag of raffle tickets | No | Yes | **No** | `{1, 2, 3}` |
| **dict** | A labelled drawer | Yes | Yes | Keys unique | `{"a": 1}` |

**Nine times out of ten the answer is a list or a dict.**

## 📘 Examples

**Example 1 — lists, the workhorse**

```python
marks = [78, 92, 65, 88, 45]

print(marks[0], marks[-1], marks[1:3])   # 78 45 [92, 65]

marks.append(97)      # add to the end
marks[0] = 80         # change one item
marks.remove(45)      # remove by VALUE
marks.sort()          # sort in place

print(sum(marks), max(marks), min(marks))
print(f"{sum(marks) / len(marks):.1f}")
```

**Example 2 — dicts, looked up by name**

```python
student = {"name": "Asha", "age": 21, "cgpa": 8.45}

print(student["name"])
print(student.get("phone"))                # None, no crash
print(student.get("phone", "not given"))   # a default

student["semester"] = 6                    # add
student["cgpa"] = 8.60                     # change

for key, value in student.items():
    print(f"{key:<10}: {value}")
```

**Example 3 — sets and tuples earn their place**

```python
cities = ["Delhi", "Mumbai", "Delhi", "Chennai", "Mumbai"]
print(set(cities), len(set(cities)))       # distinct values, fast

python_students = {"Asha", "Ravi", "Meera"}
ml_students = {"Ravi", "Meera", "John"}
print(python_students & ml_students)       # in BOTH
print(python_students | ml_students)       # in EITHER
print(python_students - ml_students)       # Python only

location = (10.85, 76.27)                  # fixed: cannot be changed
try:
    location[0] = 99
except TypeError as error:
    print("Cannot change a tuple:", error)
```

## ✏️ Practice

1. `temps = [31, 28, 35, 29, 33]`. Print the average to 1 decimal place.
2. Add `30` to the end, then print the highest and lowest.
3. Create a dict for a book (title, author, year, pages) and print each key and value on its own line.
4. `colours = ["red","blue","red","green","blue"]`. How many **distinct** colours?
5. `a = {1,2,3,4}`, `b = {3,4,5}`. Print what is in both, in either, and in `a` only.

<details><summary>Solutions</summary>

```python
temps = [31, 28, 35, 29, 33]                                  # 1
print(f"{sum(temps) / len(temps):.1f}")

temps.append(30); print(max(temps), min(temps))               # 2

book = {"title": "Python Programming", "author": "John Smith",  # 3
        "year": 2020, "pages": 300}
for k, v in book.items():
    print(f"{k:<8}: {v}")

print(len(set(["red","blue","red","green","blue"])))          # 4 -> 3

a, b = {1,2,3,4}, {3,4,5}                                     # 5
print(a & b, a | b, a - b)
```
</details>

## ❓ MCQs

**Q1.** You need to store a student's name, age and CGPA together. Which collection?
- (a) list  (b) tuple  (c) dict  (d) set

**Q2.** Why does `{1, 2, 2, 3}` print as `{1, 2, 3}`?
- (a) Python sorted it  (b) Sets refuse duplicates  (c) An error was suppressed  (d) `2` is reserved

**Q3.** `student["phone"]` when the key is missing does what?
- (a) Returns `None`  (b) Returns `""`  (c) Raises a `KeyError`  (d) Adds the key

**Q4.** When would you deliberately choose a tuple over a list?
- (a) When you need it sorted
- (b) When the values must never change
- (c) When you need duplicates
- (d) Tuples are always faster, so always

**Q5.** What is the quickest way to count distinct values in a list?
- (a) `len(my_list)`  (b) `count(my_list)`  (c) `len(set(my_list))`  (d) `my_list.distinct()`

<details><summary>Answers</summary>

**A1 — (c) dict.** Each value has a meaning, so look it up by name (`student["cgpa"]`) rather than by position. Position-based access breaks the moment someone reorders the data.

**A2 — (b).** Sets silently refuse duplicates — no error, the second `2` is simply dropped.

**A3 — (c) `KeyError`**, and it crashes. Use `.get()` when a key might be absent.

**A4 — (b).** Coordinates, an RGB colour, a database row. Trying to modify a tuple raises an error, which turns a silent bug into an obvious one.

**A5 — (c) `len(set(my_list))`.**
</details>

## 🎯 Tasks

**Task 1 — Student score analyser.** Store `{"Alice": 85, "Bob": 78, "Charlie": 92, "Daisy": 61, "Evan": 78}`. Print each name and score; compute the average; print who scored **above** average; count distinct scores with a set and explain why it is fewer than five; put the top two names in a tuple; assign each a grade.

**Task 2 — Grocery billing.** Prices `{"milk": 45, "bread": 30, "eggs": 60, "butter": 80}`, cart `["milk","eggs","bread","milk","jam"]`. Use `in` to check each item, total only the ones you have prices for, **warn for each you skip**, find unique products with a set, print a receipt, and apply 5% off above 200.

---

# 6. Strings

A **string** is text. In ML you meet strings constantly: column names, categories, reviews, prompts.

🧠 **Analogy: a row of numbered letter tiles.** Each character has a position, starting at **0**.

> ⚠️ **Strings are immutable.** Every string method returns a **new** string — it never changes the original. Forgetting to capture the result is a very common bug.

## 📘 Examples

**Example 1 — indexing and slicing**

```python
text = "Machine Learning"
#       0123456789...

print(len(text))       # 16
print(text[0])         # M      first character
print(text[-1])        # g      last character
print(text[0:7])       # Machine   includes 0, EXCLUDES 7
print(text[8:])        # Learning  from 8 to the end
print(text[:7])        # Machine   from the start
```

**Example 2 — the methods you will actually use**

```python
messy = "  Machine Learning IS Fun!  "

print(repr(messy.strip()))              # removes surrounding spaces
print(messy.strip().lower())
print(messy.strip().upper())
print(messy.strip().replace("Fun", "Useful"))
print(messy.split())                    # -> a list of words
print(messy.strip().startswith("Machine"))
print(messy.lower().count("n"))

# THE IMMUTABILITY TRAP
t = "hello"
t.upper()          # returns "HELLO" - and throws it away
print(t)           # still "hello"
t = t.upper()      # capture it
print(t)           # "HELLO"
```

**Example 3 — splitting, joining, and cleaning categories**

```python
row = "Asha,21,Computer Science,8.45"
name, age, course, cgpa = row.split(",")
print(f"{name} is {age} and studies {course}")

words = "Python is a great language".split()
print("-".join(words))

# The real ML use: making inconsistent categories match
raw = ["  Delhi", "delhi ", "DELHI", "Delhi"]
cleaned = [c.strip().lower() for c in raw]
print(raw)
print(cleaned)          # now they all match
print(len(set(raw)), "->", len(set(cleaned)), "distinct cities")
```

## ✏️ Practice

1. `s = "Data Science"`. Print its length, first character and last character.
2. Print `"Data"` from that string using a slice.
3. Convert `"  HELLO world  "` to lowercase with no surrounding spaces.
4. Count the words in `"machine learning is fun and useful"`.
5. `row = "Ravi,19,Physics"`. Split it and print `Ravi is 19 and studies Physics`.

<details><summary>Solutions</summary>

```python
s = "Data Science"                                            # 1
print(len(s), s[0], s[-1])

print(s[0:4])                                                 # 2

print("  HELLO world  ".strip().lower())                      # 3

print(len("machine learning is fun and useful".split()))      # 4

name, age, subject = "Ravi,19,Physics".split(",")             # 5
print(f"{name} is {age} and studies {subject}")
```
</details>

## ❓ MCQs

**Q1.** In `text[0:7]`, is character 7 included?
- (a) Yes  (b) No  (c) Only if the string is long enough  (d) Only for lists

**Q2.** `"Hello".replace("l", "L")` — does the original string change?
- (a) Yes  (b) No, strings are immutable; it returns a new one  (c) Only if reassigned in place  (d) It raises an error

**Q3.** Why call `.strip().lower()` on category data before modelling?
- (a) To save memory
- (b) So `"  Delhi"`, `"delhi "` and `"DELHI"` are treated as one city
- (c) It is required by Pandas
- (d) To sort the values

**Q4.** What does `"a,b,c".split(",")` return?
- (a) `"abc"`  (b) `["a", "b", "c"]`  (c) `("a","b","c")`  (d) `{"a","b","c"}`

**Q5.** What does `text[-1]` give you?
- (a) An error  (b) The first character  (c) The last character  (d) The whole string reversed

<details><summary>Answers</summary>

**A1 — (b) No.** A slice includes the start and **excludes** the stop, so `text[0:7]` gives seven characters, 0 to 6.

**A2 — (b).** You must capture it: `text = text.replace(...)`.

**A3 — (b).** A model treating `Delhi`, `delhi` and `DELHI` as three cities learns three weaker patterns instead of one strong one.

**A4 — (b) a list.** `split` cuts a string up; `join` glues a list back together.

**A5 — (c) the last character.** Negative indexes count backwards from the end.
</details>

## 🎯 Tasks

**Task 1 — Text analyser.** For a paragraph, print: total characters (with and without spaces), total words, **unique** words case-insensitively, the five longest words, vowel counts, and the most common word. **Then test it on text with punctuation** — does your word count still work? Fix it if not.

**Task 2 — Clean the messy column.** `cities = ["  Delhi", "delhi ", "DELHI", "Mumbai", " mumbai", "Chennai", "chennai  ", "MUMBAI", "New  Delhi", "new delhi"]`. How many distinct values does Python see now? Clean them so identical cities match. How many now? Handle the **double space** in `"New  Delhi"` too. Print a count of each cleaned city.

---

# 7. Functions

🧠 **Analogy: a recipe card.** A function is instructions you write **once**, give a name, and reuse. You hand it ingredients (arguments) and it hands back a dish (the return value).

**Why it matters:** if you write the same six lines in four places and later find a bug, you must fix it four times. In a function you fix it once.

## 📘 Examples

**Example 1 — defining, calling, returning**

```python
def greet(name):
    """Print a greeting. This text is the docstring."""
    print(f"Hello, {name}!")

greet("Asha")           # `def` only DEFINES; calling is what runs it

def calculate_percentage(obtained, total=100):
    """Return the percentage, rounded to 2 places."""
    return round((obtained / total) * 100, 2)

result = calculate_percentage(87)
print(result)
print(calculate_percentage(87, total=120))

# print() SHOWS a value. return HANDS IT BACK so you can keep using it.
print(calculate_percentage(45) * 2)
```

**Example 2 — defaults and keyword arguments**

```python
def student_report(name, marks, passing=40, show_grade=True):
    status = "PASS" if marks >= passing else "FAIL"
    line = f"{name:<10} {marks:>3}  {status}"
    if show_grade:
        grade = "A" if marks >= 90 else "B" if marks >= 75 else "C"
        line += f"  grade {grade}"
    return line

print(student_report("Asha", 87))                      # positional
print(student_report(name="Ravi", marks=35))           # keyword - clearer
print(student_report("Meera", 92, show_grade=False))
```

**Example 3 — small functions that build on each other**

```python
def average(numbers):
    """Return the average, or 0 for an empty list."""
    if not numbers:
        return 0
    return sum(numbers) / len(numbers)

def grade_from(mark):
    if mark >= 90: return "A"
    if mark >= 75: return "B"
    if mark >= 60: return "C"
    return "F"

def class_report(students):
    all_avgs = []
    for name, marks in students.items():
        avg = average(marks)
        all_avgs.append(avg)
        print(f"  {name:<10} {avg:>6.1f}  {grade_from(avg)}")
    print(f"  {'CLASS':<10} {average(all_avgs):>6.1f}")

class_report({"Asha": [78, 92, 65], "Ravi": [88, 74, 91]})
```

## ✏️ Practice

1. Write `square(n)` returning `n` squared. Test on 7.
2. Write `is_even(n)` returning `True`/`False`. Test on 4 and 7.
3. Write `bigger(a, b)` returning the larger, **without** `max()`.
4. Write `greet(name, greeting="Hello")` with a default. Call it both ways.
5. Write `count_vowels(text)` returning how many vowels are in a string.

<details><summary>Solutions</summary>

```python
def square(n): return n ** 2                                  # 1
print(square(7))

def is_even(n): return n % 2 == 0                             # 2
print(is_even(4), is_even(7))

def bigger(a, b):                                             # 3
    if a > b:
        return a
    return b
print(bigger(10, 25))

def greet(name, greeting="Hello"):                            # 4
    return f"{greeting}, {name}!"
print(greet("Asha")); print(greet("Ravi", "Good morning"))

def count_vowels(text):                                       # 5
    return sum(1 for ch in text.lower() if ch in "aeiou")
print(count_vowels("Machine Learning"))
```
</details>

## ❓ MCQs

**Q1.** What is the difference between `print()` and `return` inside a function?
- (a) None
- (b) `print` shows a value; `return` hands it back so the caller can use it
- (c) `return` shows a value; `print` hands it back
- (d) `return` only works with numbers

**Q2.** Does `def greet(name):` run the function?
- (a) Yes, immediately  (b) No, it only defines it  (c) Only if it has a docstring  (d) Only inside a loop

**Q3.** In `def f(x, total=100):`, what is `total=100`?
- (a) A required argument  (b) A default argument  (c) A return value  (d) A global variable

**Q4.** Why prefer `student_report(name="Ravi", marks=35)` over `student_report("Ravi", 35)`?
- (a) It runs faster
- (b) It is self-documenting and order-independent
- (c) Positional arguments are deprecated
- (d) It uses less memory

**Q5.** You have written the same six lines in four places. What should you do?
- (a) Leave it; duplication is fine
- (b) Copy it into a fifth place for safety
- (c) Put it in a function, so a bug is fixed in one place
- (d) Turn it into a class

<details><summary>Answers</summary>

**A1 — (b).** A function that only prints cannot be built upon.

**A2 — (b).** `def` creates the recipe card. Nothing runs until you **call** it.

**A3 — (b) a default argument.** Callers may leave it out.

**A4 — (b).** With four or five arguments, positional calls become unreadable and easy to get wrong.

**A5 — (c).** Otherwise you must fix the bug four times — and you will miss one.
</details>

## 🎯 Tasks

**Task 1 — Marks toolkit.** Write `average`, `highest`, `lowest` (no `max`/`min`), `grade`, `passed(marks, cutoff=40)`, then `summary(name, marks)` that uses them all, then `class_summary(students)` for `{name: [marks]}`. Give every function a docstring.

**Task 2 — Temperature converter.** Write `c_to_f`, `f_to_c` (check `f_to_c(c_to_f(25)) == 25`), `describe(c)` returning Cold/Pleasant/Hot/Very Hot, a table of day/°C/°F/description using `zip`, `hottest_day(days, temps)` returning the **day name**, and `weekly_report`. **Every function should `return`, not print** — print only in the calling code.

---

# 8. Classes and objects

🧠 **Analogy: the blueprint and the houses.**

A **class** is a blueprint. It does not exist as a thing you can live in — it describes what every house of that kind will have: rooms, doors, a roof.

An **object** is a house built from that blueprint. You can build a hundred houses from one blueprint, and each has its **own** rooms with its own furniture in them.

```text
class Student:          <- the blueprint (written once)
      │
      ├── asha  = Student("Asha", 21)     <- an object
      ├── ravi  = Student("Ravi", 22)     <- another object
      └── meera = Student("Meera", 20)    <- another
```

**Why bother?** Once you have three or four related things — a name, marks, a method that grades them — passing them around as separate variables gets messy. A class keeps the **data and the things you do to it** in one place.

> You will meet this constantly in ML without writing it yourself. `LinearRegression()` is a class; `model = LinearRegression()` builds an object; `model.fit(...)` calls one of its methods. **Recognising the pattern matters more than writing your own.**

## 📘 Examples

**Example 1 — your first class**

```python
class Student:
    """A student with a name, an age, and some marks."""

    def __init__(self, name, age):
        # __init__ runs automatically when you build an object.
        # `self` is the object being built - "this particular student".
        self.name = name        # these are ATTRIBUTES:
        self.age = age          # data belonging to THIS object
        self.marks = []

    def add_mark(self, mark):
        """A METHOD: a function that belongs to the class."""
        self.marks.append(mark)

    def average(self):
        if not self.marks:
            return 0
        return sum(self.marks) / len(self.marks)


asha = Student("Asha", 21)        # build an object from the blueprint
ravi = Student("Ravi", 22)        # a second, completely separate object

asha.add_mark(78)
asha.add_mark(92)
ravi.add_mark(55)

print(asha.name, asha.average())   # Asha 85.0
print(ravi.name, ravi.average())   # Ravi 55.0
print(asha.marks, ravi.marks)      # separate lists - they do not share
```

**Example 2 — `__str__`, so printing an object is readable**

```python
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages

    def __str__(self):
        """What print() should show for this object."""
        return f"'{self.title}' by {self.author} ({self.pages}p)"

    def is_long(self):
        return self.pages > 300


b = Book("Python Basics", "Smith", 420)
print(b)              # 'Python Basics' by Smith (420p)
print(b.is_long())    # True

# Without __str__ you would get something like:
#   <__main__.Book object at 0x10f3c2d50>
```

**Example 3 — the pattern you already use in scikit-learn**

```python
class SimpleAverager:
    """A deliberately tiny model, to show the shape of the sklearn API."""

    def __init__(self):
        self.prediction = None      # nothing learned yet

    def fit(self, y):
        """Learn from data - here, just remember the average."""
        self.prediction = sum(y) / len(y)
        return self                 # sklearn returns self so you can chain

    def predict(self, n):
        """Use what was learned."""
        if self.prediction is None:
            raise ValueError("Call fit() before predict()")
        return [self.prediction] * n


model = SimpleAverager()            # 1. create, with settings
model.fit([10, 20, 30, 40])         # 2. learn from data
print(model.predict(3))             # 3. predict -> [25.0, 25.0, 25.0]

# This is EXACTLY the shape of every scikit-learn model you will meet:
#   model = LinearRegression()
#   model.fit(X_train, y_train)
#   model.predict(X_test)
```

## ✏️ Practice

1. Write a class `Dog` with `name` and `breed`, and a method `bark()` printing `"<name> says Woof!"`. Create two dogs and bark both.
2. Add a `Rectangle` class with `width` and `height`, and methods `area()` and `perimeter()`.
3. Add `__str__` to `Rectangle` so `print(r)` shows `Rectangle 3x4, area 12`.
4. Give `Rectangle` a method `is_square()` returning `True`/`False`.
5. Create a `BankAccount` class with `deposit(amount)`, `withdraw(amount)` and `balance`. **Refuse a withdrawal that would go below zero**, and print a message instead.

<details><summary>Solutions</summary>

```python
class Dog:                                                    # 1
    def __init__(self, name, breed):
        self.name = name
        self.breed = breed
    def bark(self):
        print(f"{self.name} says Woof!")

Dog("Rex", "Labrador").bark()
Dog("Bruno", "Beagle").bark()


class Rectangle:                                              # 2, 3, 4
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def area(self):
        return self.width * self.height
    def perimeter(self):
        return 2 * (self.width + self.height)
    def is_square(self):
        return self.width == self.height
    def __str__(self):
        return f"Rectangle {self.width}x{self.height}, area {self.area()}"

r = Rectangle(3, 4)
print(r, r.perimeter(), r.is_square())


class BankAccount:                                            # 5
    def __init__(self, owner, balance=0):
        self.owner = owner
        self.balance = balance
    def deposit(self, amount):
        self.balance += amount
        return self.balance
    def withdraw(self, amount):
        if amount > self.balance:
            print(f"Refused: balance is {self.balance}, cannot withdraw {amount}")
            return self.balance
        self.balance -= amount
        return self.balance

acc = BankAccount("Asha", 100)
acc.deposit(50)
acc.withdraw(500)      # refused
print(acc.balance)     # 150
```
</details>

## ❓ MCQs

**Q1.** What is the difference between a class and an object?
- (a) They are the same thing
- (b) A class is the blueprint; an object is a thing built from it
- (c) An object is the blueprint; a class is built from it
- (d) A class holds data; an object holds functions

**Q2.** What does `__init__` do?
- (a) Deletes the object
- (b) Runs automatically when an object is created, to set it up
- (c) Prints the object
- (d) Must be called manually

**Q3.** What is `self` inside a method?
- (a) The class itself
- (b) The particular object the method was called on
- (c) A required but meaningless keyword
- (d) The return value

**Q4.** `asha = Student("Asha", 21)` and `ravi = Student("Ravi", 22)`. If you do `asha.marks.append(78)`, what happens to `ravi.marks`?
- (a) It also gets 78  (b) Nothing — each object has its own attributes  (c) An error  (d) It is cleared

**Q5.** Why add a `__str__` method?
- (a) It is required for every class
- (b) So `print(obj)` shows something readable instead of a memory address
- (c) To make the class faster
- (d) To allow the object to be sorted

<details><summary>Answers</summary>

**A1 — (b).** One blueprint, many houses. `Student` is the class; `asha` and `ravi` are objects.

**A2 — (b).** It is the setup step: it gives the new object its starting attributes.

**A3 — (b).** It means "this particular object", which is how `asha.average()` knows to use *Asha's* marks and not Ravi's.

**A4 — (b) Nothing.** Each object has its own attributes. That separation is the whole point of building objects from a blueprint.

**A5 — (b).** Without it you get `<__main__.Book object at 0x10f3c2d50>`, which tells a reader nothing.
</details>

## 🎯 Tasks

**Task 1 — Library system.** Write a `Book` class (title, author, ISBN, `available`) and a `Library` class holding a list of books, with methods `add_book`, `find_by_author`, `borrow(isbn)`, `return_book(isbn)` and `available_count()`. `borrow` must **return a message** handling two problems: the book does not exist, and it is already borrowed. Add a `borrower` attribute and track who has what.

**Task 2 — Build a tiny model class.** Extend `SimpleAverager` into `SimpleMedianModel` with the same `fit` / `predict` shape but predicting the **median**. Then write a function that takes *any* model object and some data, calls `fit` and `predict` on it, and reports the result — **without knowing which class it was given.**

> That last part is the real lesson: because both classes share the same method names, code that uses them does not care which one it got. **That is why every scikit-learn model has `fit` and `predict`** — you can swap models without rewriting anything around them.

---

# ✅ Before you move on

- [ ] I can create variables and print them with f-strings
- [ ] I can explain why `"21" + 5` fails, and fix it
- [ ] I can use `%` to test whether a number is even
- [ ] I can index and slice a string or a list
- [ ] I can choose between a list, a dict, a set and a tuple, and say why
- [ ] I know why string methods must be **captured**, not just called
- [ ] I can write an `if`/`elif`/`else` chain that behaves correctly
- [ ] I can write a function that **returns** a value
- [ ] I can define a class, create two objects from it, and explain why their data is separate
- [ ] I can recognise `model = X()`, `model.fit(...)`, `model.predict(...)` as the class pattern
- [ ] I read the **last line** of an error first

Anything unticked? Redo that topic's five practice exercises.

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-01-python-refresher.ipynb) | Every example above, runnable |
| [Scenario worksheets](../notebooks/00c_python_scenarios.ipynb) | Ten real-world problems, task by task |
| [Python drills](https://github.com/tech4alltraining/aiml/blob/main/python-internship/Python_Exercise1.md) | Topic-by-topic exercises |
| [W3Schools Python](https://www.w3schools.com/python/) | Reference with a try-it editor |
