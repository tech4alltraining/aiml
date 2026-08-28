# Session 1 — Python Refresher

**Syntax · Comments · Variables · Data types · Numbers · Input & Output · Strings · Operators · Lists · Tuples & Sets · Dictionaries · Conditionals · Loops · Functions · Classes & Objects**

| | |
|---|---|
| **Notebook** | [session-01-python-refresher.ipynb](../notebooks/session-01-python-refresher.ipynb) |
| **Next** | [Session 2 — NumPy & Pandas](session-02-numpy-pandas.md) |
| **Stuck?** | [Troubleshooting](../troubleshooting.md) |

> **This session assumes you have never written a line of Python.** Every topic uses only what came before it. If something looks unfamiliar, it has not been taught yet — keep going in order and it will be.

---

## 🎯 Learning outcomes

By the end of this session you can:

1. Read and write correctly indented Python
2. Store values in variables and know what type each one is
3. Read input from a person and print formatted output
4. Slice, modify and format strings
5. Use every Python operator confidently
6. Store many values in lists, tuples, sets and dictionaries
7. Make decisions with `if`, `elif`, `else` and `match`
8. Repeat work with `for` and `while`
9. Package code into functions
10. Build your own types with classes and objects

---

## How this session is built

**Fifteen topics, in strict teaching order.** Each one introduces exactly one new idea and uses only what you already know.

| # | Topic | | # | Topic |
|---|---|---|---|---|
| 1 | [Python syntax](#1-python-syntax) | | 9 | [Lists](#9-lists) |
| 2 | [Comments](#2-comments) | | 10 | [Tuples & Sets](#10-tuples-and-sets) |
| 3 | [Variables](#3-variables) | | 11 | [Dictionaries](#11-dictionaries) |
| 4 | [Data types](#4-data-types) | | 12 | [Conditionals](#12-conditionals-if-elif-else-and-match) |
| 5 | [Numbers](#5-numbers) | | 13 | [Loops](#13-loops-for-and-while) |
| 6 | [Input & Output](#6-input-and-output) | | 14 | [Functions](#14-functions) |
| 7 | [Strings](#7-strings) | | 15 | [Classes & Objects](#15-classes-and-objects) |
| 8 | [Operators](#8-operators) | | | |

**Five checkpoint problems** sit between the topics. Each uses only what you have learned up to that point:

| After topic | Problem |
|---|---|
| 6 | [⭐ Rectangle area calculator](#-checkpoint-problem-1--rectangle-area) |
| 7 | [⭐ Initials maker](#-checkpoint-problem-2--initials-maker) |
| 12 | [⭐ Even or odd](#-checkpoint-problem-3--even-or-odd) |
| 13 | [⭐ Multiplication table](#-checkpoint-problem-4--multiplication-table) |
| 14 | [⭐ Prime number checker](#-checkpoint-problem-5--prime-number-checker) |

**Every topic has the same shape:**

```text
📘 Examples          3-4 short examples of the new idea
🌍 Scenarios         3 examples from real situations
✏️ Tasks             5 scenario-based tasks, with solutions
❓ MCQs              5 questions, with answers and why
```

---

# 1. Python syntax

**Syntax means the rules for how code must be written.** Python has fewer of them than most languages, and one that is unusual.

🧠 **Analogy: a recipe card.** The steps run top to bottom, one per line. Sub-steps are written indented underneath the step they belong to. **Python works exactly like this — and unlike most languages, the indentation is not decoration. It is the grammar.**

## The three rules

**1. One statement per line.**

```python
print("Hello")
print("World")
```

**2. Indentation shows what belongs inside what.** Use **four spaces**.

```python
if 5 > 3:
    print("This line is INSIDE the if")
    print("So is this one")
print("This line is OUTSIDE - back to the left margin")
```

> ⚠️ **The most common beginner error.** Get the indentation wrong and Python stops with `IndentationError`. It is not being fussy — the indentation is how it knows what belongs where.

**3. A colon `:` opens a block.** Whenever a line ends in `:`, the next line must be indented.

```python
if True:          # <- colon opens the block
    print("in")   # <- so this must be indented
```

## Printing

`print()` puts something on the screen. It is how you see what your program is doing.

```python
print("Hello, world!")
print(42)
print("Two", "things", "at", "once")
```

## 📘 Examples

**Example 1 — your first program**

```python
print("Hello, world!")
```

Output:
```text
Hello, world!
```

**Example 2 — several lines run in order, top to bottom**

```python
print("Line one")
print("Line two")
print("Line three")
```

Output:
```text
Line one
Line two
Line three
```

**Example 3 — quotes: single or double, both fine**

```python
print("Double quotes work")
print('Single quotes work too')
print("Use single 'inside' double when you need a quote mark")
```

**Example 4 — indentation shows grouping**

```python
print("Always runs")
if True:
    print("Runs because the condition is True")
    print("Also inside the block")
print("Always runs - we are back at the left margin")
```

## 🌍 Scenarios

**Scenario 1 — a shop's opening banner**

```python
print("=" * 30)
print("   WELCOME TO CAMPUS CAFE")
print("=" * 30)
print("Fresh coffee served all day")
```

**Scenario 2 — a receipt header, with blank lines for spacing**

```python
print("CAMPUS BOOKSTORE")
print()                        # an empty print() gives you a blank line
print("Receipt")
print("-" * 20)
```

**Scenario 3 — an error message a program might show**

```python
print("ERROR: could not open the file")
print("Check that the filename is spelled correctly.")
```

## ✏️ Tasks

1. Print your name, your course, and your college on three separate lines.
2. Print a box: a line of 20 `*`, then `* HELLO *`, then another line of 20 `*`.
3. Print a shop banner with a blank line between the shop name and the tagline.
4. Write four `print` lines that display a short poem.
5. Write a program that prints a menu: a title, a dashed line, and three items.

<details><summary>Solutions</summary>

```python
# 1
print("Priya Sharma")
print("B.Tech Computer Science")
print("Government Engineering College")

# 2
print("*" * 20)
print("* HELLO *")
print("*" * 20)

# 3
print("CAMPUS CAFE")
print()
print("Fresh coffee served all day")

# 4
print("The code runs top to bottom,")
print("one line at a time,")
print("indented where it belongs -")
print("and that is the whole rhyme.")

# 5
print("TODAY'S MENU")
print("-" * 20)
print("1. Tea")
print("2. Coffee")
print("3. Samosa")
```
</details>

## ❓ MCQs

**Q1.** What does Python use to show that code belongs inside a block?
- (a) Curly braces `{}`  (b) Indentation  (c) Semicolons  (d) Brackets `[]`

**Q2.** How many spaces are conventionally used for one level of indentation?
- (a) 1  (b) 2  (c) 4  (d) 8

**Q3.** What must follow a line that ends in a colon `:`?
- (a) A blank line  (b) An indented line  (c) A closing brace  (d) A semicolon

**Q4.** What does `print()` with nothing inside it do?
- (a) An error  (b) Prints a blank line  (c) Prints "None"  (d) Prints nothing at all

**Q5.** Which is valid Python?
- (a) `print("Hi");`  (b) `print "Hi"`  (c) `print("Hi")`  (d) `Print("Hi")`

<details><summary>Answers</summary>

**A1 — (b) Indentation.** This is Python's most distinctive rule. Most other languages use `{}`.

**A2 — (c) 4 spaces.** Any consistent amount works, but four is the universal convention.

**A3 — (b) An indented line.** The colon opens a block, and the block must be indented.

**A4 — (b) Prints a blank line.** Useful for spacing your output.

**A5 — (c).** `Print` with a capital P is a different name and gives `NameError`. The semicolon in (a) is legal but never used in Python.
</details>

---

# 2. Comments

**A comment is a note to a human. Python ignores it completely.**

🧠 **Analogy: notes in the margin of a textbook.** They help the reader; they are not part of the text itself.

```python
# This is a comment. Python skips this entire line.
print("This runs")          # A comment can also sit after code
```

## Why comments matter

**You are writing them for yourself in three months**, when you have forgotten why you did something.

```python
x = 0

# Bad comment - says WHAT the code already says
x = x + 1     # add 1 to x

# Good comment - says WHY
x = x + 1     # rows are numbered from 1, but Python counts from 0
```

> **Comment the *why*, not the *what*.** The code already says what it does.

## Multi-line notes

```python
"""
This is a triple-quoted string.
Python does not ignore it, but if nothing uses it,
it behaves like a multi-line comment.
Commonly used to describe what a file does.
"""
```

## 📘 Examples

**Example 1 — a comment on its own line**

```python
# Print a greeting for the user
print("Welcome!")
```

**Example 2 — a comment after code**

```python
print("Total: 250")      # 250 is in rupees
```

**Example 3 — commenting out a line while testing**

```python
print("This runs")
# print("This does not run - it is commented out")
print("This runs too")
```

**Example 4 — a header block describing a file**

```python
"""
fee_calculator.py
Works out the total fee including late charges.
Written for the college accounts office.
"""
print("Fee calculator starting...")
```

## 🌍 Scenarios

**Scenario 1 — explaining a value nobody would guess**

```python
# 18 is the minimum voting age in India
print("Minimum age: 18")
```

**Scenario 2 — leaving a note for a teammate**

```python
# TODO: this needs to handle the case where the shop is closed
print("Shop is open")
```

**Scenario 3 — temporarily disabling a line during debugging**

```python
print("Step 1 complete")
# print("Step 2 complete")     # switched off while we test step 3
print("Step 3 complete")
```

## ✏️ Tasks

1. Write a program that prints a shop name, with a comment above explaining what it does.
2. Add a comment explaining why a discount is 15%.
3. Write a three-line file header describing a program that calculates marks.
4. Write two `print` lines and comment out the second one.
5. Take this line and add a *why* comment: `print("Rounding to 2 decimal places")`.

<details><summary>Solutions</summary>

```python
# 1 - Display the shop's name banner at startup
print("CAMPUS CAFE")

# 2
# 15% is the standard student discount agreed with the college
print("Discount: 15%")

# 3
"""
marks_calculator.py
Adds up marks from five subjects and prints the percentage.
Used by the class teacher at the end of each term.
"""

# 4
print("This runs")
# print("This is switched off")

# 5
# Money is always shown to 2 decimal places on receipts
print("Rounding to 2 decimal places")
```
</details>

## ❓ MCQs

**Q1.** What symbol starts a single-line comment in Python?
- (a) `//`  (b) `#`  (c) `--`  (d) `/*`

**Q2.** What does Python do with a comment?
- (a) Prints it  (b) Ignores it completely  (c) Saves it to a file  (d) Runs it slowly

**Q3.** Which is the better comment?
- (a) `x = x + 1  # add one to x`
- (b) `x = x + 1  # rows are numbered from 1, Python counts from 0`
- (c) Both are equally good
- (d) Neither

**Q4.** Can a comment appear on the same line as code?
- (a) No  (b) Yes, after the code  (c) Yes, before the code  (d) Only in functions

**Q5.** What is the main reason to write comments?
- (a) To make the file longer  (b) To explain *why*, for whoever reads it next — often you  (c) Python requires them  (d) To slow the program down

<details><summary>Answers</summary>

**A1 — (b) `#`.**

**A2 — (b) Ignores it completely.** It has no effect on how the program runs.

**A3 — (b).** **Comment the *why*, not the *what*.** The code already says it adds one.

**A4 — (b) Yes, after the code.** Leave two spaces before the `#`.

**A5 — (b).** The next reader is usually you, three months later.
</details>

---

# 3. Variables

**A variable is a name for a value.** You give a value a name so you can use it later.

🧠 **Analogy: a labelled jar in a kitchen.** You write "sugar" on a jar and put sugar in it. Later you ask for "the sugar" rather than "the white granular thing in the third jar". **The label is the variable name; the contents are the value.**

```python
name = "Priya"
age = 20
```

**The `=` sign means "put this value into this name".** It is not the equals sign from mathematics.

```python
count = 5        # count now holds 5
count = 8        # count now holds 8 - the old value is gone
```

## Naming rules

| Rule | ✅ Good | ❌ Bad |
|---|---|---|
| Letters, digits, underscores only | `total_marks` | `total-marks` |
| Cannot start with a digit | `student1` | `1student` |
| Case matters | `age` and `Age` are **different** | — |
| No Python keywords | `class_name` | `class` |
| Describe the contents | `student_count` | `x` |

> **Use `snake_case`:** lowercase words joined by underscores. This is the Python convention and everyone follows it.

## Assigning several at once

```python
x, y, z = 1, 2, 3            # three names, three values
a = b = c = 0                # all three get 0
```

## 📘 Examples

**Example 1 — store and use**

```python
city = "Kochi"
print(city)
```

**Example 2 — a variable can be reassigned**

```python
score = 10
print(score)          # 10
score = 25
print(score)          # 25 - the old value is replaced
```

**Example 3 — use one variable to build another**

```python
price = 150
quantity = 3
total = price * quantity
print(total)          # 450
```

**Example 4 — several assignments in one line**

```python
length, width = 10, 4
print(length)         # 10
print(width)          # 4
```

## 🌍 Scenarios

**Scenario 1 — a student's details**

```python
student_name = "Arun"
roll_number = 42
course = "B.Tech"
print(student_name)
print(roll_number)
print(course)
```

**Scenario 2 — a shop calculating a bill**

```python
item_price = 250
item_count = 4
bill_total = item_price * item_count
print(bill_total)         # 1000
```

**Scenario 3 — swapping two values**

```python
first = "A"
second = "B"
first, second = second, first      # Python swaps them in one line
print(first)              # B
print(second)             # A
```

## ✏️ Tasks

1. Store your name, age and city in three variables and print each one.
2. A book costs 320 and you buy 3. Store both, compute the total, print it.
3. Store a temperature of 25, then change it to 30, printing before and after.
4. Store the length and width of a room and print both using one assignment line.
5. Store two students' marks and swap them, printing before and after.

<details><summary>Solutions</summary>

```python
# 1
name = "Priya"
age = 20
city = "Kochi"
print(name)
print(age)
print(city)

# 2
book_price = 320
book_count = 3
total = book_price * book_count
print(total)              # 960

# 3
temperature = 25
print(temperature)        # 25
temperature = 30
print(temperature)        # 30

# 4
length, width = 12, 9
print(length)
print(width)

# 5
arun_marks, priya_marks = 78, 91
print(arun_marks)         # 78
print(priya_marks)        # 91
arun_marks, priya_marks = priya_marks, arun_marks
print(arun_marks)         # 91
print(priya_marks)        # 78
```
</details>

## ❓ MCQs

**Q1.** What does `=` mean in Python?
- (a) "Is equal to"  (b) "Put this value into this name"  (c) "Compare these"  (d) "Print this"

**Q2.** Which variable name is invalid?
- (a) `student_1`  (b) `_total`  (c) `1student`  (d) `totalMarks`

**Q3.** After `x = 5` then `x = 9`, what does `x` hold?
- (a) 5  (b) 9  (c) 14  (d) Both

**Q4.** Are `age` and `Age` the same variable?
- (a) Yes  (b) No — case matters  (c) Only inside functions  (d) Only for numbers

**Q5.** What is the Python naming convention for variables?
- (a) `snake_case`  (b) `camelCase`  (c) `PascalCase`  (d) `SCREAMING_CASE`

<details><summary>Answers</summary>

**A1 — (b).** It is an instruction, not a statement of fact. The comparison operator is `==`, which you meet in Topic 8.

**A2 — (c) `1student`.** A name cannot start with a digit.

**A3 — (b) 9.** Reassigning replaces the old value entirely.

**A4 — (b) No.** Python is case-sensitive, and this catches beginners often.

**A5 — (a) `snake_case`.** Lowercase words joined by underscores.
</details>

---

# 4. Data types

**Every value in Python has a type.** The type decides what you can do with the value.

🧠 **Analogy: the difference between a number and a phone number.** Both are made of digits. You can add two numbers meaningfully; adding two phone numbers is nonsense. **The type is what tells Python which operations make sense.**

## The four types you need first

| Type | Name | Example | For |
|---|---|---|---|
| **int** | Integer | `42`, `-7`, `0` | Whole numbers |
| **float** | Floating point | `3.14`, `-0.5`, `2.0` | Numbers with decimals |
| **str** | String | `"hello"`, `'A'` | Text |
| **bool** | Boolean | `True`, `False` | Yes/no, on/off |

> ⚠️ **`True` and `False` are capitalised.** `true` gives a `NameError`.

## Checking a type

```python
print(type(42))          # <class 'int'>
print(type(3.14))        # <class 'float'>
print(type("hello"))     # <class 'str'>
print(type(True))        # <class 'bool'>
```

## Converting between types

```python
int("42")        # 42     - text to whole number
float("3.14")    # 3.14   - text to decimal
str(42)          # "42"   - number to text
int(3.9)         # 3      - CHOPS the decimal, does not round
bool(0)          # False  - 0 is False, every other number is True
```

> ⚠️ **`int(3.9)` gives `3`, not `4`.** It removes the decimal part rather than rounding. Use `round(3.9)` if you want 4.

## Other types you will meet later

```python
print(type([1, 2, 3]))        # <class 'list'>   -> Topic 9
print(type((1, 2)))           # <class 'tuple'>  -> Topic 10
print(type({"a": 1}))         # <class 'dict'>   -> Topic 11
```

**You do not need these yet.** They are here so the word `list` is not a surprise when it appears.

## 📘 Examples

**Example 1 — the four basic types**

```python
count = 10               # int
price = 99.50            # float
name = "Kochi"           # str
is_open = True           # bool

print(type(count))
print(type(price))
print(type(name))
print(type(is_open))
```

**Example 2 — text and number are genuinely different**

```python
print("5" + "3")         # 53   - joins two pieces of TEXT
print(5 + 3)             # 8    - adds two NUMBERS
```

**Example 3 — converting text to a number**

```python
text_value = "25"
number_value = int(text_value)
print(number_value + 5)      # 30
```

**Example 4 — `int()` chops, `round()` rounds**

```python
print(int(3.9))          # 3
print(round(3.9))        # 4
print(int(-2.7))         # -2
```

## 🌍 Scenarios

**Scenario 1 — a form field arrives as text and must become a number**

```python
age_from_form = "20"                # forms always give you TEXT
age = int(age_from_form)
print(age + 1)                      # 21 - next birthday
```

**Scenario 2 — building a message by converting a number to text**

```python
score = 87
message = "Your score is " + str(score)
print(message)                      # Your score is 87
```

**Scenario 3 — a yes/no flag**

```python
library_card_valid = True
print(library_card_valid)
print(type(library_card_valid))     # <class 'bool'>
```

## ✏️ Tasks

1. Create one variable of each of the four basic types and print each type.
2. Add the strings `"10"` and `"20"`. Then convert both to `int` and add again. Explain the difference.
3. Convert `4.99` to an integer. What do you get, and why is it not 5?
4. A form gives you the text `"150"`. Convert it and add 50.
5. Build the sentence `"I have 3 books"` from the number `3` using `str()`.

<details><summary>Solutions</summary>

```python
# 1
count, price, name, flag = 10, 99.5, "Kochi", True
print(type(count))       # int
print(type(price))       # float
print(type(name))        # str
print(type(flag))        # bool

# 2
print("10" + "20")       # 1020 - STRING joining, placed side by side
print(int("10") + int("20"))   # 30 - real addition
# "+" means JOIN for text and ADD for numbers. The type decides.

# 3
print(int(4.99))         # 4 - int() CHOPS the decimal part, it does not round
print(round(4.99))       # 5

# 4
value = int("150")
print(value + 50)        # 200

# 5
books = 3
print("I have " + str(books) + " books")
```
</details>

## ❓ MCQs

**Q1.** What is the type of `3.0`?
- (a) int  (b) float  (c) str  (d) bool

**Q2.** What does `"5" + "3"` produce?
- (a) `8`  (b) `"53"`  (c) An error  (d) `"8"`

**Q3.** What does `int(3.9)` return?
- (a) 4  (b) 3  (c) 3.9  (d) An error

**Q4.** Which is written correctly?
- (a) `true`  (b) `TRUE`  (c) `True`  (d) `"true"` for a boolean

**Q5.** Why must a value read from a form be converted before doing arithmetic?
- (a) It is always wrong  (b) It arrives as a string, and `+` joins strings instead of adding  (c) Forms are slow  (d) It does not need converting

<details><summary>Answers</summary>

**A1 — (b) float.** The decimal point makes it a float even though the value is a whole number.

**A2 — (b) `"53"`.** For strings, `+` joins.

**A3 — (b) 3.** It chops the decimal. Use `round()` to round.

**A4 — (c) `True`.** Capital T. Lowercase `true` is a `NameError`.

**A5 — (b).** **This is the single most common beginner bug** and you will meet it again in Topic 6.
</details>

---

# 5. Numbers

**Now that you know `int` and `float`, here is what you can do with them.**

## Arithmetic

| Operator | Meaning | Example | Result |
|---|---|---|---|
| `+` | Add | `7 + 3` | `10` |
| `-` | Subtract | `7 - 3` | `4` |
| `*` | Multiply | `7 * 3` | `21` |
| `/` | Divide | `7 / 3` | `2.333...` |
| `//` | Floor divide | `7 // 3` | `2` |
| `%` | Remainder | `7 % 3` | `1` |
| `**` | Power | `7 ** 3` | `343` |

> ⚠️ **`/` always gives a float**, even when it divides evenly: `10 / 2` is `5.0`, not `5`. Use `//` when you want a whole number.

## The two that beginners underuse

**`//` gives you how many whole times something fits.**

```python
print(17 // 5)      # 3   - five fits into seventeen 3 whole times
```

**`%` gives you what is left over.** This one is genuinely useful:

```python
print(17 % 5)       # 2   - 2 left over
print(10 % 2)       # 0   - EVEN numbers leave 0 when divided by 2
print(7 % 2)        # 1   - ODD numbers leave 1
```

> **`% 2` is how you test whether a number is even.** You will use this in Checkpoint Problem 3.

## Useful built-in functions

```python
print(round(3.567, 2))   # 3.57   - round to 2 decimal places
print(abs(-15))          # 15     - distance from zero
print(min(4, 9, 2))      # 2
print(max(4, 9, 2))      # 9
print(pow(2, 10))        # 1024   - same as 2 ** 10
```

## 📘 Examples

**Example 1 — the seven operators**

```python
a, b = 17, 5
print(a + b)      # 22
print(a - b)      # 12
print(a * b)      # 85
print(a / b)      # 3.4
print(a // b)     # 3
print(a % b)      # 2
print(a ** 2)     # 289
```

**Example 2 — the order of operations, and how to control it**

```python
print(2 + 3 * 4)        # 14 - multiplication happens first
print((2 + 3) * 4)      # 20 - brackets force the addition first
```

**Example 3 — rounding money**

```python
price = 19.999
print(round(price, 2))       # 20.0
print(round(2.34567, 3))     # 2.346
```

**Example 4 — testing even and odd with `%`**

```python
print(8 % 2)      # 0  -> even
print(9 % 2)      # 1  -> odd
```

## 🌍 Scenarios

**Scenario 1 — splitting a bill**

```python
total_bill = 1750
people = 4
each_pays = total_bill / people
print(round(each_pays, 2))       # 437.5
```

**Scenario 2 — converting minutes into hours and minutes**

```python
total_minutes = 200
hours = total_minutes // 60       # how many whole hours
minutes = total_minutes % 60      # what is left over
print(hours)                      # 3
print(minutes)                    # 20
```

**Scenario 3 — a percentage**

```python
marks_scored = 437
total_marks = 500
percentage = (marks_scored / total_marks) * 100
print(round(percentage, 2))       # 87.4
```

## ✏️ Tasks

1. A shop sells pens at 12 each. Compute and print the cost of 7 pens.
2. Convert 350 minutes into hours and remaining minutes using `//` and `%`.
3. A student scored 412 out of 550. Print the percentage rounded to 2 decimals.
4. Compute the area of a circle with radius 7 (use `3.14159`), rounded to 2 decimals.
5. A shopkeeper has 100 sweets and 7 children. How many does each get, and how many are left over?

<details><summary>Solutions</summary>

```python
# 1
pen_price, pen_count = 12, 7
print(pen_price * pen_count)              # 84

# 2
total_minutes = 350
print(total_minutes // 60)                # 5  hours
print(total_minutes % 60)                 # 50 minutes

# 3
percentage = (412 / 550) * 100
print(round(percentage, 2))               # 74.91

# 4
radius = 7
area = 3.14159 * radius ** 2
print(round(area, 2))                     # 153.94
# ** happens before *, so this is 3.14159 * (7**2). Brackets would make
# that clearer to a reader.

# 5
sweets, children = 100, 7
print(sweets // children)                 # 14 each
print(sweets % children)                  # 2  left over
```
</details>

## ❓ MCQs

**Q1.** What does `10 / 2` return?
- (a) `5`  (b) `5.0`  (c) `"5"`  (d) An error

**Q2.** What does `17 % 5` return?
- (a) 3  (b) 2  (c) 3.4  (d) 12

**Q3.** How do you test whether `n` is even?
- (a) `n / 2 == 0`  (b) `n % 2 == 0`  (c) `n // 2 == 0`  (d) `n ** 2 == 0`

**Q4.** What is `2 + 3 * 4`?
- (a) 20  (b) 14  (c) 24  (d) 9

**Q5.** What does `round(3.567, 2)` return?
- (a) 3.56  (b) 3.57  (c) 3.6  (d) 4

<details><summary>Answers</summary>

**A1 — (b) `5.0`.** `/` always produces a float. Use `//` for a whole number.

**A2 — (b) 2.** 5 goes into 17 three times with 2 left over.

**A3 — (b) `n % 2 == 0`.** An even number leaves no remainder.

**A4 — (b) 14.** Multiplication before addition. Use brackets when in doubt.

**A5 — (b) 3.57.** The second argument is the number of decimal places.
</details>

---

# 6. Input and Output

**A program that always prints the same thing is not very useful.** `input()` lets a person type something in.

🧠 **Analogy: a form at a counter.** You hand it over, the person writes on it, and hands it back. **Everything they write comes back as text — even if they wrote a number.**

```python
name = input("What is your name? ")
print("Hello, " + name)
```

## ⚠️ The rule that catches everyone

> **`input()` ALWAYS returns a string.** Even when the person typed `25`, you get the text `"25"`.

```python
age = input("Your age: ")        # person types 25
# print(age + 5)                 # TypeError! You cannot add a number to text.
                                 # Uncomment it and run to see the real error.
print(age + "5")                 # "255" - it JOINED them, as text
```

**The fix is `int()` or `float()`, from Topic 4:**

```python
age = int(input("Your age: "))       # now it is a real number
print(age + 5)                       # 30
```

| Reading | Use |
|---|---|
| Text | `input("prompt")` |
| Whole number | `int(input("prompt"))` |
| Decimal number | `float(input("prompt"))` |

## Printing nicely: f-strings

Joining with `+` needs `str()` everywhere and gets ugly. **An f-string is far cleaner:**

```python
name = "Priya"
age = 20

print("Name: " + name + ", Age: " + str(age))    # works, but clumsy
print(f"Name: {name}, Age: {age}")               # f-string - much better
```

**Put `f` before the quote, then write `{variable}` anywhere inside.**

```python
price = 19.5
count = 3
print(f"{count} items at {price} each = {count * price}")
```

**You can format numbers inside the braces:**

```python
value = 3.14159
print(f"{value:.2f}")            # 3.14   - 2 decimal places
print(f"{1234567:,}")            # 1,234,567 - thousands separators
```

## Controlling `print`

```python
print("A", "B", "C")                 # A B C      - space between by default
print("A", "B", "C", sep="-")        # A-B-C      - change the separator
print("no newline", end=" ")         # stays on the same line
print("continued here")
```

## 📘 Examples

**Example 1 — reading text**

```python
city = input("Which city do you live in? ")
print(f"{city} is a great city!")
```

**Example 2 — reading a number correctly**

```python
age = int(input("Enter your age: "))
print(f"Next year you will be {age + 1}")
```

**Example 3 — reading a decimal**

```python
price = float(input("Enter the price: "))
print(f"With 18% GST: {price * 1.18:.2f}")
```

**Example 4 — reading two values**

```python
length = float(input("Length in metres: "))
width = float(input("Width in metres: "))
print(f"Area: {length * width:.2f} square metres")
```

## 🌍 Scenarios

**Scenario 1 — a canteen order**

```python
item = input("What would you like? ")
quantity = int(input("How many? "))
price = float(input("Price per item: "))
print(f"{quantity} x {item} = {quantity * price:.2f}")
```

**Scenario 2 — a temperature converter**

```python
celsius = float(input("Temperature in Celsius: "))
fahrenheit = (celsius * 9 / 5) + 32
print(f"{celsius}C is {fahrenheit:.1f}F")
```

**Scenario 3 — a simple interest calculator**

```python
principal = float(input("Principal amount: "))
rate = float(input("Interest rate (%): "))
years = int(input("Number of years: "))

interest = (principal * rate * years) / 100
print(f"Interest: {interest:,.2f}")
print(f"Total to repay: {principal + interest:,.2f}")
```

## ✏️ Tasks

1. Ask for a name and a favourite colour, then print a sentence using both in an f-string.
2. Ask for two numbers and print their sum, difference and product.
3. Ask for a price and print it with 5% discount applied, to 2 decimals.
4. Ask for a distance in kilometres and print it in metres, with thousands separators.
5. Ask for marks in three subjects and print the total and the average to 2 decimals.

<details><summary>Solutions</summary>

```python
# 1
name = input("Your name: ")
colour = input("Favourite colour: ")
print(f"{name}'s favourite colour is {colour}.")

# 2
a = float(input("First number: "))
b = float(input("Second number: "))
print(f"Sum:        {a + b}")
print(f"Difference: {a - b}")
print(f"Product:    {a * b}")

# 3
price = float(input("Price: "))
print(f"After 5% discount: {price * 0.95:.2f}")

# 4
km = float(input("Distance in km: "))
print(f"{km} km = {km * 1000:,.0f} metres")

# 5
m1 = float(input("Marks in subject 1: "))
m2 = float(input("Marks in subject 2: "))
m3 = float(input("Marks in subject 3: "))
total = m1 + m2 + m3
print(f"Total:   {total:.2f}")
print(f"Average: {total / 3:.2f}")
```
</details>

## ❓ MCQs

**Q1.** What type does `input()` always return?
- (a) int  (b) float  (c) str  (d) It depends what was typed

**Q2.** A person types `25`. What does `age = input()` then `age + 5` do?
- (a) Gives 30  (b) Raises a `TypeError`  (c) Gives `"255"`  (d) Gives 25

**Q3.** How do you read a whole number from the user?
- (a) `input()`  (b) `int(input())`  (c) `number(input())`  (d) `input(int)`

**Q4.** What does `f"{3.14159:.2f}"` produce?
- (a) `"3.14159"`  (b) `"3.14"`  (c) `"3.142"`  (d) An error

**Q5.** What does `print("A", "B", sep="-")` print?
- (a) `A B`  (b) `A-B`  (c) `AB`  (d) `A - B`

<details><summary>Answers</summary>

**A1 — (c) str.** **Always**, no matter what was typed.

**A2 — (b) `TypeError`.** You cannot add an integer to a string. This is the most common beginner error in the whole session.

**A3 — (b) `int(input())`.** Read the text, then convert it.

**A4 — (b) `"3.14"`.** `:.2f` means two decimal places.

**A5 — (b) `A-B`.** `sep` sets what goes between the items.
</details>

---

## ⭐ Checkpoint Problem 1 — Rectangle area

> **Uses only:** variables, numbers, input, f-strings. Nothing you have not seen.

**The problem.** Ask the user for the length and width of a room in metres. Print the area and the perimeter, each to two decimal places.

**Try it yourself before opening the solution.**

<details><summary>Solution</summary>

```python
# Read the two measurements. float() because rooms are rarely whole metres.
length = float(input("Length in metres: "))
width = float(input("Width in metres: "))

area = length * width
perimeter = 2 * (length + width)

print(f"Area:      {area:.2f} square metres")
print(f"Perimeter: {perimeter:.2f} metres")
```

**Sample run:**
```text
Length in metres: 5.5
Width in metres: 4
Area:      22.00 square metres
Perimeter: 19.00 metres
```
</details>

**Make it harder:**

1. Also print the cost of flooring at 850 per square metre, with thousands separators.
2. Print the area in square feet as well (1 square metre = 10.7639 square feet).
3. Ask for the room's height too, and print the volume.

---

# 7. Strings

**A string is text.** You have used strings since Topic 1 — now here is everything they can do.

🧠 **Analogy: a row of numbered boxes, one letter per box.** You can look inside any box, take a run of boxes, or make a new row from an old one. **What you cannot do is change a box — Python strings never change. Every "modification" builds a new string.**

## Creating strings

```python
single = 'Hello'
double = "Hello"
multi = """This string
spans several lines."""
```

## Indexing: getting one character

**Positions start at 0**, not 1.

```python
text = "PYTHON"
#       012345      <- positions from the left
#      -654321      <- positions from the right

print(text[0])      # P   - the first character
print(text[3])      # H
print(text[-1])     # N   - the LAST character
print(text[-2])     # O   - second from the end
```

> **`text[-1]` for the last character** is a Python idiom you will use constantly.

## Slicing: getting a run of characters

**`text[start:stop]` — `start` is included, `stop` is not.**

```python
text = "PYTHON"
print(text[0:3])    # PYT   - positions 0, 1, 2  (NOT 3)
print(text[2:5])    # THO
print(text[:3])     # PYT   - from the beginning
print(text[3:])     # HON   - to the end
print(text[:])      # PYTHON - the whole thing
print(text[-3:])    # HON   - the last three
```

> ⚠️ **The stop position is excluded.** `text[0:3]` gives you three characters: 0, 1 and 2. This trips up everyone at first, and it is why `[0:3]` and `[3:]` fit together perfectly with no overlap.

## Modifying: a new string comes back

```python
name = "priya sharma"

print(name.upper())        # PRIYA SHARMA
print(name.lower())        # priya sharma
print(name.title())        # Priya Sharma
print(name.capitalize())   # Priya sharma

print(name)                # priya sharma  <- the ORIGINAL is unchanged
```

> **Strings are immutable.** Every method returns a *new* string. If you want to keep the result, assign it: `name = name.upper()`.

## The string methods you will actually use

| Method | Does | Example → result |
|---|---|---|
| `.upper()` `.lower()` | Change case | `"aB".upper()` → `"AB"` |
| `.title()` | Capitalise Each Word | `"john doe".title()` → `"John Doe"` |
| `.strip()` | Remove surrounding spaces | `"  hi  ".strip()` → `"hi"` |
| `.replace(a, b)` | Swap text | `"a-b".replace("-", " ")` → `"a b"` |
| `.split(sep)` | Break into a list | `"a,b".split(",")` → `["a", "b"]` |
| `.find(x)` | Position, or −1 | `"hello".find("l")` → `2` |
| `.count(x)` | How many times | `"hello".count("l")` → `2` |
| `.startswith(x)` | True/False | `"hello".startswith("he")` → `True` |
| `.endswith(x)` | True/False | `"a.csv".endswith(".csv")` → `True` |
| `len(s)` | Length (not a method) | `len("hello")` → `5` |

## Escape characters

**Some characters need a backslash to type.**

| Escape | Gives |
|---|---|
| `\n` | A new line |
| `\t` | A tab |
| `\"` | A double quote |
| `\'` | A single quote |
| `\\` | A backslash |

```python
print("Line one\nLine two")
print("Name:\tPriya")
print("She said \"hello\" to me")
print("Path: C:\\Users\\priya")
```

## Formatting

You met f-strings in Topic 6. Here is the fuller picture:

```python
name, score = "Priya", 87.456

print(f"{name} scored {score:.1f}")      # f-string   - use this
print("{} scored {:.1f}".format(name, score))   # .format() - older style
```

**Alignment, useful for tables:**

```python
print(f"{'Item':<10}{'Price':>8}")       # < left, > right, ^ centre
print(f"{'Tea':<10}{15:>8}")
print(f"{'Samosa':<10}{25:>8}")
```

## 📘 Examples

**Example 1 — indexing and length**

```python
word = "PYTHON"
print(word[0])         # P
print(word[-1])        # N
print(len(word))       # 6
```

**Example 2 — slicing**

```python
date = "2026-08-28"
print(date[:4])        # 2026   - the year
print(date[5:7])       # 08     - the month
print(date[-2:])       # 28     - the day
```

**Example 3 — cleaning up messy input**

```python
messy = "   Priya Sharma   "
clean = messy.strip().title()
print(f"[{clean}]")    # [Priya Sharma]
```

**Example 4 — escape characters in action**

```python
print("Name:\tPriya\nCity:\tKochi")
```

Output:
```text
Name:	Priya
City:	Kochi
```

## 🌍 Scenarios

**Scenario 1 — validating an email address, roughly**

```python
email = "priya@college.edu"
print(email.count("@"))              # 1   - should be exactly one
print(email.endswith(".edu"))        # True
print(email.find("@"))               # 5   - where the @ sits
```

**Scenario 2 — formatting a name badge from messy data**

```python
raw = "  aRUN kUMAR  "
badge = raw.strip().title()
print(f"| {badge:^20} |")            # centred in 20 characters
```

**Scenario 3 — pulling apart a filename**

```python
filename = "sales_report_2026.csv"
print(filename.endswith(".csv"))     # True
print(filename[:-4])                 # sales_report_2026 - drop the extension
print(filename.replace("_", " "))    # sales report 2026.csv
```

## ✏️ Tasks

1. Take the string `"MACHINE LEARNING"` and print the first word, the last word, and the total length.
2. A date arrives as `"28/08/2026"`. Print the day, month and year on separate lines using slicing.
3. Clean the messy name `"   rAJESH kUMAR  "` into `"Rajesh Kumar"`.
4. From `"student_marks_final.csv"`, print the filename without its extension and with underscores replaced by spaces.
5. Print a two-row table with a left-aligned item name in 12 characters and a right-aligned price in 8.

<details><summary>Solutions</summary>

```python
# 1
text = "MACHINE LEARNING"
print(text[:7])            # MACHINE
print(text[8:])            # LEARNING
print(len(text))           # 16

# 2
date = "28/08/2026"
print(date[:2])            # 28
print(date[3:5])           # 08
print(date[6:])            # 2026

# 3
messy = "   rAJESH kUMAR  "
print(messy.strip().title())     # Rajesh Kumar

# 4
f = "student_marks_final.csv"
print(f[:-4])                          # student_marks_final
print(f[:-4].replace("_", " "))        # student marks final

# 5
print(f"{'Item':<12}{'Price':>8}")
print(f"{'Tea':<12}{15:>8}")
print(f"{'Samosa':<12}{25:>8}")
```
</details>

## ❓ MCQs

**Q1.** What does `"PYTHON"[1]` return?
- (a) `P`  (b) `Y`  (c) `PY`  (d) An error

**Q2.** What does `"PYTHON"[0:3]` return?
- (a) `PYTH`  (b) `PYT`  (c) `YTH`  (d) `PYTHO`

**Q3.** What does `"PYTHON"[-1]` return?
- (a) `P`  (b) `N`  (c) An error  (d) `O`

**Q4.** After `name = "abc"` and `name.upper()`, what does `name` hold?
- (a) `"ABC"`  (b) `"abc"` — strings are immutable, so the method returned a new string  (c) An error  (d) `None`

**Q5.** What does `\n` do inside a string?
- (a) Prints a backslash and an n  (b) Starts a new line  (c) Inserts a tab  (d) Nothing

<details><summary>Answers</summary>

**A1 — (b) `Y`.** Positions start at 0, so position 1 is the second character.

**A2 — (b) `PYT`.** **The stop position is excluded** — you get 0, 1 and 2.

**A3 — (b) `N`.** Negative indexing counts from the end.

**A4 — (b) `"abc"`.** **Strings never change.** You must assign the result: `name = name.upper()`.

**A5 — (b) Starts a new line.**
</details>

---

## ⭐ Checkpoint Problem 2 — Initials maker

> **Uses only:** input, strings, slicing, methods, f-strings.

**The problem.** Ask for a full name like `"priya raj sharma"` and print the initials in the form `P.R.S.` — even if the user types extra spaces or the wrong case.

**Hint:** `.split()` with no argument breaks text on any whitespace and handles repeated spaces for you.

<details><summary>Solution</summary>

```python
full_name = input("Enter your full name: ")

# .strip() removes surrounding spaces; .split() breaks on whitespace and
# copes with double spaces on its own.
parts = full_name.strip().split()

# We do not know loops yet, so handle the common case of three names.
first_initial = parts[0][0].upper()
middle_initial = parts[1][0].upper()
last_initial = parts[2][0].upper()

print(f"{first_initial}.{middle_initial}.{last_initial}.")
```

**Sample run:**
```text
Enter your full name:    priya   raj sharma
P.R.S.
```

**Note `parts[0][0]`:** `parts[0]` is the first name, and `[0]` on that is its first letter. Two indexes, one after the other.
</details>

**Make it harder:**

1. Print the full surname after the initials: `P.R. Sharma`.
2. Print the name reversed: `Sharma, Priya Raj`.
3. Print how many characters the name has, ignoring spaces. *(Hint: `.replace(" ", "")` then `len()`.)*

> **This solution only works for exactly three names.** After Topic 13 you will be able to handle any number — come back and improve it.

---

# 8. Operators

You already use `+`, `-`, `*`, `/` from Topic 5. **Here is the complete set.**

## Arithmetic operators

Covered in Topic 5: `+` `-` `*` `/` `//` `%` `**`

## Comparison operators — these produce `True` or `False`

| Operator | Means | Example | Result |
|---|---|---|---|
| `==` | Equal to | `5 == 5` | `True` |
| `!=` | Not equal to | `5 != 3` | `True` |
| `>` | Greater than | `5 > 3` | `True` |
| `<` | Less than | `5 < 3` | `False` |
| `>=` | Greater or equal | `5 >= 5` | `True` |
| `<=` | Less or equal | `5 <= 3` | `False` |

> ⚠️ **`=` assigns. `==` compares.** One equals sign puts a value into a name; two ask a question. Mixing them up is the classic beginner bug.

```python
age = 20            # ASSIGN 20 to age
print(age == 20)    # ASK: is age 20?  ->  True
```

## Logical operators — combining conditions

| Operator | True when | Example |
|---|---|---|
| `and` | **Both** are true | `age > 18 and has_id` |
| `or` | **At least one** is true | `is_student or is_senior` |
| `not` | Flips it | `not is_closed` |

```python
age = 20
has_id = True

print(age >= 18 and has_id)     # True  - both conditions hold
print(age >= 60 or has_id)      # True  - the second one holds
print(not has_id)               # False - flipped
```

🧠 **Analogy for `and` versus `or`:** to enter an exam hall you need a hall ticket **and** an ID card — miss either and you are out. To get a discount you might need to be a student **or** a senior citizen — either one is enough.

## Assignment operators — shorthand

```python
total = 10
total += 5      # same as: total = total + 5   -> 15
total -= 3      # 12
total *= 2      # 24
total /= 4      # 6.0
total //= 2     # 3.0
total **= 2     # 9.0
```

## Membership operators — `in` and `not in`

```python
text = "machine learning"
print("learn" in text)          # True
print("java" in text)           # False
print("java" not in text)       # True
```

**`in` also works on lists, tuples, sets and dictionaries** — Topics 9 to 11.

## Identity operators — `is` and `is not`

```python
x = None
print(x is None)         # True   - the correct way to test for None
print(x is not None)     # False
```

> **Use `is` only for `None`, `True` and `False`.** For comparing values, use `==`.

## Operator precedence

**What happens first, when there are no brackets:**

```text
1.  **                    power
2.  * / // %              multiply, divide
3.  + -                   add, subtract
4.  == != > < >= <=       comparisons
5.  not
6.  and
7.  or                    lowest
```

> **When it is not obvious, use brackets.** They cost nothing and make your intent clear to the next reader.

## 📘 Examples

**Example 1 — comparisons return booleans**

```python
marks = 78
print(marks > 40)         # True
print(marks == 100)       # False
print(marks != 78)        # False
print(type(marks > 40))   # <class 'bool'>
```

**Example 2 — combining conditions**

```python
age = 22
is_member = True

print(age >= 18 and is_member)      # True
print(age < 18 or is_member)        # True
print(not is_member)                # False
```

**Example 3 — the `+=` shorthand**

```python
score = 0
score += 10
score += 25
print(score)              # 35
```

**Example 4 — membership**

```python
password = "Summer2026!"
print("!" in password)             # True
print("2026" in password)          # True
print(" " not in password)         # True  - no spaces, good
```

## 🌍 Scenarios

**Scenario 1 — checking eligibility to vote**

```python
age = 19
is_citizen = True
can_vote = age >= 18 and is_citizen
print(can_vote)              # True
```

**Scenario 2 — a discount rule**

```python
is_student = True
is_senior = False
total_spend = 1200

gets_discount = (is_student or is_senior) and total_spend > 1000
print(gets_discount)         # True
# The brackets matter: without them, "and" binds tighter than "or"
# and the meaning changes completely.
```

**Scenario 3 — a running total across a day**

```python
day_total = 0
day_total += 250        # morning sale
day_total += 175        # afternoon sale
day_total += 400        # evening sale
print(day_total)        # 825
```

## ✏️ Tasks

1. A student passes with 40 or more. Store marks of 38 and print whether they passed.
2. A cinema ticket is free for under-5s or over-65s. Store an age and print whether it is free.
3. A password must be longer than 8 characters **and** contain `"!"`. Test `"Summer2026!"`.
4. Start a bank balance at 5000, add 1200, subtract 800 using `+=` and `-=`, and print it.
5. Check whether the word `"error"` appears in the message `"System error: disk full"`.

<details><summary>Solutions</summary>

```python
# 1
marks = 38
print(marks >= 40)                          # False

# 2
age = 70
print(age < 5 or age > 65)                  # True

# 3
password = "Summer2026!"
print(len(password) > 8 and "!" in password)   # True

# 4
balance = 5000
balance += 1200
balance -= 800
print(balance)                              # 5400

# 5
message = "System error: disk full"
print("error" in message)                   # True
```
</details>

## ❓ MCQs

**Q1.** What is the difference between `=` and `==`?
- (a) None  (b) `=` assigns a value, `==` compares two values  (c) `=` compares, `==` assigns  (d) `==` is invalid

**Q2.** What does `5 >= 5` return?
- (a) `True`  (b) `False`  (c) `5`  (d) An error

**Q3.** When is `A and B` true?
- (a) When either is true  (b) When both are true  (c) Never  (d) When neither is true

**Q4.** What does `total += 5` mean?
- (a) `total = 5`  (b) `total = total + 5`  (c) `total == 5`  (d) `total = total * 5`

**Q5.** Which operator tests whether something appears inside a string?
- (a) `has`  (b) `in`  (c) `contains`  (d) `==`

<details><summary>Answers</summary>

**A1 — (b).** **The classic beginner bug.** One assigns, two ask a question.

**A2 — (a) `True`.** `>=` is "greater than *or equal to*", and 5 equals 5.

**A3 — (b) When both are true.** Hall ticket **and** ID card.

**A4 — (b).** Shorthand for adding to what is already there.

**A5 — (b) `in`.** It works on lists, tuples, sets and dictionaries too.
</details>

---

# 9. Lists

**A list holds many values in one variable, in order.**

🧠 **Analogy: a shopping list on paper.** Items sit in a definite order, you can read the third one, cross one out, add one at the bottom, or change one. **A Python list does all of that.**

```python
fruits = ["apple", "banana", "cherry"]
numbers = [10, 20, 30, 40]
mixed = ["Priya", 20, 87.5, True]        # a list can hold any types
empty = []
```

## Accessing items

**Exactly like string indexing from Topic 7 — positions start at 0.**

```python
fruits = ["apple", "banana", "cherry", "date"]
#           0         1         2        3
#          -4        -3        -2       -1

print(fruits[0])       # apple
print(fruits[2])       # cherry
print(fruits[-1])      # date      - the last item
print(len(fruits))     # 4
```

**Slicing works too, and the stop is still excluded:**

```python
fruits = ["apple", "banana", "cherry", "date"]

print(fruits[1:3])     # ['banana', 'cherry']
print(fruits[:2])      # ['apple', 'banana']
print(fruits[-2:])     # ['cherry', 'date']
```

## Changing items

> **Unlike strings, lists CAN be changed in place.** This is the single most important difference between them.

```python
fruits = ["apple", "banana", "cherry"]
fruits[1] = "blueberry"
print(fruits)          # ['apple', 'blueberry', 'cherry']
```

## Adding items

```python
fruits = ["apple", "banana"]

fruits.append("cherry")            # add to the END
print(fruits)                      # ['apple', 'banana', 'cherry']

fruits.insert(1, "apricot")        # insert AT position 1
print(fruits)                      # ['apple', 'apricot', 'banana', 'cherry']

fruits.extend(["date", "elderberry"])   # add several at once
print(fruits)
```

## Removing items

```python
fruits = ["apple", "banana", "cherry", "banana"]

fruits.remove("banana")     # removes the FIRST "banana" only
print(fruits)               # ['apple', 'cherry', 'banana']

last = fruits.pop()         # removes and RETURNS the last item
print(last)                 # banana
print(fruits)               # ['apple', 'cherry']

del fruits[0]               # delete by position
print(fruits)               # ['cherry']

fruits.clear()              # empty it completely
print(fruits)               # []
```

## Useful list operations

```python
marks = [78, 92, 45, 88, 61]

print(len(marks))          # 5
print(max(marks))          # 92
print(min(marks))          # 45
print(sum(marks))          # 364
print(sum(marks) / len(marks))   # 72.8   - the average
print(sorted(marks))       # [45, 61, 78, 88, 92] - a NEW sorted list
print(marks)               # unchanged
marks.sort()               # sorts the list ITSELF
print(marks)               # [45, 61, 78, 88, 92]
marks.reverse()
print(marks)               # [92, 88, 78, 61, 45]
print(78 in marks)         # True
print(marks.index(78))     # 2   - where it sits
print(marks.count(78))     # 1   - how many times
```

## 🔓 New syntax: looping through a list

**To do something with *every* item, you need a loop.** Here is your first one — Topic 13 covers loops fully.

```python
fruits = ["apple", "banana", "cherry"]

for fruit in fruits:
    print(fruit)
```

Output:
```text
apple
banana
cherry
```

**Read it as an English sentence:** *"for each fruit in fruits, print that fruit."*

- `fruit` is a name **you choose**. It holds one item at a time.
- The `:` opens a block, and the indented lines run once per item — exactly the indentation rule from Topic 1.

```python
marks = [78, 92, 45]

for m in marks:
    print(f"Mark: {m}, doubled: {m * 2}")
```

## 📘 Examples

**Example 1 — create and access**

```python
cities = ["Kochi", "Delhi", "Mumbai", "Chennai"]
print(cities[0])       # Kochi
print(cities[-1])      # Chennai
print(len(cities))     # 4
```

**Example 2 — change, add, remove**

```python
todo = ["wash", "cook"]
todo.append("study")           # ['wash', 'cook', 'study']
todo[0] = "clean"              # ['clean', 'cook', 'study']
todo.remove("cook")            # ['clean', 'study']
print(todo)
```

**Example 3 — statistics without writing any maths**

```python
marks = [78, 92, 45, 88, 61]
print(f"Highest: {max(marks)}")
print(f"Lowest:  {min(marks)}")
print(f"Total:   {sum(marks)}")
print(f"Average: {sum(marks) / len(marks):.2f}")
```

**Example 4 — looping**

```python
prices = [120, 85, 260]

for price in prices:
    print(f"Price with tax: {price * 1.18:.2f}")
```

## 🌍 Scenarios

**Scenario 1 — a canteen's daily sales**

```python
sales = [450, 380, 620, 510, 295]
print(f"Best day:  {max(sales)}")
print(f"Worst day: {min(sales)}")
print(f"Week total: {sum(sales)}")
print(f"Daily average: {sum(sales) / len(sales):.2f}")
```

**Scenario 2 — a waiting list where people join and leave**

```python
waiting = ["Arun", "Priya", "Ravi"]

waiting.append("Meera")            # Meera joins the back
served = waiting.pop(0)            # the front person is served
print(f"Now serving: {served}")    # Arun
print(f"Still waiting: {waiting}") # ['Priya', 'Ravi', 'Meera']
```

**Scenario 3 — a class result summary**

```python
students = ["Arun", "Priya", "Ravi"]
marks = [78, 92, 45]

print(f"Class average: {sum(marks) / len(marks):.2f}")

for name in students:
    print(f"Student: {name}")
```

## ✏️ Tasks

1. Make a list of five subjects. Print the first, the last, and how many there are.
2. Make a list of six marks. Print the highest, lowest, total and average to 2 decimals.
3. Start with `["milk", "eggs"]`. Add `"bread"` at the end, `"tea"` at the front, then remove `"eggs"`. Print the result.
4. From `[45, 92, 78, 61, 88]`, print a sorted copy **and** show the original is unchanged. Then sort the original itself.
5. Loop through a list of three prices and print each with 18% tax added.

<details><summary>Solutions</summary>

```python
# 1
subjects = ["Maths", "Physics", "Chemistry", "English", "Computer"]
print(subjects[0])            # Maths
print(subjects[-1])           # Computer
print(len(subjects))          # 5

# 2
marks = [78, 92, 45, 88, 61, 73]
print(f"Highest: {max(marks)}")
print(f"Lowest:  {min(marks)}")
print(f"Total:   {sum(marks)}")
print(f"Average: {sum(marks) / len(marks):.2f}")

# 3
shopping = ["milk", "eggs"]
shopping.append("bread")
shopping.insert(0, "tea")
shopping.remove("eggs")
print(shopping)               # ['tea', 'milk', 'bread']

# 4
nums = [45, 92, 78, 61, 88]
print(sorted(nums))           # [45, 61, 78, 88, 92]  - a NEW list
print(nums)                   # [45, 92, 78, 61, 88]  - unchanged
nums.sort()
print(nums)                   # [45, 61, 78, 88, 92]  - now changed

# 5
prices = [120, 85, 260]
for price in prices:
    print(f"{price} + tax = {price * 1.18:.2f}")
```
</details>

## ❓ MCQs

**Q1.** What does `["a", "b", "c"][1]` return?
- (a) `"a"`  (b) `"b"`  (c) `"c"`  (d) An error

**Q2.** What is the key difference between a list and a string?
- (a) Lists are faster  (b) A list can be changed in place; a string cannot  (c) Lists hold only numbers  (d) There is none

**Q3.** Which method adds an item to the **end** of a list?
- (a) `.insert()`  (b) `.append()`  (c) `.add()`  (d) `.extend()`

**Q4.** What is the difference between `sorted(marks)` and `marks.sort()`?
- (a) None  (b) `sorted()` returns a new list; `.sort()` changes the original  (c) `.sort()` returns a new list  (d) `sorted()` is invalid

**Q5.** In `for fruit in fruits:`, what is `fruit`?
- (a) The whole list  (b) A name you choose, holding one item at a time  (c) A built-in keyword  (d) The number of items

<details><summary>Answers</summary>

**A1 — (b) `"b"`.** Positions start at 0.

**A2 — (b).** **Lists are mutable; strings are immutable.** The most important difference between them.

**A3 — (b) `.append()`.** `.insert()` needs a position; `.extend()` adds several.

**A4 — (b).** `sorted()` leaves the original alone. Use it when you still need the original order.

**A5 — (b).** You pick the name — `fruit`, `item`, `x`. Choose something readable.
</details>

---

# 10. Tuples and Sets

**Two more ways to hold several values, each with one distinctive property.**

## Tuples — a list that cannot change

🧠 **Analogy: a printed certificate versus a whiteboard.** You can rewrite a whiteboard (a list). A printed certificate is fixed (a tuple). **You use a tuple to say "this must not change".**

```python
point = (3, 7)
colours = ("red", "green", "blue")
single = (5,)                # NOTE the comma - without it this is just 5
```

**Reading works exactly like a list:**

```python
colours = ("red", "green", "blue")
print(colours[0])        # red
print(colours[-1])       # blue
print(len(colours))      # 3
print("red" in colours)  # True

for c in colours:
    print(c)
```

**Changing does not:**

```python
colours = ("red", "green", "blue")

# colours[0] = "yellow"
# -> TypeError: 'tuple' object does not support item assignment
#    Uncomment it to see the error. That refusal is the POINT of a tuple.
print(colours)
```

**Unpacking is where tuples shine:**

```python
point = (3, 7)
x, y = point             # unpack into two names
print(x)                 # 3
print(y)                 # 7
```

> **You have already used this.** `first, second = second, first` in Topic 3 works because the right-hand side builds a tuple.

| Use a **tuple** when | Use a **list** when |
|---|---|
| The values must not change | You need to add, remove or edit |
| Coordinates, RGB colours, database rows | A shopping list, a queue, results |

## Sets — no duplicates, no order

🧠 **Analogy: a guest list where each name appears once.** Adding a name that is already there changes nothing. **And nobody is "third on the list" — a set has no order at all.**

```python
numbers = {1, 2, 3, 2, 1}
print(numbers)            # {1, 2, 3}  - duplicates gone automatically

empty_set = set()         # NOTE: {} makes an empty DICTIONARY, not a set
```

**No indexing** — there is no "first" item:

```python
numbers = {1, 2, 3}

# numbers[0]
# -> TypeError: 'set' object is not subscriptable
#    There is no "first" item in a set, so there is nothing to ask for.
print(numbers)
```

**Adding and removing:**

```python
s = {1, 2, 3}
s.add(4)                  # {1, 2, 3, 4}
s.add(2)                  # unchanged - 2 is already there
s.discard(1)              # {2, 3, 4}   - no error if missing
s.discard(99)             # still fine  - discard never complains
print(s)                  # {2, 3, 4}

# s.remove(99)            # KeyError - remove() DOES complain if it is missing.
                          # Use discard() when you do not care either way.
```

**Set maths, which is what makes them worth learning:**

```python
maths = {"Arun", "Priya", "Ravi"}
physics = {"Priya", "Ravi", "Meera"}

print(maths | physics)    # union        - in EITHER class
print(maths & physics)    # intersection - in BOTH classes
print(maths - physics)    # difference   - maths only
print(maths ^ physics)    # symmetric    - in one but not both
```

**The most common real use: removing duplicates.**

```python
marks = [78, 92, 78, 45, 92, 61]
print(list(set(marks)))       # duplicates removed
```

## 📘 Examples

**Example 1 — a tuple protects a value**

```python
screen_size = (1920, 1080)
width, height = screen_size
print(f"{width} x {height}")       # 1920 x 1080
```

**Example 2 — a set removes duplicates instantly**

```python
visitors = ["Arun", "Priya", "Arun", "Ravi", "Priya"]
unique = set(visitors)
print(len(visitors))       # 5 visits
print(len(unique))         # 3 people
```

**Example 3 — set operations answer real questions**

```python
monday = {"Arun", "Priya", "Ravi"}
tuesday = {"Priya", "Meera"}

print(monday & tuesday)    # {'Priya'}          - came both days
print(monday | tuesday)    # everyone who came at all
print(monday - tuesday)    # {'Arun', 'Ravi'}   - Monday only
```

**Example 4 — a tuple cannot be modified**

```python
config = ("localhost", 8080)
print(config[0])           # localhost
# config[0] = "example.com"    # TypeError - and that is the point
```

## 🌍 Scenarios

**Scenario 1 — a database row, which should not be edited by accident**

```python
student_record = (42, "Priya Sharma", "B.Tech", 8.7)
roll, name, course, cgpa = student_record
print(f"{name} ({roll}) is studying {course} with CGPA {cgpa}")
```

**Scenario 2 — counting unique visitors to a library**

```python
entries = ["A101", "B202", "A101", "C303", "B202", "A101"]
print(f"Total entries: {len(entries)}")           # 6
print(f"Unique people: {len(set(entries))}")      # 3
```

**Scenario 3 — which students are in both clubs**

```python
coding_club = {"Arun", "Priya", "Ravi", "Sneha"}
music_club = {"Priya", "Sneha", "Vikram"}

print(f"In both:      {coding_club & music_club}")
print(f"In either:    {coding_club | music_club}")
print(f"Coding only:  {coding_club - music_club}")
```

## ✏️ Tasks

1. Store a screen resolution as a tuple and unpack it into `width` and `height`.
2. Try to change an item in a tuple. What error do you get? Why is that useful?
3. From `[5, 3, 5, 8, 3, 1, 8]`, produce a list with duplicates removed.
4. Two shops sell overlapping items. Find items sold by both, by either, and by only the first.
5. Store a student record as a tuple of four fields and unpack it into four names.

<details><summary>Solutions</summary>

```python
# 1
resolution = (1366, 768)
width, height = resolution
print(f"{width} x {height}")

# 2
point = (3, 7)
# point[0] = 5
# -> TypeError: 'tuple' object does not support item assignment
# Useful because it makes accidental modification IMPOSSIBLE, not just
# discouraged. Use a tuple to say "this must not change".

# 3
nums = [5, 3, 5, 8, 3, 1, 8]
print(sorted(set(nums)))              # [1, 3, 5, 8]

# 4
shop_a = {"pen", "book", "bag", "ruler"}
shop_b = {"book", "bag", "bottle"}
print(shop_a & shop_b)                # {'book', 'bag'}
print(shop_a | shop_b)                # everything
print(shop_a - shop_b)                # {'pen', 'ruler'}

# 5
record = (42, "Priya Sharma", "B.Tech", 8.7)
roll, name, course, cgpa = record
print(f"{name} ({roll}), {course}, CGPA {cgpa}")
```
</details>

## ❓ MCQs

**Q1.** What is the defining property of a tuple?
- (a) It is faster  (b) It cannot be changed after creation  (c) It holds only numbers  (d) It has no order

**Q2.** What does `{1, 2, 3, 2, 1}` produce?
- (a) `{1, 2, 3, 2, 1}`  (b) `{1, 2, 3}`  (c) An error  (d) `[1, 2, 3]`

**Q3.** Why does `numbers[0]` fail on a set?
- (a) Sets are empty  (b) Sets have no order, so there is no "first" item  (c) Sets hold only strings  (d) It does not fail

**Q4.** What does `{"a", "b"} & {"b", "c"}` return?
- (a) `{"a", "b", "c"}`  (b) `{"b"}`  (c) `{"a", "c"}`  (d) `set()`

**Q5.** What does `(5)` create?
- (a) A one-item tuple  (b) Just the integer 5 — you need `(5,)` for a tuple  (c) A set  (d) A list

<details><summary>Answers</summary>

**A1 — (b) It cannot be changed.** That immutability is the whole point.

**A2 — (b) `{1, 2, 3}`.** Sets discard duplicates automatically.

**A3 — (b).** No order means no positions.

**A4 — (b) `{"b"}`.** `&` is intersection: what is in both.

**A5 — (b) Just the integer 5.** **The trailing comma is what makes a tuple:** `(5,)`.
</details>

---

# 11. Dictionaries

**A dictionary stores pairs: a key, and the value it points to.**

🧠 **Analogy: an actual dictionary.** You look up a *word* and get its *meaning*. You do not ask for "the 47th word" — you ask for a word by name. **A Python dictionary works the same way: you look things up by key, not by position.**

```python
student = {
    "name": "Priya",
    "age": 20,
    "course": "B.Tech",
    "cgpa": 8.7,
}
```

## Getting values

```python
student = {"name": "Priya", "age": 20, "course": "B.Tech", "cgpa": 8.7}

print(student["name"])          # Priya
print(student["cgpa"])          # 8.7

# print(student["email"])       # KeyError! There is no "email" key.
                                # Uncomment it to see the crash for yourself.
print(student.get("email"))     # None - safe, no crash
print(student.get("email", "not provided"))    # your own default
```

> **Use `.get()` when the key might be missing.** Square brackets crash; `.get()` hands you a default.

## Adding and changing

```python
student = {"name": "Priya", "age": 20, "course": "B.Tech"}

student["age"] = 21                     # change an existing key
student["email"] = "priya@college.edu"  # a NEW key is simply created
print(student)
```

## Removing

```python
student = {"name": "Priya", "email": "priya@college.edu", "cgpa": 8.7}

del student["email"]              # delete by key
removed = student.pop("cgpa")     # remove and RETURN the value
print(removed)                    # 8.7
student.clear()                   # empty it
print(student)                    # {}
```

## Looking through a dictionary

```python
student = {"name": "Priya", "age": 20, "course": "B.Tech"}

print(student.keys())      # dict_keys(['name', 'age', 'course'])
print(student.values())    # dict_values(['Priya', 20, 'B.Tech'])
print(student.items())     # pairs

print("name" in student)   # True  - `in` checks the KEYS
print(len(student))        # 3
```

**Looping — using the `for` you met in Topic 9:**

```python
student = {"name": "Priya", "age": 20, "course": "B.Tech"}

for key in student:
    print(key)                       # just the keys

for key, value in student.items():
    print(f"{key}: {value}")         # both at once
```

## Nesting

**Values can themselves be lists or dictionaries.** This is how real data looks.

```python
classroom = {
    "teacher": "Mrs Nair",
    "subject": "Physics",
    "students": ["Arun", "Priya", "Ravi"],
    "marks": {"Arun": 78, "Priya": 92, "Ravi": 45},
}

print(classroom["students"][0])       # Arun
print(classroom["marks"]["Priya"])    # 92
```

**Read it left to right:** `classroom["marks"]` gives you the inner dictionary, and `["Priya"]` looks up inside that.

## 📘 Examples

**Example 1 — create and read**

```python
prices = {"tea": 15, "coffee": 25, "samosa": 20}
print(prices["coffee"])          # 25
print(len(prices))               # 3
```

**Example 2 — add, change, remove**

```python
prices = {"tea": 15, "coffee": 25}
prices["juice"] = 40             # add
prices["tea"] = 18               # change
del prices["coffee"]             # remove
print(prices)                    # {'tea': 18, 'juice': 40}
```

**Example 3 — safe lookup**

```python
prices = {"tea": 15}
print(prices.get("coffee"))                  # None
print(prices.get("coffee", "not on menu"))   # not on menu
```

**Example 4 — looping over pairs**

```python
prices = {"tea": 15, "coffee": 25, "samosa": 20}

for item, price in prices.items():
    print(f"{item:<10} {price:>5}")
```

## 🌍 Scenarios

**Scenario 1 — a canteen price list**

```python
menu = {"tea": 15, "coffee": 25, "samosa": 20, "sandwich": 45}

print(f"Items on menu: {len(menu)}")
print(f"Cheapest: {min(menu.values())}")
print(f"Total if you buy one of each: {sum(menu.values())}")

for item, price in menu.items():
    print(f"{item:<12}{price:>5}")
```

**Scenario 2 — a student record with nested data**

```python
student = {
    "name": "Priya Sharma",
    "roll": 42,
    "marks": {"maths": 92, "physics": 78, "chemistry": 85},
}

print(student["name"])
print(student["marks"]["maths"])                    # 92
print(sum(student["marks"].values()) / 3)           # average
```

**Scenario 3 — counting things**

```python
# A dictionary is the natural way to count occurrences
attendance = {"Arun": 18, "Priya": 20, "Ravi": 12}

for name, days in attendance.items():
    percentage = (days / 20) * 100
    print(f"{name:<8} {days:>3} days  ({percentage:.0f}%)")
```

## ✏️ Tasks

1. Build a dictionary of three cities and their populations. Print one, and print how many entries there are.
2. Build a menu dictionary. Add an item, change a price, and delete an item.
3. Look up a key that does not exist, once with `[]` and once with `.get()`. Describe the difference.
4. Loop through a price dictionary and print each item and price in a neat aligned table.
5. Build a nested dictionary holding a student's name and their marks in three subjects, then print the average.

<details><summary>Solutions</summary>

```python
# 1
cities = {"Kochi": 677381, "Delhi": 16787941, "Mumbai": 12442373}
print(cities["Kochi"])
print(len(cities))                     # 3

# 2
menu = {"tea": 15, "coffee": 25}
menu["juice"] = 40
menu["tea"] = 18
del menu["coffee"]
print(menu)                            # {'tea': 18, 'juice': 40}

# 3
menu = {"tea": 15}
print(menu.get("pizza"))               # None - safe
# print(menu["pizza"])                 # KeyError - the program STOPS
# Square brackets crash on a missing key; .get() returns None (or a
# default you supply). Use .get() whenever the key might be absent.

# 4
prices = {"tea": 15, "coffee": 25, "samosa": 20}
for item, price in prices.items():
    print(f"{item:<12}{price:>5}")

# 5
student = {"name": "Priya", "marks": {"maths": 92, "physics": 78, "chemistry": 85}}
marks = student["marks"]
print(f"{student['name']} average: {sum(marks.values()) / len(marks):.2f}")
```
</details>

## ❓ MCQs

**Q1.** How do you get a value out of a dictionary?
- (a) By position, like `d[0]`  (b) By key, like `d["name"]`  (c) With `.find()`  (d) You cannot

**Q2.** What happens with `d["missing"]` when the key is not there?
- (a) Returns `None`  (b) Raises `KeyError`  (c) Returns `0`  (d) Creates the key

**Q3.** What does `d.get("missing", "default")` return?
- (a) `KeyError`  (b) `None`  (c) `"default"`  (d) `"missing"`

**Q4.** What does `"name" in student` check?
- (a) The values  (b) The keys  (c) Both  (d) The length

**Q5.** In `classroom["marks"]["Priya"]`, what is happening?
- (a) An error  (b) Look up `"marks"` to get an inner dictionary, then look up `"Priya"` in that  (c) Two keys at once  (d) A slice

<details><summary>Answers</summary>

**A1 — (b) By key.** Dictionaries have no positions — you look things up by name.

**A2 — (b) `KeyError`**, and the program stops.

**A3 — (c) `"default"`.** The second argument is your fallback.

**A4 — (b) The keys.** To search the values, use `in student.values()`.

**A5 — (b).** Read it left to right: each `[...]` looks up one level deeper.
</details>

---

# 12. Conditionals: if, elif, else and match

**A program that always does the same thing is a list of instructions. A program that decides is software.**

🧠 **Analogy: a signboard at a road junction.** *If the road is clear, go straight. Otherwise if there is a diversion, turn left. Otherwise, wait.* **Exactly one of those happens.**

## `if`

```python
age = 20

if age >= 18:
    print("You may vote")
```

**The condition is one of the comparisons from Topic 8**, producing `True` or `False`. If it is `True`, the indented block runs. If not, it is skipped entirely.

## `if` / `else`

```python
age = 15

if age >= 18:
    print("You may vote")
else:
    print("You are too young to vote")
```

## `if` / `elif` / `else`

**`elif` is short for "else if". Python checks each in order and runs the FIRST one that is true, then stops.**

```python
marks = 78

if marks >= 90:
    print("Grade A")
elif marks >= 75:
    print("Grade B")
elif marks >= 60:
    print("Grade C")
elif marks >= 40:
    print("Grade D")
else:
    print("Fail")
```

> ⚠️ **Order matters enormously.** If you put `marks >= 40` first, then *every* passing mark gets Grade D and the later branches never run. **Always write the strictest condition first.**

## Combining conditions

**Use `and`, `or`, `not` from Topic 8:**

```python
age = 20
has_id = True

if age >= 18 and has_id:
    print("Entry allowed")
else:
    print("Entry refused")
```

## Nesting

```python
age = 20
has_ticket = False

if age >= 18:
    if has_ticket:
        print("Enjoy the film")
    else:
        print("Please buy a ticket")
else:
    print("This film is 18+")
```

> **A nested `if` can often be flattened with `and`.** Prefer the flat version when it reads well — it is easier to follow.

## The one-line form

```python
age = 20
status = "adult" if age >= 18 else "minor"
print(status)                # adult
```

## `match` — Python's newest branching tool

**When you are comparing one value against several fixed options, `match` reads better than a chain of `elif`s.** *(Python 3.10 and later.)*

```python
day = "SAT"

match day:
    case "SAT" | "SUN":
        print("Weekend")
    case "MON":
        print("Start of the week")
    case _:                       # _ means "anything else"
        print("A working day")
```

| Use | When |
|---|---|
| `if` / `elif` | Ranges and complex conditions: `marks >= 75` |
| `match` | One value against fixed options: a menu choice, a status code |

## 📘 Examples

**Example 1 — a simple `if`**

```python
temperature = 38

if temperature > 37.5:
    print("You have a fever")
```

**Example 2 — `if` / `else`**

```python
password = "abc"

if len(password) >= 8:
    print("Password accepted")
else:
    print("Password too short")
```

**Example 3 — a grade ladder**

```python
marks = 84

if marks >= 90:
    grade = "A"
elif marks >= 75:
    grade = "B"
elif marks >= 60:
    grade = "C"
else:
    grade = "F"

print(f"Marks {marks} -> Grade {grade}")     # Grade B
```

**Example 4 — `match` on a menu choice**

```python
choice = "2"

match choice:
    case "1":
        print("You chose Tea")
    case "2":
        print("You chose Coffee")
    case "3":
        print("You chose Samosa")
    case _:
        print("Not on the menu")
```

## 🌍 Scenarios

**Scenario 1 — ticket pricing by age**

```python
age = int(input("Your age: "))

if age < 5:
    price = 0
elif age < 18:
    price = 80
elif age < 60:
    price = 150
else:
    price = 100

print(f"Ticket price: {price}")
```

**Scenario 2 — checking exam eligibility**

```python
attendance = 72
fees_paid = True

if attendance >= 75 and fees_paid:
    print("You may sit the exam")
elif not fees_paid:
    print("Please clear your fees first")
else:
    print(f"Attendance is {attendance}% - you need 75%")
```

**Scenario 3 — an ATM menu**

```python
option = input("1=Balance 2=Withdraw 3=Deposit: ")

match option:
    case "1":
        print("Your balance is 5,240")
    case "2":
        print("How much would you like to withdraw?")
    case "3":
        print("Please insert your cash")
    case _:
        print("Invalid option - please try again")
```

## ✏️ Tasks

1. Ask for a number and print whether it is positive, negative or zero.
2. Ask for marks and print the grade using the ladder A/B/C/D/Fail.
3. A shop gives 10% off above 1000 and 20% off above 5000. Ask for a bill and print the final amount.
4. Ask for a day name and use `match` to print whether it is a weekend or a working day.
5. Ask for age and whether the person has a licence, then print whether they may drive (18+ **and** licensed).

<details><summary>Solutions</summary>

```python
# 1
n = float(input("Enter a number: "))
if n > 0:
    print("Positive")
elif n < 0:
    print("Negative")
else:
    print("Zero")

# 2
marks = float(input("Enter marks: "))
if marks >= 90:
    print("Grade A")
elif marks >= 75:
    print("Grade B")
elif marks >= 60:
    print("Grade C")
elif marks >= 40:
    print("Grade D")
else:
    print("Fail")
# The strictest condition MUST come first, or every mark falls into
# the first branch that happens to match.

# 3
bill = float(input("Bill amount: "))
if bill > 5000:
    discount = 0.20
elif bill > 1000:
    discount = 0.10
else:
    discount = 0.0
print(f"You pay: {bill * (1 - discount):.2f}")

# 4
day = input("Day (MON-SUN): ").upper()
match day:
    case "SAT" | "SUN":
        print("Weekend")
    case "MON" | "TUE" | "WED" | "THU" | "FRI":
        print("Working day")
    case _:
        print("Not a valid day")

# 5
age = int(input("Your age: "))
has_licence = input("Do you have a licence (yes/no)? ").lower() == "yes"
if age >= 18 and has_licence:
    print("You may drive")
else:
    print("You may not drive")
```
</details>

## ❓ MCQs

**Q1.** What does `elif` mean?
- (a) "End if"  (b) "Else if" — checked only when the earlier conditions were false  (c) "Every if"  (d) "Exit if"

**Q2.** In a chain of `if`/`elif`, how many branches run?
- (a) All that are true  (b) Exactly one — the first that is true  (c) None  (d) The last one

**Q3.** Why must `marks >= 90` come before `marks >= 40`?
- (a) Style  (b) Python runs the FIRST true branch, so a loose condition first would catch everything  (c) It does not matter  (d) 90 is bigger

**Q4.** What does `case _:` mean in a `match`?
- (a) An error  (b) "Anything else" — the catch-all  (c) An empty case  (d) The first case

**Q5.** When is `match` a better fit than `if`/`elif`?
- (a) Always  (b) When comparing one value against several fixed options  (c) For ranges like `x > 10`  (d) Never

<details><summary>Answers</summary>

**A1 — (b) "Else if".** It is only checked if everything above it was false.

**A2 — (b) Exactly one.** Once a branch runs, Python skips the rest of the chain.

**A3 — (b).** **Order matters enormously.** Write the strictest condition first.

**A4 — (b).** The underscore is the catch-all, like `else`.

**A5 — (b).** For ranges and compound conditions, `if`/`elif` is still the right tool.
</details>

---

## ⭐ Checkpoint Problem 3 — Even or odd

> **Uses only:** input, numbers, `%`, conditionals.

**The problem.** Ask the user for a whole number and print whether it is even or odd. Then extend it: also say whether the number is positive, negative or zero.

<details><summary>Solution</summary>

```python
number = int(input("Enter a whole number: "))

# % gives the remainder. An even number divided by 2 leaves nothing over.
if number % 2 == 0:
    print(f"{number} is EVEN")
else:
    print(f"{number} is ODD")

if number > 0:
    print("It is positive")
elif number < 0:
    print("It is negative")
else:
    print("It is zero")
```

**Sample run:**
```text
Enter a whole number: -7
-7 is ODD
It is negative
```

**Why `% 2 == 0` and not `/ 2`?** `/` gives `3.5` for odd numbers and `4.0` for even ones — both are truthy, so it tells you nothing directly. `%` gives exactly `0` or `1`, which is precisely the question you are asking.
</details>

**Make it harder:**

1. Also print whether the number is divisible by 3, by 5, or by both.
2. Print the largest of three numbers the user enters.
3. Ask for a year and print whether it is a leap year. *(A leap year is divisible by 4, except centuries, unless divisible by 400.)*

<details><summary>Solution to the leap year extension</summary>

```python
year = int(input("Enter a year: "))

if year % 400 == 0:
    print(f"{year} IS a leap year")       # 2000 -> yes
elif year % 100 == 0:
    print(f"{year} is NOT a leap year")   # 1900 -> no
elif year % 4 == 0:
    print(f"{year} IS a leap year")       # 2024 -> yes
else:
    print(f"{year} is NOT a leap year")   # 2023 -> no
```

**Note the order.** Checking `% 400` first, then `% 100`, then `% 4` is what makes the rule work. Reverse them and 2000 gets the wrong answer.
</details>

---

# 13. Loops: for and while

**You met `for` briefly in Topic 9. Here is the full picture, plus `while`.**

🧠 **Analogy: two kinds of repeated instruction.**
- *"Greet every guest on this list"* — you know exactly how many. **That is a `for` loop.**
- *"Keep stirring until it thickens"* — you do not know how many stirs. **That is a `while` loop.**

## `for` — repeat once per item

```python
for fruit in ["apple", "banana", "cherry"]:
    print(fruit)
```

**Loop over a string, and you get its characters:**

```python
for letter in "PYTHON":
    print(letter)
```

## `range()` — counting

```python
for i in range(5):
    print(i)                # 0 1 2 3 4  - starts at 0, STOPS BEFORE 5

for i in range(1, 6):
    print(i)                # 1 2 3 4 5

for i in range(0, 10, 2):
    print(i)                # 0 2 4 6 8  - step of 2

for i in range(5, 0, -1):
    print(i)                # 5 4 3 2 1  - counting down
```

> ⚠️ **`range(5)` gives 0, 1, 2, 3, 4 — five numbers, not reaching 5.** The same "stop is excluded" rule as slicing in Topic 7.

## `enumerate()` — when you need the position too

```python
students = ["Arun", "Priya", "Ravi"]

for position, name in enumerate(students):
    print(f"{position}: {name}")           # 0: Arun, 1: Priya, ...

for position, name in enumerate(students, start=1):
    print(f"{position}. {name}")           # 1. Arun, 2. Priya, ...
```

## `while` — repeat while a condition holds

```python
count = 1

while count <= 5:
    print(count)
    count += 1              # WITHOUT this line, the loop never ends
```

> ⚠️ **Every `while` loop must contain something that eventually makes the condition false.** Forget it and your program hangs forever. If that happens, press **Ctrl+C** to stop it.

## `break` and `continue`

```python
for i in range(1, 10):
    if i == 5:
        break               # leave the loop entirely
    print(i)                # 1 2 3 4

for i in range(1, 6):
    if i == 3:
        continue            # skip just this one, carry on
    print(i)                # 1 2 4 5
```

## Building up a result

**The pattern you will use constantly:**

```python
marks = [78, 92, 45, 88]

total = 0                   # start with an empty result
for m in marks:
    total += m              # add each item to it
print(total)                # 303
```

```python
names = ["arun", "priya"]

capitalised = []            # start with an empty list
for n in names:
    capitalised.append(n.title())
print(capitalised)          # ['Arun', 'Priya']
```

## Nested loops

**A loop inside a loop. The inner one runs completely for each turn of the outer one.**

```python
for i in range(1, 4):
    for j in range(1, 4):
        print(f"{i} x {j} = {i * j}")
    print("---")
```

## 📘 Examples

**Example 1 — `for` over a list**

```python
for city in ["Kochi", "Delhi", "Mumbai"]:
    print(f"Welcome to {city}")
```

**Example 2 — `range` to count**

```python
for i in range(1, 6):
    print(f"{i} squared is {i ** 2}")
```

**Example 3 — `while` until a condition changes**

```python
balance = 1000
year = 0

while balance < 2000:
    balance = balance * 1.10       # 10% growth each year
    year += 1

print(f"It took {year} years to double")     # 8 years
```

**Example 4 — `enumerate` for a numbered list**

```python
tasks = ["wash", "cook", "study"]

for number, task in enumerate(tasks, start=1):
    print(f"{number}. {task}")
```

## 🌍 Scenarios

**Scenario 1 — a canteen bill**

```python
prices = [15, 25, 20, 45]

total = 0
for p in prices:
    total += p

print(f"Items: {len(prices)}")
print(f"Total: {total}")
print(f"With 5% tax: {total * 1.05:.2f}")
```

**Scenario 2 — counting how many students passed**

```python
marks = [78, 32, 92, 45, 38, 61]

passed = 0
failed = 0
for m in marks:
    if m >= 40:
        passed += 1
    else:
        failed += 1

print(f"Passed: {passed}, Failed: {failed}")     # Passed: 4, Failed: 2
```

**Scenario 3 — a password retry loop**

```python
correct = "python123"
attempts = 0

while attempts < 3:
    guess = input("Password: ")
    if guess == correct:
        print("Access granted")
        break
    attempts += 1
    print(f"Wrong. {3 - attempts} attempts remaining.")
else:
    # This runs only if the while ended WITHOUT a break
    print("Account locked")
```

## ✏️ Tasks

1. Print the numbers 1 to 10, each on its own line.
2. Given `[12, 45, 7, 89, 23]`, use a loop to find the total and the average without using `sum()`.
3. Print the 7 times table from 7×1 to 7×10.
4. Count how many vowels appear in a word the user types.
5. Use a `while` loop to keep asking for a number until the user enters one above 100.

<details><summary>Solutions</summary>

```python
# 1
for i in range(1, 11):
    print(i)

# 2
numbers = [12, 45, 7, 89, 23]
total = 0
for n in numbers:
    total += n
print(f"Total: {total}")                       # 176
print(f"Average: {total / len(numbers):.2f}")  # 35.20

# 3
for i in range(1, 11):
    print(f"7 x {i} = {7 * i}")

# 4
word = input("Enter a word: ").lower()
vowel_count = 0
for letter in word:
    if letter in "aeiou":
        vowel_count += 1
print(f"'{word}' has {vowel_count} vowels")

# 5
number = 0
while number <= 100:
    number = int(input("Enter a number above 100: "))
print(f"Thank you - {number} is above 100")
```
</details>

## ❓ MCQs

**Q1.** What does `range(5)` produce?
- (a) 1, 2, 3, 4, 5  (b) 0, 1, 2, 3, 4  (c) 0, 1, 2, 3, 4, 5  (d) 5

**Q2.** When should you use `while` instead of `for`?
- (a) Always  (b) When you do not know in advance how many repeats you need  (c) For lists  (d) Never

**Q3.** What does `break` do?
- (a) Skips one turn  (b) Leaves the loop entirely  (c) Restarts the loop  (d) Causes an error

**Q4.** What does `continue` do?
- (a) Leaves the loop  (b) Skips the rest of this turn and moves to the next  (c) Restarts from the beginning  (d) Nothing

**Q5.** What is wrong with `while count <= 5: print(count)`?
- (a) Nothing  (b) `count` never changes, so it loops forever  (c) The syntax is invalid  (d) `while` needs `range`

<details><summary>Answers</summary>

**A1 — (b) 0, 1, 2, 3, 4.** Five numbers, starting at 0 and stopping before 5.

**A2 — (b).** Known count → `for`. Unknown → `while`.

**A3 — (b) Leaves the loop entirely.**

**A4 — (b).** It skips to the next turn rather than leaving.

**A5 — (b) An infinite loop.** It needs `count += 1` inside. Press **Ctrl+C** to escape one.
</details>

---

## ⭐ Checkpoint Problem 4 — Multiplication table

> **Uses only:** input, loops, f-strings.

**The problem.** Ask the user for a number and print its multiplication table from 1 to 10, neatly aligned.

<details><summary>Solution</summary>

```python
number = int(input("Which table would you like? "))

print(f"\nMultiplication table for {number}")
print("-" * 22)

for i in range(1, 11):
    print(f"{number:>3} x {i:>2} = {number * i:>4}")
```

**Sample run:**
```text
Which table would you like? 7

Multiplication table for 7
----------------------
  7 x  1 =    7
  7 x  2 =   14
  7 x  3 =   21
...
  7 x 10 =   70
```

**The `:>3` and `:>4` right-align each number in a fixed width**, which is what makes the columns line up.
</details>

**Make it harder:**

1. Print tables for every number from 1 to 5 using a nested loop.
2. Ask for a start and an end, and print the table only for that range.
3. Print a full 10×10 grid, with each row on one line.

<details><summary>Solution to the 10×10 grid</summary>

```python
# The header row
print("    ", end="")
for col in range(1, 11):
    print(f"{col:>4}", end="")
print()
print("    " + "-" * 40)

# One line per row
for row in range(1, 11):
    print(f"{row:>3}|", end="")
    for col in range(1, 11):
        print(f"{row * col:>4}", end="")
    print()                     # end the line after the inner loop finishes
```

**`end=""` stops `print` from starting a new line**, so the inner loop builds one row across. The bare `print()` afterwards ends it.
</details>

---

# 14. Functions

**A function is a named piece of code you can run whenever you like.**

🧠 **Analogy: a recipe you have written down.** You write it once. After that you say *"make chai"* rather than repeating all the steps. **And if you improve the recipe, every future cup improves — you fix it in one place.**

## Defining and calling

```python
def greet():                     # define it
    print("Hello!")

greet()                          # call it - now it runs
greet()                          # call it again
```

**`def` starts the definition, the name follows, then `()` and a `:`. The body is indented** — the same rule as everything else.

## Parameters — giving the function information

```python
def greet(name):
    print(f"Hello, {name}!")

greet("Priya")            # Hello, Priya!
greet("Arun")             # Hello, Arun!
```

## `return` — getting an answer back

```python
def add(a, b):
    return a + b

result = add(3, 5)
print(result)             # 8
```

> ⚠️ **`print` shows a value; `return` hands it back.** A function that prints cannot have its answer used in a calculation. **A function that returns can.**

```python
def show(a, b):
    print(a + b)          # displays it, gives back None

def give(a, b):
    return a + b          # hands the value back

total = give(2, 3) * 10   # 50 - works
print(total)

print(show(2, 3))         # prints 5, then prints None - THAT is what it gave back
# total = show(2, 3) * 10 # TypeError: unsupported operand type(s) for *:
                          # 'NoneType' and 'int'.  Uncomment it to see.
```

## Default values

```python
def greet(name, greeting="Hello"):
    print(f"{greeting}, {name}!")

greet("Priya")                        # Hello, Priya!
greet("Priya", "Good morning")        # Good morning, Priya!
```

**Parameters with defaults must come after those without.**

## Keyword arguments

```python
def describe(name, age, city):
    print(f"{name}, {age}, from {city}")

describe("Priya", 20, "Kochi")                      # by position
describe(age=20, city="Kochi", name="Priya")        # by name - order free
```

## Returning several values

```python
def statistics(numbers):
    return min(numbers), max(numbers), sum(numbers) / len(numbers)

low, high, average = statistics([78, 92, 45])       # a tuple, unpacked
print(f"{low}, {high}, {average:.2f}")
```

**This works because of tuple unpacking from Topic 10.**

## `*args` and `**kwargs` — any number of arguments

```python
def total(*numbers):             # collects them into a TUPLE
    return sum(numbers)

print(total(1, 2))               # 3
print(total(1, 2, 3, 4, 5))      # 15

def show_details(**details):     # collects them into a DICTIONARY
    for key, value in details.items():
        print(f"{key}: {value}")

show_details(name="Priya", age=20, city="Kochi")
```

## Scope — where a variable lives

```python
def my_function():
    inside = "I only exist in here"
    print(inside)

my_function()
# print(inside)          # NameError - it does not exist out here
```

> **A variable created inside a function is invisible outside it.** This is a feature: it stops functions accidentally interfering with each other.

## Docstrings

```python
def calculate_area(length, width):
    """Return the area of a rectangle in square metres."""
    return length * width

print(calculate_area.__doc__)
```

## Lambda — a tiny one-line function

```python
square = lambda x: x ** 2
print(square(5))          # 25

# The same thing, written normally - usually clearer
def square(x):
    return x ** 2
```

**Use `lambda` only for very short throwaway functions**, typically passed to `sorted()` or similar.

## 📘 Examples

**Example 1 — no parameters**

```python
def print_banner():
    print("=" * 30)
    print("  CAMPUS CAFE")
    print("=" * 30)

print_banner()
```

**Example 2 — parameters and a return**

```python
def area_of_rectangle(length, width):
    return length * width

print(area_of_rectangle(5, 3))      # 15
print(area_of_rectangle(2.5, 4))    # 10.0
```

**Example 3 — a default value**

```python
def apply_discount(price, percent=10):
    return price * (1 - percent / 100)

print(apply_discount(1000))         # 900.0  - uses the default
print(apply_discount(1000, 25))     # 750.0
```

**Example 4 — returning several values**

```python
def analyse(numbers):
    return min(numbers), max(numbers), sum(numbers) / len(numbers)

low, high, avg = analyse([45, 78, 92, 61])
print(f"Low {low}, High {high}, Average {avg:.2f}")
```

## 🌍 Scenarios

**Scenario 1 — a reusable grade calculator**

```python
def grade_for(marks):
    """Return the letter grade for a mark out of 100."""
    if marks >= 90:
        return "A"
    elif marks >= 75:
        return "B"
    elif marks >= 60:
        return "C"
    elif marks >= 40:
        return "D"
    return "F"

for m in [95, 82, 67, 45, 30]:
    print(f"{m:>3} -> {grade_for(m)}")
```

**Scenario 2 — a bill calculator used everywhere in a shop**

```python
def final_price(amount, tax_percent=18, discount_percent=0):
    """Apply a discount, then tax, and return the amount to pay."""
    after_discount = amount * (1 - discount_percent / 100)
    return after_discount * (1 + tax_percent / 100)

print(f"{final_price(1000):.2f}")                        # 1180.00
print(f"{final_price(1000, discount_percent=10):.2f}")   # 1062.00
```

**Scenario 3 — cleaning up messy names**

```python
def clean_name(raw):
    """Strip surrounding spaces and fix the capitalisation."""
    return raw.strip().title()

messy = ["  priya sharma ", "ARUN KUMAR", "  ravi  "]
for name in messy:
    print(f"[{clean_name(name)}]")
```

## ✏️ Tasks

1. Write `celsius_to_fahrenheit(c)` that returns the converted temperature. Test it on three values.
2. Write `is_even(n)` that returns `True` or `False`, and use it on a list of numbers.
3. Write `calculate_bill(price, quantity, tax=18)` returning the total. Call it with and without the tax argument.
4. Write `name_stats(full_name)` returning the number of words, the number of letters, and the initials.
5. Write `count_vowels(text)` that returns how many vowels a piece of text has.

<details><summary>Solutions</summary>

```python
# 1
def celsius_to_fahrenheit(c):
    """Convert Celsius to Fahrenheit."""
    return (c * 9 / 5) + 32

for t in [0, 37, 100]:
    print(f"{t}C = {celsius_to_fahrenheit(t):.1f}F")

# 2
def is_even(n):
    return n % 2 == 0

for n in [4, 7, 10, 13]:
    print(f"{n}: {is_even(n)}")

# 3
def calculate_bill(price, quantity, tax=18):
    return price * quantity * (1 + tax / 100)

print(f"{calculate_bill(150, 3):.2f}")            # 531.00
print(f"{calculate_bill(150, 3, tax=5):.2f}")     # 472.50

# 4
def name_stats(full_name):
    parts = full_name.strip().split()
    letters = len(full_name.replace(" ", ""))
    initials = ""
    for p in parts:
        initials += p[0].upper() + "."
    return len(parts), letters, initials

words, letters, initials = name_stats("priya raj sharma")
print(f"{words} words, {letters} letters, initials {initials}")

# 5
def count_vowels(text):
    count = 0
    for letter in text.lower():
        if letter in "aeiou":
            count += 1
    return count

print(count_vowels("Machine Learning"))           # 6
```
</details>

## ❓ MCQs

**Q1.** What keyword defines a function?
- (a) `function`  (b) `def`  (c) `func`  (d) `define`

**Q2.** What is the difference between `print` and `return`?
- (a) None  (b) `print` displays a value; `return` hands it back so it can be used  (c) `return` displays it  (d) `print` is faster

**Q3.** What does a function return if it has no `return` statement?
- (a) `0`  (b) `None`  (c) An error  (d) An empty string

**Q4.** In `def greet(name, greeting="Hello")`, what is `greeting`?
- (a) Required  (b) A parameter with a default value  (c) Invalid  (d) A return value

**Q5.** A variable created inside a function is…
- (a) Available everywhere  (b) Only available inside that function  (c) Automatically printed  (d) Deleted immediately

<details><summary>Answers</summary>

**A1 — (b) `def`.**

**A2 — (b).** **`total = show(2, 3) * 10` fails because `show` returned `None`.** This trips up beginners constantly.

**A3 — (b) `None`.**

**A4 — (b).** Parameters with defaults must come after those without.

**A5 — (b).** That isolation is a feature — functions cannot accidentally interfere with each other.
</details>

---

## ⭐ Checkpoint Problem 5 — Prime number checker

> **Uses only:** functions, loops, conditionals, `%`.

**The problem.** Write a function `is_prime(n)` that returns `True` if `n` is a prime number. Then use it to print every prime below 50.

**A prime number is greater than 1 and divides evenly only by 1 and itself.**

<details><summary>Solution</summary>

```python
def is_prime(n):
    """Return True if n is a prime number."""
    if n < 2:                    # 0, 1 and negatives are not prime
        return False

    # Try every possible divisor from 2 up to n-1.
    for divisor in range(2, n):
        if n % divisor == 0:     # it divided evenly, so it is NOT prime
            return False         # return immediately - no need to keep going

    return True                  # nothing divided it: it is prime


for number in range(1, 50):
    if is_prime(number):
        print(number, end=" ")
print()
```

**Output:**
```text
2 3 5 7 11 13 17 19 23 29 31 37 41 43 47
```

**Two things worth noticing:**

1. **`return False` inside the loop stops immediately.** Once you find one divisor, the answer is settled — carrying on would be wasted work.
2. **`return True` is outside the loop.** It only runs if the loop finished without ever finding a divisor.
</details>

**Make it harder:**

1. Make it faster: you only need to test divisors up to the square root of `n`. *(Use `int(n ** 0.5) + 1` as the range end.)*
2. Write `next_prime(n)` returning the first prime larger than `n`.
3. Count how many primes there are below 1000.

<details><summary>Solution to the faster version</summary>

```python
def is_prime_fast(n):
    """Return True if n is prime. Only tests divisors up to sqrt(n)."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:               # every other even number is out at once
        return False

    # If n has a divisor larger than its square root, it must also have
    # a matching one SMALLER than the square root - which we would have
    # found already. So there is no point looking any higher.
    for divisor in range(3, int(n ** 0.5) + 1, 2):
        if n % divisor == 0:
            return False
    return True


count = 0
for n in range(2, 1000):
    if is_prime_fast(n):
        count += 1
print(f"There are {count} primes below 1000")      # 168
```
</details>

---

# 15. Classes and Objects

**Everything so far has kept data and the code that works on it separate.** A class puts them together.

🧠 **Analogy: a cookie cutter and the cookies.** The cutter is the **class** — one design. Each cookie you press out is an **object** — same shape, its own icing. **You define the class once; you make as many objects as you like.**

## Why bother?

Suppose you track three students. Without classes:

```python
name1, marks1 = "Arun", 78
name2, marks2 = "Priya", 92
name3, marks3 = "Ravi", 45
```

**This falls apart quickly.** Add a third field and you are editing everywhere. A class bundles the fields together and keeps the related behaviour with them.

## Defining a class

```python
class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks


arun = Student("Arun", 78)
priya = Student("Priya", 92)

print(arun.name)        # Arun
print(priya.marks)      # 92
```

**Class names use `PascalCase`** — capitals, no underscores. Variables stay `snake_case`.

## `__init__` and `self`

**`__init__` runs automatically when you create an object.** It is where you set the starting values. The name has two underscores on each side.

**`self` is the object being worked on.** When you write `arun.name`, Python passes `arun` in as `self`.

```python
class Student:
    def __init__(self, name, marks):
        self.name = name        # store on THIS object
        self.marks = marks
```

> ⚠️ **`self` must be the first parameter of every method.** You never pass it yourself — Python supplies it.

## Methods — functions that belong to the class

```python
class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def grade(self):                      # a METHOD
        if self.marks >= 90:
            return "A"
        elif self.marks >= 75:
            return "B"
        elif self.marks >= 40:
            return "C"
        return "F"

    def has_passed(self):
        return self.marks >= 40


arun = Student("Arun", 78)
print(arun.grade())          # B
print(arun.has_passed())     # True
```

**A method is just a function that lives inside the class and takes `self` first.**

## Properties (attributes) and methods

| | What it is | Example |
|---|---|---|
| **Attribute** | A piece of data on the object | `arun.name` |
| **Method** | Something the object can do | `arun.grade()` |

**Attributes have no brackets; methods do.**

```python
arun = Student("Arun", 78)         # from the class defined just above

print(arun.name)     # attribute - no brackets
print(arun.grade())  # method    - brackets
```

**Attributes can be changed:**

```python
arun.marks = 85
print(arun.grade())    # B - recalculated from the new value
```

> These two blocks continue from the `Student` class above. If you are running
> them on their own, define the class first.

## `__str__` — how the object prints

```python
class Student:
    def __init__(self, name, marks):
        self.name = name
        self.marks = marks

    def __str__(self):
        return f"{self.name} ({self.marks} marks)"


arun = Student("Arun", 78)
print(arun)            # Arun (78 marks)   - instead of an ugly memory address
```

## A complete, useful class

```python
class BankAccount:
    def __init__(self, holder, balance=0):
        self.holder = holder
        self.balance = balance

    def deposit(self, amount):
        if amount <= 0:
            return "Deposit must be positive"
        self.balance += amount
        return f"Deposited {amount}. Balance: {self.balance}"

    def withdraw(self, amount):
        if amount > self.balance:
            return "Insufficient funds"
        self.balance -= amount
        return f"Withdrew {amount}. Balance: {self.balance}"

    def __str__(self):
        return f"{self.holder}: {self.balance}"


account = BankAccount("Priya", 1000)
print(account.deposit(500))       # Deposited 500. Balance: 1500
print(account.withdraw(2000))     # Insufficient funds
print(account)                    # Priya: 1500
```

**Notice what the class gives you:** the balance and the rules that protect it live together. **Nothing outside can withdraw more than the balance**, because the check sits inside `withdraw`.

## Inheritance — one class built on another

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"I am {self.name}, {self.age} years old"


class Student(Person):                   # Student INHERITS from Person
    def __init__(self, name, age, course):
        super().__init__(name, age)      # run Person's __init__ first
        self.course = course

    def introduce(self):                 # replace the inherited version
        return f"{super().introduce()} and I study {self.course}"


p = Person("Ravi", 45)
s = Student("Priya", 20, "B.Tech")
print(p.introduce())     # I am Ravi, 45 years old
print(s.introduce())     # I am Priya, 20 years old and I study B.Tech
```

## Why this matters for the rest of the course

**Every scikit-learn model you meet from Session 4 onwards is an object of a class.** When you write:

```python
# This is Session 4 code - it will not run yet. Read it for the SHAPE.
model = RandomForestClassifier(n_estimators=100)   # __init__ runs
model.fit(X_train, y_train)                        # a method
model.predict(X_test)                              # another method
model.feature_importances_                         # an attribute
```

**you are doing exactly what you just did with `BankAccount`.** Here is the shape, in miniature:

```python
class SimpleAverager:
    """Predicts the average of whatever it was trained on."""

    def __init__(self):
        self.prediction = None            # nothing learned yet

    def fit(self, y):
        """Learn from the data."""
        self.prediction = sum(y) / len(y)
        return self                       # so you can chain .fit().predict()

    def predict(self, n):
        """Predict for n new items."""
        if self.prediction is None:
            raise ValueError("Call fit() before predict()")
        return [self.prediction] * n


model = SimpleAverager()
model.fit([10, 20, 30, 40])
print(model.predict(3))         # [25.0, 25.0, 25.0]
```

> **That is the entire scikit-learn pattern.** Create with settings, `fit` to learn, `predict` to use. Every model in Sessions 4 to 8 follows it.

## 📘 Examples

**Example 1 — the simplest useful class**

```python
class Dog:
    def __init__(self, name, breed):
        self.name = name
        self.breed = breed

    def bark(self):
        return f"{self.name} says woof!"


d = Dog("Bruno", "Labrador")
print(d.name)       # Bruno
print(d.bark())     # Bruno says woof!
```

**Example 2 — several objects from one class**

```python
class Rectangle:
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * (self.length + self.width)


for r in [Rectangle(5, 3), Rectangle(10, 2)]:
    print(f"Area {r.area()}, perimeter {r.perimeter()}")
```

**Example 3 — attributes can change**

```python
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count


c = Counter()
c.increment()
c.increment()
print(c.count)      # 2
```

**Example 4 — `__str__` for readable output**

```python
class Book:
    def __init__(self, title, author, price):
        self.title = title
        self.author = author
        self.price = price

    def __str__(self):
        return f"'{self.title}' by {self.author} - {self.price}"


print(Book("Python Basics", "R. Nair", 450))
```

## 🌍 Scenarios

**Scenario 1 — a library book that can be borrowed**

```python
class LibraryBook:
    def __init__(self, title, copies):
        self.title = title
        self.copies = copies

    def borrow(self):
        if self.copies == 0:
            return f"'{self.title}' is not available"
        self.copies -= 1
        return f"Borrowed '{self.title}'. {self.copies} left"

    def return_book(self):
        self.copies += 1
        return f"Returned '{self.title}'. {self.copies} available"


book = LibraryBook("Python Basics", 2)
print(book.borrow())        # 1 left
print(book.borrow())        # 0 left
print(book.borrow())        # not available
print(book.return_book())   # 1 available
```

**Scenario 2 — a student with several marks**

```python
class Student:
    def __init__(self, name):
        self.name = name
        self.marks = []                    # start with no marks

    def add_mark(self, mark):
        self.marks.append(mark)

    def average(self):
        if not self.marks:
            return 0
        return sum(self.marks) / len(self.marks)

    def __str__(self):
        return f"{self.name}: {len(self.marks)} subjects, average {self.average():.2f}"


s = Student("Priya")
s.add_mark(92)
s.add_mark(78)
s.add_mark(85)
print(s)                    # Priya: 3 subjects, average 85.00
```

**Scenario 3 — a shopping cart**

```python
class Cart:
    def __init__(self):
        self.items = {}

    def add(self, item, price, quantity=1):
        self.items[item] = self.items.get(item, 0) + quantity
        self.prices = getattr(self, "prices", {})
        self.prices[item] = price

    def total(self):
        return sum(self.prices[i] * q for i, q in self.items.items())

    def __str__(self):
        return f"Cart with {len(self.items)} items, total {self.total()}"


cart = Cart()
cart.add("tea", 15, 2)
cart.add("samosa", 20, 3)
print(cart)                 # Cart with 2 items, total 90
```

## ✏️ Tasks

1. Write a `Circle` class with a radius, and methods `area()` and `circumference()`.
2. Write a `BankAccount` class that refuses to withdraw more than the balance.
3. Write a `Student` class storing a name and a list of marks, with `average()` and `grade()` methods.
4. Add a `__str__` to your `Student` class so `print(student)` shows something readable.
5. Write a `Person` class and a `Teacher` class that inherits from it, adding a subject.

<details><summary>Solutions</summary>

```python
# 1
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        return 3.14159 * self.radius ** 2

    def circumference(self):
        return 2 * 3.14159 * self.radius

c = Circle(7)
print(f"Area {c.area():.2f}, circumference {c.circumference():.2f}")

# 2
class BankAccount:
    def __init__(self, holder, balance=0):
        self.holder = holder
        self.balance = balance

    def withdraw(self, amount):
        if amount > self.balance:
            return "Insufficient funds"
        self.balance -= amount
        return f"Balance now {self.balance}"

a = BankAccount("Priya", 1000)
print(a.withdraw(2000))       # Insufficient funds
print(a.withdraw(400))        # Balance now 600

# 3, 4
class Student:
    def __init__(self, name):
        self.name = name
        self.marks = []

    def add_mark(self, mark):
        self.marks.append(mark)

    def average(self):
        return sum(self.marks) / len(self.marks) if self.marks else 0

    def grade(self):
        avg = self.average()
        if avg >= 90: return "A"
        if avg >= 75: return "B"
        if avg >= 40: return "C"
        return "F"

    def __str__(self):
        return f"{self.name}: average {self.average():.2f}, grade {self.grade()}"

s = Student("Priya")
s.add_mark(92); s.add_mark(78); s.add_mark(85)
print(s)                      # Priya: average 85.00, grade B

# 5
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def introduce(self):
        return f"I am {self.name}, {self.age}"

class Teacher(Person):
    def __init__(self, name, age, subject):
        super().__init__(name, age)
        self.subject = subject

    def introduce(self):
        return f"{super().introduce()} and I teach {self.subject}"

print(Teacher("Mrs Nair", 42, "Physics").introduce())
```
</details>

## ❓ MCQs

**Q1.** What is the difference between a class and an object?
- (a) None  (b) The class is the design; an object is one thing made from it  (c) An object is the design  (d) Classes are faster

**Q2.** When does `__init__` run?
- (a) When you call it directly  (b) Automatically, when an object is created  (c) At the end  (d) Never

**Q3.** What is `self`?
- (a) A keyword  (b) The object the method is being called on  (c) The class  (d) A parameter you pass in yourself

**Q4.** Which is an attribute and which is a method?
- (a) `arun.name` is a method, `arun.grade()` is an attribute
- (b) `arun.name` is an attribute, `arun.grade()` is a method
- (c) Both are attributes
- (d) Both are methods

**Q5.** `model.fit(X, y)` in scikit-learn is…
- (a) A special ML-only syntax  (b) An ordinary method call on an object, exactly like `account.deposit(500)`  (c) A function  (d) A class

<details><summary>Answers</summary>

**A1 — (b).** Cookie cutter and cookies. One class, as many objects as you like.

**A2 — (b) Automatically**, the moment you create the object.

**A3 — (b) The object being worked on.** Python passes it in; you never do.

**A4 — (b).** **Attributes have no brackets; methods do.**

**A5 — (b).** **Every scikit-learn model is an object of a class.** You have just built the same pattern yourself.
</details>

---

# ✅ Before you move on

**Topics**

- [ ] I can write correctly indented Python and know why indentation matters
- [ ] I write comments that explain *why*, not *what*
- [ ] I can create variables and follow the `snake_case` convention
- [ ] I know `int`, `float`, `str` and `bool`, and how to convert between them
- [ ] I can use `+ - * / // % **` and know that `%` tests for even numbers
- [ ] I know `input()` always returns a string, and I convert it
- [ ] I use f-strings, including `:.2f` and alignment
- [ ] I can index and slice strings, and I know they are immutable
- [ ] I can use every comparison, logical and assignment operator
- [ ] I can create, access, change, add to and remove from a list
- [ ] I know when to use a tuple and when to use a set
- [ ] I can use a dictionary, including `.get()` and `.items()`
- [ ] I can write `if` / `elif` / `else` in the right order, and use `match`
- [ ] I can write `for` and `while` loops, with `break` and `continue`
- [ ] I can write functions with parameters, defaults and return values
- [ ] I know the difference between `print` and `return`
- [ ] I can write a class with `__init__`, `self`, attributes and methods
- [ ] **I understand that `model.fit()` is just a method call on an object**

**Checkpoint problems**

- [ ] ⭐ 1 — Rectangle area
- [ ] ⭐ 2 — Initials maker
- [ ] ⭐ 3 — Even or odd
- [ ] ⭐ 4 — Multiplication table
- [ ] ⭐ 5 — Prime number checker

## More practice

| Where | What |
|---|---|
| [Notebook](../notebooks/session-01-python-refresher.ipynb) | Every example above, runnable |
| [Exercises & assignments](../exercises-assignments.md) | Longer graded work |
| [Session 2 — NumPy & Pandas](session-02-numpy-pandas.md) | Where this all starts paying off |
