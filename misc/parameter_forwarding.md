# Parameter forwarding

In Python, there exist a mechanism that allow the *forwarding* of
parameters from a function to another one. Let's see how it works.

First, let's define a very simple function

```python
def add(a, b):
  return a + b

print(add(5, 7))
```

```
12
```

Nothing spectacular yet. We can create a dictionary containing the
parameters values we want to give to the function.

```python
kwargs = {
    'a' : 5,
    'b' : 7
}
print(add(**kwargs))
```

```
12
```

Neat!

A function can also recieve a variable number of *named* parameters

```python
def print_kwargs(a, b, **kwargs):
  print(a)
  print(b)
  print(kwargs)
```

We can call it by simply giving `a` and `b`

```python
print_kwargs('hi', 'everyone')
```

```
hi
everyone
{}
```

`kwargs` is empty as no extra named parameters have been given.

```python
print_kwargs('hi', 'everyone', c = 'foo', banana = 16)
```

```
hi
everyone
{'c': 'foo', 'banana': 16}
```

`kwargs` now contains all the extra parameter names and their
corresponding values.

We can use these two behaviors in combination to *forward* the
parameters of one function to another one.

Let's say we want to write a function that measures the execution time
of another function call.

```python
import time

def add(a, b):
  return a + b

def time_function(fun, **kwargs):
  start_time = time.time()
  fun(**kwargs)
  end_time   = time.time()

  return end_time - start_time

print(time_function(add, a = 5, b = 7))
```

```
1.1920928955078125e-06
```
