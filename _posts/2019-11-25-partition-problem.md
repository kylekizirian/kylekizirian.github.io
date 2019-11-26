# An implementation of Euler's Partition Function P(n) in Python

This post looks at various ways to implement Euler's Partition Function P(n) (outlined [here on Wolfram's MathWorld](http://mathworld.wolfram.com/PartitionFunctionP.html) in Figure 11) in Python. The description of Euler's function here is taken entirely from the MathWorld reference above, this post will simply look at a Python implementation and how to test both the correctness and performance of our implementations.

From MathWorld:

P(n) gives the number of ways to sum positive integers to n. For example, 4 can be written as the following sums

$4 = 4$

$\ \ = 3 + 1$

$\ \ = 2 + 2$

$\ \ = 2 + 1 + 1$

$\ \ = 1 + 1 + 1 +1$

This gives 5 ways to sum positive integers to 4, thus P(4) = 5.

## Testing Our Partition Function for Correctness

The Online Encyclopedia of Integer Sequences ([OEIS A000041](http://oeis.org/A000041)) gives P(n) for n=0...49 which we will use for testing our implementation of P(n).

Our test function will take a function, P(n), which in turn takes a single integer argument. The test function will iterate over the list of known P(n) from OEIS above and assert that calling our function returns the correct value.


```python
from typing import Callable

def test_pn(pn_implementation: Callable[[int], int]) -> None:
    pn_truths = [
        1, 1, 2, 3, 5, 7, 11, 15, 22, 30, 42, 56, 77, 101, 135, 176, 231,
        297, 385, 490, 627, 792, 1002, 1255, 1575, 1958, 2436, 3010,
        3718, 4565, 5604, 6842, 8349, 10143, 12310, 14883, 17977,
        21637, 26015, 31185, 37338, 44583, 53174, 63261, 75175,
        89134, 105558, 124754, 147273, 173525
    ]
    for n, pn_truth in enumerate(pn_truths):
        assert pn_implementation(n) == pn_truth, (
            f"Error: Implementation gave P({n}) == {pn_implementation(n)}. "
            f"Correct P({n}) == {pn_truth}"
        )
```

## Implementing P(n)

Figure 11 in the [MathWorld reference on Partition Function P](http://mathworld.wolfram.com/PartitionFunctionP.html) gives P(n) as

$P(n) = \sum_{k=1}^{n} (-1)^{k+1}[P(n - \frac{1}{2}k(3k-1))+P(n - \frac{1}{2}k(3k+1))]$

This recurrence equation is naturally expressed as a recursive function with a base case of P(0) = 1

Let's look first at an implementation that most closely resembles the the function above


```python
def P(n: int) -> int:

    if n == 0:
        return 1

    total = 0
    for k in range(1, n+1):
        total += pow(-1, k+1) * (P(n - k*(3*k-1)//2) + P(n - k*(3*k+1)//2))

    return total
```

This implementation, while simple, is awfully slow. Even our test suite to check all cases for n < 50 will not run in a reasonable amount of time. Instead, we will spot check our results with a few known P(n)


```python
assert P(0) == 1
assert P(1) == 1
assert P(5) == 7
assert P(10) == 42
```

To get a baseline for measuring our function's performance, we will use the Jupyter cell magic `%%time` to measure the execution time of P(30)


```python
%%time
P(30)
```

    CPU times: user 40 s, sys: 269 ms, total: 40.3 s
    Wall time: 41.8 s





    5604



As we can see, our implementation of P(n) for even moderately large values of n takes on the order of tens of seconds! To get an understanding of what is taking up so much of our function's execution time, we can profile it with the cell magic `%%prun`


```python
%%prun
P(30)
```

     


             77370568 function calls (25790192 primitive calls) in 56.326 seconds
    
       Ordered by: internal time
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    51580377/1   44.909    0.000   56.326   56.326 <ipython-input-2-2bfb35d290f4>:1(P)
     25790188   11.416    0.000   11.416    0.000 {built-in method builtins.pow}
            1    0.000    0.000   56.326   56.326 {built-in method builtins.exec}
            1    0.000    0.000   56.326   56.326 <string>:1(<module>)
            1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}


As we can see, our function P(n) was called a whopping 51,580,377 times! A major reason for the slow computation time is that the same P(n) will be repeatedly calculated within the recursive calls. To resolve this, we can cache computed values of P(n) to avoid repeated calculations.

To do this, we will create a class that generates a callable object which stores computed values of P(n) in a member variable.


```python
class PWithCache:

    def __init__(self):
        self.computed_pn = {0: 1}

    def __call__(self, n: int) -> int:

        if n in self.computed_pn:
            return self.computed_pn[n]

        total = 0
        for k in range(1, n+1):
            # note: the call to self() here actually invokes self.__call__ (i.e. this function)
            # so this function is also recursive, but with self.computed_pn stored between
            # calls
            total += pow(-1, k+1) * (self(n - k*(3*k-1)//2) + self(n - k*(3*k+1)//2))

        self.computed_pn[n] = total

        return total
```


```python
P = PWithCache()
```

Our first call begins without any P(n) computed, but the caching between recursive calls will immediately result in a dramatic speed-up, with P(30) now taking on the order of milliseconds.


```python
%%time
assert P(30) == 5604
```

    CPU times: user 1.94 ms, sys: 225 µs, total: 2.17 ms
    Wall time: 6.36 ms


Of course, on subsequent calls, P(30) has already been computed which means the call will simply result in a dictionary look-up which happens on the order of hundreds of nanoseconds.


```python
%%timeit
assert P(30) == 5604
```

    311 ns ± 11.7 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)


Finally, we'll run our implementation over our test suite


```python
test_pn(P)
```

## Additional Speed-Ups

Let us profile P(n) once again for n = 1000 to characterize its performance.


```python
%%timeit
P = PWithCache()
P(1000)
```

    1.28 s ± 196 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
P = PWithCache()
%prun P(1000)
```

     


             1501504 function calls (500504 primitive calls) in 1.603 seconds
    
       Ordered by: internal time
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    1001001/1    1.347    0.000    1.603    1.603 <ipython-input-6-38e9931c38ea>:6(__call__)
       500500    0.256    0.000    0.256    0.000 {built-in method builtins.pow}
            1    0.000    0.000    1.603    1.603 {built-in method builtins.exec}
            1    0.000    0.000    1.603    1.603 <string>:1(<module>)
            1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}


One interesting thing we will notice right away is that P(n) is called 1,001,001 times while pow() is called only 500,500 times even though pow() should be called n times for each call to P(n). This is because the the terms $n - \frac{1}{2}k(3k-1)$ and $n - \frac{1}{2}k(3k+1)$ frequently evaluate to negative numbers, in which case the summation loop is bypassed and 0 is returned.

Because function calls in Python are expensive, it would be better to simply replace the term with 0 rather than call P(n) with n < 0. The implementation below first calculates the terms $n - \frac{1}{2}k(3k-1)$ and $n - \frac{1}{2}k(3k+1)$ and replaces the respective P(n) terms with 0 if they are negative.


```python
class PWithCache:

    def __init__(self):
        self.computed_pn = {0: 1}

    def __call__(self, n: int) -> int:

        if n in self.computed_pn:
            return self.computed_pn[n]

        total = 0
        for k in range(1, n+1):
            minus_one_term = n - k*(3*k-1)//2
            plus_one_term = n - k*(3*k+1)//2
            first_term = 0 if minus_one_term < 0 else self(minus_one_term)
            second_term = 0 if plus_one_term < 0 else self(plus_one_term)
            total += pow(-1, k+1) * (first_term + second_term)

        self.computed_pn[n] = total

        return total

P = PWithCache()
test_pn(P)
```


```python
%%timeit
P = PWithCache()
P(1000)
```

    615 ms ± 43.2 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
P = PWithCache()
%prun P(1000)
```

     


             533979 function calls (500504 primitive calls) in 0.667 seconds
    
       Ordered by: internal time
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      33476/1    0.440    0.000    0.667    0.667 <ipython-input-13-14388ae67b71>:6(__call__)
       500500    0.226    0.000    0.226    0.000 {built-in method builtins.pow}
            1    0.000    0.000    0.667    0.667 {built-in method builtins.exec}
            1    0.000    0.000    0.667    0.667 <string>:1(<module>)
            1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}


We can see that by cleverly avoiding unnecessary calls to P(n), we have decreased the number of times it's called from 1,001,001 to a mere 33,476 times resulting in a roughly 2x speed-up!

## Examing our calls to pow()

From our profiling metrics above, we see that our calls to pow() are now taking up nearly half of our execution time. Re-examining our function, we can take a second look at the term $(-1)^{k+1}$

$P(n) = \sum_{k=1}^{n} (-1)^{k+1}[P(n - \frac{1}{2}k(3k-1))+P(n - \frac{1}{2}k(3k+1))]$

Careful consideration will show us that the purpose of the term $(-1)^{k+1}$ is that we will subtract the intermediate result when k is even and add the intermediate result when k is odd. That is, $(-1)^{k+1}$ evaluates to -1 when k is even and to +1 when k is odd.

If we simply want to know whether k is odd or even in order to add or subtract our intermediate result, we can do this more quickly with the modulus operator than by computing $(-1)^{k+1}$. Let us analyze this by taking a random positive integer under 1000 and comparing the time to calculate $pow(-1,\ random\_num)$ and $random\_num\ \%\ 2$.


```python
from random import randint
random_num = randint(1, 1000)
```


```python
%%timeit
pow(-1, random_num)
```

    451 ns ± 30.5 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)



```python
%%timeit
random_num % 2
```

    50.2 ns ± 0.591 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)


As we can see, $random\_num\ \%\ 2$ is roughly an order of magnitude faster than $pow(-1,\ random\_num)$. Let us go back and replace the term $(-1)^{k+1}$ with the use of a modulus operator to govern whether we add or subtract the intermediate result.


```python
class PWithCache:

    def __init__(self):
        self.computed_pn = {0: 1}

    def __call__(self, n: int) -> int:

        if n in self.computed_pn:
            return self.computed_pn[n]

        total = 0
        for k in range(1, n+1):
            minus_one_term = n - k*(3*k-1)//2
            plus_one_term = n - k*(3*k+1)//2
            first_term = 0 if minus_one_term < 0 else self(minus_one_term)
            second_term = 0 if plus_one_term < 0 else self(plus_one_term)
            if k % 2:
                total += first_term + second_term
            else:
                total -= first_term + second_term

        self.computed_pn[n] = total

        return total

P = PWithCache()
test_pn(P)
```


```python
%%timeit
P = PWithCache()
P(1000)
```

    352 ms ± 44 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)



```python
P = PWithCache()
%prun P(1000)
```

     


             33479 function calls (4 primitive calls) in 0.445 seconds
    
       Ordered by: internal time
    
       ncalls  tottime  percall  cumtime  percall filename:lineno(function)
      33476/1    0.445    0.000    0.445    0.445 <ipython-input-19-78a7394da91f>:6(__call__)
            1    0.000    0.000    0.445    0.445 {built-in method builtins.exec}
            1    0.000    0.000    0.445    0.445 <string>:1(<module>)
            1    0.000    0.000    0.000    0.000 {method 'disable' of '_lsprof.Profiler' objects}


Once again, we see a nearly 2x speed-up by eliminating the 500,500 function calls to pow(), with all remaining calls now being the recursive calls to P(n).

## An Analysis of Results

Our final implementatation has grown significantly in lines of code and no longer closely resembles the mathematical form of our function. We've seen signficant speed-ups but at a cost of additional complexity and lines of code. The purpose of this post is to look at how profiling can help us to understand (and potentially improve upon) the performance of our code. Whether optimization is a worthwhile endeavor is situationally dependent and the old adage that "premature optimization is the root of all evil" is always worth keeping in mind.
