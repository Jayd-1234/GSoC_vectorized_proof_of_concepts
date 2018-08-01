## Argpairs

### Introduction

`Argpairs` is a special case of [argproduct](./argproduct.md). Unlike `argproduct`, `argpairs` can only be applied for self-pairing, that is, pair one collection with itself. The operation of `argpairs` is undefined for pairing between multiple different arrays.

**Then why do we need it?** 

Well, it turns out that `argproduct` is pretty wasteful in terms of space requirements, if we apply it to the same array. If we consider an array `arr1` with n elements, then `argproduct(arr1, arr1)` will lead to a total of <img src="/tex/671384664426f82966adbde77d983140.svg?invert_in_darkmode&sanitize=true" align=middle width=16.41942389999999pt height=26.76175259999998pt/> pairings, and thus, a corresponding requirement to store and maintain two such arrays, which will take up a space of <img src="/tex/29bfef6c6050d16079ac7bdb1be8ab45.svg?invert_in_darkmode&sanitize=true" align=middle width=24.63863324999999pt height=26.76175259999998pt/> elements.

`Argpairs` aims to rectify that. It can be observed that `argproduct` calculates a pair _twice_. This is because an element will be paired tpo every other element both as the first element of the pair, as well as the last element. In other words if <img src="/tex/70a073591b43ab4f5040431407ad1fd2.svg?invert_in_darkmode&sanitize=true" align=middle width=78.09548834999998pt height=21.18721440000001pt/>, then both <img src="/tex/7392a8cd69b275fa1798ef94c839d2e0.svg?invert_in_darkmode&sanitize=true" align=middle width=38.135511149999985pt height=24.65753399999998pt/> and <img src="/tex/bff3925497e17fa99b4993508b1e6e7e.svg?invert_in_darkmode&sanitize=true" align=middle width=38.135511149999985pt height=24.65753399999998pt/> will be included in `argproduct`. `Argpairs` takes advantage of the symmetry in such pairing. There is no need to calculate both <img src="/tex/7392a8cd69b275fa1798ef94c839d2e0.svg?invert_in_darkmode&sanitize=true" align=middle width=38.135511149999985pt height=24.65753399999998pt/> and <img src="/tex/bff3925497e17fa99b4993508b1e6e7e.svg?invert_in_darkmode&sanitize=true" align=middle width=38.135511149999985pt height=24.65753399999998pt/>, as they are ( in most cases) equivalent; only one would suffice. This saves a considerable amount of memory, and by extension, code runtime on future operations done on them. Since every pair is calculated only once ( including pair of an element with itself), so the total memory requirement is <img src="/tex/3adbb4fb23cc47e54c2926672b15b7ed.svg?invert_in_darkmode&sanitize=true" align=middle width=68.45933489999999pt height=28.40558820000002pt/> elements, that is <img src="/tex/132122235da7e27dfed089f00ba361db.svg?invert_in_darkmode&sanitize=true" align=middle width=60.82958639999999pt height=24.65753399999998pt/> elements, which is a saving of <img src="/tex/a257df36454c2808de28081d8c1149c2.svg?invert_in_darkmode&sanitize=true" align=middle width=175.49835764999997pt height=26.76175259999998pt/> elements. For large arrays, the value can be quite large.

TODO: Add some more if needed.

### Sequential implementation

In order to explain how to implement `argpairs()`, we consider an array `arr1`.

```python
arr1 = [a,b,c,d]
```

The `argpairs()` of this array will have <img src="/tex/b0c08f9b595a704efb907fc688034d80.svg?invert_in_darkmode&sanitize=true" align=middle width=16.438418699999993pt height=21.18721440000001pt/> pairs. In order to represent the pairs, we can output the index of the first element in each pair in `first` array, while we can store the index of second element in `second` array. This has the advantage of being readily portable to languages that don't support tuples out of the box. Also, they make vectorization possible.

In our specific case, the `first` and `second` will be:

```python
first = [0,0,0,0,1,1,1,2,2,3]
second = [0,1,2,3,1,2,3,2,3,3]
```

For example, the first pair is `(arr1[first[0]], arr1[second[0]])`, which is `(a,a)`, and so on.

A sequential code to generate the pairs can be implemented as a double-looped version as follows:

```python
pairs_i = 0
for i in range(len(arr1)):
    for j in range(i, len(arr1)):
        first[pairs_i] = i
        second[pairs_i] = j
        pairs_i +=1
```

A practical data array may be composed of multiple sub-arrays. One would have to create pairs for each sub-array, which belong to a particular event.  The sub-arrays tend to be very irregular in shape. In order to represent them the sub-arrays in the array, the individual sub-arrays are usually stacked in a contiguous fashion, with the start and stop indices of each sub-array are described by two arrays `start` and `stop`. 

Using these information, we can implement a sequential version to implement `argpairs` as follows:

```python
import numpy

counts = stop - starts
pairs_length = numpy.zeros(len(starts) + 1, dtype=numpy.int)
pairs_length[1:] = numpy.cumsum(counts*(counts+1)/2, dtype=numpy.int)
for i in range(len(starts)):
    pairs_i = pairs_length[i] 
    for j in range(starts[i], stops[i]):
        for k in range(i, stops[i]):
            first[pairs_i] = j
            second[pairs_i] = k
            pairs_i +=1
```

Note that the dependence on `pairs_i`, which counts the current pair index, leads to this problem being non-trivially vectorizable. In the next section, we will explain how we can achieve it.

### Vectorized implementation


