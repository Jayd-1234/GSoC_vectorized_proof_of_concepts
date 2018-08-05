## Local Reduction

### Introduction

Local reduction is an important algorithm in relation to Physics applications. In many cases, we would like to find the best match between particles in a set of events for example, or select the best result per event with respect to a given selection criteria, quickly. The selection criteria can be from simple `max()`, `min()` operation to any set of complex criteria. We want the best match *per event* that satisfies the criteria. This is what local reduction helps in.

Reduction is an inheretly sequential algorithm, which makes it very hard to vectorize properly. The fact that the **local** reduction is performed per event or segment-wise, makes it all the more complex, because data isn't necessarily localized in fixed length segements. Under such a situation, it becomes essential to make the algorithm as much vectorized/parallelized as possible. 

This is what we will explore in this report. 

Also, please note that we shall be using `max()` as the selection criteria for explaining the algorithm, but it will work for any associative operator, like `min()` or a custom function.


### Sequential Implementation

Let us consider a linear data array `arr` which contains the data for all the segments. The segments/events are separated with the `starts` and `stops` array, which stores the starting index and stopping index for the arrays in `arr`. 

Let us define an array `maxarr`, which will store the results of the `max()` operations. The sequential algorithm which dictates how it will work goes as follows:

```python

for i in range(len(starts)):
    for j in range(starts[i]:stops[i])
        maxarr[i] = max(arr[j], maxarr[i])
```

For example, if the `arr`, `starts` and `stops` are defined as 

```python

arr = [1,2,3,2,5,1,4]
starts = [0,2,3]
stops = [2,3,6]
```

Then the array segments are 

```python

arr1 = [1,2,3]
arr2 = [2]
arr3 = [5,1,4]
```

Then the local reduction with `max()`operation will result in the maxarr as

```python
maxarr = [3,2,5]
```

Let's see how to vectorize it now. Also, please keep in mind that given the eextremely sequential nature of the algorithm, it may not be possible to get a lot of performance benefits from vectorizing this problem, especially for smallish data.

### Vectorized Implementation

The earliest works on vectorizing a similar algorithm,  which goes by the name of parallel-scan, or cumulative sum, or prefix sum, comes from the work of two researchers Daniel Hilli and Guy L Steele. Their work was refined and popularized by Guy E Blelloch who created the classical parallel version of the solution, known as Blelloch scan. The algorithm works on two phases, on first phase,called "up-sweep" phase, it involves decomposition of the array into multiple offsets, like a tree like structure, and ecomposing it to a single value.  In the next stage, called the "down-sweep" stage, the prefix sum is estimated by adding the decomposed value to the array results in offsets. 

We draw our inspiration from their work. There are some similarities between prefix-sum and local reduction. The prefix sum is same as local reduction, if:

- There is only one segment.

- The reduction operator is `+`.

The algorithm can be described by the pseudocode:

