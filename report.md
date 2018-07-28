# GSoC 2018 report

This markdown file is meant to be an intermediate content for the more complete Google doc report.  

<!--<p align='center'> <b>Product </b> </p> -->

## Product

### Introduction  

To be done by Jim/David as per Jim's suggestion.

### Sequential Implementation

`Product()` creates the cartesian product of two arrays, by combining them element-wise, grouping every element in first array ( say `arr1`) with every other element in second array ( say `arr2`) . To emulate this grouping, the function return two arrays, `first` and `second`, which contains the indices for the elements from `arr1` that get paired with elements in `arr2`.

To explain how `first` and `second` dictate how the grouping work, let's consider the simpler case first, where there is a single __event__. Consider the simple arrays:

```python
arr1 = [a,b,c]
arr2 = [1,2]
```

The product of these two arrays will have `len(arr1)*len(arr2)` elements, which will be the size of `first` and `second` arrays. Considering 0-indexing, the arrays `first` and `second` will be

```python
first = [0,0,1,1,2,2]
second = [0,1,0,1,0,1]
```

This arrays serve two purposes:

1. They can be used to emulate product on languages which do not support list of tuples out of the box.

2. They will allow vectorization of the `product()` function. This will be explained shortly.

A sequential implementation of the `product()` can be implemented as a double-looped version (We will be using `python` for explaining the codes, but the general idea holds true for any language).

```python
for i in range(len(arr1)):
    for j in range(len(arr2)):
        first[i*len(arr2) + j] = i
        second[i*len(arr2) + j] = j
```

This serves a theoretical background for `product`. In real world use-case, we usually care about `product` between sub-arrays of arrays `arr1` and `arr2`. These sub-arrays have some property in common, like they may belong to a particular class or event. The sub-arrays tend to be very irregular in shape. In order to represent them the sub-arrays in the array, the individual sub-arrays are usually stacked in a contiguous fashion, with the start and stop indices of each sub-array are described by two arrays `start` and `stop`. 

Let's denote the `starts` and `stops` arrays of `arr1` and `arr2` as `starts1`,`stops1` and `starts2`, `stops2` respectively. Now, we can modify the above sequential implementation to account for sub-arrays as follows:

```python
for i in range(len(starts2)):
    for j in range(starts1[i], stops1[i]):
        for k in range(starts2[i], stops2[i]):
            first[j*(stops2[i] - starts2[i]) + k] = j
            second[j*(stops2[i] - starts2[i]) + k] = k
```

Note that the triple loop structure will slow down the runtime of the code significantly. This calls for the need of vectorizing the code, which we shall discuss next.

### Vectorized Implementation

The vectorized implementation will help us reduce two of the loops above. In order to derive the vectorized implementation, we shall first start with the simpler case, where there is only one sub-array ( same as original array) which corresponds to use case 1 described earlier. 

The implementation can be vectorized, using a key idea: The `first` and `second` arrays are actually the row index and column index of a matrix that is formed from the linear index `idx` of the arrays. This can be done easily if we know the number of elements in `arr2`. Let's call it `counts2`. 

Then the row index is `rid` = `idx/counts2`, which corresponds to `first[index]`; 

and the column index is `cid` = `idx%counts2`, which corresponds to `second[index]`.

So, the vectorized code is

```python
for idx in range(len(arr1) * len(arr2)):
    first[idx] = idx/counts2
    second[idx] = idx%counts2
```

Note that the time complexity is still **O**(__m__*__n__), where m is the number of elements in `arr1` and n is the number of elements in `arr2`. However, now, the code can be readily parallelized.

Armed with this idea, we can now extend it to the more general case with sub-arrays. In fact, it turns out that this is also quite easy. We need two things for that:

- The offset for the sub-array element index.
- The number of elements in `arr2` at ith sub-array, denoted by `counts2[i]`.
- A `parents` array, which links each sub-array element to the index of the sub-array in the original array.
- A `pairs_indices` array, which links every pair element with the starting index of of any pair in that event.

The offset can be determined from `starts` array. This will give us the beginning of the ith sub-array. The value of `counts2[i]` can be determined from `starts2[i]` and `stops2[i]`. 

Then `counts2[i] = stops2[i] - starts2[i]`.

The process of implementing `parents` will be described in it's own section.

This amounts to generating the sub-matrix indices from the linear-index of the sub-arrays. A sequential code illustrating the idea can be 

```python
for idx in range(stops1[-1] * stops2[-1]):        #loop until last element pair

    first[idx] = starts1[parents[idx]] + (idx - pairs_indices[parents[idx]])//counts2[parents[idx]]

    second[idx] = starts2[parents[idx]] + (idx - pairs_indices[parents[idx]])%counts2[parents[idx]]
```

Note that this code is also parallel in nature.

#### GPU Implementation

The inherently parallel nature of the algorithm makes it perfectly suited for a GPU. With a high number of cores, the GPU is tailor-made for such a purpose. We chose **CUDA** for the programming, but **OpenCL** should work equally well.

In the CUDA kernel, we can pass the arrays required for the computation, and assign a thread to every pair, to calculate the `first` and `second` arrays efficiently.

A simple implementation in CUDA is shown below

```cpp
__global__ void combinations(int* starts1,int* starts2,int* counts2,int* pairs_parents,int* pairs_indices,int* left,int* right,int* numpairs)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx>numpairs[0])
        return;
    

    if (counts2[pairs_parents[idx]]>0)
    {
        int temp = (idx-pairs_indices[pairs_parents[idx]])*(1/(counts2[pairs_parents[idx]]));
        left[idx] = starts1[pairs_parents[idx]] + temp;
        right[idx] = starts2[pairs_parents[idx]] + (idx-pairs_indices[pairs_parents[idx]])-counts2[pairs_parents[idx]]*temp;
    }        
}
```
Note that `num_pairs` gives the total number of pairs that can be formed between the array elements.

The kernel was launched with a thread block size of 512 threads. 

#### CPU implementation

The CPU implementation of the vectorized code was done as both a pure `numpy` implementation, as well as a C++ code. The C++ code was compiled with gcc, with all optimizations turned on.

**__Numpy Implementation__** 

TODO

**__CPU Implementation__**

TODO

### Performance tests

#### Testing conditions

The GPU implementation was tested on an Amazon AWS instance with a Nvidia card of kepler architecture (TODO: Add more details regarding the card).

The CPU implementation was tested on the same instance, which had an Intel Xeon processor with 8 cores, and a base clock speed of 2.60 GHz.

#### Results

Some timing results are shown in table below


|      |   Events   |   Average number of elements   |   Runtime   |
|-------|:----------:|:------------------------------:|:-----------:|
| GPU code   | 5000   | 100   | 8ms   |
| C++ code   | 5000   | 100   | 2.1s  |
| `numpy` code | 5000 | 100 | 6s  |


TODO: Prettify the table, or use some other method of displaying the results?