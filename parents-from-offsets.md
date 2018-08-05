## Parents from Offsets

### Introduction

Given an array, which is divided into multiple segments or "events", we need a way to relate every element in any segment with the event id for that segment. The `parents` array serves this purpose. It has the same size as the original array, and for every element in data array `arr`, there is the corresponding event value in `parents`. The array can be generated from `offsets`, which gives the segment boundary indices.

The `parents` array is very important to the class of algorithms we will be dealing with. This is because:

- Some algorithms can be vectorized more efficiently in the presence of `parents`. An example of this is `nonzero` function.

- Some other algorithms are impossible to vectorize without `parents`. The best example of this is the `product` function.

In fact, the `product` array is so fundamental that it is absolutely necessary to find an vectorized implementation of parents for use.

We shall explore  how to do this now.

### Sequential implementation

Before we move on to vectorized implementation, it is however useful to look at a sequential implementation first. This will help understand what `parents` works.

Let us consider an array `arr`, of length $n$, which is divided into $m$ segments. The offsets of the segments are given by the `offsets` array.

The parents can then be set by the following loop:

```python
for i in range(len(offsets)):
    parents[offsets[i]:offsets[i+1]] = i
```

Unrolling the inner loop will give an even explicit looping:

```python
counter = 0
for i in range(len(offsets)):
    for j in range(offsets[i]:offsets[i+1]):
        parents[counter] = i
        counter += 1
```

As an example output, say

```python
arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
offsets = [0, 3, 5, 8]
```

Which corresponds to the following three segments

```python
seg1 = ['a', 'b', 'c']
seg2 = ['d', 'e']
seg3 = ['f', 'g', 'h']
```

The `parents` array will be:

```python
parents = [0, 0, 0, 1, 1, 2, 2, 2]
```

Note how the `parents` links the segment elements to it's respective event index or segment number.

We shall discuss it's vectorization next.

### Vectorized Implementation

The main objective in vectorizing the code is removing the two loops:

1. The outer loop iterating over the offsets.

2. An inner loop which sets the values for each event.

There are two strategies we can take in vectorizing the loops, depending on the situation:

1. **Small number of Segments**:  In this case, there is a relatively smaller number of segments/events compared to the segment size. This requires special treatment, as the vectorization must be done across all elements to have fast timings.

2. **Large number of Segments**: In this case, the relative number of segments is quite high. Under that case, the approach taken in earlier case will be slower than just vectorizing across events, and setting the elements with a simple for-loop. This however, may break the total vectorization approach that we are trying to describe.

Te two cases will be described now.

#### _Small number of Segments_

This situation occurs, when the number of events is around 5000, and the number of elements per sgment is less than or eual to 100 on the average.

For such a case, vectorizing over the events ( that is, letting a single thread act over a single event), and looping over the elements of that segment, will give poor performance. 

We can take a different approach here. 

The setting of the values can be done in a vectorized loop. The major steps involved there are:

- Initialize two arrays `middle`, `below` and `above`. `below` and `above` will store the upper limit indexes for the segments, while `middle` will store the `parents`.

- Consider any particle at index `idx`. 

- Set $middle[idx] = \frac{below[idx] + above[idx]}{2}$

- **Convergence check:** Check if `offsets[middle[idx] + 1] > idx` or `offsets[middle[idx]] <= idx` is true. If it is, we are sure that the `middle` array has been succesfully set, as for any valid element in the segment, it's `idx` will be bounded by the two offsets extreme. Then the loop has converged, and can be broken.

- If the above condition is not true, and:

    - `offsets[middle[idx] + 1] <= idx`, then update `below[idx] = middle[idx] + 1`.

    - `offsets[middle[idx]] > idx`, then update `above[idx] = middle[idx] - 1`.

- Do the above steps until convergence is met.

- Return `middle` as the `parents` array.

For small arrays, the approach is quite efficient, as it is expected to complete is $\mathcal{O} (\log_{2}n)$, where $n$ is the number of elemeents in the array. Note that it is true if the data can be processed by all the processors in the GPU. In general, if there are p processors available, then the expected convergence is in $\displaystyle \sum_{p} \mathcal{O} (\log_{2}(\frac{n}{p}))$.

So, why is only suitable for _small_ number of segments? 

Well, it turns out that this process is not the most cache friendly data access pattern on the GPU, as it requires strided data access. Also, the brach predication that it needs also leads to inefficient execution. For large arrays, this non-cache friendly data access and write can be quite costly. For this, we choose the $2^{nd}$ option, which is vectorizing across all the events, and setting the data by a loop. This is suitable for large number of segments, which we will discuss shrotly.

##### GPU implementation

A simple GPU kernel illustrating the idea is given below:

```cpp
__global__ void parents(int* offsets, int* middle,int* len_content,int* below,int* above)
{
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    if ( index >= len_content[0])
        return;

    __shared__ int soffsets[512], smiddle[512], sbelow[512], sabove[512];


    // Load data into shared memory 
    smiddle[tid] = middle[index];
    sbelow[tid] = below[index];
    sabove[tid] = above[index];
    __syncthreads();

    // Loop
    while (1)
    {
        smiddle[tid] = int((sbelow[tid] + sabove[tid])/2);

        // Check for convergence.
        if (offsets[smiddle[tid]+1]<=index || offsets[smiddle[tid]]>index)
        {
            sbelow[tid] = (offsets[smiddle[tid]+1]<=index)? smiddle[tid]+1 :sbelow[tid];
            sabove[tid] = (offsets[smiddle[tid]]>index) ? smiddle[tid]-1: sabove[tid];
        }
        else
            break;
    }
    middle[index] = smiddle[tid];
    __syncthreads();
}
```

#### _Large number of Segments_

In this case, the number of events or segments is quite large, typically larger than 50,000. Under such cases, it is much more useful to give a thread to handle each event, while setting the data via a loop. Since the parents data is layed out in contigous fashion, this leads to very efficient data access from global memory. 

##### GPU Implementation

The kernel illustrating the idea is given below:

```cpp
__global__ void parents(int* starts,int* stops, int* parents,int* numevents)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

    if (idx>=numevents[0])
        return;
    
    
    // Note that starts and stops aren't necessary; offsets alone can be used, since 
    // starts[idx] = offsets[idx] and stops[idx] = offsets[idx+1]. They are given for clarity.

    // store the value from starts.
    const int val = starts[idx];

    // try to unroll the loop. 
    #pragma unroll
    for (int i=val; i<stops[idx]; i++)
    {
        // Non-strided data access.
        parents[i] = idx;
    }

}
```
Performance-wise, the above kernel ( which is simpler than last kernel) is faster than the kernel for small number of segments in this case. The complexity of the algorithm is $\frac{m}{p} \mathcal{O}(k)$, where $m$ is the number of segments, and $k$ is the number of elements in $m^{th}$ segment.

While it's theoritical complexity is higher, it wins on runtime, because of the disadvantages of earlier approach that was mentioned a while back. 

#### CPU Implementation

The code was implemented as a `numba + numpy` code, as well as a pure Cpp code.

The cpp code was compiled with 

```bash
gcc -O3 -march=native -o parents -fopenmp parents_cpp.cpp
```

### Timings

- [ ] Add after testing

