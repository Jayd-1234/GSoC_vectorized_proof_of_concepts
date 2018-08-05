## Argpairs

### Introduction

`Argpairs` is a special case of [argproduct](./argproduct.md). Unlike `argproduct`, `argpairs` can only be applied for self-pairing, that is, pair one collection with itself. The operation of `argpairs` is undefined for pairing between multiple different arrays.

**Then why do we need it?** 

Well, it turns out that `argproduct` is pretty wasteful in terms of space requirements, if we apply it to the same array. If we consider an array `arr1` with n elements, then `argproduct(arr1, arr1)` will lead to a total of $n^{2}$ pairings, and thus, a corresponding requirement to store and maintain two such arrays, which will take up a space of $2n^{2}$ elements.

`Argpairs` aims to rectify that. It can be observed that `argproduct` calculates a pair _twice_. This is because an element will be paired tpo every other element both as the first element of the pair, as well as the last element. In other words if $x, y \in arr1$, then both $(x,y)$ and $(y,x)$ will be included in `argproduct`. `Argpairs` takes advantage of the symmetry in such pairing. There is no need to calculate both $(x,y)$ and $(y,x)$, as they are ( in most cases) equivalent; only one would suffice. This saves a considerable amount of memory, and by extension, code runtime on future operations done on them. Since every pair is calculated only once ( including pair of an element with itself), so the total memory requirement is $\displaystyle ^{n+1}C_{2}*2$ elements, that is $n(n+1)$ elements, which is a saving of $2n^{2} - n(n+1) = n^{2}-n$ elements. For large arrays, the value can be quite large.

TODO: Add some more if needed.

### Sequential implementation

In order to explain how to implement `argpairs()`, we consider an array `arr1`.

```python
arr1 = [a,b,c,d]
```

The `argpairs()` of this array will have $10$ pairs. In order to represent the pairs, we can output the index of the first element in each pair in `first` array, while we can store the index of second element in `second` array. This has the advantage of being readily portable to languages that don't support tuples out of the box. Also, they make vectorization possible.

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

#### Mathematical basis

The vectorized implementation is much involved in the case of `argpairs`. A key element to notice is that the `argpairs` generated pairs, which are not repeated. We can visualize this as generating the upper triangular matrix indices, from the linear indices of the array. We shall consider the case for single event first.

To derive the necessary relationship, not that given the row index $i$ and column index $j$ of the upper triangular matrix, the linear index of the matrix is given by 

$$ m = n*i - \frac{i(i+1)}{2} +j$$

Where $n$ is the total number of elements in `arr1`, that is $n = \mathrm{counts}  = \mathrm{(stops-starts)}$.

Now, rearranging the equation, we get 

$$ j = m-n*i+\frac{i(i+1)}{2} $$.

Since j (column index) must always be positive, so we are finding the minimum value of `i`, which gives positive `j` for every `m`.

That is, row index, 

$$ i = \min\{i :  m-n*i+\frac{i(i+1)}{2} \geq 0\}$$ 

Which leads to the quadratic equation 

$$ i^2 - (2n-1)i +m \geq 0   $$

$$ \Rightarrow i \geq \frac{2n+1 - \sqrt{ (2n-1)(2n-1) -8m}}{2}  $$

$$ \Rightarrow i = floor \{ \frac{2n+1 - \sqrt{ (2n-1)(2n-1) -8m}}{2} \} $$ 

The j can easily be calculated from 

$$j = m-n*i+\frac{i(i+1)}{2}$$

#### Implementation

##### Single event case

The algorithm described above can easily be implemented, as shown:

```python
import numpy 

counts = stops-starts
numpairs = counts*(counts+1)/2

# Swith to our notation

n = counts

for m in range(numpairs):
    first[m] = numpy.floor((2*n+1 - numpy.sqrt((2*n+1)*(2*n+1) - 8*m)) /2)
    second[m] = m - n*first[m] + first[m]*(first[m]+1)/2
```

Note that we are using $2n+1$ as the factor instead of $2n-1$. This has to do with the fact that the the usage of latter sometimes gives imaginary roots. 

##### Multiple events case

Now, let's consider the harder part, where the linear data array is divided into segments.

We can simulate the same pairing logic here too. We just have to offset each pair to the start of the array segment, given by `starts` array. 

Also, we need the help of 
- `pairs_parents`, which relates every pair element to the event index in which the parent is in.The process of generation of `pairs_parents` is similar to [parents]() function.

    TODO: link to parents after it's done.

- `pairs_indices`, which gives the index of the first pair in the event.

- `pairs_contents` which gives the running index of all possible pairs.

The implementation is 

```python
import numpy

# Calculate counts
counts = stops-starts

# Find pairs_indices
pairs_indices = np.zeros(len(starts)+1)
pairs_indices[1:] = np.cumsum(counts*(counts+1)/2, dtype=numpy.int)

# Define pairs_contents
pairs_contents = np.arange(pairs_indices[-1])

# Find pair_parents
pairs_parents = parents(pairs_indices, pairs_contents)

# Initialize the first and second arrays
first = np.ones_like(pairs_contents)*-1
second = np.empty_like(pairs_contents)

# Calculate the values
n = counts[pairs_parents[pairs_contents]]
k = pairs_contents-pairs_indices[pairs_parents[pairs_contents]]

# Add offset to the pairs_indices
first[pairs_contents] = starts[pairs_parents[pairs_contents]]+ np.floor((2*n+1 - np.sqrt((2*n-+1)*(2*n+1) - 8*k))/2)
i = first[pairs_contents] - starts[pairs_parents[pairs_contents]]
second[pairs_contents] = starts[pairs_parents[pairs_contents]] + k - n*i + i*(i+1)/2
```

#### GPU Implementation

The GPU implementation follows the similar implementation procedure.  We used CUDA for the implementation.

The implementation is straightforward:

```cpp
__global__ void argpairs(int* first,int* second,int* starts,int* counts,int* pairs_indices,int* pairs_parents, int* numpairs)
{
    unsigned int tid = threadIdx.x + blockIdx.x*blockDim.x;

    // Restrict threads to legal pairs
    if (tid > numpairs[0])
        return;

    int n = counts[pairs_parents[tid]];
    int x = pairs_parents[tid];
    int k = idx - pairs_indices[x];

    // Check for events with no elements
    if (n <= 0)
        return;

    // Calculate 
    int i = int(floor((2*n+1 - sqrt((2*n+1)*(2*n+1) -8*k))/2));
    first[tid] = starts[x] + i;
    second[tid] = starts[x] + k -n*i + i*(i+1)/2 ;

}
```

#### CPU implementation

The CPU implementation was done in C++, and is available [here]()

TODO: Add link to cpp code

It was compiled with

```bash
g++ -O3 -march=native -o argpairs -fopenmp argpairs_cpp.cpp 
```

### Performance 

#### Execution environment

TODO: Add bout AWS instance

#### Timings

TODO: Add graph after timings are done





