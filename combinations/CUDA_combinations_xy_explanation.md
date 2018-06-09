## Idea behind the CUDA Function

To determine the appropriate CUDA function, it is useful to first study the sequential version of the combinations code:


```python
for i in range(base_len-1):
    pairs_i = pairs_lengths[i]
    for j in range(start1[i], stop1[i]):
        for k in range(start2[i], stop2[i]):
            left[pairs_i] = j
            right[pairs_i] = k
            pairs_i +=1
```
There are some important points to consider here:


- `base_len = total number of events +1` so i is index of a particular event and `pairs_i` is the index of current pair. Please note that it draws its value from the pairs_lengths array, which is dependent on **event index i only**, and **independent of j and k**. 
- `j` and `k` is the index of 1st and 2nd data arrays. j varies from start of 1st particle for the event i to end for that particle. Similarly, k does the same for 2nd particle. 
- The next part is self-explanatory. The pairs will be drawn from first array whose indices are stored in `left[]`, and second array, whose indices are stored in `right[]`. Since j corresponds to 1st particle and k for second, so for each pair index `pairs_i`, `left[pairs_i] = j` and `right[pairs_i] = k`. Finally, we increment pairs_i as each pair is processed.

To parallelize this code, we need every thread block to perform identical and independent operations. The main barriers to this condition here are:


1. `pairs_i`, whose incerment depends on j,k both, and may raise a race condition. 
2.  variable length `start` and `stop` arrays.


#### How do we solve this?

It turns out that we dont need to explicitly increment pairs_i everytime. A unique pairs index can be generated given the values of 


- `pairs_lengths[i]`, which serves as the start index for a particular event i.
- `lengths1[i]`, which will give us the length of 1st array for event i.
- `j` and `k`.

The unique index is thus `pairs_i = pairs_lengths[i] + j*lengths1[i] + k`. Note that this index will be unique no matter whatever way the threads are accessed. 

#### Forming the CUDA thread blocks

In CUDA, Nvidia provides us the option of using 1D, 2D or 3D grids and thread blocks. The threads ( and blocks) are indexed via a 3 element struct (x,y,z). The important thing to note here is that:

- The Nvidia CUDA driver assigns the 3D blocks the required number of threads. The x,y,z threads are independent of each other, and have their own limitations in the maximum number of threads possible. For more information, refer to the `deviceQuery()` function that comes bundled with CUDA toolkit.
- The threads in dims x,y,z start from index zero, and upto the total number of threads allocated in each dimension.  

We can use the above facts for creating the CUDA function. We form 3D grid and blocks, and use i,j,k to refer to thread indices in x,y,z dimensions. 

Then 
```c
int i =  blockIdx.x*blockDim.x+threadIdx.x;
int j =  blockIdx.y*blockDim.y+threadIdx.y; 
int k =  blockIdx.z*blockDim.z+threadIdx.z;
```

We now do the following:

- `i` refers to events, so that it runs `base_len-1` number of threads. We can either explicitly assign that number of threads, or use if condition to restrict it.
- `j` and `k` refers to particles 1 and 2. Since the lengths of the arrays are variable length, so we must explicitly restrict the `j` and `k` indices ( possible performance loss). Since `j` and `k` start from 0, so we add `start1[i]` to `j` and `start2[i]` to `k` to provide the necessary offsets.

Thus, we can now calculate the `left` and `right` arrays as:

```c
left[pairs_lengths[i] + j*lengths2[i] + k] = j + start1[i];
right[pairs_lengths[i] + j*lengths2[i] + k] = k + start2[i];
```

The full CUDA c code is given here:
```c
__global__ void comb_events(int* left,int* right,int* start1,int* start2,int* length,int* lengths1,int* lengths2,int* pairs_lengths)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int j =  blockIdx.y*blockDim.y+threadIdx.y;
    int k =  blockIdx.z*blockDim.z+threadIdx.z;
    if (i <length[0])
    {
    if (j< lengths1[i] && k<lengths2[i])
        {
            left[pairs_lengths[i] + j*lengths2[i] + k] = j + start1[i];
            right[pairs_lengths[i] + j*lengths2[i] + k] = k + start2[i];
        }
    }
}
```


Simple, right :smile: ?
