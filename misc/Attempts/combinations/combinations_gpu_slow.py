
import pycuda.autoinit
import pycuda.gpuarray as gpuarray
from pycuda.compiler import *
import pycuda.driver as cuda
import numpy as np
import numba


# In[61]:


NUMEVENTS = 50
AVENUMJETS = 10

numjets1 = np.random.poisson(AVENUMJETS, NUMEVENTS).astype(np.int32)
stops1 = np.cumsum(numjets1, dtype=np.int32)
starts1 = np.zeros_like(stops1)
starts1[1:] = stops1[:-1]

counts1 = stops1-starts1
offsets1 = np.zeros(len(numjets1)+1)
offsets1[1:] = stops1[:]

numjets2 = np.random.poisson(AVENUMJETS, NUMEVENTS).astype(np.int32)
stops2 = np.cumsum(numjets2, dtype=np.int32)
starts2 = np.zeros_like(stops2)
starts2[1:] = stops2[:-1]


counts2 = stops2-starts2
offsets2 = np.zeros(len(numjets2)+1)
offsets2[1:] = stops2[:]


# In[62]:


@numba.jit()
def vectorized_search(offsets, content):
    index = np.arange(len(content), dtype=np.int32)                     
    below = np.zeros(len(content), dtype=np.int32)                      
    above = np.ones(len(content), dtype=np.int32) * (len(offsets) - 1)  
    while True:
        middle = (below + above) // 2

        change_below = offsets[middle + 1] <= index                  
        change_above = offsets[middle] > index                        

        if not np.bitwise_or(change_below, change_above).any():    
            break
        else:
            below = np.where(change_below, middle + 1, below)      
            above = np.where(change_above, middle - 1, above)      

    return middle


# In[63]:


pairs_indices = np.zeros(NUMEVENTS+1, dtype=np.int32)
pairs_indices[1:] = np.cumsum(counts1*counts2, dtype=np.int32)
pairs_indices = pairs_indices


# In[64]:


pairs_contents = np.arange(pairs_indices[-1]).astype(np.int32)
pairs_parents = vectorized_search(pairs_indices, pairs_contents)
pairs_parents = pairs_parents.astype(np.int32)


# In[65]:


mod = SourceModule('''
__global__ void combinations(int* starts1,int* starts2,int* counts2,int* pairs_parents,int* pairs_indices,int* left,int* right,int* numpairs)
{
    unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;

    
    if (idx>numpairs[0])
        return;
    
    if (counts2[pairs_parents[idx]]>0)
    {
        int temp = int((idx-pairs_indices[pairs_parents[idx]])/counts2[pairs_parents[idx]]);
        left[idx] = starts1[pairs_parents[idx]] + temp;
        right[idx] = starts2[pairs_parents[idx]] + (idx-pairs_indices[pairs_parents[idx]])-counts2[pairs_parents[idx]]*temp;
    }    
    
}
''',options=['-use_fast_math'])


# In[66]:


func = mod.get_function('combinations')


# In[67]:


gpu_starts1 = gpuarray.to_gpu(starts1)
gpu_starts2 = gpuarray.to_gpu(starts2)
gpu_counts2 = gpuarray.to_gpu(counts2)
gpu_pairs_parents = gpuarray.to_gpu(pairs_parents)
gpu_pairs_indices = gpuarray.to_gpu(pairs_indices)
left = gpuarray.zeros(pairs_indices[-1], dtype=np.int32)-1
right = gpuarray.zeros_like(left)-1
numpairs = gpuarray.to_gpu(np.array([pairs_indices[-1]]).astype(np.int32))
numthreads = 512
numblocks = int(np.ceil(pairs_indices[-1]/numthreads))


# In[68]:


#import time
#start_time = time.time()
#start = cuda.Event()
#stop = cuda.Event()
#start.record()
func(gpu_starts1,gpu_starts2,gpu_counts2,gpu_pairs_parents,gpu_pairs_indices,left,right,numpairs, block=(numthreads,1,1), grid = (numblocks,1))
#stop_time = time.time()
#stop.record()
#stop.synchronize()
#print ("Total time taken = {} milliseconds".format(start.time_till(stop)))


# In[69]:


#for i in range(6):
#   print("Event {}\n Left {}\nRight {}\n\n".format(i, left[pairs_indices[i]:pairs_indices[i+1]], right[pairs_indices[i]:pairs_indices[i+1]]))


# In[70]:

#pycuda.autoinit.context.detach()
#np.count_nonzero(counts2)

