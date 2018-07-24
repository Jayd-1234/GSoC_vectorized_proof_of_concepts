#include <iostream>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <cstddef>
#include <cstdlib>
#include <vector>
#include <iterator>

std::vector<int> readfile(const std::string& name)
{
  std::ifstream is(name.c_str(), std::ios::in | std::ios::binary);
  std::istream_iterator<int> start(is), end;
  return std::vector<int>(start, end);
}

int main()
{
    int numevents = 50000;
    std::vector<int> starts1,starts2,stops1,stops2, pairs_indices, counts1(numevents), counts2(numevents);
    
    // Read from files
    starts1 = readfile("starts1.txt");
    starts2 = readfile("starts2.txt");
    stops1 = readfile("stops1.txt");
    stops2 = readfile("stops2.txt");
    pairs_indices = readfile("pairs_indices.txt");


    // Set the counts
    for(int i = 0; i < numevents; i++)
    {
        counts1[i] = stops1[i] - starts1[i];
        counts2[i] = stops2[i] - starts2[i];
    }
    

    std::vector<float> left(pairs_indices[numevents]),right(pairs_indices[numevents]);

    double starts_time = omp_get_wtime();
    for (int i=0; i<numevents; i++)
    {
        if (counts2[i]>0)
        {
            for(int j = pairs_indices[i]; j < pairs_indices[i+1]; j++)
            {
                left[j] = starts1[i] + (j-pairs_indices[i])/counts2[i];
                right[j] = starts2[i] + (j-pairs_indices[i])%counts2[i];       
            }
        }
        
    }

    int* lefti = reinterpret_cast<int *> (&left);
    int* righti = reinterpret_cast<int *>(&right);

    double stop_time = omp_get_wtime();

    std::cout << "Time taken: "<< stop_time-starts_time<<std::endl ;

    return 0;
}
