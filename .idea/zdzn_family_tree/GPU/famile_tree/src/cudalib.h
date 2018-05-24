/*
 * cudalib.h
 *
 *  Created on: 2018年5月17日
 *      Author: root
 */


#ifndef CUDALIB_H_
#define CUDALIB_H_
#include <stdlib.h>
#include <stdio.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <stdio.h>
typedef char byte;
typedef unsigned char ubyte;
template <class T>
__host__ __device__ void len(const char*info,T* result);
template <class T>
__global__ void split_global(T* dum, char* info,long start,long length,int dimblock);


#endif /* CUDALIB_H_ */
