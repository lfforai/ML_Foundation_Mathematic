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
//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include <stdio.h>
typedef char byte;
typedef unsigned char ubyte;
template <class T>
__host__ __device__ void len(const char*info,T* result);
template <class T>
__global__ void split_global(T* dum, char* info,long start,long length);


#endif /* CUDALIB_H_ */
