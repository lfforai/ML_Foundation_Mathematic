/*
 * cudalib.cpp
 *
 *  Created on: 2018年5月17日
 *      Author: root
 */

#include <src/cudalib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <string>
#include "src/cudalib.h"

using namespace std;

__global__ void split_global(char* dum, char* info,long length)
{       extern __shared__ char s[];
		long length_N = length;
		int step = gridDim.x*blockDim.x;
		long start_N =threadIdx.x+blockIdx.x*blockDim.x;
		if (start_N==0){
			printf("d:=%d \n",gridDim.x);
		}
		for(long start=start_N;start<length_N;start=+step)
		   {
		   }
}

