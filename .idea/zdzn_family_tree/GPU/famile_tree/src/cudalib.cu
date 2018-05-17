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

namespace gpu {
__global__ void split_global(Matrix A, Matrix B, Matrix C)
{
		float Cvalue = 0;
		int row = blockIdx.y * blockDim.y + threadIdx.y;
		int col = blockIdx.x * blockDim.x + threadIdx.x;
		for (int e = 0; e < A.width; ++e)
			Cvalue += A.elements[row * A.width + e]
					* B.elements[e * B.width + col];
		C.elements[row * C.width + col] = Cvalue;
}

} /* namespace gpu */
