/*
 * split.cpp
 *
 *  Created on: 2018年5月17日
 *      Author: root
 */

#include <src/split.h>
#include <stdlib.h>
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <string>
#include <cudalib.h>

namespace gpu {

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


split::split() {
	// TODO Auto-generated constructor stub

}

split::~split() {
	// TODO Auto-generated destructor stub
}

//返回所有key值中分解出的最多的一个祖先个数
//例如：MDNY.SZWQ.11MD#1.MlyGnCptLo,祖先个数为3
//返回最大祖先个数
int split::max_ancestors_num(file_input::info info_of_key,int GPU_num=-1){

    int max=0;
    char* keys_data=info_of_key.data;
    long row_num=info_of_key.total_row;

    if (GPU_num==-1)
	   {int deviceCount;
	    CUDACHECK(cudaGetDeviceCount(&deviceCount));
	   }

    //#记录每个key的最大祖先个数的数组
    char* h_num=(char *)malloc(row_num*sizeof(char));
    char** d_num=(char **)malloc(row_num*sizeof(char*));

    long yu= row_num%deviceCount;
    long sub_length=row_num/deviceCount;

    //此处假设所有的祖先不超过255个，所以采用char
    for(int i=0;i<deviceCount;i++){
        CUDACHECK(cudaMalloc(d_num + i, row_num * sizeof(char)));
    }

    for(int i=0;i<deviceCount;i++){
       if (i!=deviceCount-1)
          {split_global<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);             }
       else
          {                }
    }
	return max=0;
};

} /* namespace gpu */
