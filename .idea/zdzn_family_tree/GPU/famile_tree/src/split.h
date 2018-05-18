/*
 * split.h
 *
 *  Created on: 2018年5月17日
 *       Author: 罗锋
 *  基于多GPU集群的族谱分解：
 *  例如：MDNY.SZWQ.11MD#1.MlyGnCptLo拆分为
 *       grand-grand-father：MDNY
 *       grand-father：MDNY.SZWQ
 *       father：MDNY.SZWQ.11MD#1
 */


#ifndef SPLIT_H_
#define SPLIT_H_

#include "file_input.h"
using namespace std;


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

class split {
public:
	split();
	virtual ~split();
	static int* max_ancestors_num(file_input::info* info_of_key,int GPU_num);
};
#endif /* SPLIT_H_ */
