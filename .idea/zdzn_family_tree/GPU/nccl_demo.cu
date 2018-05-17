//demo作用：多gpu可以通过nccl构建gpu集群并行运算，nccl是nvidia开发的多gpu的信息交互工具类似MPI
//demo内容利用nccl中的例子中的=单thread对多device的模式，对reduce，allreduce,allgather,ReduceScatter进行了测试
//参考文档:https://docs.nvidia.com/deeplearning/sdk/nccl-developer-guide/index.html
//作者：罗锋
//demo归类：数据族谱gpu实现的子项目测试

#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <string>
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

int main(int argc, char* argv[])
{
  ncclComm_t comms[2];

  //managing 4 devices
  int nDev = 2;
  int size = 10;
  int devs[2] = { 0, 1};

  //allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));
  float** h_sendbuff_N=(float **)malloc(nDev* sizeof(float*));
  cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);

  for(int i = 0; i < nDev; ++i)
  {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    for(int j=0;j<size;j++){
       h_sendbuff_N[i]=(float *)malloc(size * sizeof(float));
       for(int j=0;j<size;j++)
          {if (i==0 and j<5)
    	      *(h_sendbuff_N[i]+j)=1;
           else if(i==1 and j>=5)
                  *(h_sendbuff_N[i]+j)=2;
                else
                  {*(h_sendbuff_N[i]+j)=-1;}
          }
    }
    CUDACHECK(cudaMemcpy(sendbuff[i], h_sendbuff_N[i], size*sizeof(float), cudaMemcpyHostToDevice));
   // CUDACHECK(cudaMemset(sendbuff[i], 1, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i],0,size * sizeof(float)));
    CUDACHECK(cudaStreamCreate(s+i));
  }

  //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));

  //calling NCCL communication API. Group API is required when using
  //multiple devices per thread
  NCCLCHECK(ncclGroupStart());

  size_t sendcount=(int)(size/2);
  for (int i = 0; i < nDev; ++i)
//     NCCLCHECK(ncclAllReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
//         comms[i], s[i]));
	   NCCLCHECK(ncclReduce((const void*)sendbuff[i], (void*)recvbuff[i], size, ncclFloat, ncclSum,
	           1,comms[i], s[i]));
//	   NCCLCHECK(ncclAllGather((const void*)(sendbuff[i]+i*sendcount), (void*)recvbuff[i], sendcount, ncclFloat,
//	       comms[i], s[i]));
//	   NCCLCHECK(ncclReduceScatter((const void*)sendbuff[i], (void*)(recvbuff[i]+i*sendcount), sendcount, ncclFloat,ncclSum,
//		   comms[i], s[i]));
  NCCLCHECK(ncclGroupEnd());

  //synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  //输出结果产看
  float ** h_recvbuff=(float **)malloc(nDev*sizeof(float*));
  for(int i = 0; i < nDev; ++i){
	  *(h_recvbuff+i)=(float *)malloc(size*sizeof(float));
  }

  for (int i = 0; i < nDev; ++i) {
      CUDACHECK(cudaSetDevice(i));
      CUDACHECK(cudaMemcpy(*(h_recvbuff+i),recvbuff[i],size*sizeof(float), cudaMemcpyDeviceToHost));
      for (int j=0;j<size;j++)
          {printf("result:=%f \n",*(*(h_recvbuff+i)+j));}
      printf("i:=%d \n",i);
  }

  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }

  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);

  printf("Success \n");
}
