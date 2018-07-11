#备注：多gpu之间信息交互的nccl2实现了:allreduce Reduce Broadcast ReduceScatter AllGather

/* Includes, system */
#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
using namespace std;

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
}while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
}while(0)


int main(int argc, char* argv[])
{
  ncclComm_t comms[2];
  //managing 4 devices
  int nDev = 2;
  int size = 32*1024*1024;
  int devs[4] = {0, 1};
  //allocating and initializing device buffers
  float** sendbuff = (float**)malloc(nDev * sizeof(float*));
  float** recvbuff = (float**)malloc(nDev * sizeof(float*));

  float** h_recvbuff=(float**)malloc(nDev * sizeof(float*));//host
  float** h_sendbuff=(float**)malloc(nDev * sizeof(float*));//host

  for (int i = 0; i < nDev; ++i){
	  h_sendbuff[i]=(float*)malloc(size * sizeof(float));
	  for(int j=0;j<size;++j){
		  if(j<100)
		  {h_sendbuff[i][j]=1;}
		  else
		  {if(j>=100 && j<200)
			  h_sendbuff[i][j]=2;
		   else
			  h_sendbuff[i][j]=3;
		  }
	  }
  }

  cudaStream_t* s =(cudaStream_t*)malloc(sizeof(cudaStream_t)*nDev);
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaMalloc(sendbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMalloc(recvbuff + i, size * sizeof(float)));
    CUDACHECK(cudaMemset(recvbuff[i], 0.0, size * sizeof(float)));
    h_recvbuff[i]=(float*)malloc(size * sizeof(float));
    CUDACHECK(cudaStreamCreate(s+i));
  }

  //initializing NCCL
  NCCLCHECK(ncclCommInitAll(comms, nDev, devs));
  //calling NCCL communication API. Group API is required when using

  //multiple devices per thread
  for(int i = 0; i < nDev; ++i){
	 CUDACHECK(cudaSetDevice(i));
	 CUDACHECK(cudaMemcpyAsync(sendbuff[i],h_sendbuff[i],size*sizeof(float),cudaMemcpyHostToDevice,s[i]));
  }

  NCCLCHECK(ncclGroupStart());
  for (int i = 0; i < nDev; ++i)
    { //1、把每个recvbuff[i]=sendbuff[1]+sendbuff[2]+.......sendbuff[nDev]
	  //NCCLCHECK(ncclAllReduce((const void*)sendbuff[i],(void*)sendbuff[i],
    		  //size, ncclFloat, ncclSum,comms[i], s[i]));

	  //2、第root个recvbuff[i]=sendbuff[1]+sendbuff[2]+.......sendbuff[nDev]
	  //NCCLCHECK(ncclReduce((const void*)sendbuff[i],(void*)recvbuff[i],size, ncclFloat, ncclSum,0,comms[i], s[i]));

	 //3、第root个sendbuff[i]的前count个数据，分发到所有其他每个recvbuff[i]上去
	 //recvbuff[1023]=1,而recvbuff[1024]=0
	 //NCCLCHECK(ncclBroadcast((const void*)sendbuff[i],(void*)recvbuff[i],1024,ncclFloat,0,comms[i], s[i]));

	 //4、在(void*)recvbuff[i]+100*i,recvbuff[i]+100*i+100范围内的reduce，以外的为0
	   //NCCLCHECK(ncclReduceScatter((const void*)sendbuff[i],(void*)recvbuff[i]+100*i,100,ncclFloat,ncclSum,comms[i],s[i]));
	 //5、
	  NCCLCHECK(ncclAllGather((const void*)sendbuff[i]+100*i,(void*)recvbuff[i],100,ncclFloat,comms[i],s[i]));
    }
  NCCLCHECK(ncclGroupEnd());

  //synchronizing on CUDA streams to wait for completion of NCCL operation
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaStreamSynchronize(s[i]));
  }

  for(int i = 0; i < nDev; ++i){
	 CUDACHECK(cudaSetDevice(i));
	 CUDACHECK(cudaMemcpyAsync(h_recvbuff[i],recvbuff[i],size*sizeof(float),cudaMemcpyDeviceToHost,s[i]));
	 CUDACHECK(cudaMemcpyAsync(h_sendbuff[i],sendbuff[i],size*sizeof(float),cudaMemcpyDeviceToHost,s[i]));
  }

  //free device buffers
  for (int i = 0; i < nDev; ++i) {
    CUDACHECK(cudaSetDevice(i));
    CUDACHECK(cudaFree(sendbuff[i]));
    CUDACHECK(cudaFree(recvbuff[i]));
  }

  for (int i = 0; i < nDev; ++i) {
	  printf("%d,recvbuff=%f,sendbuff=%f \n",i,h_recvbuff[i][100*i+9],h_sendbuff[i][10]);
	  printf("out:=%d,recvbuff=%f,sendbuff=%f \n",i,h_recvbuff[i][100*i+110],h_sendbuff[i][10]);
  }

  //finalizing NCCL
  for(int i = 0; i < nDev; ++i)
      ncclCommDestroy(comms[i]);
  printf("Success \n");
  return 0;
}
