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
#include "cuda_runtime.h"
//include的输入是有顺序
#include "nccl.h"
#include <src/split.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <src/cudalib.h>

//#include <src/cudalib.h>
using namespace std;

split::split() {
	// TODO Auto-generated constructor stub

}

split::~split() {
	// TODO Auto-generated destructor stub
}

//返回所有key值中分解出的最多的一个祖先个数
//例如：MDNY.SZWQ.11MD#1.MlyGnCptLo,祖先个数为3
//返回keys列表中单个key最大祖先个数和最大字符串长度
int* split::max_ancestors_num(file_input::info* info_of_key,int GPU_num){
	int dimGrid_N=224;
	int dimBlock_N=256;

	int* max=(int*)malloc(2*sizeof(int));//max[0]:祖先最大数，max[1]:key的最大长度
    char* keys_data=info_of_key->data;
    long row_num=info_of_key->total_row;
    long buffer_size=info_of_key->total_size;//字节数
    printf("sizeof:=%ld",buffer_size);
    int  deviceCount=GPU_num;

    if (GPU_num==-1)
	   {CUDACHECK(cudaGetDeviceCount(&deviceCount));}

//    #记录每个key的最大祖先个数和最大长度的数组
    char** h_num=(char **)malloc(deviceCount*sizeof(char*));
    //每个gpu一个数组记录最大值
    char** d_num=(char **)malloc(deviceCount*sizeof(char*));
    char** d_info=(char **)malloc(deviceCount*sizeof(char*));

    long yu= row_num%deviceCount;
    long sub_length= row_num/deviceCount;

    //此处假设所有的祖先不超过255个，所以采用char
    for(int i=0;i<deviceCount;i++){
    	CUDACHECK(cudaSetDevice(i));
    	//每个gpu一个数组记录：每个线程获取的部分key值中最大的祖先数
        CUDACHECK(cudaMalloc(d_num + i,dimGrid_N*dimBlock_N*sizeof(char)));
        h_num[i]=(char *)malloc(dimGrid_N*dimBlock_N*sizeof(char*));
        CUDACHECK(cudaMalloc(d_info + i,buffer_size* sizeof(char)));
        if (i==0){//gpu较多情况下使用nccl，在gpu较少情况下可以不使用nccl，这里统一使用nccl无论gpu个数
           CUDACHECK(cudaMemcpy(d_info[i],keys_data,buffer_size*sizeof(char), cudaMemcpyHostToDevice));
        }
    }
//把 d_info[0]通过nccl的boadcast到d_info[i]上去--------------------开始
     ncclComm_t* comms=(ncclComm_t*)malloc(deviceCount*sizeof(ncclComm_t));
     //managing deviceCount devices
     int* devs=(int *)malloc(deviceCount*sizeof(int));
     for(int i=0;i<deviceCount;i++){
    	  devs[i]=i;
     }
     cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*deviceCount);

     //initializing NCCL
     NCCLCHECK(ncclCommInitAll(comms,deviceCount, devs));
     //calling NCCL communication API. Group API is required when using
     //multiple devices per thread

     NCCLCHECK(ncclGroupStart());
     for (int i = 0; i < deviceCount;i++)
     	  NCCLCHECK(ncclBcast((void*)d_info[i],(size_t)buffer_size,ncclChar,0,comms[i], s[i]));

     NCCLCHECK(ncclGroupEnd());

     //synchronizing on CUDA streams to wait for completion of NCCL operation
     for (int i = 0; i < deviceCount;i++) {
       CUDACHECK(cudaSetDevice(i));
       CUDACHECK(cudaStreamSynchronize(s[i]));
     }
     //finalizing NCCL
     for(int i = 0; i <deviceCount;i++)
          ncclCommDestroy(comms[i]);
//把 d_info[0]通过nccl的boadcast到d_info[i]上去--------------------开始
     printf("kaishi1 \n");
     for(int i=0;i<deviceCount;i++){
       if (i!=deviceCount-1)
          {CUDACHECK(cudaSetDevice(i));
    	   split_global<<<dimGrid_N, dimBlock_N,dimBlock_N>>>(d_num[i],d_info[i]+i*sub_length,sub_length);}
       else
          {CUDACHECK(cudaSetDevice(i));
    	   split_global<<<dimGrid_N, dimBlock_N,dimBlock_N>>>(d_num[i],d_info[i]+(deviceCount-1)*sub_length,sub_length+yu);}
    }
	   CUDACHECK(cudaDeviceSynchronize());

   	return max;
};

