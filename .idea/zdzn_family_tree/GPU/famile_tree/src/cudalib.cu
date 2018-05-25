/*
 * cudalib.cpp
 *
 *  Created on: 2018年5月17日
 *      Author: root
 */

//#include <src/cudalib.h>
//#include <stdio.h>
//#include "cuda_runtime.h"
//#include "nccl.h"
//#include <string>
//#include "src/cudalib.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "nccl.h"
#include "file_input.h"
#include <src/cudalib.h>
#include <cuda_fp16.h>

#include <string>
using namespace std;

//统计最大祖先个数、最大祖先长度和最大单条长度
template <class T>
__host__ __device__ void len(const char*info,T*result)
{
  int i=0;//index
  char frist_mark=(byte)0;
  while(*(info+i)!='@'){
	  if((*(info+i)=='.' or *(info+i)=='_')){
		  result[0]=(T)((int)result[0]+(int)1);
	  }

	  if(frist_mark==(byte)1)//if frist '.'
	  {result[1]=(T)((int)result[1]+(int)1);}

	  if(frist_mark==(byte)0 and *(info+i)=='.')//if frist '.'
	  { frist_mark=(byte)1;
	  }
	  i=i-1;
  }
  result[2]=(T)(abs(i)-2);
}

template <class T>
__global__ void split_global(T* dum, char* info,long start,long length,int dimblock)
{       extern __shared__ byte s[];
        if (threadIdx.x==0){
           memset(s,(byte)0,3*dimblock*sizeof(T));
        }
    	__syncthreads();
//		T* temp=(T*)malloc(2*sizeof(T));
		T temp[3];
		long length_N = length;
		int step = gridDim.x*blockDim.x;
		const long start_P=start;//开始的位置
		long start_N =threadIdx.x+blockIdx.x*blockDim.x;
		for(long start=start_N;start<length_N;start=start+step)
		   {
			  if((char)*(info+start+start_P)=='\n')
		        {  temp[0]=0;
				   temp[1]=0;
		           len(info+start+start_P,temp);
		           if((int)temp[0]>(int)s[threadIdx.x*3]){
		        	   s[threadIdx.x*3]=temp[0];
		           }


		           if((int)temp[1]>(int)s[threadIdx.x*3+2]){
		        	   s[threadIdx.x*3+1]=temp[1];
		           }

		           if((int)temp[2]>(int)s[threadIdx.x*3+2]){
		        	   s[threadIdx.x*3+2]=temp[2];
		           }

		        }
		    }
		delete temp;

		//同步
		__syncthreads();
		if(threadIdx.x==0)
		memcpy(dum+3*blockIdx.x*blockDim.x*sizeof(T),s,3*blockDim.x*sizeof(T));
		__syncthreads();
}

//切割出所有祖先，为放入hash表用
//dimGrid_N, dimBlock_N,0,s[i]>>>(d_result[i]+h_len_result[deviceCount-2]*max_an_len*max_an_num,d_info[i],max_an_len,max_an_num,(deviceCount-1)*sub_length,sub_length+yu,dimBlock_N
template <class T>
__global__ void scut2ancestors(char* des,long max_an_len,char* info,long start,long length)
{
		long length_N = length;
		int step = gridDim.x*blockDim.x;
		const long start_P=start;//开始的位置
		long start_N =threadIdx.x+blockIdx.x*blockDim.x;
		for(long start=start_N;start<length_N;start=start+step)
		   {
			  if((char)*(info+start+start_P)=='\n')
		        {

		        }
		    }
		delete temp;

		//同步
		__syncthreads();
		if(threadIdx.x==0)
		memcpy(dum+3*blockIdx.x*blockDim.x*sizeof(T),s,3*blockDim.x*sizeof(T));
		__syncthreads();
}



template __host__ __device__ void len<ubyte>(const char*,ubyte *);
template __host__ __device__ void len<byte>(const char*,byte *);
template __global__ void split_global<ubyte>(ubyte*, char*,long,long,int);
template __global__ void split_global<byte>(byte*, char*,long,long,int);

template __host__ __device__ void len<int>(const char*,int *);
template __global__ void split_global<int>(int*, char*,long,long,int);


