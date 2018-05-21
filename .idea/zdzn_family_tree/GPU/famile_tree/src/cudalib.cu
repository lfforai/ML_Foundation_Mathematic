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

template <class T>
__host__ __device__ void len(const char*info,T*result)
{
  int i=0;//index
  char frist_mark=0;
  while(*(info+i)!='@'){

	  if((*(info+i)=='.' or *(info+i)=='_')){
		  result[0]=result[0]+1;
	  }

	  if(frist_mark==1)//if frist '.'
	  {result[1]=result[1]+1;}

	  if(frist_mark==0 and *(info+i)=='.')//if frist '.'
	  { frist_mark=1;
	  }
	  i=i-1;
  }
}

template <class T>
__global__ void split_global(T* dum, char* info,long start,long length)
{       extern __shared__ byte s[];
		long length_N = length;
		T* temp=(T*)malloc(2*sizeof(T));
		int step = gridDim.x*blockDim.x;
		long start_P=start;//开始的位置
		long start_N =threadIdx.x+blockIdx.x*blockDim.x;
		for(long start=start_N;start<length_N;start=start+step)
		   {  if((char)*(info+start+start_P)=='\n')
		        {
		           len(info+start+start_P,temp);
		           if((int)temp[0]>(int)s[threadIdx.x*2]){
		        	   s[threadIdx.x*2]=temp[0];
		        	   if((int)temp[0]<1)
		        	     {T* temp_s=(T*)malloc(temp[1]*sizeof(T)+1+20);
						  memcpy(temp_s,info+start+start_P-(int)temp[1]*sizeof(T)-20,(int)temp[1]*sizeof(T)+1+20);
						  printf("s0:%c,s1:%d,s2:%d,小于:=%s \n",(char)(*(info+start+start_P-1)),(int)temp[0],(int)temp[1],temp_s);
						  printf("------------------------ \n");
						  delete temp_s;
		        	     }
		           }

		           if(threadIdx.x==27 and start==start_N){
		        	   T* temp_s=(T*)malloc(temp[1]*sizeof(T));
		        	   						  memcpy(temp_s,info+start+start_P-(int)temp[1]*sizeof(T),(int)temp[1]*sizeof(T));
		        	   						  printf("s0:%c,s1:%d,s2:%d,:=%s \n",(char)(*(info+start+start_P-1)),(int)temp[0],(int)temp[1],temp_s);
		        	   						  printf("------------------------ \n");
		        	   						  delete temp_s;
		           		        	   }

		           if((int)temp[1]>(int)s[threadIdx.x*2+1]){
		        	   s[threadIdx.x*2+1]=temp[1];
		           }

		           if((int)temp[0]>6)
		         				 {
		         				  T* temp_s=(T*)malloc(temp[0]*sizeof(T)+1);
		         				  memcpy(temp_s,info+start+start_P-temp[0]*sizeof(T),temp[0]*sizeof(T));
		         				  printf("大于6：=thread:=%d,%d,%s \n",(int)threadIdx.x,(int)temp[0],temp_s);
		         				  printf("--------------- \n");
		         				  delete temp_s;
		         				 }
		         			   temp[0]=0;
		         			   temp[1]=0;
		        }
		    }
		   delete temp;
		//同步
		__syncthreads();
		if(threadIdx.x==0)
		memcpy(dum+2*blockIdx.x*blockDim.x*sizeof(T),s,2*blockDim.x*sizeof(T));
		__syncthreads();
}

template __host__ __device__ void len<ubyte>(const char*,ubyte *);
template __host__ __device__ void len<byte>(const char*,byte *);
template __global__ void split_global<ubyte>(ubyte*, char*,long,long);
template __global__ void split_global<byte>(byte*, char*,long,long);

template __host__ __device__ void len<int>(const char*,int *);
template __global__ void split_global<int>(int*, char*,long,long);


