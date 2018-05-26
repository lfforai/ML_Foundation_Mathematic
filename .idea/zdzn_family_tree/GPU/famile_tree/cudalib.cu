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
				   temp[2]=0;
		           len(info+start+start_P,temp);
		           if((int)temp[0]>(int)s[threadIdx.x*3]){
		        	   s[threadIdx.x*3]=temp[0];
		           }

		           if((int)temp[1]>(int)s[threadIdx.x*3+1]){
		        	   s[threadIdx.x*3+1]=temp[1];
		           }

		           if((int)temp[2]>(int)s[threadIdx.x*3+2]){
		        	   s[threadIdx.x*3+2]=temp[2];
		           }

		        }
		    }
		free(temp);

		//同步
		__syncthreads();
		if(threadIdx.x==0)
		memcpy(dum+3*blockIdx.x*blockDim.x*sizeof(T),s,3*blockDim.x*sizeof(T));
		__syncthreads();
}

//切割出所有祖先，为放入hash表用
//dimGrid_N, dimBlock_N,0,s[i]>>>(d_result[i]+h_len_result[deviceCount-2]*max_an_len*max_an_num,d_info[i],max_an_len,max_an_num,(deviceCount-1)*sub_length,sub_length+yu,dimBlock_N
template <class T>
__global__ void scut2ancestors(char* des,int max_an_len,int max_an_num,char* info,long start,long length,unsigned long long int* mark,int dimblock)
{       //为每个thread分配空间记录当前分解记录的各"_"和“.”的位置
	    extern __shared__ byte s[];
		long length_N = length;
		int step = gridDim.x*blockDim.x;
		const long start_P=start;//开始的位置
		long start_N =threadIdx.x+blockIdx.x*blockDim.x;
		if(threadIdx.x==0){
			*mark=0;
			memset(s,(byte)0,max_an_num*dimblock*sizeof(T));
		}
		__syncthreads();

		for(long start=start_N;start<length_N;start=start+step)
		   {
			  if((char)*(info+start+start_P)=='\n')
		        { //所有的相对位置都是相对于每个‘\n’回车符号来的
				  //1、-----------------------------------------------------
				  int dian_i=0;//第一个"."前面的最大祖先的根位置
				  int last_i=0;//开始的"@"

				  int first_dian_mark=0;//是否第一个"."
				  int i=0;
				  while(*(info+start+start_P+i)!='@')
		          {if(*(info+start+start_P+i)=='.' and first_dian_mark==0)
		        	  {dian_i=i;
		        	   first_dian_mark=1;
		        	  }
		           i=i-1;
		          }
				  last_i=i;
//				  printf("a:=%c \n",*(info+start+start_P+i));
				  dian_i=dian_i-1;//最大祖先的开始位置info+start+start_P+dian_i
				  last_i=last_i+1;//最大祖先的结束位置info+start+start_P+last_i
				  //----------------------------------------------------


//				  //2、记录各级祖先节点的位置----"."和“—”
				  int last_i_N=last_i;//保留祖先开始位置记录
				  int dian_i_N=dian_i+1;//“.”位置
				  int an_num=0;//祖先数目
				  while(last_i<=dian_i_N and an_num<max_an_num){
					  if(*(info+start+start_P+last_i)=='.' or *(info+start+start_P+last_i)=='_'){
						s[threadIdx.x*max_an_num+an_num]=last_i;
//						printf("*(info+start+start_P+s[w]):%c,theid:=%d \n",*(info+start+start_P+s[threadIdx.x*max_an_num+an_num]),threadIdx.x*max_an_num+an_num);
						an_num=an_num+1;
					  }
					  last_i=last_i+1;
				  }

//				  //3、依次输出各祖先节点
				  an_num=0;
				  unsigned long long int  position=0;
				  //(*(info+start+start_P+s[threadIdx.x*max_an_num+an_num])=='.' or *(info+start+start_P+s[threadIdx.x*max_an_num+an_num])=='_')  and
				  while((*(info+start+start_P+s[threadIdx.x*max_an_num+an_num])=='.' or *(info+start+start_P+s[threadIdx.x*max_an_num+an_num])=='_') and an_num<max_an_num){
					   position=(unsigned long long int )atomicAdd((unsigned long long int  *)mark,(unsigned long long int )1);
					   memcpy(des+position*max_an_len,info+start+start_P+last_i_N,s[threadIdx.x*max_an_num+an_num]-last_i_N);
					   *(des+position*max_an_len+s[threadIdx.x*max_an_num+an_num]-last_i_N)='\0';
//					   if(*(des+position*max_an_len+s[threadIdx.x*max_an_num+an_num]-last_i_N-1)=='.')
//					   {char* temp=(char *)malloc(-last_i_N);
//					    memcpy(temp,info+start+start_P+last_i_N,-last_i_N);
//					    *(temp-last_i_N)='\0';
//						printf("n:=%s||%s||%c,%c,%d,%d \n",des+position*max_an_len,temp,*(info+start+start_P+dian_i),*(info+start+start_P+last_i_N),(int)s[threadIdx.x*max_an_num+an_num],an_num);
//					   }
					   an_num=an_num+1;
				  }

				  //放入整条记录
				  position=(unsigned long long int )atomicAdd((unsigned long long int  *)mark,(unsigned long long int )1);
				  memcpy(des+position*max_an_len,info+start+start_P+last_i_N,-last_i_N+1);
				  *(des+position*max_an_len-last_i_N)='\0';
//				  if(*(des+position*max_an_len)=='\0')
//				  {char* temp=(char *)malloc(-last_i_N+1);
//				   memcpy(temp,info+start+start_P+last_i_N,-last_i_N);
//				   *(temp-last_i_N)='\0';
//				   printf("ok::%s||%s||%c,%c,%d,%d \n",des+position*max_an_len,temp,*(info+start+start_P+dian_i),*(info+start+start_P+last_i_N),(int)s[threadIdx.x*max_an_num+an_num],an_num);
//				   delete temp;
//				  }
//				  printf("ok:=%s \n",des+position*max_an_len);

				  __syncthreads();
				  if(threadIdx.x==0){memset(s,(byte)0,max_an_num*dimblock*sizeof(T));}
				  __syncthreads();
		        }
		    }
}

template __host__ __device__ void len<ubyte>(const char*,ubyte *);
template __host__ __device__ void len<byte>(const char*,byte *);
template __host__ __device__ void len<int>(const char*,int *);

template __global__ void split_global<ubyte>(ubyte*, char*,long,long,int);
template __global__ void split_global<byte>(byte*, char*,long,long,int);
template __global__ void split_global<int>(int*, char*,long,long,int);

template __global__ void scut2ancestors<byte>(char*,int ,int ,char*,long,long,unsigned long long int *,int);
template __global__ void scut2ancestors<ubyte>(char*,int ,int ,char*,long,long,unsigned long long int *,int);
template __global__ void scut2ancestors<int>(char*,int ,int ,char*,long,long,unsigned long long int *,int);
