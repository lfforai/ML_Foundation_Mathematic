/*
 * hashtablegpu.cpp
 *
 *  Created on: 2018年5月17日
 *      Author: 罗锋
 *  基于多GPU的hash表实现
 */
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include "nccl.h"
#include "file_input.h"
#include <src/cudalib.h>
#include <cuda_fp16.h>
#include <src/hashtablegpu.h>
#include <cmath>

//递归调用搜索所有的bucketList中index个桶分裂出的所有桶的相关信息更新
//Gd_max为全局最大深度
template<typename T1,typename T2>
void hashtable_gpu<T1,T2>::Recursion_Ld(bucket* bucketList,int i)
 {   int index=i+pow(2,bucketList[i].Ld_init);
     bucketList[index].Ld_init=bucketList[i].Ld_init+1;
     bucketList[index].first_Ld=bucketList[i].Ld_init+1;
     bucketList[index].gpu_index=bucketList[i].gpu_index;
     bucketList[i].Ld_init=bucketList[i].Ld_init+1;
 }

template<typename T1,typename T2>
hashtable_gpu<T1,T2>::hashtable_gpu(int Gd,int Gd_max,int e_key_len,int e_value_len,int GPU_num) {

    (this->extendiblehashtable_N).Gd=Gd;
    (this->extendiblehashtable_N).Gd_max=Gd_max;


    (this->extendiblehashtable_N).bucketNum_init=pow(2,Gd);
    int bucketNum_init=(this->extendiblehashtable_N).bucketNum_init;
    (this->extendiblehashtable_N).bucketNum_max=pow(2,Gd_max);
    int bucketNum_max=(this->extendiblehashtable_N).bucketNum_max;

    (this->extendiblehashtable_N).bucketList=(bucket*)malloc((this->extendiblehashtable_N).bucketNum_max*sizeof(bucket));
    bucket* init_bucketList=(this->extendiblehashtable_N).bucketList;

    (this->extendiblehashtable_N).e_key_len=e_key_len;
    (this->extendiblehashtable_N).e_value_len=e_value_len;

//根据Gd转化为二叉树的层次
//  Ld=0                                       node(root)
//  Ld=1                      node(0)                                  node(1)
//  Ld=2          node(00)              node(10)             node(01)             node(11)
//  Ld=3     node(000) node(100)   node(010) node(110) | node(001) node(101)  node(011) node(111)
//  gpu分配   -------------------gpu0-----------------    ---------------------gpu1--------------
//在cpu上初始化为一个深度为Gd_max的满二叉树，并且分配其所属的不同GPU
    int deviceCount=-1;
    if (deviceCount==-1)
	   {CUDACHECK(cudaGetDeviceCount(&deviceCount));}

//bucketNum_init分段到每个gpu上,此后从每个属于特定gpu分裂出去的桶都属于该特定gpu
    int sub_len=(int)(bucketNum_init/deviceCount);
    //int yu_len=bucketNum_init%deviceCount;
    printf("sub_len:%d \n",sub_len);

    int* cpu_start_p=(int* )malloc(deviceCount*sizeof(int));
    for(int i=0;i<deviceCount;i++){
    	cpu_start_p[i]=i*sub_len;//属于每个gpu的桶的开始位置
//      printf("cpu_start_p[i]:=%d \n",cpu_start_p[i]);
    }

    //从1号桶开始存储数据-------bucketNum_max号桶
    for(int i=0;i<bucketNum_max;i++)
     {  if(i<bucketNum_init)
        {  init_bucketList[i].Ld=Gd;//初始化时候所有桶都在全局深度默认为Gd=3
           init_bucketList[i].first_Ld=Gd;
           init_bucketList[i].Ld_init=Gd;
           init_bucketList[i].is_reliable=1;

           for(int j=0;j<deviceCount-1;j++){
        	  if(i>=cpu_start_p[j] and i<cpu_start_p[j+1])
        		  {init_bucketList[i].gpu_index=j;
//        	       printf("gpu:=%d,%d \n",init_bucketList[i].gpu_index,i);
        		  }
        	  else
        	     { init_bucketList[i].gpu_index=deviceCount-1;
//                   printf("gpu:=%d,%d \n",init_bucketList[i].gpu_index,i);
        	     }
             }
        }
     }

//遍历所有已经初始的桶，分解出由这个桶分裂出所有子桶，并赋值同一个gpu序号
//i:=0,gpu_index=0,Ld=3,Ld_init=5,frist=3
//i:=1,gpu_index=0,Ld=3,Ld_init=5,frist=3
    int tmp_i=bucketNum_init;
    while(tmp_i<bucketNum_max)
    { for(int i=0;i<tmp_i;i++)
         Recursion_Ld(init_bucketList,i);
      tmp_i=tmp_i*2;
    }

//按所属gpu将桶分解到不同的device上去

//    for(int i=0;i<bucketNum_max;i++){
//    	printf("i:=%d,gpu_index=%d,Ld=%d,Ld_init=%d,frist=%d \n",i,init_bucketList[i].gpu_index,init_bucketList[i].Ld,init_bucketList[i].Ld_init,init_bucketList[i].first_Ld);
//    }

//       CUDACHECK(cudaSetDevice(i));
 }

//分解输入集，提前计算每个桶存储的数据是否超标，是否需要分裂并实现分裂
template<typename T1,typename T2>
void hashtable_gpu<T1,T2>::Predo(){

}


template<typename T1,typename T2>
hashtable_gpu<T1,T2>::~hashtable_gpu() {
	// TODO Auto-generated destructor stub
}

template class hashtable_gpu<byte,byte>;
