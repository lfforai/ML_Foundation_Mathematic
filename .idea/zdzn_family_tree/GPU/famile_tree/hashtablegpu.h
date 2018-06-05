/*
 * hashtablegpu.h
 *  基于多GPU的hash
 *  Created on: 2018年5月17日
 *      Author: 罗锋
 */

#ifndef HASHTABLEGPU_H_
#define HASHTABLEGPU_H_


#include <stdlib.h>
#include <stdio.h>

template<class T1,class T2>
class hashtable_gpu {

public:
      struct bucket{
				T1* key;
				T2* value;
				int index; //在segList中的位置
				int Ld;//当前局部深度（会变的）
				int Ld_init;//用于在init时候初始化桶所属gpu和层次
				int first_Ld;//该桶首次出现的深度（不变的）
				int gpu_index;//在多GPU下每个桶属于第gpu_index的gpu
				int is_reliable;//默认为虚桶
				int mark;//桶中存放的数据量
        };

		struct extendiblehashtable{
		  bucket * bucketList;

		  int Gd;//当前（初始化）全局深度
	      int bucketNum_init;//当前（初始化）桶的数量

	      int Gd_max;//最大全局深度
	      int bucketNum_max;//最大桶的数量

	      int e_key_len;//每个桶key长度
		  int e_value_len;//每个桶value的长度

		  int b;//每个桶可以存放的数据总量
		};

	extendiblehashtable extendiblehashtable_N;
	hashtable_gpu(int Gd,int Gd_max,int e_key_len,int e_value_len,int GPU);
	void Recursion_Ld(bucket* bucketList,int i);
	void Predo(char* input,int max_an_len);
	virtual ~hashtable_gpu();
};



#endif /* HASHTABLEGPU_H_ */
