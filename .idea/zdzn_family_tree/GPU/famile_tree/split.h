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

using namespace std;


#ifndef SPLIT_H_
#define SPLIT_H_
#include <src/file_input.h>

template<class T>
class split {
public:
	split();
	virtual ~split();
	static char* cut2ancestors(file_input::info* info_of_key,int max_an_num,int max_an_len,int GPU_num);
	static T* max_ancestors_num(file_input::info* info_of_key,int GPU_num);
};
#endif /* SPLIT_H_ */
