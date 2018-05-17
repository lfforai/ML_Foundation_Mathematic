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

namespace gpu {

class split {
public:
	split();
	virtual ~split();
	int max_ancestors_num(file_input::info info_of_key,int GPU_num=-1);
};

} /* namespace gpu */

#endif /* SPLIT_H_ */
