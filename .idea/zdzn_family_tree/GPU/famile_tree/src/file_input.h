/*
 * fileinput.h
 *
 *  Created on: 2018年5月17日
 *      Author: root
 */


#ifndef FILEINPUT_H_
#define FILEINPUT_H_

#include <string.h>
using namespace std;

class file_input {
private:

public:
	struct info{
			char* data;
			long  total_row;
			long  total_size;
			int*  split_mark;//记录每个<\n>的位置
		}info_N;

	file_input();
	static info* input();
	virtual ~file_input();
};

#endif /* FILEINPUT_H_ */


