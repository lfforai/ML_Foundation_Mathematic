/*
 * fileinput.h
 *
 *  Created on: 2018年5月17日
 *      Author: root
 */

#pragma once
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
		}info_N;
	file_input();
	static info* input();
	virtual ~file_input();
};

#endif /* FILEINPUT_H_ */


