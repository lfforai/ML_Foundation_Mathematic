/*
 * fileinput.cpp
 *
 *  Created on: 2018年5月17日
 *      Author: root
 */

#include <src/file_input.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <linux/fb.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string>
using namespace std;

file_input::file_input() {

}

//内存映射方式读取测点名录
file_input::info* file_input::input(){
	    string szFileName="/lf/tree/szwq1.txt";
	    int m_nFile = open(szFileName.c_str(),O_RDWR | O_APPEND | O_CREAT);
	    if (m_nFile < 0)
	    {   m_nFile = 0;
	        printf("input file <0");
	    }

	    struct stat status;
	    fstat(m_nFile, &status);

	    long m_uSize = status.st_size;

	    char *m_pData =(char *)mmap(0, m_uSize, PROT_READ, MAP_SHARED, m_nFile, 0);
	    close(m_nFile);

	    char *result=(char *)malloc((m_uSize+1)*(sizeof(char)));
	    memcpy(result,m_pData,m_uSize);
	    result[m_uSize]='\0';
	    munmap(m_pData,m_uSize);
	    long i=0;
	    long j=0;
	    int* enter=(int *)malloc(m_uSize*sizeof(int));
	    while(result[i]!='\0'){
	    	 if (result[i]=='\n'){
	    		 enter[j]=i;
	    		 j++;
	    	 }
	    	 i++;
	    }

	    info * info_return=(info *)malloc(sizeof(info));
        info_return->data=result;
        info_return->total_row=j-1;
        info_return->total_size=m_uSize+1;
        info_return->split_mark=(int*)malloc((j-1)*sizeof(int));
        memcpy(info_return->split_mark,enter,(j-1)*sizeof(int));
        delete enter;
	    return info_return;
}

file_input::~file_input() {
	// TODO Auto-generated destructor stub
}

