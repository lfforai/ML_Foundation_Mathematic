/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This example demonstrates how to call CUBLAS library
 * functions both from the HOST code and from the DEVICE code
 * running on the GPU (the latter is available only for the compute
 * capability >= 3.5). The single-precision matrix-matrix
 * multiplication operation, SGEMM, will be performed 3 times:
 * 1) once by calling a method defined in this file (simple_sgemm),
 * 2) once by calling the cublasSgemm library routine from the HOST code
 * 3) and once by calling the cublasSgemm library routine from
 *    the DEVICE code.
 */

/* Includes, system */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>

/* Includes, cuda helper functions */
#include <helper_cuda.h>
#include "file_input.h"
#include <src/split.h>
#include <src/cudalib.h>

using namespace std;


/* Main */
int main(int argc, char **argv)
{file_input::info* re =file_input::input();
 // printf("%s",re->data);
// printf("input total rows is:=%ld \n",re->total_row);
// for (int i=0;i<re->total_row;i++)
//     {printf("i:=%d,re:=%d \n",i,*(re->split_mark+i));}
//
// byte* a=(byte*)malloc(2*sizeof(byte));
// byte* rex="1_DJ#3.Tmp\n@MDNY.SZWQ.11MD#10_A#1_YP#3.ActuAng\n";
// len<byte>((byte *)rex+strlen((char *)rex),a);
// printf("%d,%d \n",(int)a[0],(int)a[1]);
	// char* re_temp=(char *)malloc(100);
	// memcpy(re_temp,re->data,100);
	// printf("%s",re_temp);
 byte* max=split<byte>::max_ancestors_num(re,2);
 delete re->data;
 delete re;
 printf("start!");
}