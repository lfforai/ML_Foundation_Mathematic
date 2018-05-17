/*
 * cudalib.h
 *
 *  Created on: 2018年5月17日
 *      Author: root
 */

#ifndef CUDALIB_H_
#define CUDALIB_H_

namespace gpu {
int dimGrid=32;
int dimBlock=256;
__global__ void split_global(Matrix A, Matrix B, Matrix C);
} /* namespace gpu */

#endif /* CUDALIB_H_ */
