
/**
 * 作用：求所有key的最大祖先个数和最大key长度
 * 作者：罗峰
 */
#include "nccl.h"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#include <src/cudalib.h>
#include <src/split.h>

template<typename T>
split<T>::split() {
	// TODO Auto-generated constructor stub
}

template<typename T>
split<T>::~split() {
	// TODO Auto-generated destructor stub
}

//返回所有key值中分解出的最多的一个祖先个数
//例如：MDNY.SZWQ.11MD#1.MlyGnCptLo,祖先个数为3
//返回keys列表中单个key最大祖先个数、最大祖先字符串长度，本身字符串最大长度
template<typename T>
T* split<T>::max_ancestors_num(file_input::info* info_of_key,int GPU_num){
	int dimGrid_N=224;
	int dimBlock_N=256;
	int result_n=3;//需要输出几种结果：目前为3

    T* max=(T*)malloc(result_n*sizeof(T));//max[0]:祖先最大数，max[1]:key的最大长度
    max[0]=(T)0;
    max[1]=(T)0;
    max[2]=(T)0;

    char* keys_data=info_of_key->data;
//  long row_num=info_of_key->total_row;
    long buffer_size=info_of_key->total_size;//字节数
    int deviceCount=GPU_num;

    if (GPU_num==-1)
	   {CUDACHECK(cudaGetDeviceCount(&deviceCount));}

    //#记录每个key的最大祖先个数和最大长度的数组
    T** h_num=(T **)malloc(deviceCount*sizeof(T*));
    T** d_num=(T **)malloc(deviceCount*sizeof(T*));
    char** d_info=(char **)malloc(deviceCount*sizeof(char*));

    long yu= buffer_size%deviceCount;
    long sub_length=buffer_size/deviceCount;

    for(int i=0;i<2;i++)
    {cudaOccupancyMaxPotentialBlockSize(
		&dimGrid_N,
		&dimBlock_N,
		(void*)split_global<T>,
		result_n*dimBlock_N*sizeof(T),
		2048);
        printf("第%d次：dimGrid_N=:%d,dimBlock_N:=%d \n",i,dimGrid_N,dimBlock_N);
    }

    //此处假设所有的祖先不超过255个，所以采用char
    for(int i=0;i<deviceCount;i++){
    	CUDACHECK(cudaSetDevice(i));
    	h_num[i]=(T *)malloc(result_n*dimGrid_N*dimBlock_N*sizeof(T));
        CUDACHECK(cudaMalloc(d_num+i,result_n*dimGrid_N*dimBlock_N*sizeof(T)));
        CUDACHECK(cudaMalloc(d_info+i,buffer_size* sizeof(char)));
        if (i==0){//gpu较多情况下使用nccl，在gpu较少情况下可以不使用nccl，这里统一使用nccl无论gpu个数
           CUDACHECK(cudaMemcpy(d_info[i],keys_data,buffer_size*sizeof(char), cudaMemcpyHostToDevice));
        }
    }

//把 d_info[0]通过nccl的boadcast到d_info[i]上去--------------------开始
    if(deviceCount>0)
     {cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*deviceCount);
     ncclComm_t* comms=(ncclComm_t*)malloc(deviceCount*sizeof(ncclComm_t));
     //managing deviceCount devices
     int* devs=(int *)malloc(deviceCount*sizeof(int));
     for(int i=0;i<deviceCount;i++){
    	  CUDACHECK(cudaSetDevice(i));
    	  devs[i]=i;
    	  CUDACHECK(cudaStreamCreate(s+i));
     }
     //initializing NCCL
     NCCLCHECK(ncclCommInitAll(comms,deviceCount, devs));
     //calling NCCL communication API. Group API is required when using
     //multiple devices per thread
     NCCLCHECK(ncclGroupStart());
     for (int i = 0; i < deviceCount; ++i)
   	     NCCLCHECK(ncclBcast((void*)d_info[i],buffer_size,ncclChar,0,comms[i], s[i]));
     NCCLCHECK(ncclGroupEnd());
     //synchronizing on CUDA streams to wait for completion of NCCL operation
     for (int i = 0; i < deviceCount; ++i) {
       CUDACHECK(cudaSetDevice(i));
       CUDACHECK(cudaStreamSynchronize(s[i]));
     }
     //finalizing NCCL
      for(int i = 0; i <deviceCount; ++i)
          {ncclCommDestroy(comms[i]);
           CUDACHECK(cudaStreamDestroy(s[i]));}
     }
//把 d_info[0]通过nccl的boadcast到d_info[i]上去--------------------开始

      cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*deviceCount);
      for(int i=0;i<deviceCount;i++){
     	  CUDACHECK(cudaSetDevice(i));
     	  CUDACHECK(cudaStreamCreate(s+i));
      }

      for(int i=0;i<deviceCount;i++)
      {   CUDACHECK(cudaSetDevice(i));
          if (i!=deviceCount-1)
          {
    	   split_global<T><<<dimGrid_N, dimBlock_N,result_n*dimBlock_N*sizeof(T),s[i]>>>(d_num[i],d_info[i],i*sub_length,sub_length,dimBlock_N);
    	   //2*dimBlock_N*sizeof(T)
          }
          else
          {
    	   split_global<T><<<dimGrid_N, dimBlock_N,result_n*dimBlock_N*sizeof(T),s[i]>>>(d_num[i],d_info[i],(deviceCount-1)*sub_length,sub_length+yu,dimBlock_N);
          }
      }

      for (int i = 0;i < deviceCount;i++)
        {
            CUDACHECK(cudaSetDevice(i));
            CUDACHECK(cudaMemcpyAsync(h_num[i],d_num[i],result_n*dimGrid_N*dimBlock_N*sizeof(T),cudaMemcpyDeviceToHost,s[i]));
        }


      for (int i = 0;i < deviceCount;i++)
      {
          CUDACHECK(cudaSetDevice(i));
          CUDACHECK(cudaStreamSynchronize(s[i]));
      }

      for(int i = 0; i <deviceCount; ++i)
           {CUDACHECK(cudaStreamDestroy(s[i]));}

      for (int i = 0; i < deviceCount;i++)
      {  for (int j = 0; j < dimGrid_N*dimBlock_N;j++)
          {
        	 if((int)h_num[i][j*result_n]>(int)max[0]){
        		 max[0]=h_num[i][j*result_n];
//        		 printf("max[0]：=%d \n",max[0]);
        	 }

        	 if((int)h_num[i][j*result_n+1]>(int)max[1]){
        		 max[1]=h_num[i][j*result_n+1];
//        		 printf("max[1]：=%d \n",max[1]);
        	 }

        	 if((int)h_num[i][j*result_n+2]>(int)max[2]){
        		 max[2]=h_num[i][j*result_n+2];
//        		 printf("max[1]：=%d \n",max[1]);
        	 }
//        	     if(j<2 and (h_num[i][j*2]!=0 or h_num[i][j*2+1]!=0))
//        	     printf("max[0]=%d,max[1]=%d,i:=%d,j:=%d \n",(int)h_num[i][j*2],(int)h_num[i][j*2+1],i,j);
          }
      }
      max[2]=max[2]+1;//程序有问题统计出来最大值少1？
      max[1]=max[1]+1;//
      printf("%d,%d,%d \n",max[0],max[1],max[2]);

      for (int i = 0;i < deviceCount;i++)
      {
//        CUDACHECK(cudaSetDevice(i));
          CUDACHECK(cudaFree(d_info[i]));
          CUDACHECK(cudaFree(d_num[i]));
      }
      return max;
};


//int num记录条数，max_an_len是祖先长度,max_an_num祖先个数，
//按6个祖先，每个32最大长度，有1000万个点计算需要消耗内存为1.8g存储
template<typename T>
char* split<T>::cut2ancestors(file_input::info* info_of_key,int max_an_num,int max_an_len,int GPU_num){
    int num=info_of_key->total_row;
    char* keys_data=info_of_key->data;
    long buffer_size=info_of_key->total_size;//字节数
	int dimGrid_N=224;
	int dimBlock_N=256;

	int deviceCount=2;
	if (GPU_num==-1)
	   {CUDACHECK(cudaGetDeviceCount(&deviceCount));}

	//每个GPU分解一部分
	char** h_result=(char**)malloc(deviceCount*sizeof(char*));
    char** d_result=(char **)malloc(deviceCount*sizeof(char*));
    char** d_info=(char **)malloc(deviceCount*sizeof(char*));

    //对输入数据的切割
    long yu= buffer_size%deviceCount;
    long sub_length=buffer_size/deviceCount;

    //对应输入数据对输出数据进行切割------------------------------
    //存放每个gpu开始的位置索引
    int* h_len_result=(int*)malloc(deviceCount*sizeof(int));
    for(int i=0;i<deviceCount;i++)
    {  if(i!=0)
       {for(int j=0;j<num-2;j++)
          {if(i*sub_length>=info_of_key->split_mark[j] and i*sub_length<info_of_key->split_mark[j+1])
        	  if(info_of_key->split_mark[info_of_key->split_mark[j]]!='\n')
        	     h_len_result[i]=j+1;
        	  else
        		 h_len_result[i]=j;
    	   }
       }
       else
       {h_len_result[0]=0;}
       printf(" h_len_result[i]:=%d \n",h_len_result[i]);
    }

    //-------------------------切割结束----------------------------

    for(int i=0;i<2;i++)
    {cudaOccupancyMaxPotentialBlockSize(
		&dimGrid_N,
		&dimBlock_N,
		(void*)scut2ancestors<T>,
		max_an_num*dimBlock_N*sizeof(T),
		2048);
        printf("第%d次：dimGrid_N=:%d,dimBlock_N:=%d \n",i,dimGrid_N,dimBlock_N);
    }

    //初始化所有gpu上数据
    for(int i=0;i<deviceCount;i++){
    	CUDACHECK(cudaSetDevice(i));
    	h_result[i]=(char *)malloc(num*max_an_len*max_an_num*sizeof(char));
        CUDACHECK(cudaMalloc(d_result+i,num*max_an_len*max_an_num*sizeof(char)));
        CUDACHECK(cudaMalloc(d_info+i,buffer_size* sizeof(char)));
        if (i==0){//gpu较多情况下使用nccl，在gpu较少情况下可以不使用nccl，这里统一使用nccl无论gpu个数
           CUDACHECK(cudaMemcpy(d_info[i],keys_data,buffer_size*sizeof(char), cudaMemcpyHostToDevice));
        }
    }

//把 d_info[0]通过nccl的boadcast到d_info[i]上去--------------------开始
    if(deviceCount>0)
    { cudaStream_t* s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*deviceCount);
     ncclComm_t* comms=(ncclComm_t*)malloc(deviceCount*sizeof(ncclComm_t));
     //managing deviceCount devices
     int* devs=(int *)malloc(deviceCount*sizeof(int));
     for(int i=0;i<deviceCount;i++){
    	  CUDACHECK(cudaSetDevice(i));
    	  devs[i]=i;
    	  CUDACHECK(cudaStreamCreate(s+i));
     }

     //initializing NCCL
     NCCLCHECK(ncclCommInitAll(comms,deviceCount, devs));
     //calling NCCL communication API. Group API is required when using
     //multiple devices per thread
     NCCLCHECK(ncclGroupStart());
     for (int i = 0; i < deviceCount; ++i)
   	     NCCLCHECK(ncclBcast((void*)d_info[i],buffer_size,ncclChar,0,comms[i], s[i]));
     NCCLCHECK(ncclGroupEnd());
     //synchronizing on CUDA streams to wait for completion of NCCL operation
     for (int i = 0; i < deviceCount; ++i) {
       CUDACHECK(cudaSetDevice(i));
       CUDACHECK(cudaStreamSynchronize(s[i]));
     }
     //finalizing NCCL
     for(int i = 0; i <deviceCount; ++i)
          {ncclCommDestroy(comms[i]);
           CUDACHECK(cudaStreamDestroy(s[i]));}
    }
//把 d_info[0]通过nccl的boadcast到d_info[i]上去--------------------开始

        cudaStream_t*  s = (cudaStream_t*)malloc(sizeof(cudaStream_t)*deviceCount);
        for(int i=0;i<deviceCount;i++){
       	  CUDACHECK(cudaSetDevice(i));
       	  CUDACHECK(cudaStreamCreate(s+i));
        }

        //记录当前记录存放位置的变量
        unsigned long long int ** p_mark=(unsigned long long int ** )malloc(deviceCount*sizeof(unsigned long long int *));
        for(int i=0;i<deviceCount;i++){
           CUDACHECK(cudaSetDevice(i));
           CUDACHECK(cudaMalloc(p_mark+i,sizeof(unsigned long long int)));
        }


        for(int i=0;i<deviceCount;i++)
        {   CUDACHECK(cudaSetDevice(i));
            if (i==0)
            {
            scut2ancestors<T><<<dimGrid_N, dimBlock_N,max_an_num*dimBlock_N*sizeof(T),s[i]>>>(d_result[i],max_an_len,max_an_num,d_info[i],i*sub_length,sub_length,p_mark[i],dimBlock_N);
            }
            else
            { if(i==deviceCount-1)
              scut2ancestors<T><<<dimGrid_N, dimBlock_N,max_an_num*dimBlock_N*sizeof(T),s[i]>>>(d_result[i]+h_len_result[i]*max_an_len*max_an_num,max_an_len,max_an_num,d_info[i],i*sub_length,sub_length+yu,p_mark[i],dimBlock_N);
              else
              scut2ancestors<T><<<dimGrid_N, dimBlock_N,max_an_num*dimBlock_N*sizeof(T),s[i]>>>(d_result[i]+h_len_result[i]*max_an_len*max_an_num,max_an_len,max_an_num,d_info[i],i*sub_length,sub_length,p_mark[i],dimBlock_N);
            }
        }


        for (int i = 0;i < deviceCount;i++)
          {
              CUDACHECK(cudaSetDevice(i));
              CUDACHECK(cudaMemcpyAsync(h_result[i],d_result[i],num*max_an_len*max_an_num*sizeof(char),cudaMemcpyDeviceToHost,s[i]));
          }


        unsigned long long int** h_mark=(unsigned long long int**)malloc(sizeof(unsigned long long int *));
        for(int i=0;i<deviceCount;i++){
                h_mark[i]=(unsigned long long int*)malloc(sizeof(unsigned long long int));
         }
        for (int i = 0;i < deviceCount;i++)
             {
                 CUDACHECK(cudaSetDevice(i));
                 CUDACHECK(cudaMemcpyAsync(h_mark[i],p_mark[i],sizeof(unsigned long long int), cudaMemcpyDeviceToHost,s[i]));
             }

        for (int i = 0;i < deviceCount;i++)
        {
            CUDACHECK(cudaSetDevice(i));
            CUDACHECK(cudaStreamSynchronize(s[i]));
        }

        unsigned long long int total=0;
        for(int i = 0; i <deviceCount;i++)
             {CUDACHECK(cudaStreamDestroy(s[i]));
              total=total+*h_mark[i];
              printf("%dl  \n",*h_mark[i]);
             }
        printf("%dl  \n",total);
        char* result=(char *)malloc(total*max_an_len*sizeof(char));

        total=0;
        for(int i = 0; i<deviceCount;i++)
         {  memcpy(result+total*max_an_len,h_result[i]+h_len_result[i]*max_an_len*max_an_num,*h_mark[i]*max_an_len*sizeof(char));
        	total=total+*h_mark[i];
         }

       memset(result+total*max_an_len,'&',max_an_len);

       for (int i = 0;i < deviceCount;i++)
       {
           CUDACHECK(cudaFree(d_info[i]));
           CUDACHECK(cudaFree(d_result[i]));
           CUDACHECK(cudaFree(p_mark[i]));
       }

//       int j=0;
//       while(*(result+j*max_an_len)!='&'){
////    	   if(*(result+j*max_an_len)=='\0')
//              printf("i=:%d,%s \n",j,result+j*max_an_len);
//              j++;
//        }
//       printf("i=:%d,%c \n",j,(char)*(result+j*max_an_len+1));
//       printf("max:=%d",max_an_len);
       return result;
}

template class split<byte>;
template class split<ubyte>;
template class split<int>;

