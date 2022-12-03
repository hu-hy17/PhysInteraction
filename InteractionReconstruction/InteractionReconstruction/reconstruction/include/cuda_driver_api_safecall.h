#ifndef _CUDA_DRIVER_API_SAFECALL_H_
#define _CUDA_DRIVER_API_SAFECALL_H_

#include <cuda.h>
#include <iostream>

#define cuSafeCall(expr) \
{ \
	CUresult error = expr; \
	if (error != CUDA_SUCCESS) { \
		const char *error_msg; \
		cuGetErrorString(error, &error_msg); \
		printf("cuda error: %s, at LINE %d, in FILE \"%s\"\n", error_msg, __LINE__, __FILE__); \
		std::exit(1); \
	} \
}

#endif