#ifndef _CUDALIB_HELPER_H_
#define _CUDALIB_HELPER_H_

#include <cusparse.h>
#include <cusolver_common.h>
#include <cublas_v2.h>

#define cusparseSafeCall(expr) \
{ \
	cusparseStatus_t error = expr; \
	if (error != CUSPARSE_STATUS_SUCCESS) { \
		printf("cusparse error: error code %d at LINE %d, in FILE \"%s\"\n", error, __LINE__, __FILE__); \
		std::exit(0); \
	} \
}

#define cusolverSafeCall(expr) \
{ \
	cusolverStatus_t error = expr; \
	if (error != CUSOLVER_STATUS_SUCCESS) { \
		printf("cusolver error: error code %d at LINE %d, in FILE \"%s\"\n", error, __LINE__, __FILE__); \
		std::exit(0); \
	} \
}

#define cublasSafeCall(expr) \
{ \
	cublasStatus_t error = expr; \
	if (error != CUBLAS_STATUS_SUCCESS) { \
		printf("cublas error: error code %d at LINE %d, in FILE \"%s\"\n", error, __LINE__, __FILE__); \
		std::exit(0); \
	} \
}

#endif