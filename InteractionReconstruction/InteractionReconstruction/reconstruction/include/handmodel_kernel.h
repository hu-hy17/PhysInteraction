#pragma once
#include <cuda_runtime.h>
#include "safe_call.hpp"
#include "recon_externs.h"

namespace reconstruction {

	texture<ushort1, 2, cudaReadModeElementType> depth_tex;

	cudaArray* sensor_depth_array = NULL;
}