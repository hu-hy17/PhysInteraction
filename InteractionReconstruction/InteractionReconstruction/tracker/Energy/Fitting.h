#pragma once
#include "tracker/ForwardDeclarations.h"
#include "tracker/Types.h"
#include "tracker/Energy/Fitting/Settings.h"
#include "Fitting/DistanceTransform.h"

#include "cudax/CudaHelper.h"
#include "cudax/CublasHelper.h"
#include "../CommonVariances.h"

#include <cuda_gl_interop.h>
//struct MappedResource;//ZH

struct MappedResource {
	struct cudaGraphicsResource* resouce = NULL;
	cudaArray* array = NULL;
	GLuint texid = 0;

	void init(GLuint texid) {
		this->texid = texid;
		checkCudaErrors(cudaGraphicsGLRegisterImage(&resouce, texid, GL_TEXTURE_2D, cudaGraphicsMapFlagsReadOnly));
	}
	void cleanup() {
		checkCudaErrors(cudaGraphicsUnregisterResource(resouce));
	}

	cudaArray* bind() {
		checkCudaErrors(cudaGraphicsMapResources(1, &resouce, 0));
		checkCudaErrors(cudaGraphicsSubResourceGetMappedArray(&array, resouce, 0, 0));
		return array;
	}
	void unbind() {
		checkCudaErrors(cudaGraphicsUnmapResources(1, &resouce, 0));
	}
};

namespace energy{
class Fitting {
protected:
    Camera* camera = NULL;
    DepthTexture16UC1* sensor_depth_texture = NULL;
    HandFinder* handfinder = NULL;
	Model * model = NULL;
	int current_processed_frame_id = 0;

public:
    fitting::Settings _settings;
    fitting::Settings* settings = &_settings;
	DistanceTransform distance_transform;
	MappedResource sensor_depth_map;

public:
#ifdef WITH_CUDA
	void track(DataFrame &frame, LinearSystem &sys, bool rigid_only, bool eval_error, 
			   float &push_error, float &pull_error, int iter, int frame_id, cv::Mat& silhouette, std::vector<int>& real_ADT, 
			   camera_intr intr_para, Eigen::Matrix4f camera_view, bool cal_conf);
	void track3dOnly(DataFrame &frame, LinearSystem &sys, bool rigid_only, bool eval_error, float &push_error, float &pull_error, int iter);
    void init(Worker* worker);
    void cleanup();
	~Fitting();
#else
    void track(DataFrame &frame, LinearSystem &sys, bool rigid_only, bool eval_error, float &push_error, float &pull_error){}
    void init(Worker* worker){}
    void cleanup(){}
#endif
};
} /// energy::
