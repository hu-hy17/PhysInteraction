#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

// define all constants used in both host and device code
const int c_knn = 4; /*constant number of k-nearest nodes*/
const int c_nn = 8; /*constant number of node neighbors*/

#define c_eps 10.0f//10.0f /*sample radius of nodes*/ //0.025
#define c_eps_sq (c_eps*c_eps) /*squared sample radius*/

#define LENGTH_X 1.0f
#define LENGTH_Y 1.0f
#define LENGTH_Z 1.0f

#define RESOLUTION_X 256//512
#define RESOLUTION_Y 256//512
#define RESOLUTION_Z 256//512

#define c_knnRatioX 1
#define c_knnRatioY 1
#define c_knnRatioZ 1
#define c_knnResX (RESOLUTION_X / c_knnRatioX)
#define c_knnResY (RESOLUTION_Y / c_knnRatioY)
#define c_knnResZ (RESOLUTION_Z / c_knnRatioZ)
#define c_knnNum (c_knnResX * c_knnResY * c_knnResZ)

#define c_knnPerIdxX 8
#define c_knnPerIdxY 8
#define c_knnPerIdxZ 8

#define KNN_BLOCK_X (c_knnPerIdxX / c_knnRatioX)
#define KNN_BLOCK_Y (c_knnPerIdxY / c_knnRatioY)
#define KNN_BLOCK_Z (c_knnPerIdxZ / c_knnRatioZ)
#define KNN_BLOCK_SIZE (KNN_BLOCK_X * KNN_BLOCK_Y * KNN_BLOCK_Z)

#define RESOLUTION_X_B (RESOLUTION_X/c_knnPerIdxX)
#define RESOLUTION_Y_B (RESOLUTION_Y/c_knnPerIdxY)
#define RESOLUTION_Z_B (RESOLUTION_Z/c_knnPerIdxZ)

//#define CELL_X 0.005f//(LENGTH_X/RESOLUTION_X)  0.005f
//#define CELL_Y 0.005f//(LENGTH_Y/RESOLUTION_Y)  0.005f
//#define CELL_Z 0.005f//(LENGTH_Z/RESOLUTION_Z)  0.005f

#define PAUSE { printf("FILE \"%s\", LINE %d\n, ", __FILE__, __LINE__); system("pause"); }
#define EXIT { printf("FILE \"%s\", LINE %d\n", __FILE__, __LINE__); std::exit(0); }

#define MAX_NODES 1500//4096//3072   *4

#define solver_max_outer_iteration_times 2
#define solver_outer_iteration_times 2
#define solver_max_icp_iteration_times 3
#define solver_gn_iter_0 20
#define solver_gn_iter_1 5
#define solver_gn_iter_2 3
#define solver_gn_iter_3 1
#define solver_gn_iter_4 1
#define solver_gn_iter_5 1

#define HUBER_K 0.004f


//ZH add
//#define MU 0.02f
//#define MAX_W 100
//#define VOXEL_SIZE 0.005f
//#define rectify_constant 0.0005  //0.0005

#endif