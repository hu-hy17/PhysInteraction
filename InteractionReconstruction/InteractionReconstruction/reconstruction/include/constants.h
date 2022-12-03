#pragma once
// define all constants used in both host and device code
const int c_knn = 4; /*constant number of k-nearest nodes*/
const int c_nn = 8; /*constant number of node neighbors*/

#define c_eps 20.0f /*sample radius of nodes: mm  */  
#define c_eps_sq (c_eps*c_eps) /*squared sample radius*/

#define MAX_NODES 4096//4096//3072