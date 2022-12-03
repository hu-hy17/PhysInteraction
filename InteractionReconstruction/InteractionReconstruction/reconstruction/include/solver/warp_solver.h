#ifndef _WARP_SOLVER_VER3_
#define _WARP_SOLVER_VER3_

#include <vector>
#include <array>
#include <Eigen/Eigen>
#include "image_2d.hpp"
#include "camera.hpp"
#include "./gpu/warp_field.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

class WarpSolver {
	typedef Eigen::Matrix<float, 6, 1> Vector6f;
public:
	WarpSolver() {}
	WarpSolver(int _frame_idx, int _iter_idx, int _max_iter_times, const std::string &_results_dir) :
		frame_idx(_frame_idx), iter_idx(_iter_idx), max_iter_times(_max_iter_times), results_dir(_results_dir) {} /*debug ctor*/

	// Set regularization coefficient
	void set_reg_coeff(float _reg_coeff) { m_reg_coeff = _reg_coeff; }

	// Input camera, depth vertex and normal maps, warped vertex and normal maps,
	// canonical vertex and normals maps (camera pose will be updated during Gauss-Newton iteration)
	void input(const Camera &_camera,
		const cv::Mat &_depth_vertex_map, const cv::Mat &_depth_normal_map,
		const cv::Mat &_warp_vertex_map, const cv::Mat &_warp_normal_map,
		const cv::Mat &_can_vertex_map, const cv::Mat &_can_normal_map);

	/*each Gauss-Newton iteration uses linearized twist, and also update warping field*/
	void solve(WarpField &_warp_field);

	// Dump data pairs
	void dump_data_pairs(const std::string &_data_pair_file) const;

	// Dump data pairs in 3d
	void dump_data_pairs_3d(const std::string &_data_pair_file) const;

	// visualize data weight on depth
	void visualize_data_weight(const Eigen::VectorXd &_data_weight,
		const std::string &_data_weight_file) const;

private:
	// Transform Vector6f to matrix4f
	Eigen::Matrix4f vec6_to_mat44(const Vector6f &_vec6f) const;

	// Find pair set {(u, u')} between render vertex map and depth vertex map
	void associate_data_pairs();

	// find valid data pairs without unnecessary re-projection
	void associate_data_pairs2();

	/*used with solve, this MUST be called after updating warping field*/
	std::vector<double> evaluate_energy(const WarpField &_warp_field,
		const std::vector<std::array<int, c_nn>> &_node_knn_index) const;

	std::vector<std::pair<Eigen::Vector2i, Eigen::Vector2i>> m_data_pairs;
	cv::Mat m_depth_vertex;
	cv::Mat m_depth_normal;
	cv::Mat m_warped_vertex;
	cv::Mat m_warped_normal;
	cv::Mat m_can_vertex;
	cv::Mat m_can_normal;
	Camera m_camera;

	float m_reg_coeff = 5.0f;

	// Debug variables
	int frame_idx = 0;
	int iter_idx = 0;
	int max_iter_times = 0;
	std::string results_dir;
};

#endif