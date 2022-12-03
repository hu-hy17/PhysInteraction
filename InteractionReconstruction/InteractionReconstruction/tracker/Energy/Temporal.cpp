#include "Temporal.h"
#include "tracker/Data/DataFrame.h"
#include "tracker/TwSettings.h"
#include "tracker/HModel/Model.h"

/// Queue for temporal coherence
class SolutionQueue {
public:
	typedef std::vector<Scalar> Solution;
	typedef std::map<int, Solution> Solutions;
	Solutions solutions;

	bool valid(int id) {
		return id >= 2
			&& solutions.find(id - 1) != solutions.end()
			&& solutions.find(id - 2) != solutions.end();
	}
	void set(int id, const Solution &s) {
		solutions[id] = s;
	}
	void update(int id, const Solution &s) {
		set(id, s);
		for (auto it = solutions.begin(); it != solutions.end();) {
			int d = id - it->first;
			if (d > 2 || d < 0) {
				it = solutions.erase(it);
			}
			else {
				++it;
			}
		}
	}
};

namespace energy {

	void Temporal::init(Model * model) {
		this->model = model;
		this->solution_queue = new SolutionQueue();
		tw_settings->tw_add(settings->temporal_coherence1_enable, "1st Order", "group=Temporal");
		tw_settings->tw_add(settings->temporal_coherence2_enable, "2nd Order", "group=Temporal");
		tw_settings->tw_add(settings->temporal_coherence1_weight, "weight(1st)", "group=Temporal");
		tw_settings->tw_add(settings->temporal_coherence2_weight, "weight(2nd)", "group=Temporal");
	}

	Temporal::~Temporal() {
		if (solution_queue) delete solution_queue;
	}

	void Temporal::update(int frame_id, const std::vector<Scalar> & theta) {
		solution_queue->update(frame_id, theta);
	}

	void extract_positions(Model * model, const vector<int> & joint_ids, const vector<int> & center_ids,
		const vector<Scalar> & theta_previous, const vector<Scalar> & theta_current, vector<Vector3> & positions) {
		if (theta_previous.empty()) return;			
		positions.resize(joint_ids.size());

		model->move(theta_previous);
		model->update_centers();
		//cout << "HMODEL" << endl;
		for (size_t i = 0; i < joint_ids.size(); ++i) {
			glm::vec3 c = model->centers[center_ids[i]];
			positions[i] = Vector3(c[0], c[1], c[2]);			
		}
		model->move(theta_current);
		model->update_centers();
	}

	void Temporal::temporal_coherence_init() {
		joint_ids.clear();
		center_ids.clear();
		pos_prev1.clear();
		pos_prev2.clear();

		for (size_t i = 0; i < num_phalanges - 1; i++) {
			joint_ids.push_back(model->phalanges[i].segment_id);
			center_ids.push_back(model->phalanges[i].center_id);
			phalange_ids.push_back(i); // remove when removing children
			pos_prev1.push_back(model->phalanges[i].global.block(0, 3, 3, 1).cast<float>());
			pos_prev2.push_back(model->phalanges[i].global.block(0, 3, 3, 1).cast<float>());

			if (model->phalanges[i].children_ids.empty()) {
				glm::vec3 c = model->centers[model->phalanges[i].attachments[0]];

				center_ids.push_back(model->phalanges[i].attachments[0]);
				joint_ids.push_back(model->phalanges[i].segment_id + 1);
				phalange_ids.push_back(i); // remove when removing children
				pos_prev1.push_back(Vector3(c[0], c[1], c[2]));
				pos_prev2.push_back(Vector3(c[0], c[1], c[2]));

				center_ids.push_back(model->phalanges[i].attachments[0]);
				joint_ids.push_back(model->phalanges[i].segment_id + 1);
				pos_prev1.push_back(Vector3(c[0], c[1], c[2]));
				phalange_ids.push_back(i); // remove when removing children
				pos_prev2.push_back(Vector3(c[0], c[1], c[2]));
			}
			else {
				Phalange child = model->phalanges[model->phalanges[i].children_ids[0]];
				center_ids.push_back(child.center_id);
				joint_ids.push_back(child.segment_id);
				phalange_ids.push_back(i); // remove when removing children
				pos_prev1.push_back(child.global.block(0, 3, 3, 1).cast<float>());
				pos_prev2.push_back(child.global.block(0, 3, 3, 1).cast<float>());
			}
		}
		/*cout << endl << "HMODEL" << endl;
		for (size_t i = 0; i < joint_ids.size(); i++) {
		cout << "joint_ids[" << i << "] = " << joint_ids[i] << ", t = " << pos_prev1[i].transpose() << endl;
		}*/
	}

	void Temporal::track(LinearSystem& system, int fid, bool first_order, bool store_error, float &tempor_error) {
		if (first_order) 
			if (!settings->temporal_coherence1_enable) 
				return;
		else if (!settings->temporal_coherence2_enable) 
			return;

		// TIMED_BLOCK(timer,"Worker::temporal_coherence_track(extract positions)")
		{
			if (joint_ids.empty()) {
				temporal_coherence_init();
			}

			if (solution_queue->valid(fid)) {
				if (fid != fid_curr) {
					vector<float> theta_current = model->get_theta();
					extract_positions(model, joint_ids, center_ids, solution_queue->solutions[fid - 1], theta_current, pos_prev1);
					extract_positions(model, joint_ids, center_ids, solution_queue->solutions[fid - 2], theta_current, pos_prev2);
					fid_curr = fid;
				}
			}
			else return;
		}

		// TIMED_BLOCK(timer,"Worker::temporal_coherence_track(compute jacobian)")
		{
			Eigen::Matrix<Scalar, d * num_temporal, num_thetas> J = Eigen::Matrix<Scalar, d * num_temporal, num_thetas>::Zero(d * num_temporal, num_thetas);
			Eigen::Matrix<Scalar, d * num_temporal, 1> e = Eigen::Matrix<Scalar, d * num_temporal, 1>::Zero(d * num_temporal, 1);

			if (store_error)
				tempor_error = 0;
					
			for (size_t i = 0; i < joint_ids.size(); ++i) {
				glm::vec3 c = model->centers[center_ids[i]];
				Vector3 pos_frame = Vector3(c[0], c[1], c[2]);				

				J.block(3 * i, 0, 3, num_thetas) = model->jacobian(pos_frame, phalange_ids[i]);	

				if (first_order) e.block(3 * i, 0, 3, 1) = pos_prev1[i] - pos_frame;
				else e.block(3 * i, 0, 3, 1) = 2 * pos_prev1[i] - pos_prev2[i] - pos_frame;

				if (store_error)
				{
					Vector3 temporal_error_temp = e.block(3 * i, 0, 3, 1);
					tempor_error += sqrt(temporal_error_temp[0] * temporal_error_temp[0] + temporal_error_temp[1] * temporal_error_temp[1] + temporal_error_temp[2] * temporal_error_temp[2]);
				}
			}

			Scalar omega = 1.0f;
			if (first_order) omega = settings->temporal_coherence1_weight;
			else omega = settings->temporal_coherence2_weight;
			//std::cout << "omega:" << omega << std::endl;

			Eigen::Matrix<Scalar, num_thetas, d * num_temporal> JT = J.transpose();
			system.lhs += omega * JT * J;
			system.rhs += omega * JT * e;

			if (store_error)
				tempor_error *= omega;
		}

		///--- Check
		if (Energy::safety_check) Energy::has_nan(system);
	}

	void Temporal::track_theta(LinearSystem& system, int fid, bool first_order, bool store_error, float &tempor_error) {
		if (first_order) if (!settings->temporal_coherence1_enable) return;
		else if (!settings->temporal_coherence2_enable) return;

		// TIMED_BLOCK(timer,"Worker::temporal_coherence_track(extract positions)")
		{
			/*if (joint_ids.empty()) {
				temporal_coherence_init();
			}*/

			if (solution_queue->valid(fid)) {
				if (fid != fid_curr) {
					/*vector<float> theta_current = model->get_theta();
					extract_positions(model, joint_ids, center_ids, solution_queue->solutions[fid - 1], theta_current, pos_prev1);
					extract_positions(model, joint_ids, center_ids, solution_queue->solutions[fid - 2], theta_current, pos_prev2);*/
					theta_prev1 = solution_queue->solutions[fid - 1];
					theta_prev2 = solution_queue->solutions[fid - 2];
					fid_curr = fid;
				}
			}
			else return;
		}

		// TIMED_BLOCK(timer,"Worker::temporal_coherence_track(compute jacobian)")
		float factor = 400.0f;
		{
			Eigen::Matrix<Scalar, num_thetas, num_thetas> J = Eigen::Matrix<Scalar, num_thetas, num_thetas>::Zero(num_thetas, num_thetas);
			Eigen::Matrix<Scalar, num_thetas, 1> e = Eigen::Matrix<Scalar, num_thetas, 1>::Zero(num_thetas, 1);

			if (store_error)
				tempor_error = 0;
			vector<float> theta_current = model->get_theta();
			for (size_t i = 0; i < num_thetas; ++i) {
				J(i,i) = 1;

				if (first_order) e(i) = theta_prev1[i] - theta_current[i];
				else e(i) = 2 * theta_prev1[i] - theta_prev2[i] - theta_current[i];

				if (i >= 3)
				{
					J(i, i) *= factor;
					e(i) *= factor;
				}

				if (store_error)
				{
					float temporal_error_temp = e(i);
					tempor_error += temporal_error_temp*temporal_error_temp;
				}
			}

			Scalar omega = 1.0f;
			if (first_order) omega = settings->temporal_coherence1_weight;
			else omega = settings->temporal_coherence2_weight;

			Eigen::Matrix<Scalar, num_thetas, num_thetas> JT = J.transpose();
			system.lhs += omega * JT * J;
			system.rhs += omega * JT * e;

			if (store_error)
				tempor_error *= omega;
		}

		///--- Check
		if (Energy::safety_check) Energy::has_nan(system);
	}

	void Temporal::track(LinearSystem & system, DataFrame & frame, bool store_error, float &first_order_tempor_error, float &second_order_tempor_error) {

		track(system, frame.id, true, store_error, first_order_tempor_error); ///< 1st order
		track(system, frame.id, false, store_error, second_order_tempor_error); ///< 2nd order

		//ZH
		// track_theta(system, frame.id, true, store_error, first_order_tempor_error); ///< 1st order
		// track_theta(system, frame.id, false, store_error, second_order_tempor_error); ///< 2nd order*/

		if (Energy::safety_check) Energy::has_nan(system);
	}

}




