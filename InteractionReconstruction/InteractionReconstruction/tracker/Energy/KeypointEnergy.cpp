#include "KeypointEnergy.h"

Eigen::Matrix<float, 2, 3 > energy::Keypoint2D::calculate_projectionJacobian(glm::vec3 sphere_pos, camera_intr para_cam)
{
	Eigen::Matrix<float, 2, 3 > projectionJacobian = Eigen::Matrix<float, 2, 3>::Zero(2, 3);
	
	projectionJacobian(0, 0) = para_cam.fx / sphere_pos[2];
	projectionJacobian(0, 2) = -sphere_pos[0] * para_cam.fx / (sphere_pos[2] * sphere_pos[2]);

	projectionJacobian(1, 1) = -para_cam.fy / sphere_pos[2];
	projectionJacobian(1, 2) = sphere_pos[1] * para_cam.fy / (sphere_pos[2] * sphere_pos[2]);

	return projectionJacobian;
}

void energy::Keypoint2D::track(LinearSystem& sys, const Model *model, const std::vector<int> &keypoint2block, const std::vector<int> &keypoint2SphereCenter, 
	const std::vector<float2> &keypoint_2D_GT_vec, const std::vector<int> &using_keypoint, camera_intr para_cam, int frame_id, int itr_id)
{
	Eigen::Matrix<float, 21 * 2, num_thetas> J = Eigen::Matrix<float, 21 * 2, num_thetas>::Zero(21 * 2, num_thetas);
	Eigen::Matrix<float, 21 * 2, 1> e = Eigen::Matrix<float, 21 * 2, 1>::Zero(21 * 2, 1);
	JointTransformations jointinfos = model->transformations;
	KinematicChain chains = model->kinematic_chain;

	//static ofstream keypoint_2D_debug_file(keypoint_2D_debug_file_path);
	//keypoint_2D_debug_file << frame_id << "." << itr_id<<" ";// << std::endl;

//	cout << "keypoint_2D_GT_vec size:" << keypoint_2D_GT_vec.size() << endl;
	if (keypoint_2D_GT_vec.size() > 0)
	{
		/*keypoint_2D_debug_file << "GT- projection_pixel:" << std::endl;*/
		for (int i = 0; i < using_keypoint.size(); i++)
		{
//			cout << "i:" << i << endl;
			static int keypoint_id;
			keypoint_id = using_keypoint[i];
			static int sphere_id;
			sphere_id = keypoint2SphereCenter[keypoint_id];
			static int block_id;
			block_id = keypoint2block[keypoint_id];
			static int joint_id;
			joint_id = model->blockid_to_jointid_map[block_id];
			float2 keypoint_2D_GT = keypoint_2D_GT_vec[keypoint_id];

			if (keypoint_2D_GT.x > 0 && keypoint_2D_GT.y > 0)
			{
				glm::vec3 sphere_center = model->centers[sphere_id];
				Eigen::Matrix<float, 2, 3 > J_proj = calculate_projectionJacobian(sphere_center, para_cam);

				///--- Compute LHS
				for (int i_column = 0; i_column < CHAIN_MAX_LENGTH; i_column++) {
					int jointinfo_id = chains[joint_id].data[i_column];
					if (jointinfo_id == -1) break;
					const CustomJointInfo& jinfo = jointinfos[jointinfo_id];
					Eigen::Vector3f axis = model->dofs[jointinfo_id].axis;
					int joint_idx = model->dofs[jointinfo_id].joint_id;

					switch (jinfo.type) {
					case 1:
					{
						Eigen::Vector3f col_4f = axis;
						Eigen::Vector2f jcol = J_proj * Eigen::Vector3f(col_4f[0], col_4f[1], col_4f[2]);

						J.block(2 * i, joint_idx, 2, 1) = jcol;

						break;
					}
					case 0: // ROT
					{
						Eigen::Vector3f t(model->phalanges[model->dofs[jointinfo_id].phalange_id].global(0, 3), model->phalanges[model->dofs[jointinfo_id].phalange_id].global(1, 3), model->phalanges[model->dofs[jointinfo_id].phalange_id].global(2, 3));
						Eigen::Vector4f axis_4f = model->phalanges[model->dofs[jointinfo_id].phalange_id].global * Eigen::Vector4f(axis[0], axis[1], axis[2], 1);
						Eigen::Vector3f axis_3f(axis_4f(0), axis_4f(1), axis_4f(2));
						Eigen::Vector3f a = (axis_3f - t).normalized();
						Eigen::Vector3f col = a.cross(Eigen::Vector3f(sphere_center[0], sphere_center[1], sphere_center[2]) - t);
						Eigen::Vector2f jcol = J_proj * col;

						J.block(2 * i, joint_idx, 2, 1) = jcol;

						break;
					}
					}
				}

				float2 projection_sphere;
				projection_sphere.x = para_cam.fx*sphere_center[0] / sphere_center[2] + para_cam.cx;
				projection_sphere.y = -para_cam.fy*sphere_center[1] / sphere_center[2] + para_cam.cy;

				e(2 * i, 0) = -(projection_sphere.x - keypoint_2D_GT.x);
				e(2 * i + 1, 0) = -(projection_sphere.y - keypoint_2D_GT.y);

				/*keypoint_2D_debug_file << "(" << keypoint_2D_GT.x << "," << keypoint_2D_GT.y << ")-(" << projection_sphere.x << "," << projection_sphere.y << ") ";*/
			}
		}

		/*keypoint_2D_debug_file << std::endl;*/

		Eigen::Matrix<float, num_thetas, 21 * 2> JT = J.transpose();
		sys.lhs += Keypoint2D_weight*JT*J;
		sys.rhs += Keypoint2D_weight*JT*e;

		/*keypoint_2D_debug_file << "J:" << endl;
		keypoint_2D_debug_file << J << endl;

		keypoint_2D_debug_file << "e:" << endl;
		keypoint_2D_debug_file << e.transpose() << endl;

		keypoint_2D_debug_file << "eTe:" << endl;
		keypoint_2D_debug_file << e.transpose()*e << endl;*/

		///--- Check
		if (Energy::safety_check) Energy::has_nan(sys);
	}
}


void energy::Keypoint3D::track(LinearSystem& sys, const Model *model, 
							   const std::vector<int> &keypoint2block, 
							   const std::vector<int> &keypoint2SphereCenter,
							   const std::vector<float3> &keypoint_3D_GT_vec, 
							   const std::vector<int> &using_keypoint, 
							   Eigen::Matrix4f camera_pose, 
							   int frame_id, int itr_id, float weight_factor)
{
	Eigen::Matrix<float, 21 * 3, num_thetas> J = Eigen::Matrix<float, 21 * 3, num_thetas>::Zero(21 * 3, num_thetas);
	
	Eigen::Matrix<float, 21 * 3, 1> e = Eigen::Matrix<float, 21 * 3, 1>::Zero(21 * 3, 1);
//	Eigen::VectorXf e(21 * 3);

	JointTransformations jointinfos = model->transformations;
	KinematicChain chains = model->kinematic_chain;

	// keypoint_3D_debug_file << frame_id << "." << itr_id << " " << std::endl;

//	cout << "keypoint_2D_GT_vec size:" << keypoint_2D_GT_vec.size() << endl;
	if (keypoint_3D_GT_vec.size() > 0)
	{
		/*cout << "J-before:" << endl;
		cout << J;
		cout << endl;*/

		//keypoint_3D_debug_file << "GT - sphere position:" << std::endl;

		/*if (frame_id > 210 && frame_id < 218)
		{
			keypoint_3D_debug_file << "GT 3D:" << endl;
			for (int i = 0; i < using_keypoint.size(); i++)
			{
				keypoint_3D_debug_file << keypoint_3D_GT_vec[using_keypoint[i]].x << " " << keypoint_3D_GT_vec[using_keypoint[i]].y << " " << keypoint_3D_GT_vec[using_keypoint[i]].z << "   ";
			}
			keypoint_3D_debug_file << endl;

			keypoint_3D_debug_file << "solve-before 3D:" << endl;
			for (int i = 0; i < using_keypoint.size(); i++)
			{
				keypoint_3D_debug_file << model->centers[keypoint2SphereCenter[using_keypoint[i]]].x << " " << model->centers[keypoint2SphereCenter[using_keypoint[i]]].y << " " << model->centers[keypoint2SphereCenter[using_keypoint[i]]].z << "   ";
			}
			keypoint_3D_debug_file << endl;
		}*/

//		cout.precision(6);
		for (int i = 0; i < using_keypoint.size(); i++)
		{
			//			cout << "i:" << i << endl;
			static int keypoint_id;
 			keypoint_id = using_keypoint[i];
			static int sphere_id;
			sphere_id = keypoint2SphereCenter[keypoint_id];
			static int block_id;
			block_id = keypoint2block[keypoint_id];
			static int joint_id;
			joint_id = model->blockid_to_jointid_map[block_id];
			float3 local_keypoint3D = keypoint_3D_GT_vec[keypoint_id];
			Eigen::Vector4f global_keypoint3D = camera_pose*Eigen::Vector4f(local_keypoint3D.x, local_keypoint3D.y, local_keypoint3D.z, 1.0f);

			float3 keypoint_3D_GT;
			keypoint_3D_GT.x = global_keypoint3D[0];
			keypoint_3D_GT.y = -global_keypoint3D[1];
			keypoint_3D_GT.z = global_keypoint3D[2];

			/*if (itr_id == 0)
				keypoint_3D_debug_file << keypoint_3D_GT.x << " " << keypoint_3D_GT.y << " " << keypoint_3D_GT.z << " ";*/

			/*cout << "J-before-" << i << ":" << endl;
			cout << J;
			cout << endl;*/

			if (keypoint_3D_GT.z > 0)
			{
				glm::vec3 sphere_center = model->centers[sphere_id];

				///--- Compute LHS
				for (int i_column = 0; i_column < CHAIN_MAX_LENGTH; i_column++) {
					int jointinfo_id = chains[joint_id].data[i_column];
					if (jointinfo_id == -1) break;
					const CustomJointInfo& jinfo = jointinfos[jointinfo_id];
					Eigen::Vector3f axis = model->dofs[jointinfo_id].axis;
					static int joint_idx;
					joint_idx = model->dofs[jointinfo_id].joint_id;

					/*cout << "J-" << i << "-" << i_column << ":" << endl;
					cout << J;
					cout << endl;*/

					switch (jinfo.type)
					{
						case 1:
						{
							/*if (keypoint_id != 4)
								return;*/

							//Eigen::Vector4f col_4f = model->phalanges[model->dofs[jointinfo_id].phalange_id].global * Eigen::Vector4f(axis[0], axis[1], axis[2], 1);
							//Eigen::Vector3f col_4f = model->phalanges[model->dofs[jointinfo_id].phalange_id].global.block(0,0,3,3) * axis;
							Eigen::Vector3f col_4f = axis;
							/*J(2 * i, joint_idx) = jcol.x;
							J(2 * i + 1, joint_idx) = jcol.y;*/

							//J.block(3 * i, joint_idx, 3, 1) = Eigen::Vector3f(col_4f[0], col_4f[1], col_4f[2]);
							J(3 * i, joint_idx) = col_4f(0);
							J(3 * i + 1, joint_idx) = col_4f(1);
							J(3 * i + 2, joint_idx) = col_4f(2);

							break;
						}
						case 0: // ROT
						{
							Eigen::Vector3f t(model->phalanges[model->dofs[jointinfo_id].phalange_id].global(0, 3), model->phalanges[model->dofs[jointinfo_id].phalange_id].global(1, 3), model->phalanges[model->dofs[jointinfo_id].phalange_id].global(2, 3));
							Eigen::Vector4f axis_4f = model->phalanges[model->dofs[jointinfo_id].phalange_id].global * Eigen::Vector4f(axis[0], axis[1], axis[2], 1);
							Eigen::Vector3f axis_3f(axis_4f(0), axis_4f(1), axis_4f(2));
							Eigen::Vector3f a = (axis_3f - t).normalized();
							Eigen::Vector3f col = a.cross(Eigen::Vector3f(sphere_center[0], sphere_center[1], sphere_center[2]) - t);

							/*J(2 * i, joint_idx) = jcol.x;
							J(2 * i + 1, joint_idx) = jcol.y;*/

							/*cout << model->phalanges[model->dofs[jointinfo_id].phalange_id].global << endl;
							cout << t << endl;*/

							//J.block(3 * i, joint_idx, 3, 1) = col;
							J(3 * i, joint_idx) = col(0);
							J(3 * i + 1, joint_idx) = col(1);
							J(3 * i + 2, joint_idx) = col(2);

							break;
						}
					}

					/*cout << "J-"<<i<<"-"<<i_column<<":"<< endl;
					cout << J;
					cout << endl;*/
				}

				/*cout << "J-after-1-" << i << ":" << endl;
				cout << J;
				cout << endl;*/

				e(3 * i, 0) = -(sphere_center[0] - keypoint_3D_GT.x);
				e(3 * i + 1, 0) = -(sphere_center[1] - keypoint_3D_GT.y);
				e(3 * i + 2, 0) = -(sphere_center[2] - keypoint_3D_GT.z);

				/*cout << "J-after-2-" << i << ":" << endl;
				cout << J;
				cout << endl;*/
				/*if (frame_id > 210 && frame_id < 218)
				{
					keypoint_3D_debug_file << sphere_center[0] << " " << sphere_center[1] << " " << sphere_center[2] << "   ";
				}*/
//				keypoint_3D_debug_file << "(" << keypoint_3D_GT.x << "," << keypoint_3D_GT.y << "," << keypoint_3D_GT.z << ")-(" << sphere_center[0] << "," << sphere_center[1] << "," << sphere_center[2] << ") ";
			}
		}

		/*if (frame_id>210 && frame_id<218)
			keypoint_3D_debug_file << endl;*/

		Eigen::Matrix<float, num_thetas, 21 * 3> JT = J.transpose();
		Eigen::Matrix<float, num_thetas, num_thetas> JTJ = JT * J;
		Eigen::Matrix<float, num_thetas, 1> JTe = JT * e;

		//for (int i = 0; i < num_thetas; i++)
		//{
		//	if(JTJ(i,i)==0)
		//		JTJ(i, i) = 1;
		//}

//		 JTJ(7, 7) = 1; JTJ(8, 8) = 1;

		/*if ((frame_id > 750 && frame_id < 770))
		{
			weight_factor /= 100;
		}*/

		float final_weight = weight_factor * Keypoint3D_weight;
		sys.lhs += final_weight * JTJ;
		sys.rhs += final_weight * JTe;

//		sys.lhs(6, 6) = Keypoint3D_weight; sys.lhs(7, 7) = Keypoint3D_weight; sys.lhs(8, 8) = Keypoint3D_weight;

		/*keypoint_3D_debug_file << "J:" << endl;
		keypoint_3D_debug_file << J << endl;*/

		/*cout << "J-after:" << endl;
		cout << J << endl;*/

		/*keypoint_3D_debug_file << "e:" << endl;
		keypoint_3D_debug_file << e.transpose() << endl;*/
		/*if (frame_id > 60)
		{
			cout << "inner-" << frame_id << "-" << itr_id << endl;
			cout << "J" << endl;
			cout << J << endl;
			cout << "e" << endl;
			cout << e.transpose() << endl;
			cout << "JTJ" << endl;
			cout << Keypoint3D_weight*JT*J << endl;
			cout << "JTe" << endl;
			cout << Keypoint3D_weight*JT*e << endl;
		}*/

//		keypoint_3D_debug_file << "eTe:" << e.transpose()*e << endl;
//
//		if (frame_id>210&&frame_id<218)
//		{
//			keypoint_3D_debug_file << "J:" << endl;
//			keypoint_3D_debug_file << J.block(0,0,12,6) << endl;
//			keypoint_3D_debug_file << "JTJ:" << endl;
//			keypoint_3D_debug_file << sys.lhs.block(0,0,6,6) << endl;
//			keypoint_3D_debug_file << "JTr:" << endl;
//			keypoint_3D_debug_file << sys.rhs.block(0,0,6,1) << endl;
////			keypoint_3D_debug_file << endl;
//		}

		///--- Check
		if (Energy::safety_check) Energy::has_nan(sys);
	}

	/*if (itr_id == 0)
		keypoint_3D_debug_file << std::endl;*/
}

void energy::Keypoint3D::track_with_weight(LinearSystem& sys, const Model *model, const std::vector<int> &keypoint2block, const std::vector<int> &keypoint2SphereCenter,
										   const std::vector<float3> &keypoint_3D_vec, const std::vector<float> &keypoint_3D_visible, const std::vector<int> &using_keypoint, Eigen::Matrix4f camera_pose, int frame_id, int itr_id)
{
	Eigen::Matrix<float, 21 * 3, num_thetas> J = Eigen::Matrix<float, 21 * 3, num_thetas>::Zero(21 * 3, num_thetas);

	Eigen::Matrix<float, 21 * 3, 1> e = Eigen::Matrix<float, 21 * 3, 1>::Zero(21 * 3, 1);
	//	Eigen::VectorXf e(21 * 3);

	JointTransformations jointinfos = model->transformations;
	KinematicChain chains = model->kinematic_chain;

	if (keypoint_3D_vec.size() > 0)
	{

		//		cout.precision(6);
		for (int i = 0; i < using_keypoint.size(); i++)
		{
			//			cout << "i:" << i << endl;
			float weight = 1.0;
			if (keypoint_3D_visible[i] > 0.7)
				weight = 1.414;

			static int keypoint_id;
			keypoint_id = using_keypoint[i];
			static int sphere_id;
			sphere_id = keypoint2SphereCenter[keypoint_id];
			static int block_id;
			block_id = keypoint2block[keypoint_id];
			static int joint_id;
			joint_id = model->blockid_to_jointid_map[block_id];
			float3 local_keypoint3D = keypoint_3D_vec[keypoint_id];
			Eigen::Vector4f global_keypoint3D = camera_pose*Eigen::Vector4f(local_keypoint3D.x, local_keypoint3D.y, local_keypoint3D.z, 1.0f);

			float3 keypoint_3D_GT;
			keypoint_3D_GT.x = global_keypoint3D[0];
			keypoint_3D_GT.y = -global_keypoint3D[1];
			keypoint_3D_GT.z = global_keypoint3D[2];

			if (keypoint_3D_GT.z > 0)
			{
				glm::vec3 sphere_center = model->centers[sphere_id];

				///--- Compute LHS
				for (int i_column = 0; i_column < CHAIN_MAX_LENGTH; i_column++) {
					int jointinfo_id = chains[joint_id].data[i_column];
					if (jointinfo_id == -1) break;
					const CustomJointInfo& jinfo = jointinfos[jointinfo_id];
					Eigen::Vector3f axis = model->dofs[jointinfo_id].axis;
					static int joint_idx;
					joint_idx = model->dofs[jointinfo_id].joint_id;

					/*cout << "J-" << i << "-" << i_column << ":" << endl;
					cout << J;
					cout << endl;*/

					switch (jinfo.type)
					{
						case 1:
						{
							Eigen::Vector3f col_4f = axis;

							//J.block(3 * i, joint_idx, 3, 1) = Eigen::Vector3f(col_4f[0], col_4f[1], col_4f[2]);
							J(3 * i, joint_idx) = col_4f(0);
							J(3 * i + 1, joint_idx) = col_4f(1);
							J(3 * i + 2, joint_idx) = col_4f(2);

							break;
						}
						case 0: // ROT
						{
							Eigen::Vector3f t(model->phalanges[model->dofs[jointinfo_id].phalange_id].global(0, 3), model->phalanges[model->dofs[jointinfo_id].phalange_id].global(1, 3), model->phalanges[model->dofs[jointinfo_id].phalange_id].global(2, 3));
							Eigen::Vector4f axis_4f = model->phalanges[model->dofs[jointinfo_id].phalange_id].global * Eigen::Vector4f(axis[0], axis[1], axis[2], 1);
							Eigen::Vector3f axis_3f(axis_4f(0), axis_4f(1), axis_4f(2));
							Eigen::Vector3f a = (axis_3f - t).normalized();
							Eigen::Vector3f col = a.cross(Eigen::Vector3f(sphere_center[0], sphere_center[1], sphere_center[2]) - t);


							//J.block(3 * i, joint_idx, 3, 1) = col;
							J(3 * i, joint_idx) = weight*col(0);
							J(3 * i + 1, joint_idx) = weight*col(1);
							J(3 * i + 2, joint_idx) = weight*col(2);

							break;
						}
					}

				}

				e(3 * i, 0) = -(sphere_center[0] - keypoint_3D_GT.x)*weight;
				e(3 * i + 1, 0) = -(sphere_center[1] - keypoint_3D_GT.y)*weight;
				e(3 * i + 2, 0) = -(sphere_center[2] - keypoint_3D_GT.z)*weight;

			}
		}

		Eigen::Matrix<float, num_thetas, 21 * 3> JT = J.transpose();
		Eigen::Matrix<float, num_thetas, num_thetas> JTJ = JT*J;
		Eigen::Matrix<float, num_thetas, 1> JTe = JT*e;

		for (int i = 0; i < num_thetas; i++)
		{
			/*if (JTJ(i, i) == 0)
				JTJ(i, i) = 1;*/
			JTJ(i, i) += 100;
		}

		sys.lhs += Keypoint3D_weight*JTJ;
		sys.rhs += Keypoint3D_weight*JTe;

		///--- Check
		if (Energy::safety_check) Energy::has_nan(sys);
	}

	if (itr_id == 0)
		keypoint_3D_debug_file << std::endl;
}

void energy::KeypointTips::track(LinearSystem& sys, const Model *model,
								 const std::vector<int> &keypoint2block,
								 const std::vector<int> &keypoint2SphereCenter,
								 const std::vector<Eigen::Vector3f> &keypoint_3D_GT_vec,
								 const std::vector<int> &using_keypoint)
{
	Eigen::Matrix<float, 5 * 3, num_thetas> J = Eigen::Matrix<float, 5 * 3, num_thetas>::Zero(5 * 3, num_thetas);

	Eigen::Matrix<float, 5 * 3, 1> e = Eigen::Matrix<float, 5 * 3, 1>::Zero(5 * 3, 1);

	JointTransformations jointinfos = model->transformations;
	KinematicChain chains = model->kinematic_chain;

	for (int i = 0; i < using_keypoint.size(); i++)
	{
		static int keypoint_id;
		keypoint_id = using_keypoint[i];
		static int sphere_id;
		sphere_id = keypoint2SphereCenter[keypoint_id];
		static int block_id;
		block_id = keypoint2block[keypoint_id];
		static int joint_id;
		joint_id = model->blockid_to_jointid_map[block_id];
		Eigen::Vector3f local_keypoint3D = keypoint_3D_GT_vec[i];
		// Eigen::Vector4f global_keypoint3D = Eigen::Vector4f(local_keypoint3D.x, local_keypoint3D.y, local_keypoint3D.z, 1.0f);

		float3 keypoint_3D_GT;
		keypoint_3D_GT.x = local_keypoint3D[0];
		keypoint_3D_GT.y = local_keypoint3D[1]; // -global_keypoint3D[1]; by hhy
		keypoint_3D_GT.z = local_keypoint3D[2];

		if (keypoint_3D_GT.z > 0)	// by hhy 
		{
			glm::vec3 sphere_center = model->centers[sphere_id];

			///--- Compute LHS
			for (int i_column = 0; i_column < CHAIN_MAX_LENGTH; i_column++) {
				int jointinfo_id = chains[joint_id].data[i_column];
				if (jointinfo_id == -1) break;
				const CustomJointInfo& jinfo = jointinfos[jointinfo_id];
				Eigen::Vector3f axis = model->dofs[jointinfo_id].axis;
				static int joint_idx;
				joint_idx = model->dofs[jointinfo_id].joint_id;

				switch (jinfo.type)
				{
					case 1:
					{
						/*if (keypoint_id != 4)
						return;*/

						//Eigen::Vector4f col_4f = model->phalanges[model->dofs[jointinfo_id].phalange_id].global * Eigen::Vector4f(axis[0], axis[1], axis[2], 1);
						//Eigen::Vector3f col_4f = model->phalanges[model->dofs[jointinfo_id].phalange_id].global.block(0,0,3,3) * axis;
						Eigen::Vector3f col_4f = axis;
						/*J(2 * i, joint_idx) = jcol.x;
						J(2 * i + 1, joint_idx) = jcol.y;*/

						//J.block(3 * i, joint_idx, 3, 1) = Eigen::Vector3f(col_4f[0], col_4f[1], col_4f[2]);
						J(3 * i, joint_idx) = col_4f(0);
						J(3 * i + 1, joint_idx) = col_4f(1);
						J(3 * i + 2, joint_idx) = col_4f(2);

						break;
					}
					case 0: // ROT
					{
						Eigen::Vector3f t(model->phalanges[model->dofs[jointinfo_id].phalange_id].global(0, 3), model->phalanges[model->dofs[jointinfo_id].phalange_id].global(1, 3), model->phalanges[model->dofs[jointinfo_id].phalange_id].global(2, 3));
						Eigen::Vector4f axis_4f = model->phalanges[model->dofs[jointinfo_id].phalange_id].global * Eigen::Vector4f(axis[0], axis[1], axis[2], 1);
						Eigen::Vector3f axis_3f(axis_4f(0), axis_4f(1), axis_4f(2));
						Eigen::Vector3f a = (axis_3f - t).normalized();
						Eigen::Vector3f col = a.cross(Eigen::Vector3f(sphere_center[0], sphere_center[1], sphere_center[2]) - t);

						J(3 * i, joint_idx) = col(0);
						J(3 * i + 1, joint_idx) = col(1);
						J(3 * i + 2, joint_idx) = col(2);

						break;
					}
				}
			}

			e(3 * i, 0) = -(sphere_center[0] - keypoint_3D_GT.x);
			e(3 * i + 1, 0) = -(sphere_center[1] - keypoint_3D_GT.y);
			e(3 * i + 2, 0) = -(sphere_center[2] - keypoint_3D_GT.z);
		}
	}

	// global position and rotation are not considered
	if(false)
	{
		for (int i = 0; i < using_keypoint.size(); i++)
		{
			for (int j = 0; j < 7; j++)
			{
				J(3 * i, j) = 0;
				J(3 * i + 1, j) = 0;
				J(3 * i + 2, j) = 0;
			}
		}
	}

	Eigen::Matrix<float, num_thetas, 5 * 3> JT = J.transpose();
	// cout << J << endl;
	Eigen::Matrix<float, num_thetas, num_thetas> JTJ = JT * J;
	Eigen::Matrix<float, num_thetas, 1> JTe = JT * e;

	for (int i = 0; i < num_thetas; i++)
	{
		// L-M damping
		JTJ(i, i) += 1e3;
	}

	sys.lhs += KeypointTips_weight * JTJ;
	sys.rhs += KeypointTips_weight * JTe;

	///--- Check
	if (Energy::safety_check) Energy::has_nan(sys);
}