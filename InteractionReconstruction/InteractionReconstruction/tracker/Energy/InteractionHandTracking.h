#pragma once
#include "../Interaction.h"
#include "tracker/ForwardDeclarations.h"
#include "Energy.h"
#include "tracker/Types.h"

#include "cudax/CudaHelper.h"
#include "cudax/CublasHelper.h"

namespace energy
{
	class InteractionHandTracking
	{
	public:
		float InteractionHand_weight = 0.3f;
	public:
//		InteractionHandTracking(Interaction& _interaction_data);
		void track_joints(Interaction& _interaction_data, LinearSystem &system, Eigen::Matrix4f rigid_motion, bool set_parameter, bool store_result, int frame_idx);

		void track_blocks(Interaction& _interaction_data, LinearSystem &system, Eigen::Matrix4f rigid_motion, bool set_parameter, bool store_result, int frame_idx);
	};
}