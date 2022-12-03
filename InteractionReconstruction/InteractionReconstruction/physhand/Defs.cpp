#include"Defs.h"
#include<string>

const int CHILD_JOINT[TOTAL_JOINT_NUM]{
	-1, 0, 1, 2,
	-1, 4, 5, 6,
	-1, 8, 9, 10,
	-1, 12, 13, 14,
	-1, 16, 17, 18
};

const int JOINT_IDX_TO_TIP[TOTAL_JOINT_NUM]{
	0, -1, -1, -1,
	1, -1, -1, -1,
	2, -1, -1, -1,
	3, -1, -1, -1,
	4, -1, -1, -1
};

const int TIPS_JOINT_IDX[5]{ 0, 4, 8, 12, 16 };

const std::string JOINT_NAME[TOTAL_JOINT_NUM]{
	"index_top", "index_middle", "index_bottom", "index_base",
	"middle_top", "middle_middle", "middle_bottom", "middle_base",
	"pinky_top", "pinky_middle", "pinky_bottom", "pinky_base",
	"ring_top", "ring_middle", "ring_bottom", "ring_base",
	"thumb_top", "thumb_middle", "thumb_bottom", "thumb_base"
};

const int TIPS_TO_JOINT_CONF_ID[5]{
	22, 18, 10, 14, 6
};

const int JOINT_TO_JOINT_CONF_ID[TOTAL_JOINT_NUM]{
	22, 21, 20, -1,
	18, 17, 16, -1,
	10,  9,  8, -1,
	14, 13, 12, 11,
	 6,  5,  4,  3
};

const double TIPS_CONF_REF[5]{
	0.115, 0.142, 0.0871, 0.125, 0.152
};

const bool g_use_friction = true;
const bool g_use_conf_on_contact_status = false;
const bool g_use_conf_on_friction = true ;

namespace CONTACT_CONTROL {
	const int g_contact_point_lower_bound = 6;
	const float g_radius_expand_rate = 1.2;
	const float g_max_potential_cp_dist = 20.0;
	const float g_min_potential_cp_dist = 0.1;
	const float g_sticky_tip_force_min_rate = 0.3;
}

namespace CONF_CONTROL {
	const double g_conf_ratio_inc = 0;
	const double g_conf_ratio_dec = 0;
	const float g_allow_slide_diff_lower_bound = 10;
	const float g_stop_slide_diff_upper_bound = 0.5;
	const float g_allow_slide_conf_lower_bound = 0.5;
	const float g_slide_ratio = 0.5;
}