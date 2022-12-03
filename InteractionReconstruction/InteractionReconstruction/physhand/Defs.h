#pragma once
#include<string>

#define TOTAL_JOINT_NUM 20

extern const int CHILD_JOINT[TOTAL_JOINT_NUM];				
extern const int TIPS_JOINT_IDX[5];							
extern const int JOINT_IDX_TO_TIP[TOTAL_JOINT_NUM];			
extern const std::string JOINT_NAME[TOTAL_JOINT_NUM];		
extern const int TIPS_TO_JOINT_CONF_ID[5];					
extern const int JOINT_TO_JOINT_CONF_ID[TOTAL_JOINT_NUM];	
extern const double TIPS_CONF_REF[5];						

extern const bool g_use_friction;		
extern const bool g_use_conf_on_contact_status;		
extern const bool g_use_conf_on_friction;			

namespace CONTACT_CONTROL {
	extern const int g_contact_point_lower_bound;		
	extern const float g_radius_expand_rate;			
	extern const float g_max_potential_cp_dist;			
	extern const float g_min_potential_cp_dist;			
	extern const float g_sticky_tip_force_min_rate;		
}

namespace CONF_CONTROL {
	extern const double g_conf_ratio_inc;						
	extern const double g_conf_ratio_dec;						
	extern const float g_allow_slide_diff_lower_bound;			
	extern const float g_stop_slide_diff_upper_bound;			
	extern const float g_allow_slide_conf_lower_bound;			
	extern const float g_slide_ratio;							
}