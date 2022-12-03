#pragma once
#include "tracker/Types.h"

struct TrackingError{
    float pull_error;
    float push_error;
	float collision_error;
	float first_order_temporal_error;
	float second_order_temporal_error;
	float limit_error;
	int push_num;
	int pull_num;
    static TrackingError zero(){ return {0, 0, 0, 0, 0, 0, 0, 0}; }
};

namespace energy{

class Energy{
protected:
    bool safety_check = false;
    bool has_nan(LinearSystem &system);

public:
    static void rigid_only(LinearSystem& system);
    static VectorN solve(LinearSystem& system);
};

}

