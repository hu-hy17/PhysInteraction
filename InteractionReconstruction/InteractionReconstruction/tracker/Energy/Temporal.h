#pragma once
#include <vector>
#include "tracker/ForwardDeclarations.h"
#include "tracker/Types.h"
#include "tracker/Energy/Energy.h"
class SolutionQueue;

namespace energy{
class Temporal : public Energy{
	Model * model = NULL;	

    SolutionQueue* solution_queue = NULL;
    std::vector<int> joint_ids;
	std::vector<int> center_ids;
	std::vector<int> phalange_ids;
    std::vector<Vector3> pos_prev1;
    std::vector<Vector3> pos_prev2;
	std::vector<float> theta_prev1;//ZH
	std::vector<float> theta_prev2;//ZH
    int fid_curr = -1;

public:
    struct Settings{
        bool temporal_coherence1_enable = true;
        bool temporal_coherence2_enable = true;
        float temporal_coherence1_weight = 0.05;
        float temporal_coherence2_weight = 0.05;
    } _settings;
    Settings*const settings = &_settings;

public:
	void init(Model * model);
    ~Temporal();
    void track(LinearSystem& system, DataFrame& frame, bool store_error, float &first_order_tempor_error, float &second_order_tempor_error);
    void update(int frame_id, const std::vector<Scalar>& Solution);
private:
	void track(LinearSystem& system, int fid, bool first_order, bool store_error, float &tempor_error);
	void track_theta(LinearSystem& system, int fid, bool first_order, bool store_error, float &tempor_error);
	void temporal_coherence_init();
};

} /// energy::
