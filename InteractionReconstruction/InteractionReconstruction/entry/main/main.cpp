#include <iostream>
#include <QApplication>

#include "tracker/Data/Camera.h"

#include "tracker/Tracker.h"
#include "tracker/GLWidget.h"
#include <json/json.h>

//#include <vld.h>

int main(int argc, char* argv[]) {
	bool htrack = false;
	bool test = false; //J' * J on CPU
	bool real_color = false;
	bool save_rastorized_model = false;

	bool benchmark = false;
	bool use_exist_obj = false;
	bool track_hand_with_phys = false;
	bool nonrigid = false;
	int user_name = 0;

	bool show_render = false;
	bool store_render = false;

	std::string sequence_path, data_path, sequence_name, exist_obj_path, render_store_path;
	std::string left_camera_id;

	/************************************************************************/
	/* Parse parameters                                                     */
	/************************************************************************/
	Json::Reader reader;
	Json::Value m_jsonRoot;
	if (argc<2)
	{
		std::cout << "Please Configure and Supply the json file!!!" << std::endl;
		exit(0);
	}

	std::cout << "(Main) Info: json file:" << argv[1] << std::endl;

	if (std::string(argv[1]).find(".json") != std::string::npos)
	{
		std::ifstream fin(argv[1]);

		if (!fin.is_open())
		{
			std::cout << "(Main) Error: Can't open json file\n";
			exit(-1);
		}

		if (!reader.parse(fin, m_jsonRoot))
		{
			std::cout << "(Main) Error: Can't parse json file\n";
			exit(-1);
		}

		sequence_path = m_jsonRoot["sequence_path"].asString();
		sequence_name = m_jsonRoot["sequence_name"].asString();
		data_path = m_jsonRoot["data_path"].asString();
		benchmark = m_jsonRoot["benchmark"].asBool();

		show_render = m_jsonRoot["show_render"].asBool();
		store_render = m_jsonRoot["store_render"].asBool();
		render_store_path = m_jsonRoot["render_store_path"].asString();
		nonrigid = m_jsonRoot["nonrigid"].asBool();

		left_camera_id = m_jsonRoot["camera_id"].asString();
	}
	else
	{
		std::cout << "Cannot find the json file!!!" << std::endl;
		exit(0);
	}

	Q_INIT_RESOURCE(shaders);
	QApplication app(argc, argv);
	
	// configure depth camera
	auto dep_intr_data = m_jsonRoot["depth_intrinsic"];
	if (dep_intr_data == Json::nullValue || dep_intr_data.size() != 4) {
		cout << "(Tracker) Error: depth intrinsic params are not correctly configured!" << endl;
		exit(-1);
	}
	Camera camera(Intel, dep_intr_data[0].asFloat(), dep_intr_data[1].asFloat(), dep_intr_data[2].asFloat(), dep_intr_data[3].asFloat(), 60);
	RealSenseSR300 sensor_sr300(left_camera_id);

	if(!benchmark)
	{
		rs2::context ctx;    // Create librealsense context for managing devices

		// Register callback for tracking which devices are currently connected
		ctx.set_devices_changed_callback([&](rs2::event_information& info)
		{
			sensor_sr300.remove_devices(info);
			for (auto&& dev : info.get_new_devices())
			{
				sensor_sr300.enable_device(dev);
			}
		});

		// Initial population of the device list
		for (auto&& dev : ctx.query_devices()) // Query the list of connected RealSense devices
		{
			sensor_sr300.enable_device(dev);
		}
	}

	DataStream datastream(&camera);
	SolutionStream solutions;

	Worker worker(&camera, test, benchmark, save_rastorized_model, user_name, data_path);
 
	GLWidget glwidget(&worker, &datastream, &solutions, false, false /*real_color*/, false /*use_mano*/, data_path);
	worker.bind_glwidget(&glwidget);
	worker.bind_gl_resrc(&glwidget);
	QString title = "InteractRecon";
	glwidget.set_store_file(show_render, store_render, render_store_path);
	glwidget.setWindowTitle(title);
	glwidget.show();

	Tracker tracker(&worker, sensor_sr300, camera.FPS(), sequence_path, m_jsonRoot, real_color, benchmark);
	tracker.datastream = &datastream;
	tracker.solutions = &solutions;

	///-- set hand tracking energy weight
	{
		float ratio = 1.0f;

		worker.settings->termination_max_iters = 5;// 5

		worker.E_fitting.settings->fit2D_enable = true;
		worker.E_fitting.settings->fit2D_weight = 1.0 * ratio;// 0.7; 1.0

		worker.E_fitting.settings->fit3D_enable = true;
		worker.E_fitting.settings->fit3D_weight = 1.0;

		worker.E_limits.jointlimits_enable = true;
		worker.E_limits.jointlimits_weight = 1e6  * ratio;		// 1e6

		worker.E_pose._settings.enable_split_pca = true;
		worker.E_pose._settings.weight_proj = 4 * 10e2  * ratio;			// 4 * 10e2

		worker.E_collision._settings.collision_enable = true;
		worker.E_collision._settings.collision_weight = 1e3  * ratio;		// 1e3

		worker.E_temporal._settings.temporal_coherence1_enable = true;//true
		worker.E_temporal._settings.temporal_coherence2_enable = true;//true
		worker.E_temporal._settings.temporal_coherence1_weight = 1.0 * ratio;// 0.05 1.0
		worker.E_temporal._settings.temporal_coherence2_weight = 1.0 * ratio;// 0.05 1.0

		worker.E_damping._settings.abduction_damping = 1500000 * ratio;		// 1500000
		worker.E_damping._settings.translation_damping = 1 * ratio;
		worker.E_damping._settings.rotation_damping = 10000 * ratio;//3000
		worker.E_damping._settings.top_phalange_damping = 100000 * ratio;//10000

		worker._settings.termination_max_rigid_iters = 1;

		// Energy term added by ZH
		worker.E_poseprediction.posepred_weight = 2e4 * ratio;	// 2e4
		worker.E_KeyPoint3D.Keypoint3D_weight = 10 * ratio;		// 10
		if (nonrigid)
		{
			worker.E_interaction.InteractionHand_weight = 0.3 * ratio; // 0.3 1.0
		}
		else
		{
			worker.E_interaction.InteractionHand_weight = 1.0 * ratio; // 0.3 1.0
		}

		// second stage hand tracking
		worker.E_PhysKinDiff.diff_weight = 10;
		worker.E_KeyPointTips.KeypointTips_weight = 100;
		worker.E_pose_second._settings.enable_split_pca = true;
		worker.E_pose_second._settings.weight_proj = 50;
		worker.E_limits_second.jointlimits_enable = true;
		worker.E_limits_second.jointlimits_weight = 1e4;
		worker.E_collision_second._settings.collision_enable = true;
		worker.E_collision_second._settings.collision_weight = 1e3;
		worker.settings->second_stage_iter_times = 20;
	}

	///-- set obj tracking energy weight
	{
		tracker.rigid_solver.force_coef = 0.01f;
		tracker.rigid_solver.sil_coe = 0.3f;
		tracker.rigid_solver.friction_coe = 0.1f;	// 0.1f
		tracker.obj_smooth_ratio = 0;
	}

	///--- Starts the tracking
	tracker.toggle_tracking(!benchmark);
	tracker.toggle_benchmark(benchmark);

	return app.exec();
}
