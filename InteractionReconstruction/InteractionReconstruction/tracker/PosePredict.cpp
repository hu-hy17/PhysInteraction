#include "PosePredict.h"

size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp)
{
	((std::string*)userp)->append((char*)contents, size * nmemb);
	return size * nmemb;
}

void PosePredictNet::get_predicted_pose(std::vector<float>& current_pose, std::vector<float>& predicted_pose)
{
	if (current_pose.size() == 29)
	{
		std::string data;
		for (int j = 7; j < 29; j++)
		{
			data += std::to_string(current_pose[j]) + " ";
		}

		std::string readBuffer;
		if (curl) {
			form = curl_mime_init(curl);

			field = curl_mime_addpart(form);
			curl_mime_name(field, "pose");
			curl_mime_data(field, data.c_str(), CURL_ZERO_TERMINATED);

			field = curl_mime_addpart(form);
			curl_mime_name(field, "id");
			curl_mime_data(field, id.c_str(), CURL_ZERO_TERMINATED);

			curl_easy_setopt(curl, CURLOPT_URL, "http://127.0.0.1:8081");
			curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);
			curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
			curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readBuffer);
			res = curl_easy_perform(curl);
		}

		predicted_pose.resize(29);

		{
			predicted_pose[0] = current_pose[0];
			predicted_pose[1] = current_pose[1];
			predicted_pose[2] = current_pose[2];

			predicted_pose[3] = current_pose[3];
			predicted_pose[4] = current_pose[4];
			predicted_pose[5] = current_pose[5];

			predicted_pose[6] = current_pose[6];
		}

		std::stringstream str(readBuffer);
		for (int col = 0; col < 22; ++col) {
			std::string elem;
			str >> elem;
			predicted_pose[col + 7] = std::stof(elem);
		}
	}
}