#pragma once
#include<curl/curl.h>
#include<string>
#include<iostream>
#include <vector>
#include <sstream>

class PosePredictNet
{
public:
	CURL *curl; 
	curl_mime *form = nullptr;
	curl_mimepart *field = nullptr;
	std::string id = "1";
	CURLcode res;

	PosePredictNet() { curl= curl_easy_init(); }
	~PosePredictNet() { curl_easy_cleanup(curl); }
public:

	void get_predicted_pose(std::vector<float>& current_pose, std::vector<float>& predicted_pose);
};
