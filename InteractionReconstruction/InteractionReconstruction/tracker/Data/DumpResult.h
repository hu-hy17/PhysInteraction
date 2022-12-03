#pragma once
#include "tracker/Tracker.h"

/**********************************************************************************************//**
 * \brief	Dumps an object model
 *
 * \param [in,out]	valid_vertices_host	The vertices of the object model.
 * \param 		  	file_path		   	(Optional) Full pathname of the file.
 **************************************************************************************************/

void dumpObjModel(std::vector<float4> &valid_vertices_host, 
	const std::string& file_path = "../../../../result/model/object_model.obj");

/**********************************************************************************************//**
 * \brief	Dumps an object model
 *
 * \param [in,out]	valid_vertices_host	The vertices of the object model.
 * \param [in,out]	valid_normal_host  	The normals of the object model.
 * \param 		  	file_path		   	(Optional) Full pathname of the file.
 **************************************************************************************************/

void dumpObjModel(std::vector<float4> &valid_vertices_host, std::vector<float4> &valid_normal_host,
	const std::string& file_path = "../../../../result/model/object_model.obj");

/**********************************************************************************************//**
 * \brief	Output contact information
 *
 * \param 		  	file_prefix	The file prefix.
 * \param [in,out]	tracker	   	The tracker.
 **************************************************************************************************/

void outputContactInfo(std::string file_prefix, Tracker& tracker);

/**********************************************************************************************//**
 * \brief	Output hand object motion
 *
 * \param 		  	filename	Filename of the file.
 * \param [in,out]	tracker 	The tracker.
 **************************************************************************************************/

void outputHandObjMotion(std::string filename, Tracker& tracker);