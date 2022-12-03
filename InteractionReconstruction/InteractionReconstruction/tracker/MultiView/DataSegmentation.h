#pragma once
#include "reconstruction/ObjectReconstruction.h"
#include "reconstruction/recon_externs.h"

class DataSegmentation
{
public:
	ReconstructDataInput Recon_DataInput;
	HandSegmentation hand_segmentation;

private:
	mat34 _camera_RT;

	DataSegmentation() : Recon_DataInput(320, 240) {}
	~DataSegmentation() {};
};
