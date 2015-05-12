#pragma once
#include "..\..\ml\base\header\TrainingData.h"

class DataReader
{
	public:

	static cvpr::TrainingSet	wine(const std::string &path);
};