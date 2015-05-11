#include "..\header\StatModelFactory.h"
#include "..\..\rf\header\ClassificationTree.h"
#include "..\..\rf\header\RandomForest.h"

using namespace cvpr;

PtrWeakLearner WeakLearnerFactory::create(const StatModelType &type)
{
	switch (type) {
		case StatModelType::CLASSIFICATION_TREE:
			return PtrWeakLearner(new ClassificationTree());
		case StatModelType::CLASSIFICATION_FOREST:
			return PtrWeakLearner(new ClassificationForest());
		default:
			return nullptr;
	}

	return nullptr;
}

std::vector<PtrWeakLearner> WeakLearnerPoolFactoryBase::create_trained_pool(const TrainingSet &datas, std::size_t pool_size) 
{
	std::vector<PtrWeakLearner>	pool;

	for (std::size_t ii = 0; ii < pool_size; ++ii) {
		PtrWeakLearner		new_model	=	next_model();
		PtrWeakLearnerParam	new_param	=	next_param();

		if (0 != new_model->train(datas, new_param.get()))
			continue;

		pool.push_back(new_model);
	}

	return pool;
}
