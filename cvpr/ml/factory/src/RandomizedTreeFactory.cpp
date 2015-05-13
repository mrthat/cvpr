#include "..\header\RandomizedTreeFactory.h"
#include "..\..\rf\header\RegressionTree.h"

using namespace cvpr;

PtrRandomizedTree	RandomizedTreeFactory::Create(TreeType tree_type)
{
	switch (tree_type) {
		default:
			return PtrRandomizedTree(nullptr);
		case TREE_TYPE_CLASSIFICATION:
			return PtrRandomizedTree(new ClassificationTree());
		case TREE_TYPE_REGRESSION:
			return PtrRandomizedTree(new RegressionTree());
	}
}

PtrWeakLearnerParam ClassificationTreePoolFactory::next_param()
{
	PtrWeakLearnerParam	param_	=	PtrWeakLearnerParam(new ClassificationTreeParameter(this->param));

	((ClassificationTreeParameter*)param_.get())->rng_seed	=	rng();

	return param_;
}

PtrWeakLearner ClassificationTreePoolFactory::next_model()
{
	return PtrWeakLearner(new ClassificationTree());
}