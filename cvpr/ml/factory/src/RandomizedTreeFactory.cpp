#include "..\header\RandomizedTreeFactory.h"

using namespace cvpr;

PtrRandomizedTree	RandomizedTreeFactory::Create(TreeType tree_type)
{
	switch (tree_type) {
		default:
			return PtrRandomizedTree(nullptr);
		case TREE_TYPE_CLASSIFICATION:
			return PtrRandomizedTree(new ClassificationTree());
	}
}

PtrWeakLearnerParam ClassificationTreePoolFactory::next_param()
{
	PtrWeakLearnerParam	param_	=	PtrWeakLearnerParam(new ClassificationTreeParameter(this->param));

	return param_;
}

PtrWeakLearner ClassificationTreePoolFactory::next_model()
{
	return PtrWeakLearner(new ClassificationTree());
}