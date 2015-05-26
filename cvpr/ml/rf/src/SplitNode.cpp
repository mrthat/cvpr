#include "..\header\SplitNode.h"

using namespace cvpr;
using namespace cvpr::TreeNode;

template<typename ty>
void SplitNodeShapeIndexed<ty>::init_params(const TrainingSet &train_set, const StaticalModelParameter *param, std::mt19937 &rnd)
{
	// 形状パラメータは全データ同じ数ある想定で欠損値考慮しない

	const ShapeIndexedSplitParameter	*param	=	dynamic_cast<const ShapeIndexedSplitParameter*>(train_set[0]->param);
	
	assert(nullptr != param);
	
	std::uniform_int_distribution<std::size_t>	distr(0, param->shape.size() - 1);

	shape_index	=	distr(rnd);
}

template<typename ty>
double SplitNodeShapeIndexed<ty>::kernel_function(const cv::Mat &feature, const SplitNodeParameterBase *param) const
{
	return 0;
}
