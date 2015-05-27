#include "..\header\SplitNode.h"
#include "..\..\..\util\include\utils.h"

using namespace cvpr;
using namespace cvpr::TreeNode;

template<typename ty>
void SplitNodeShapeIndexed<ty>::init_params(const TrainingSet &train_set, const StaticalModelParameter *param, std::mt19937 &rnd)
{
	// 形状パラメータは全データ同じ数ある想定で欠損値考慮しない

	const ShapeIndexedTrainParameter	*param_	=	dynamic_cast<const ShapeIndexedTrainParameter*>(train_set[0]->param);
	
	assert(nullptr != param_);
	
	std::uniform_int_distribution<std::size_t>	distr_idx(0, param_->num_shape);
	std::uniform_int_distribution<int>			distr_pos(-param_->radius, param_->radius);

	// 対象形状点決める
	shape_index	=	distr_idx(rnd);

	// 相対特徴点位置を決める
	for (int ii = 0; ii < NUM_FEATURE_POS; ++ii) {
		offsets.push_back(cv::Point2d(dist_pos(rnd), distr_pos(rnd)));
	}
}

template<typename ty>
double SplitNodeShapeIndexed<ty>::kernel_function(const cv::Mat &feature, const SplitNodeParameterBase *param) const
{
	std::vector<cv::Point2d>	warped_offsets;
	std::vector<cv::Point2d>	feature_pos;
	cv::Mat	transform;
	const ShapeIndexedSplitParameter	*param_	=	dynamic_cast<const ShapeIndexedSplitParameter*>(param);
	std::vector<double>	val;

	assert(null != param_);

	cv::perspectiveTransform(offsets, warped_offsets, param_->transform);

	for (std::size_t ii = 0; ii < warped_offsets.size(); ++ii) {
		cv::Point2d	pos	=	param_->shape[shape_index] + warped_offsets[ii];

		pos	=	round(feature, pos); // 範囲外なら例外でいい気もする?

		val.push_back((double)feature.at<ty>((int)pos.y, (int)pos.x));
	}

	return val[0] - val[1];
}
