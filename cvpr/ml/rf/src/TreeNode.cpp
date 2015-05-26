#include "..\header\TreeNode.h"
using namespace cvpr;
using namespace cvpr::TreeNode;

int	NodeBase::save(cv::FileStorage &cvfs) const
{
	cv::internal::WriteStructContext	wsc(cvfs, "node_base", cv::FileNode::MAP);
	
	cv::write(cvfs, "node_id", (int)this->node_id);
	cv::write(cvfs, "left_id", (int)this->left_node_id);
	cv::write(cvfs, "right_id", (int)this->right_node_id);
	
	return 0;
}

int	NodeBase::load(cv::FileStorage &cvfs)
{
	cv::FileNode	top_node	=	cvfs["node_base"];
	int				tmp;
	cv::read(top_node["node_id"], tmp, 0);
	this->node_id		=	(unsigned)tmp;
	cv::read(top_node["left_id"], tmp, 0);
	this->left_node_id	=	(unsigned)tmp;
	cv::read(top_node["right_id"], tmp, 0);
	this->right_node_id	=	(unsigned)tmp;
	return 0;
}

int	SplitNodeBase::operator()(const cv::Mat &feature, const SplitNodeParameterBase *param) const
{

	double			kernel_value	=	0;

	kernel_value	=	kernel_function(feature, param);

	return split(kernel_value);
}

void	SplitNodeBase::operator()(const TrainingSet &train_set, TrainingSet &left_set, TrainingSet &right_set) const
{
	for (unsigned ii = 0; ii < train_set.size(); ++ii) {
		const PtrTrainingExample		target_example	=	train_set[ii];
		const SplitNodeParameterBase	*param			=	dynamic_cast<const SplitNodeParameterBase*>(train_set[ii]->param.get());

		if (LEFT == this->operator()(target_example->feature, param)) {
			left_set.push_back(target_example);
		} else {
			right_set.push_back(target_example);
		}
	}
}
int	SplitNodeBase::train(const TrainingSet &train_set, const StaticalModelParameter *param, std::mt19937 &rnd, TrainingSet *left, TrainingSet *right)
{
	init_attributes(train_set, rnd);
	//パラメータランダム初期化．オーバーライドされた奴
	init_params(train_set, param, rnd);
	double	min_kernel_value	=	std::numeric_limits<double>::max();
	double	max_kernel_value	=	-std::numeric_limits<double>::max();
	double	kernel_value_range;
	std::vector<double>		kernel_vals(train_set.size(), 0);

	//全データ舐めてカーネルの値の範囲を調べる
	for (unsigned ii = 0; ii < train_set.size(); ++ii) {
		double			kernel_value;
		const SplitNodeParameterBase	*param	=	dynamic_cast<const SplitNodeParameterBase*>(train_set[ii]->param.get());


		kernel_value		=	kernel_function(train_set.operator[](ii)->feature, param);
		min_kernel_value	=	std::min<double>(min_kernel_value, kernel_value);
		max_kernel_value	=	std::max<double>(max_kernel_value, kernel_value);
		kernel_vals[ii]		=	kernel_value;
	}

	kernel_value_range	=	max_kernel_value - min_kernel_value;

	//範囲内で2箇所cut pointをランダムに決める
	for (unsigned ii = 0; ii < NUM_CUTPOINT; ++ii) {
		std::uniform_real_distribution<>	real_dst(min_kernel_value, max_kernel_value);
		double rnd_value			=	real_dst(rnd);

		this->cut_points_[ii]	=	rnd_value ;
	}

	//上限下限の大小関係逆ならswapする
	if (this->cut_points_[IDX_UPPER_CUTPOINT] < this->cut_points_[IDX_UNDER_CUTPOINT]) {
		std::swap(this->cut_points_[IDX_UNDER_CUTPOINT], this->cut_points_[IDX_UPPER_CUTPOINT]);
	}

	double	rnd_val	=	rnd() / static_cast<double>(rnd.max() - rnd.min());

	//0.5の確率でcutpointを一個にする
	if (rnd_val < 0.5) {
		this->cut_points_[IDX_UNDER_CUTPOINT]	=	-std::numeric_limits<double>::max();
	}

	if (nullptr == left || nullptr == right) {
		return 0;
	}

	for (unsigned ii = 0; ii < train_set.size(); ++ii) {
		const PtrTrainingExample target_example	=	train_set.operator[](ii);
		if (LEFT == split(kernel_vals[ii])) {
			left->push_back(target_example);
		} else {
			right->push_back(target_example);
		}
	}

	return 0;
}

int	SplitNodeBase::save(cv::FileStorage &cvfs) const
{
	NodeBase::save(cvfs);

	cv::internal::WriteStructContext	wsc(cvfs, "SplitBase", cv::FileNode::MAP);
					
	cv::write<double>(cvfs, "cut_points", this->cut_points_);

	cv::write<int>(cvfs, "attributes", this->attributes_);

	return 0;
}

int	SplitNodeBase::load(cv::FileStorage &cvfs)
{
	NodeBase::load(cvfs);

	cv::FileNode	target_node	=	cvfs["SplitBase"];

	cv::read<double>(target_node["cut_points"], this->cut_points_);

	cv::read<int>(target_node["attributes"], this->attributes_);

	return 0;
}

void	SplitNodeBase::init_attributes(const TrainingSet &train_set, std::mt19937 &rnd)
{
	std::uniform_real_distribution<>	real_dist(0, (double)train_set.get_feature_type().total()) ;

	this->attributes_.assign(get_num_attributes(), 0);

	for (unsigned ii = 0; ii < this->attributes_.size(); ++ii) {
		double		rnd_value	=	real_dist(rnd);

		this->attributes_[ii]	=	static_cast<int>(std::floor(rnd_value));
	}
}