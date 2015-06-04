#include "..\header\ClassificationTree.h"

using namespace cvpr;

double	ClassificationTree::calc_entropy_gain(const TrainingSet &train_set, const TrainingSet &left_set, const TrainingSet &right_set, const RandomizedTreeParameter &param) const
{
	double		entropy_gain	=	0;

	entropy_gain	=	train_set.compute_label_entropy()
		- left_set.size() / (float)train_set.size() * left_set.compute_label_entropy()
		- right_set.size() / (float)train_set.size() * right_set.compute_label_entropy();

	return entropy_gain;
}

bool	ClassificationTree::is_end_growth(const TrainingSet &train_set, const cvpr::RandomizedTreeParameter &param, unsigned tree_height) const
{
	if (param.max_height <= tree_height) {
		return true;
	}

	if (train_set.size() <= param.min_samples) {
		return true;
	}

	return false;
}

void	ClassificationTree::print_train_log(const TreeNode::PtrSplitNodeBase split, const TrainingSet &train_set) const
{
	MatType				ltype	=	train_set.get_label_type();
	MatType				ftype	=	train_set.get_feature_type();
	int					rows	=	(int)ltype.total();
	cv::Mat_<double>	left_tmp;
	cv::Mat_<double>	right_tmp;
	TrainingSet			left_set(ftype, ltype);
	TrainingSet			right_set(ftype, ltype);

	split->operator()(train_set, left_set, right_set);

	
	left_set.compute_target_mean(left_tmp);
	right_set.compute_target_mean(right_tmp);

	cv::Mat_<double>	left_dist(rows, 1, (double*)left_tmp.data);
	cv::Mat_<double>	right_dist(rows, 1, (double*)right_tmp.data);


	printf("left dist\n");
	for (unsigned ii = 0; ii < left_dist.total(); ++ii) {
		printf("\tlabel%d:%f\n", ii, left_dist.at<double>(ii) / left_set.size());
	}

	printf("right dist\n");
	for (unsigned ii = 0; ii < right_dist.total(); ++ii) {
		printf("\tlabel%d:%f\n", ii, right_dist.at<double>(ii) / right_set.size());
	}

}

void	ClassificationTree::print_train_log(const TreeNode::PtrLeafNodeBase leaf, const TrainingSet &train_set) const
{
	cv::Mat_<double>	label_dist;

	train_set.compute_target_mean(label_dist);

	printf("leaf dist\n");

	for (unsigned ii = 0; ii < label_dist.total(); ++ii) {
		printf("\tlabel%d:%f\n", ii, label_dist.at<double>(ii) / std::max<double>((double)train_set.size(), 1.0));
	}
}
