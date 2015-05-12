#include "..\header\RegressionTree.h"

using namespace cvpr;

double RegressionTree::calc_entropy_gain(const TrainingSet &train_set, const TrainingSet &left_set, const TrainingSet &right_set, const RandomizedTreeParameter &param) const
{
	double	prev_var	=	train_set.compute_target_var();
	double	left_var	=	left_set.compute_target_var();
	double	right_var	=	right_set.compute_target_var();
	double	gain		=	prev_var - (left_var + right_var);

	return gain;
}

bool RegressionTree::is_end_growth(const TrainingSet &train_set, const cvpr::RandomizedTreeParameter &param, unsigned tree_height) const
{
	// 一応クラス識別木の丸コピをおいておく
	if (param.max_height <= tree_height) {
		return true;
	}

	if (train_set.size() <= param.min_samples) {
		return true;
	}

	return false;
}

void RegressionTree::print_train_log(const TreeNode::PtrSplitNodeBase split, const TrainingSet &train_set) const
{
	MatType	ltype	=	train_set.get_label_type();
	MatType	ftype	=	train_set.get_feature_type();
	TrainingSet	left_set(ftype, ltype);
	TrainingSet	right_set(ftype, ltype);

	split->operator()(train_set, left_set, right_set);

	double	prev_var	=	train_set.compute_target_var();
	double	left_var	=	left_set.compute_target_var();
	double	right_var	=	right_set.compute_target_var();

	printf("\t var_total=%f\n", prev_var);
	printf("\t left_var=%f\n", left_var);
	printf("\t right_var=%f\n", right_var);
}

void RegressionTree::print_train_log(const TreeNode::PtrLeafNodeBase leaf, const TrainingSet &train_set) const
{
	double	leaf_var	=	train_set.compute_target_var();

	printf("leaf_var=%f\n", leaf);
}
