#define _CRT_SECURE_NO_WARNINGS 1
#include "..\header\RandomizedTree.h"

using namespace cvpr ;
using namespace TreeNode;

static bool _cmp_node(const PtrNodeBase &lhs, const PtrNodeBase &rhs)
{
	return lhs->node_id < rhs->node_id;
}

std::vector<TreeNode::SplitNodeType>	RandomizedTreeParameter::default_split_list()
{
	std::vector<SplitNodeType>	list;
	list.push_back(SPLIT_TYPE_AXISALIGNED);
	list.push_back(SPLIT_TYPE_LINE);
	list.push_back(SPLIT_TYPE_CONIC);
	list.push_back(SPLIT_TYPE_ATAN);
	list.push_back(SPLIT_TYPE_MIN);
	return list;
}

int	RandomizedTree::save(const std::string &save_path) const
{
	
	_mkdir(save_path.c_str());

	std::string		tree_data_path	=	save_path + "\\" + "tree_data.txt";

	save_node_type(tree_data_path);

	for (unsigned ii = 0; ii < this->nodes_.size(); ++ii) {
		
		char	node_save_path[FILENAME_MAX];

		sprintf(node_save_path, "%s\\node%d", save_path.c_str(), ii);
					
		cv::FileStorage	cvfs(node_save_path, cv::FileStorage::WRITE);

		this->nodes_[ii]->save(cvfs);
	}

	return 0;
}

int	RandomizedTree::load(const std::string &load_path)
{
	nodes_.clear();

	std::vector<int>	node_types;
	std::vector<int>	detail_types;
	std::string			tree_data_path	=	load_path + "\\" + "tree_data.txt";

	load_node_type(tree_data_path, node_types, detail_types);

	this->nodes_.assign(node_types.size(), nullptr);

	for (unsigned ii = 0; ii < this->nodes_.size(); ++ii) {
		char		node_file_path[FILENAME_MAX];
		PtrNodeBase	target_node	=	nullptr;

		sprintf(node_file_path, "%s\\node%d", load_path.c_str(), ii);

		cv::FileStorage	cvfs(node_file_path, cv::FileStorage::READ);

		if (TreeNode::NODE_TYPE_SPLIT == node_types[ii]) {
			PtrSplitNodeBase	tmp	=	TreeNodeFactory::create_split_node((TreeNode::SplitNodeType)detail_types[ii], this->feature_depth_);
			
			target_node	=	std::static_pointer_cast<NodeBase, SplitNodeBase>(tmp);
		} else if (TreeNode::NODE_TYPE_LEAF == node_types[ii]) {
			PtrLeafNodeBase		tmp	=	TreeNodeFactory::create_leaf_node((TreeNode::LeafNodeType)detail_types[ii]);

			target_node	=	std::static_pointer_cast<NodeBase, LeafNodeBase>(tmp);
		}

		target_node->load(cvfs);

		this->nodes_[ii]	=	target_node;
	}

	return 0;
}

int	RandomizedTree::predict(const cv::Mat &feature, PredictionResult *result, const PredictionParameter *param)
{
	PtrNodeBase	target_node(nullptr);
	const SplitNodeParameterBase	*param_	=	dynamic_cast<const SplitNodeParameterBase*>(param);

	if (this->nodes_.empty()) {
		return -1;
	}

	target_node	=	nodes_.front();

	while (target_node->get_node_type() == TreeNode::NODE_TYPE_SPLIT) {
		PtrSplitNodeBase	target_split	=	std::static_pointer_cast<SplitNodeBase, NodeBase>(target_node);
		int					split_result	=	target_split->operator()(feature, param_);

		if (TreeNode::SplitNodeBase::LEFT == split_result) {
			target_node	=	this->nodes_[target_node->left_node_id];
		} else {
			target_node	=	this->nodes_[target_node->right_node_id];
		}
	}

	PtrLeafNodeBase	target_leaf	=	std::static_pointer_cast<LeafNodeBase, NodeBase>(target_node);
	target_leaf->operator()(feature, result);

	return 0;
}

int	RandomizedTree::train(const TrainingSet &train_set, const StaticalModelParameter *param)
{
	const RandomizedTreeParameter	*tree_param	=	dynamic_cast<const RandomizedTreeParameter*>(param);
	if (nullptr == tree_param) {
		return -1;
	}

	if (train_set.size() == 0) {
		return 0;
	}

	nodes_.clear();

	return train(train_set, *tree_param);
}

int	RandomizedTree::train(const TrainingSet &train_set, const RandomizedTreeParameter &param)
{
	std::mt19937 rng(param.rng_seed);

	this->feature_depth_	=	train_set.get_feature_type().depth();

	grow_tree(train_set, param, 0, 0, rng);

	std::sort(this->nodes_.begin(), this->nodes_.end(), _cmp_node);

	return 0;
}

int RandomizedTree::save_node_type(const std::string &file_path) const
{
	std::ofstream	data_file(file_path);
	
	if (data_file.bad()) {
		return -1;
	}

	data_file << "feature_depth=" << this->feature_depth_ << std::endl;

	data_file << "num_nodes=" << this->nodes_.size() << std::endl;

	for (unsigned ii = 0; ii < this->nodes_.size(); ++ii) {
		int	node_type	=	this->nodes_[ii]->get_node_type();
		int	detail_type;
		
		if (TreeNode::NODE_TYPE_SPLIT == node_type) {
			PtrSplitNodeBase	tmp	=	std::static_pointer_cast<SplitNodeBase, NodeBase>(nodes_[ii]);
			detail_type	=	tmp->get_split_type();
		} else if (TreeNode::NODE_TYPE_LEAF == node_type) {
			PtrLeafNodeBase	tmp	=	std::static_pointer_cast<LeafNodeBase, NodeBase>(nodes_[ii]);
			detail_type	=	tmp->leaf_type();
		}

		data_file << node_type << "," << detail_type << std::endl;
	}

	return 0;
}

int	RandomizedTree::load_node_type(const std::string &file_path, std::vector<int> &node_types, std::vector<int> &detail_types)
{
	std::ifstream	data_file(file_path);
	std::string		line_buff;
	unsigned		num_nodes	=	0;

	if (data_file.bad()) {
		return -1;
	}
				
	std::getline(data_file, line_buff);
	
	if (1 != sscanf(line_buff.c_str(), "feature_depth=%d", &this->feature_depth_)) {
		return -1;
	}

	std::getline(data_file, line_buff);

	if (1 != sscanf(line_buff.c_str(), "num_nodes=%d", &num_nodes)) {
		return -1;
	}

	node_types.assign(num_nodes, 0);
	
	detail_types.assign(num_nodes, 0);

	for (unsigned ii = 0; ii < num_nodes && (!data_file.eof()); ++ii) {

		std::getline(data_file, line_buff);

		if (2 != sscanf(line_buff.c_str(), "%d,%d", &node_types[ii], &detail_types[ii])) {
			return -1;
		}
	}

	return 0;
}

int	RandomizedTree::grow_tree(const TrainingSet &train_set, const RandomizedTreeParameter &param, unsigned tree_height, unsigned node_id, std::mt19937 &rng)
{
	MatType	feature_type	=	train_set.get_feature_type();
	MatType	label_type		=	train_set.get_label_type();

	printf("train node_id %d, height = %d\n", node_id, tree_height);
	printf("num_samples %d\n", train_set.size());

	if (is_end_growth(train_set, param, tree_height)) {
		//leafÇÃäwèKÇÇ∑ÇÈ
		puts("reach leaf");
		//factoryÇ≈leafçÏÇÈ
		LeafNodeType	ltype	=	leaf_type();
		PtrLeafNodeBase	leaf_node	=	TreeNodeFactory::create_leaf_node(ltype);
		//leaf->train
		leaf_node->node_id	=	node_id;
		leaf_node->train(train_set);
		this->nodes_.push_back(leaf_node);

		print_train_log(leaf_node, train_set);

		return node_id;
	} else {

		//ïÅí ÇÃäwèKÇÇ∑ÇÈ
		int					last_node_id	=	node_id;
		PtrSplitNodeBase	best_split	=	nullptr;
		double				best_score	=	-std::numeric_limits<double>::max();
		TrainingSet	left_set(feature_type, label_type);
		TrainingSet	right_set(feature_type, label_type);
					
		puts("FindBestSplit ...");

		for (unsigned ii = 0; ii < param.num_splits; ++ii) {
			SplitNodeType		split_type	=	random_sample<SplitNodeType>(param.split_type_list, rng);
			PtrSplitNodeBase	new_split	=	TreeNode::TreeNodeFactory::create_split_node(split_type, feature_depth_);
			double				new_score	=	0;
			TrainingSet	left_tmp(feature_type, label_type);
			TrainingSet	right_tmp(feature_type, label_type);

			new_split->train(train_set, rng, &left_tmp, &right_tmp);
			new_score	=	calc_entropy_gain(train_set, left_tmp, right_tmp, param);//EvaluateSplitNode(train_set, param, new_split);
			
			if (best_score < new_score) {
				best_score = new_score;

				best_split	=	new_split;
				left_set	=	left_tmp;
				right_set	=	right_tmp;
			}

		}

		best_split->node_id	=	node_id;
		printf("info gain = %f\n", best_score);
		//info gainÇ™è¨Ç≥Ç©Ç¡ÇΩÇÁñ≥óùÇ‚ÇËleafÇ…Ç∑ÇÈ
		if (best_score < param.min_info_grain) {
			return grow_tree(train_set, param, param.max_height + 1, node_id, rng);
		}
		
		print_train_log(best_split, train_set);

		//ç∂ë§ÇçXÇ…äwèKÇ∑ÇÈ
		best_split->left_node_id	=	last_node_id + 1;
		last_node_id	=	grow_tree(left_set, param, tree_height + 1, best_split->left_node_id, rng);
		//âEë§ÇçXÇ…äwèKÇ∑ÇÈ
		best_split->right_node_id	=	last_node_id + 1;
		last_node_id	=	grow_tree(right_set, param, tree_height + 1, best_split->right_node_id, rng);

		this->nodes_.push_back(best_split);
		return last_node_id;
	}
	return 0;
}
