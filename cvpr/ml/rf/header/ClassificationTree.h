#pragma once

#include "RandomizedTree.h"

namespace cvpr
{
	typedef RandomizedTreeParameter ClassificationTreeParameter;

	/**
	*	ƒNƒ‰ƒXŽ¯•Ê—p‚Ìrandomized tree
	*/
	class ClassificationTree : public RandomizedTree
	{
		public:

		virtual TreeType	tree_type() const { return TREE_TYPE_CLASSIFICATION; };

		virtual int	train(const TrainingSet &train_set, const StaticalModelParameter *param)
		{
			return RandomizedTree::train(train_set, param);
		}

		protected:

		virtual void	print_train_log(const TreeNode::PtrSplitNodeBase split, const TrainingSet &train_set) const;

		TreeNode::LeafNodeType	leaf_type() const { return TreeNode::LEAF_TYPE_CLASSIFICATION; };

		virtual double	calc_entropy_gain(const TrainingSet &train_set, const TrainingSet &left_set, const TrainingSet &right_set, const RandomizedTreeParameter &param) const;

		virtual void	print_train_log(const TreeNode::PtrLeafNodeBase leaf, const TrainingSet &train_set) const;

		virtual bool	is_end_growth(const TrainingSet &train_set, const cvpr::RandomizedTreeParameter &param, unsigned tree_height) const;
	};
};