#pragma once

#include "RandomizedTree.h"

namespace cvpr
{
	/**
	*	‰ñ‹A—p–Ø
	*/
	class RegressionTree : public RandomizedTree
	{
		public:

		virtual TreeType tree_type() const { return TREE_TYPE_REGRESSION; }

		using RandomizedTree::train;

		protected:

		virtual void	print_train_log(const TreeNode::PtrSplitNodeBase split, const TrainingSet &train_set) const;

		TreeNode::LeafNodeType	leaf_type() const { return TreeNode::LEAF_TYPE_REGRESSION; };

		virtual double	calc_entropy_gain(const TrainingSet &train_set, const TrainingSet &left_set, const TrainingSet &right_set, const RandomizedTreeParameter &param) const;

		virtual void	print_train_log(const TreeNode::PtrLeafNodeBase leaf, const TrainingSet &train_set) const;

		virtual bool	is_end_growth(const TrainingSet &train_set, const cvpr::RandomizedTreeParameter &param, unsigned tree_height) const;

	};
};