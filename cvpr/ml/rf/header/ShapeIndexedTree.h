#pragma once

#include "RegressionTree.h"

namespace cvpr
{
	typedef RegressionTree ShapeIndexedTree;

	class ShapeIndexedPredictionParameter : public TreeNode::ShapeIndexedSplitParameter
	{
	};

	class ShapeIndexedTreeParameter : public RegressionTreeParameter
	{
		public:

			ShapeIndexedTreeParameter()
				: RegressionTreeParameter()
			{
				split_type_list.push_back(TreeNode::SplitNodeType::SPLIT_TYPE_SHAPE_INDEXED);
			}

		protected:

			// shape indexedのsplit nodeだけ使いたいのでprotectedにしちゃう
			RegressionTreeParameter::add_split_type;
			RegressionTreeParameter::remove_split_type;
			RegressionTreeParameter::set_default_split_list;
	};
};