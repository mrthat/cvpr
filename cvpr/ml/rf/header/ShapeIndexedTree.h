#pragma once

#include "RegressionTree.h"

namespace cvpr
{
	typedef RegressionTree ShapeIndexedTree;

	class ShapeIndexedTreeParameter : public RegressionTreeParameter
	{
		public:

			ShapeIndexedTreeParameter()
				: RegressionTreeParameter()
			{
				split_type_list.push_back(TreeNode::SplitNodeType::SPLIT_TYPE_SHAPE_INDEXED);
			}

		protected:

			// shape indexed��split node�����g�������̂�protected�ɂ����Ⴄ
			RegressionTreeParameter::add_split_type;
			RegressionTreeParameter::remove_split_type;
			RegressionTreeParameter::set_default_split_list;

	};
};