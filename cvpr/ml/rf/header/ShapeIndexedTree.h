#pragma once

#include "RegressionTree.h"

namespace cvpr
{
	typedef RegressionTree ShapeIndexedTree;

	class ShapeIndexedPredictionParameter : public TreeNode::ShapeIndexedSplitParameter
	{
	};

	class ShapeIndexedTreeParameter : public virtual RegressionTreeParameter, public virtual TreeNode::ShapeIndexedTrainParameter
	{
		public:

			ShapeIndexedTreeParameter()
				: RegressionTreeParameter()
			{
				split_type_list.push_back(TreeNode::SplitNodeType::SPLIT_TYPE_SHAPE_INDEXED);
			}

		protected:

			// shape indexed‚Ìsplit node‚¾‚¯Žg‚¢‚½‚¢‚Ì‚Åprotected‚É‚µ‚¿‚á‚¤
			RegressionTreeParameter::add_split_type;
			RegressionTreeParameter::remove_split_type;
			RegressionTreeParameter::set_default_split_list;

			virtual int		save(const std::string &save_path) const { return 0; };

			virtual int		load(const std::string &load_path) { return 0; };
	};
};