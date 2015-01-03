#pragma once
#include "TreeNode.h"
#include "SplitNode.h"
#include "LeafNode.h"

namespace cvpr
{
	namespace TreeNode
	{
		class TreeNodeFactory
		{
			public:

				static PtrSplitNodeBase	create_split_node(SplitNodeType node_type, int feature_depth) ;

				template<typename ty> static PtrSplitNodeBase create_split_node(SplitNodeType node_type) ;

				static PtrLeafNodeBase create_leaf_node(LeafNodeType node_type) ;

			protected:
		};
	};
};