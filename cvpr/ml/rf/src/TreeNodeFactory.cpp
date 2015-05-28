#include <cassert>
#include "..\header\TreeNodeFactory.h"

using namespace cvpr;
using namespace TreeNode;

PtrSplitNodeBase	TreeNodeFactory::create_split_node(SplitNodeType node_type, int feature_depth)
{
	switch (feature_depth) {
		case CV_64F:
			return create_split_node<double>(node_type);
		case CV_32F:
			return create_split_node<float>(node_type);
		case CV_32S:
			return create_split_node<int>(node_type);
		case CV_16S:
			return create_split_node<short>(node_type);
		case CV_16U:
			return create_split_node<ushort>(node_type);
		case CV_8S:
			return create_split_node<char>(node_type);
		case CV_8U:
			return create_split_node<uchar>(node_type);
		default:
			assert(!"unsupported data type");
			return 0;
	}
}

template<typename ty>
PtrSplitNodeBase TreeNodeFactory::create_split_node(SplitNodeType node_type)
{
	switch (node_type) {
		case SPLIT_TYPE_BASE:
		default:
			return PtrSplitNodeBase(nullptr);
		case SPLIT_TYPE_AXISALIGNED:
			return PtrSplitNodeBase(new SplitNodeAxisAligned<ty>());
		case SPLIT_TYPE_LINE:
			return PtrSplitNodeBase(new SplitNodeOrientedLine<ty>());
		case SPLIT_TYPE_CONIC:
			return PtrSplitNodeBase(new SplitNodeConicSection<ty>());
		case SPLIT_TYPE_ATAN:
			return PtrSplitNodeBase(new SplitNodeAtan<ty>());
		case SPLIT_TYPE_MIN:
			return PtrSplitNodeBase(new SplitNodeMin<ty>());
		case SPLIT_TYPE_HAAR:
			return PtrSplitNodeBase(new SplitNodeHaar<ty>());
		case SPLIT_TYPE_HAAR_INT:
			return PtrSplitNodeBase(new SplitNodeHaarIntegral<ty>());
		case SPLIT_TYPE_SHAPE_INDEXED:
			return PtrSplitNodeBase(new SplitNodeShapeIndexed<ty>());
	}
}

PtrLeafNodeBase TreeNodeFactory::create_leaf_node(LeafNodeType node_type)
{
	switch (node_type) {
		default:
		case LEAF_TYPE_BASE:
			return nullptr;
		case LEAF_TYPE_CLASSIFICATION:
			return PtrLeafNodeBase(new LeafNodeClassification());
		case LEAF_TYPE_REGRESSION:
			return PtrLeafNodeBase(new LeafNodeRegression());
	}
}