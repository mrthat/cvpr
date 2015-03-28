#pragma once

namespace cvpr
{
	//! 統計モデルタイプ
	typedef enum {
		CLASSIFICATION_FOREST,
		CLASSIFICATION_TREE,
		ADABOOST,
		STAT_MODEL,		//< 一応ベースクラス用
	} StatModelType;
};