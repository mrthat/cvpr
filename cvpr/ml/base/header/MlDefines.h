#pragma once

namespace cvpr
{
	//! ���v���f���^�C�v
	typedef enum {
		CLASSIFICATION_FOREST,
		CLASSIFICATION_TREE,
		REGRESSION_TREE,
		ADABOOST,
		GRADIENT_BOOST,
		STAT_MODEL,		//< �ꉞ�x�[�X�N���X�p
	} StatModelType;
};