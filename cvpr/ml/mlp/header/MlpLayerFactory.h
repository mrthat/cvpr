#pragma once
#include "MlpLayerBase.h"

namespace cvpr
{
	namespace mlp
	{
				
		/**
		*	�w�̃t�@�N�g���[�N���X
		*/
		class LayerFactory
		{
			public:
				/**
				*	�w�̎�ʂɉ�����LayerBase�̃T�u�N���X��new���ĕԂ�
				*	@param	layer_type	�w�̎��
				*	@return	���������w. �m��Ȃ���ʂ̏ꍇ�͋�
				*/
				static PtrOutputLayerBase	create(MlpOutputLayerType layer_type) ;
				/**
				*	�w�̎�ʂɉ�����LayerBase�̃T�u�N���X��new���ĕԂ�
				*	@param	layer_type	�w�̎��
				*	@return	���������w. �m��Ȃ���ʂ̏ꍇ�͋�
				*/
				static PtrHiddenLayerBase	create(MlpHiddenLayerType layer_type) ;
		};
	};
};