#pragma once
#include "MlpLayerBase.h"

namespace cvpr
{
	namespace mlp
	{
				
		/**
		*	層のファクトリークラス
		*/
		class LayerFactory
		{
			public:
				/**
				*	層の種別に応じてLayerBaseのサブクラスをnewして返す
				*	@param	layer_type	層の種別
				*	@return	生成した層. 知らない種別の場合は空
				*/
				static PtrOutputLayerBase	create(MlpOutputLayerType layer_type) ;
				/**
				*	層の種別に応じてLayerBaseのサブクラスをnewして返す
				*	@param	layer_type	層の種別
				*	@return	生成した層. 知らない種別の場合は空
				*/
				static PtrHiddenLayerBase	create(MlpHiddenLayerType layer_type) ;
		};
	};
};