#pragma once

#include "..\..\rf\header\ShapeIndexedTree.h"

namespace cvpr
{
	class FaceAlignKazemiCvpr14
	{
		public:

		/**
		*	学習パラメータクラス
		*/
		class TrainParameter : public ShapeIndexedTreeParameter
		{
			public:

			//! gradient boostのラウンド数
			std::size_t	nr_rounds;

			//! regressorの段数
			std::size_t	nr_cascades;

			//! 学習率
			double shrinkage;

			protected:
		};


		protected:
	};
};