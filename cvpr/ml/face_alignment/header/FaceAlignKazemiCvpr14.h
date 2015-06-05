#pragma once

#include "..\..\rf\header\ShapeIndexedTree.h"

namespace cvpr
{
	class FaceAlignKazemiCvpr14
	{
		public:

		/**
		*	�w�K�p�����[�^�N���X
		*/
		class TrainParameter : public ShapeIndexedTreeParameter
		{
			public:

			//! gradient boost�̃��E���h��
			std::size_t	nr_rounds;

			//! regressor�̒i��
			std::size_t	nr_cascades;

			//! �w�K��
			double shrinkage;

			protected:
		};


		protected:
	};
};