#pragma once
#include <memory>

namespace cvpr
{
	/**
	*	�\���p�����[�^�x�[�X�N���X
	*/
	class PredictionParameter
	{
		public:

			virtual ~PredictionParameter(){};

		protected:
	};

	typedef std::shared_ptr<PredictionParameter> PtrPredictionParameter;

};