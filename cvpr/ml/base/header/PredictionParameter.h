#pragma once
#include <memory>

namespace cvpr
{
	/**
	*	予測パラメータベースクラス
	*/
	class PredictionParameter
	{
		public:

			virtual ~PredictionParameter(){};

		protected:
	};

	typedef std::shared_ptr<PredictionParameter> PtrPredictionParameter;

};