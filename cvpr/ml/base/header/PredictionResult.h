#pragma once
#include <memory>
#include <vector>
#include <cmath>
#include <opencv2\core\core.hpp>
#include "utils.h"


namespace cvpr
{
	/**
	*	識別結果のタイプ
	*/
	typedef enum {
		RESULT_TYPE_CLASSIFICATION,			//< クラス識別
		RESULT_TYPE_REGRESSION,				//< 回帰
	} ResultType;

	/**
	*	識別結果ベースクラス
	*/
	class PredictionResult
	{
		public:

			/**
			*	予測結果行列のsetter
			*/
			void	set_posterior(const cv::Mat &posterior) 
			{
				this->posterior_	=	posterior;
			}

			/**
			*	予測結果行列のgetter
			*/
			const cv::Mat	get_posterior() const
			{
				return this->posterior_;
			}

			/**
			*	識別結果のタイプを取得
			*/
			virtual ResultType	type() const = 0;

			virtual				~PredictionResult() {};

		protected:
			/** 予測結果行列 */
			cv::Mat_<double>	posterior_;
	};

	/**
	*	クラス識別結果クラス
	*/
	class ClassificationResult : public PredictionResult
	{
		public:

			/**
			*	識別結果のタイプを取得
			*/
			virtual cvpr::ResultType	type() const { return RESULT_TYPE_CLASSIFICATION; };
			
			/**
			*	posterior_内の最大要素のidxを取得
			*	@return	label内の最大要素のidx
			*/
			int	label() const 
			{
				return max_idx(posterior_);
			}

		protected:
	};
	
	/**
	*	回帰予測結果クラス
	*/
	class RegressionResult : public PredictionResult
	{
		public:
			
			/**
			*	識別結果のタイプを取得
			*/
			virtual cvpr::ResultType	type() const { return RESULT_TYPE_REGRESSION; };

		protected:
	};
	
	typedef std::shared_ptr<PredictionResult> PtrPredictionResult ;

	/**
	*	予測結果のfactoryクラス
	*/
	class PredictionResultFactory
	{
		public:
			/**
			*	予測結果種別から予測結果を生成して返す
			*	@param	type	予測結果種別
			*	@return	null:知らないtype, その他:対応するクラスのインスタンス
			*/
			static PtrPredictionResult	create(ResultType type) ;

		protected:
	};
};