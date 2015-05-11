#pragma once
#include <random>
#include "LinearRegression.h"

namespace cvpr
{
	/**
	*	ロジスティック回帰のパラメータクラス
	*/
	class LogisticRegressionParameter : public LinearModelParameterBase
	{		
	};

	/**
	*	2クラス識別用ロジスティック回帰
	*/
	class LogisticRegression : public LinearModelBase
	{

		public:

			virtual int		predict(const cv::Mat &feature, PredictionResult *result, const PredictionParameter *param = nullptr);

			/**
			*	ロジスティック回帰の予測メソッド
			*	@param	feature	列ベクトルの特徴ベクトルを想定(=> 中でcvtはしない)
			*	@param	result	予測結果 各要素にそのクラスである確率が入ったk次列ベクトル
			*/
			int				predict(const cv::Mat &feature, ClassificationResult &result) const;

			/**
			*	ロジスティック回帰の学習メソッド
			*	2クラス識別用
			*	教師データは0,1の二値を想定．ラベルが多次元の場合は，各次元独立した2クラス識別で解く
			*	@param	train_set	学習データ.特徴ベクトルは列ベクトル,教師データは2値
			*	@param	param		全項目埋まっていること．デフォルト値の使用や値のチェックはしてない．
			*/
			int				train(const TrainingSet &train_set, const LogisticRegressionParameter &param)
			{
				return LinearModelBase::train(train_set, &param);
			};

		protected:

			/**
			*	訓練サンプルについて，予測結果とラベルから損失を求める
			*	@param	data	損失計算対象のサンプル
			*	@return	損失値
			*/
			virtual cv::Mat	calc_loss(const PtrTrainingExample data) const;

			/**
			*	パラメータ更新量を計算する(普通のgd)
			*	@param	train_set	更新量計算に使用する学習データ
			*	@param	delta_w		特徴ベクトルに対する重みベクトルの更新量
			*	@param	delta_w0	バイアスの更新量
			*/
			virtual void	calc_param_delta(const TrainingSet &train_set, cv::Mat &delta_w, double &delta_w0);
			
			/**
			*	学習の最初に重みをランダムに初期化する
			*	@param	train_set	学習データ
			*	@param	rnd			乱数エンジン
			*/
			void			init_weight(const TrainingSet &train_set, std::mt19937 &rnd) ;
			
			/**
			*	特徴ベクトルから重みとバイアスつかってスカラーのレスポンスを求める
			*	@param	feature	特徴ベクトル
			*	@return	応答値
			*/
			virtual cv::Mat	calc_activation(const cv::Mat &feature) const;

		private:

	};

	class SoftMaxRegression : public LogisticRegression
	{
		public:
		protected:

			virtual cv::Mat	calc_activation(const cv::Mat &feature) const;
			
			virtual cv::Mat	calc_loss(const PtrTrainingExample data) const;
	};
};