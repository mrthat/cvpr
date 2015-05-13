#pragma once

#include "..\..\base\header\StaticalModel.h"
#include "..\..\factory\header\StatModelFactory.h"

namespace cvpr
{
	/**
	*	gradient boost用パラメータ
	*/
	class GradientBoostParameter : public StaticalModelParameter
	{
		public:

			int save(const std::string &save_path) const { return 0; };

			int load(const std::string &load_path) { return 0; };

			//! 選択する弱識別器数
			unsigned nr_rounds;

			//! 乱数のシード
			//uint64 seed;

			//! ステージ毎に弱識別器生成するためのファクトリー
			StageWiseStatModelFactoryBase *factory;

			//! 学習率
			double shrinkage;

		protected:
	};

	class GradientBoost : public StaticalModel
	{
		public:
			/**
			*	パラメータのファイル出力
			*	@param	save_path	出力先のパス
			*	@return	エラーコード等
			*/
			virtual int		save(const std::string &save_path) const;
			
			/**
			*	パラメータのファイル入力
			*	@param	load_path	入力元のパス
			*	@return	エラーコード等
			*/
			virtual int		load(const std::string &load_path);

			/**
			*	特徴ベクトルから予測結果を出す
			*	@param	feature	入力の特徴ベクトル
			*	@param	result	出力の予測結果
			*/
			virtual int		predict(const cv::Mat &feature, PredictionResult *result, const PredictionParameter *param = nullptr);

			/**
			*	学習を実行する
			*	@param	train_set	学習データ
			*	@param	param		学習パラメータ
			*/
			virtual int		train(const TrainingSet &train_set, const StaticalModelParameter *param);

			/**
			*	クラスの型情報を返す
			*	@return	統計モデルの識別子
			*/
			virtual StatModelType get_type() const { return StatModelType::ADABOOST; };

		protected:

			/**
			*	学習本処理
			*	@param	datas	学習データ
			*	@param	param	パラメータ
			*/
			int train(const TrainingSet &datas, const GradientBoostParameter &param);

			/**
			*	0番目の弱識別器(-> 回帰の場合サンプルの平均(L2最小化))
			*/
			cv::Mat	f0;

			//! 弱識別器
			std::vector<PtrWeakLearner> weak_laerner;

			//! 学習率 (各識別器統合時に使用)
			double	shrinkage;
	};


};