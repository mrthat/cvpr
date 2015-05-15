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
			*	パラメータが不正でないか調べる
			*	学習セットに対して妥当かどうかを調べるかもしれないので一応データも渡す
			*	@param	datas	データ
			*	@param	param	検査対象パラメータ
			*	@param	falseなら不正
			*/
			virtual bool	is_valid_param(const TrainingSet &datas, const GradientBoostParameter &param) const;

			/**
			*	初期モデル(f0)を計算する
			*	@param	datas	学習データ
			*	@param	parama	パラメータ
			*	@return	成否
			*/
			int find_initial_model(const TrainingSet &datas, const GradientBoostParameter &param);

			/**
			*	目標ベクトルを更新した学習セットを取得
			*	各ステージで弱識別器は学習済み分と目標との残差にフィッティングするため,
			*	毎ステージ新しい学習セットを作る必要がある．
			*	特徴ベクトルはshallow copy
			*	@param	datas	更新元データ
			*	@param	param	パラメータ
			*/
			TrainingSet calc_next_target(const TrainingSet &datas, const GradientBoostParameter &param);

			/**
			*	色々開放して再学習できるようにする
			*/
			void release();

			/**
			*	GradientBoostのデータファイルパス取得
			*	@param	保存先ディレクトリパス
			*	@return	データファイルパス
			*/
			std::string get_data_path(const std::string &save_path) const;

			/**
			*	0番目の弱識別器(-> 回帰の場合サンプルの平均(L2最小化))
			*/
			cv::Mat	f0;

			//! 弱識別器
			std::vector<PtrWeakLearner> weak_learner;

			//! 学習率 (各識別器統合時に使用)
			double	shrinkage;

	};


};