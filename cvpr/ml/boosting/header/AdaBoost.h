#pragma once
//#include <tbb\blocked_range.h>
#include "..\..\base\header\StaticalModel.h"
#include "..\..\factory\header\StatModelFactory.h"
namespace cvpr
{
	//! adaboostパラメータクラス
	class AdaboostParameter : public StaticalModelParameter
	{
		public:

			int		save(const std::string &save_path) const;

			int		load(const std::string &load_path) ;

			//! 収束打ち切り用誤差最小値
			double min_error_rate;

			//! ラウンド数 => 選択する弱識別器の数
			unsigned int nr_rounds;

			//! 選択元の弱識別プールの総数
			unsigned int nr_weak_learners;

			//! 乱数のシード
			uint64 seed;
			
			//! 弱識別器ファクトリー
			mutable WeakLearnerPoolFactoryBase *factory;

		protected:
	};

	//! adaboostクラス
	class AdaBoost : public StaticalModel
	{
		public:

			/**
			*	パラメータのファイル出力
			*	@param	save_path	出力先のパス
			*	@return	エラーコード等
			*/
			virtual int		save(const std::string &save_path) const ;
			
			/**
			*	パラメータのファイル入力
			*	@param	load_path	入力元のパス
			*	@return	エラーコード等
			*/
			virtual int		load(const std::string &load_path) ;

			/**
			*	特徴ベクトルから予測結果を出す
			*	@param	feature	入力の特徴ベクトル
			*	@param	result	出力の予測結果
			*/
			virtual int		predict(const cv::Mat &feature, PredictionResult *result);

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
			*	弱識別器の候補
			*/
			struct WeakLearnerentry {
				//! 弱識別器
				PtrWeakLearner	learner;

				//! 評価値
				double			eval;
			};

			/**
			*	現在のモデルを破棄する
			*/
			void clear();

			/**
			*	学習データと重みを使用して弱識別器を評価する
			*	@param	datas	学習データ
			*	@param	weight	データの重み
			*	@param	model	評価対象のモデル
			*	@return	評価値
			*/
			double evaluate(const TrainingSet &datas, const cv::Mat &weight, PtrWeakLearner &model) ;

			/**
			*	学習サンプルの重みを更新する
			*	@param	datas		学習サンプル
			*	@param	src_weight	現在サンプルの重み
			*	@param	model		このモデルの識別結果で更新値を計算する
			*	@param	alpha		モデルの重み
			*	@param	dst_weight	更新後のサンプル重み
			*/
			void update_sample_weights(const TrainingSet &datas, const cv::Mat &src_weight,
				PtrWeakLearner &model, double alpha, cv::Mat &dst_weight) ;

			/**
			*	
			*/
			double calc_alpha(double error) const { return 0.5 * log( (1.0 - error) / error); };
			
			int train(const TrainingSet &datas, const AdaboostParameter &param, cv::RNG &rng);

			std::vector<PtrWeakLearner> weak_classifiers_;
			
			std::vector<double> alpha_t_;
			
			double sum_alpha_;

	};
};