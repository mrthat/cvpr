#pragma once
#include "PredictionResult.h"
#include "TrainingData.h"
#include "MlDefines.h"

namespace cvpr
{

	/** パラメータベースクラス */
	class StaticalModelParameter
	{
		public:
			/** 正則化のタイプ */
			typedef enum { REGULARIZE_L2, REGULARIZE_L1, REGULARIZE_NONE } RegularizeType;
			
			virtual	~StaticalModelParameter() {};
			
			/**
			*	パラメータのファイル出力
			*	@param	save_path	出力先のパス
			*	@return	エラーコード等
			*/
			virtual int		save(const std::string &save_path) const	=	0;

			/**
			*	パラメータのファイル入力
			*	@param	load_path	入力元のパス
			*	@return	エラーコード等
			*/
			virtual int		load(const std::string &load_path)	=	0;

		protected:
	};

	/** 識別器ベースクラス */
	class StaticalModel
	{
		public:
			virtual			~StaticalModel(){};
			
			/**
			*	パラメータのファイル出力
			*	@param	save_path	出力先のパス
			*	@return	エラーコード等
			*/
			virtual int		save(const std::string &save_path) const	=	0;
			
			/**
			*	パラメータのファイル入力
			*	@param	load_path	入力元のパス
			*	@return	エラーコード等
			*/
			virtual int		load(const std::string &load_path)	=	0;

			/**
			*	特徴ベクトルから予測結果を出す
			*	@param	feature	入力の特徴ベクトル
			*	@param	result	出力の予測結果
			*/
			virtual int		predict(const cv::Mat &feature, PredictionResult *result)	=	0;

			/**
			*	学習を実行する
			*	@param	train_set	学習データ
			*	@param	param		学習パラメータ
			*/
			virtual int		train(const TrainingSet &train_set, const StaticalModelParameter *param)	=	0;

			/**
			*	クラスの型情報を返す
			*	@return	統計モデルの識別子
			*/
			virtual StatModelType get_type() const { return StatModelType::STAT_MODEL; };

		protected:

		private:
	};

	//! 弱識別器パラメータクラス (不要感ある)
	class WeakLearnerParameter : public StaticalModelParameter
	{
	};

	//! 弱識別器ベースクラス
	class WeakLearner : public StaticalModel
	{
	};

	typedef std::shared_ptr<StaticalModel> PtrStaticalModel;
	typedef std::shared_ptr<WeakLearner> PtrWeakLearner;
	typedef std::shared_ptr<WeakLearnerParameter> PtrWeakLearnerParam;
};