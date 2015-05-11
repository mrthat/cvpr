#pragma once

#include "..\..\base\header\StaticalModel.h"

namespace cvpr
{
	//! 弱識別器ファクトリー なんでも作れるすごいやつ
	class WeakLearnerFactory
	{
		public:

			/**
			*	生成メソッド
			*	@param	type	生成するモデルの型
			*	@return	新しいモデル
			*/
			static PtrWeakLearner create(const StatModelType &type);

		protected:
	};

	//! 弱識別器の学習済みプールを作成するクラス
	class WeakLearnerPoolFactoryBase
	{
		public:
			/**
			*	学習済みの弱識別器プールを生成する
			*/
			virtual std::vector<PtrWeakLearner> create_trained_pool(const TrainingSet &datas, std::size_t pool_size) ;

		protected:
			
			/**
			*	次のモデル用パラメータを取得する
			*	@return	次のモデル用パラメータ
			*/
			virtual PtrWeakLearnerParam next_param() = 0;

			/**
			*	次のモデルを生成する
			*	@return	次にプールに加えるモデル
			*/
			virtual PtrWeakLearner next_model() = 0;
	};
};