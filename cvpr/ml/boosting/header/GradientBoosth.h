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
		protected:

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