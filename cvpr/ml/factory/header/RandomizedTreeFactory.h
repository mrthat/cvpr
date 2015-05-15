#include "StatModelFactory.h"
#include "..\..\rf\header\RandomizedTree.h"
#include "..\..\rf\header\ClassificationTree.h"
#include "..\..\rf\header\RegressionTree.h"

namespace cvpr
{
	/**
	*	randomized treeのファクトリークラス
	*/
	class RandomizedTreeFactory
	{
		public:
		virtual ~RandomizedTreeFactory() {};

		/**
		*	木を生成する
		*	@param	tree_type	木の種別
		*	@return	知らない種別:nullptr, その他:有効なぽいんた
		*/
		static PtrRandomizedTree	Create(TreeType tree_type);

		protected:
	};

	//! 弱識別器としてRandomizedTreeをつかってプールを生成するクラス
	class ClassificationTreePoolFactory : public WeakLearnerPoolFactoryBase
	{

		public:

		/**
		*	パラメータ設定
		*	@param	param	設定するパラメータ
		*/
		void set_param(const ClassificationTreeParameter &param_)
		{
			this->param	=	param_;
			rng	=	std::mt19937(param.rng_seed);
		}

		protected:

		ClassificationTreeParameter	param;

		std::mt19937 rng;

		virtual PtrWeakLearnerParam next_param();

		virtual PtrWeakLearner next_model();
	};

	//! GradientBoost向けステージ毎生成
	class StageWiseRegressionTreeFactory : public StageWiseStatModelFactoryBase
	{
		public:

			virtual PtrWeakLearner next(const TrainingSet &datas) ;

			/**
			*	新しい木の学習毎に使用されるパラメータを設定する
			*/
			void set_param(const RegressionTreeParameter &param_)
			{
				param	=	param_;
				rng		=	std::mt19937(param.rng_seed);
			}

		protected:

			//! 新しい木の生成毎に学習に使用されるパラメータ
			//! 乱数のシードは書き換えられる
			RegressionTreeParameter param;

			//! 乱数のシードを生成する乱数(ホントはこう使うのは良くないが)
			std::mt19937 rng;
	};
};