#include "StatModelFactory.h"
#include "..\..\rf\header\RandomizedTree.h"
#include "..\..\rf\header\ClassificationTree.h"

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
};