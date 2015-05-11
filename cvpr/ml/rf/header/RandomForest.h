#pragma once
//#include <tbb/blocked_range.h>
#include "..\..\base\header\TrainingData.h"
#include "..\..\base\header\StaticalModel.h"
#include "..\header\RandomizedTree.h"
#include <direct.h>
#include <fstream>
#include <sstream>
//#include <tbb\tbb.h>
//#include "ParallelTrain.h"

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

//#define PARALLEL_TRAIN

namespace cvpr
{
	/**
	*	Random Forestの学習パラメータクラス
	*/
	class RandomForestParameter : public cvpr::RandomizedTreeParameter
	{
		public:
			RandomForestParameter()
				: num_trees(5), resample_rate(0.4)
			{};

			virtual		~RandomForestParameter() {};

			unsigned	num_trees;			//< treeの数
			
			double		resample_rate;		//< 各tree学習時にデータセットからサンプルする割合

		protected:
	};

	/**
	*	random forest系のベースクラス
	*/
	class RandomForest : public WeakLearner
	{
		public:
			
			RandomForest() {};
			
			virtual			~RandomForest() {};

			/**
			*	学習実行メソッド
			*	@param	train_set	学習データ
			*	@param	param		学習パラメータ
			*	@return	0:成功, -1:失敗
			*/
			virtual int		train(const cvpr::TrainingSet &train_set, const cvpr::RandomForestParameter &param) ;
			
			/**
			*	学習済みのtreeをforestに追加する.
			*	@param	tree	追加するtree
			*	@return	0:成功, -1:失敗
			*/
			virtual int		add_tree(PtrRandomizedTree tree) ;

			/**
			*	学習済みのforest(内のtree)を自分に追加する
			*	@param	forest	追加するforest
			*	@return	0:成功, -1:失敗
			*/
			virtual int		merge(const RandomForest *forest) ;

			/**
			*	結果の種別を取得
			*	@return	結果の種別
			*/
			virtual ResultType	result_type() const = 0 ;

#pragma region override methods

			virtual int		save(const std::string &save_path) const ;

			virtual int		load(const std::string &load_path) ;

			virtual int		predict(const cv::Mat &feature, cvpr::PredictionResult *result, const PredictionParameter *param = nullptr);

			virtual int		train(const cvpr::TrainingSet &train_set, const cvpr::StaticalModelParameter *param) ;

#pragma endregion

		protected:
			
			/**
			*	現在保持するtreeの種別をファイルに保存する
			*	@param	data_path	出力先ファイルパス
			*	@return	0:成功, -1:失敗
			*/
			int	save_tree_type(const std::string &data_path) const ;

			/**
			*	(save_tree_typeで)ファイルに保存したtree種別を読み込む
			*	@param	data_path,	入力ファイルパス
			*	@param	tree_types	読み込んだtree種別
			*	@return	0:成功, -1:失敗
			*/
			int	load_tree_type(const std::string &data_path, std::vector<cvpr::TreeType> &tree_types) const ;

			/**
			*	学習するtreeの種別を取得する
			*/
			virtual	cvpr::TreeType	tree_type() const = 0;

			/**
			*	各treeの予測結果をマージする
			*	@param	results	各treeの予測結果
			*	@param	dst		マージ後の結果
			*/
			virtual void	merge_results(const std::vector<PtrPredictionResult> &results, PredictionResult *dst) = 0;
			
			std::vector<PtrRandomizedTree>	trees_;	//<要素のtreeを格納するコンテナ
	};
	
	/**
	*	クラス識別用のRandomForestクラス
	*/
	class ClassificationForest : public RandomForest
	{
		public:

			//using	RandomForest::predict;

			virtual ResultType	result_type() const{ return RESULT_TYPE_CLASSIFICATION; };

		protected:
			
			virtual	cvpr::TreeType	tree_type() const { return TREE_TYPE_CLASSIFICATION; };

			virtual void	merge_results(const std::vector<PtrPredictionResult> &results, PredictionResult *dst) ;
	};
};
