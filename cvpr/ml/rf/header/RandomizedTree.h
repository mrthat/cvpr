#pragma once
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <direct.h>
#include "..\..\base\header\StaticalModel.h"
#include "TreeNode.h"
#include "TreeNodeFactory.h"

namespace cvpr
{
	/**
	*	treeの種別
	*/
	typedef enum {
		TREE_TYPE_CLASSIFICATION,	//< クラス識別用tree
		TREE_TYPE_REGRESSION,		//< 回帰用木
		TREE_TYPE_UNKNOWN			//< 無効値
	} TreeType;

	/**
	*	randomized treeのパラメータクラス
	*/
	class RandomizedTreeParameter : public virtual WeakLearnerParameter
	{
		public:
			
			RandomizedTreeParameter()
				: max_height(10), min_samples(10), num_splits(1000), rng_seed(19861124), min_info_grain(0.001f)
			{};

			virtual			~RandomizedTreeParameter() {};
			
			unsigned		max_height;		//< 木の最大高さ 越える前に成長を打ち切る
			
			unsigned		min_samples;	//< 葉ノードの最小サンプル数 下回った時に成長を打ち切る

			unsigned		num_splits;		//< 最良分割を探索する際のプールの数

			unsigned long	rng_seed;		//< 乱数のシード

			float			min_info_grain;	//< 分割によって得られた情報利得の最小値 下回った場合成長を打ち切る

			/**
			*	学習時に生成する分割ノードの種類を追加する
			*	@param	type	追加する種類
			*	@return	成否
			*/
			virtual bool	add_split_type(const TreeNode::SplitNodeType &type);

			/**
			*	学習時に生成する分割ノードの種類を削除する
			*	@param	type	削除する種類
			*/
			void	remove_split_type(const TreeNode::SplitNodeType &type);

			/**
			*	分割ノードの種別リストを取得する．
			*/
			void	get_split_list(std::vector<TreeNode::SplitNodeType> &dst) const;

			/**
			*	デフォルトの分割ノードリストを設定する．
			*/
			virtual void	set_default_split_list();

		protected:

			/**
			*	分割ノードの種別のリスト
			*	リスト内の種別の分割ノードだけ学習に使う
			*/
			std::vector<TreeNode::SplitNodeType>	split_type_list;

			virtual int		save(const std::string &save_path) const { return 0; };

			virtual int		load(const std::string &load_path) { return 0; };
	};

	/**
	*	randomized treeの基本クラス
	*/
	class RandomizedTree : public WeakLearner
	{
		public:
			/**
			*	treeの種別を取得
			*	@return	treeの種別
			*/
			virtual TreeType	tree_type() const = 0;

			virtual		~RandomizedTree() { };

			virtual int	save(const std::string &save_path) const ;

			virtual int	load(const std::string &load_path) ;

			virtual int	predict(const cv::Mat &feature, PredictionResult *result, const PredictionParameter *param = nullptr);

			virtual int	train(const TrainingSet &train_set, const StaticalModelParameter *param) ;

			int			train(const TrainingSet &train_set, const RandomizedTreeParameter &param) ;

		protected:
			
			/**
			*	木内の各ノードの種別をファイル出力する
			*	@param	file_path	出力ファイルパス
			*	@return 0:成功, -1:失敗
			*/
			int	save_node_type(const std::string &file_path) const ;

			/**
			*	木内の各ノードの種別をファイルから入力する
			*	@param	file_path		入力ファイルパス
			*	@param	node_types		split or leafの種別
			*	@param	detail_types	split内やleaf内の詳細な種別
			*/
			int	load_node_type(const std::string &file_path, std::vector<int> &node_types, std::vector<int> &detail_types) ;

			/**
			*	leaf nodeの種別を取得する
			*	@return leaf nodeの種別
			*/
			virtual TreeNode::LeafNodeType	leaf_type() const	=	0;

			/**
			*	木の成長の終了条件を満たすか判断する
			*	@param	train_set	学習セット
			*	@param	param		パラメータ
			*	@param	tree_height	学習中のノードの高さ位置
			*	@return	true:終了条件を満たす, false:終了条件を満たさない
			*/
			virtual bool	is_end_growth(const TrainingSet &train_set, const RandomizedTreeParameter &param, unsigned tree_height) const = 0;
			
			/**
			*	木を成長させる = 注目位置に新しいノード追加してパラメータ最適化
			*	@param	train_set	学習セット(そのノードに来たサブセット)
			*	@param	param		学習パラメータ
			*	@param	tree_height	学習中のノードの高さ位置
			*	@param	node_id		新しく追加する
			*/
			virtual int		grow_tree(const TrainingSet &train_set, const RandomizedTreeParameter &param, unsigned tree_height, unsigned node_id, std::mt19937 &rng) ;

			/**
			*	学習経過を標準出力する
			*	@param	split		学習済みの分割ノード
			*	@param	train_set	学習セット
			*/
			virtual void	print_train_log(const TreeNode::PtrSplitNodeBase split, const TrainingSet &train_set) const = 0;

			/**
			*	学習経過を標準出力する
			*	@param	leaf		学習済みの葉ノード
			*	@param	train_set	学習セット
			*/
			virtual void	print_train_log(const TreeNode::PtrLeafNodeBase leaf, const TrainingSet &train_set) const = 0;

			/**
			*	分割前後の情報利得を計算する
			*	@param	train_set	分割前の学習セット
			*	@param	left_set	分割後の学習セット(左)
			*	@param	right_set	分割後の学習セット(右)
			*	@param	param		学習パラメータ
			*	@return	情報利得
			*/
			virtual double	calc_entropy_gain(const TrainingSet &train_set, const TrainingSet &left_set, const TrainingSet &right_set, const RandomizedTreeParameter &param) const = 0;

			std::vector<TreeNode::PtrNodeBase> nodes_;	//< 木内のノードの配列．学習後にidでソートして，id=配列のidxになるようにする

			int feature_depth_;	//< 入力特徴ベクトルの型(CV_64Fとか)
	};

	typedef std::shared_ptr<RandomizedTree>	PtrRandomizedTree ;
};