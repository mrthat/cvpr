#pragma once
#include <random>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>
#include "..\..\base\header\TrainingData.h"
#include "..\..\base\header\PredictionResult.h"
namespace cvpr
{
	namespace TreeNode
	{
		/**
		*	ノードの種別
		*/
		typedef enum {
			NODE_TYPE_BASE,		//< ベースクラス
			NODE_TYPE_SPLIT,	//< 分割ノード
			NODE_TYPE_LEAF		//< 葉ノード
		} NodeType;

		/**
		*	分割ノードの詳細な種別
		*/
		typedef enum {
			SPLIT_TYPE_BASE,		//< ベースクラス
			SPLIT_TYPE_AXISALIGNED,	//< 軸並行カット
			SPLIT_TYPE_LINE,		//< 超平面カット
			SPLIT_TYPE_CONIC,		//< 円錐カット
			SPLIT_TYPE_ATAN,		//< 2点とってatan
			SPLIT_TYPE_MIN,			//< 2点とってmin
			SPLIT_TYPE_HAAR,		//< haarの出力でカット
			SPLIT_TYPE_HAAR_INT,	//< 上と同じだが，入力がintegral imageになってる場合
			SPLIT_TYPE_LAST			//< 末尾だよ～＾＾
		} SplitNodeType;

		/**
		*	葉ノードの詳細な種別
		*/
		typedef enum {
			LEAF_TYPE_BASE,				//< ベースクラス
			LEAF_TYPE_CLASSIFICATION,	//< クラス識別用
			//LEAF_TYPE_REGRESSION,		//< 回帰用
			//LEAF_TYPE_DENSITY,			//< 密度推定用
			//LEAF_TYPE_MANIFOLD			//< 多様体用
		} LeafNodeType;

		/**
		*	ノードのベースクラス
		*/
		class NodeBase
		{
			public:

				/**
				*	デフォルトコンストラクタ
				*/
				NodeBase()
						: left_node_id(0), right_node_id(0), node_id(0) {};

				virtual				~NodeBase() {};

				unsigned			left_node_id;	//< 左子ノードのid
				
				unsigned			right_node_id;	//< 右子ノードのid

				unsigned			node_id;		//< 自分のid

				bool	operator<(const NodeBase &rhs) const
				{
					return node_id < rhs.node_id;
				}

				/**
				*	ノードの種別を取得
				*	@return	ノードの種別
				*/
				virtual NodeType	get_node_type() const	=	0;

				/**
				*	パラメータファイル出力
				*	@param	cvfs	出力先ファイル
				*	@return	0:成功, -1:失敗
				*/
				virtual int	save(cv::FileStorage &cvfs) const ;

				/**
				*	パラメータファイル入力
				*	@param	cvfs	入力ファイル
				*	@return	0:成功, -1:失敗
				*/
				virtual int	load(cv::FileStorage &cvfs) ;

			protected:
		};

		/**
		*	分割ノードベースクラス
		*/
		class SplitNodeBase
			: public NodeBase
		{
			public:

				/**
				*	分割結果を表す定数
				*/
				enum {
					LEFT = -1,	//< 左側に行け^^
					RIGHT = 1	//< 右側に行け^^
				};

				/**
				*	分割に必要な点の数
				*/
				enum {
					NUM_CUTPOINT = 2
				};

				SplitNodeBase()
					: cut_points_(NUM_CUTPOINT, 0) {};

				virtual					~SplitNodeBase() {};
				
				NodeType				get_node_type() const { return NODE_TYPE_SPLIT; }
				
				/**
				*	分割ノードの詳細種別を取得
				*	@return	分割ノードの種別
				*/
				virtual SplitNodeType	get_split_type() const	=	0;

				/**
				*	分割処理
				*	@return	LEFT or RIGHT
				*/
				virtual int				operator()(const cv::Mat &feature) const ;

				/**
				*	データセットまとめて分割
				*	@param	train_set	分割したいセット
				*	@param	left_set	分割結果LEFTのサンプルを入れるset
				*	@param	right_set	分割結果RIGHTのサンプル入れるset
				*/
				virtual void			operator()(const TrainingSet &train_set, TrainingSet &left_set, TrainingSet &right_set) const ;

				/**
				*	パラメータ学習する
				*	@param	train_set	学習セット
				*	@param	rnd			乱数エンジン
				*	@param	left		学習後時にLEFTに分割されたサンプルを入れる(オプション)
				*	@param	right		学習後時にRIGHTに分割されたサンプルを入れる(オプション)
				*	@return	0:成功, -1:失敗
				*/
				virtual int				train(const TrainingSet &train_set, std::mt19937 &rnd, TrainingSet *left = nullptr, TrainingSet *right = nullptr) ;

				virtual int	save(cv::FileStorage &cvfs) const ;

				virtual int	load(cv::FileStorage &cvfs) ;

			protected:
				
				/**
				*	
				*/
				enum {
					IDX_UNDER_CUTPOINT = 0,
					IDX_UPPER_CUTPOINT = 1
				};

				/**
				*	カーネル計算に使うattributeの数を取得
				*	@return	attributeの数
				*/
				virtual unsigned		get_num_attributes() const	=	0;
				
				/**
				*	attributes_の初期化する
				*	@param	train_set	学習せっと
				*	@param	rnd			乱数エンジン
				*/
				void					init_attributes(const TrainingSet &train_set, std::mt19937 &rnd) ;

				/**
				*	パラメータ初期化する
				*	@param	train_set	学習セット
				*	@param	rnd			乱数エンジン
				*/
				virtual void			init_params(const TrainingSet &train_set, std::mt19937 &rnd)	=	0;

				/**
				*	カーネル関数計算する
				*	入力は，元サンプルからattributeの点だけサンプルした列ベクトルなことに注意
				*	@param	feature	特徴ベクトル
				*	@return	カーネル値
				*/
				virtual double			kernel_function(const cv::Mat &feature) const	=	0;

				/**
				*	カーネル関数の計算結果から左or右を判定して返す
				*	@param	kernel_value	カーネル値
				*	@return	LEFT or RIGHT
				*/
				virtual int		split(double kernel_value) const
				{
					if (this->cut_points_[IDX_UNDER_CUTPOINT] < kernel_value
						&& kernel_value < this->cut_points_[IDX_UPPER_CUTPOINT]) {
							return RIGHT;
					} else {
							return LEFT;
					}
				}

				std::vector<int>	attributes_;	//< 特徴ベクトル中のattributeとして利用する点のidx

				std::vector<double>	cut_points_;	//< 左 or 右を判定するための値の範囲を表す配列
		};

		/**
		*	葉ノードのベースクラス
		*/
		class LeafNodeBase
			: public NodeBase
		{
			public:

				virtual					~LeafNodeBase() {};
				
				NodeType				get_node_type() const { return NODE_TYPE_LEAF; };

				/**
				*	葉ノードの種別を取得
				*	@return	葉ノードの種別
				*/
				virtual LeafNodeType	leaf_type() const	=	0;

				/**
				*	葉ノードの情報を取得する
				*	@param	feature	特徴ベクトル
				*	@param	result	葉ノードの情報入れる箱
				*	@return	0:成功, -1:失敗
				*/
				virtual int				operator()(const cv::Mat &feature, PredictionResult *result)	=	0;

				/**
				*	葉ノードに情報貯める
				*	@param	train_set	学習セット
				*	@return	0;成功, -1:失敗
				*/
				virtual int				train(const TrainingSet &train_set)	=	0;
				
				using	NodeBase::save ;

				using	NodeBase::load ;

			protected:
		};
		
		typedef std::shared_ptr<NodeBase> PtrNodeBase ;
		typedef std::shared_ptr<SplitNodeBase> PtrSplitNodeBase ;
		typedef std::shared_ptr<LeafNodeBase> PtrLeafNodeBase ;
	};
};