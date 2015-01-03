#pragma once
#include "..\..\base\header\StaticalModel.h"

namespace cvpr
{
	/**
	*	出力層の活性化関数のタイプ
	*/
	enum MlpOutputLayerType { 
		LAYER_TYPE_LINEAR,		//<	線形 回帰時に使う
		LAYER_TYPE_SIGMOID,		//< シグモイド 2クラス識別に使う
		LAYER_TYPE_SOFTMAX,		//< ソフトマックス 多クラス識別に使う
	};
	
	/**
	*	隠れ層の種別
	*/
	enum MlpHiddenLayerType
	{
		LAYER_TYPE_TANH,		//< tanh
		LAYER_TYPE_CONVOLUTION,	//< 畳み込み
	};

	namespace mlp
	{

		class LayerParameter : public cvpr::StaticalModelParameter
		{
			public:
				
			int		num_hidden_units;	//< 隠れ層内のユニット数
			
			double	update_rate;		//< 学習率
			
			double	lambda;				//< 正則化項の倍率
			
			RegularizeType	regularize_type;	//< 正則化の種類
			
			LayerParameter()
				:num_hidden_units(10), update_rate(0.01),
				lambda(0.001), regularize_type(REGULARIZE_L2) {};

			#pragma region override methods

			virtual int		save(const std::string &save_path) const;

			virtual int		load(const std::string &load_path);

			#pragma endregion

			protected:
				
		};
		
		/**
		*	MLP内のレイヤーのベースクラス
		*/
		class LayerBase
		{
			public:

				/**
				*	パラメータの内容をファイルに保存する
				*	@param	fname	出力先ファイルパス
				*	@return	0:成功, -1:失敗
				*/
				virtual int	save(const std::string &fname) const;

				/**
				*	ファイルからパラメータの内容を読み込む
				*	@param	fname	入力ファイルぱす
				*	@return	0:成功, -1:失敗
				*/
				virtual int	load(const std::string &fname) ;

				/**
				*	a_j=w_ji.z_iを計算する
				*	@param	feature	入力ベクトル
				*	@param	dst		計算結果(a_j)
				*/
				virtual void	calc_a_j(const cv::Mat &feature, cv::Mat &dst) const
				{
					cv::Mat	tmp	=	get_colvec_header(feature);

					cv::gemm(weight_, tmp, 1.0, cv::Mat(), 0, dst);

					dst	+=	bias_;
				}

				/**
				*	sum_k (dE_n / da_k) を計算する
				*	@param	err	delta_k
				*	@param	dst	出力配列
				*/
				virtual void	calc_de_da(const cv::Mat &err, cv::Mat &dst) const 
				{
					dst	=	weight_.t() * err;
				};
					
				/**
				*	活性化関数
				*	
				*	@param	a_j		w_ji * ziしたもの
				*	@param	dst		活性化関数の計算結果
				*/
				virtual void	calc_activation(const cv::Mat &a_j, cv::Mat &dst) const = 0;

				/**
				*	入力を順方向に伝搬させる
				*	@param	feature	入力ベクトル
				*	@param	dst		順伝搬後のベクトル
				*/
				virtual void	foward_prop(const cv::Mat &feature, cv::Mat &dst) const ;

				/**
				*	活性化関数の導関数
				*	@param	z_j	活性化関数の結果
				*	@param	a_j	w_ji*z_iの結果
				*	@param	dst	導関数の計算結果
				*/
				virtual void	calc_derivative(const cv::Mat &z_j, const cv::Mat &a_j, cv::Mat &dst) const = 0;

				/**
				*	エラーからパラメータ更新分を求める
				*	@param	err	エラー
				*	@param	weight_delta	重みの更新分
				*	@param	bias_delta		バイアス更新分
				*/
				virtual void	calc_param_delta(const cv::Mat &err, const cv::Mat &activation, cv::Mat &weight_delta, double &bias_delta) const;

				/**
				*	パラメータ更新分を適用する
				*	@param	weight_delta	重みの更新分
				*	@param	bias_delta		バイアス更新分
				*/
				virtual void	update(const LayerParameter &param, const cv::Mat &weight_delta, double bias_delta) ;
					
			protected:
					
				/**
				*	l2正則化をする
				*	@param	param	パラメータ
				*/
				void	regularize_l2(const LayerParameter &param) ;

				/**
				*	l1正則化をする
				*	@param	param	パラメータ
				*/
				void	regularize_l1(const LayerParameter &param) ;

				cv::Mat	weight_;	//<	重み行列

				double	bias_;		//< バイアス
		};

		/**
		*	隠れ層のベースクラス
		*/
		class HiddenLayerBase : public LayerBase
		{
			public:
				/**
				*	層の種別を取得する
				*	@return	層の種別
				*/
				virtual MlpHiddenLayerType	type() const = 0;

				/**
				*	層の初期化
				*	@param	in_type		入力行列のヘッダー
				*	@param	param		パラメータ
				*	@param	out_type	出力行列のヘッダー
				*	@param	rng			乱数エンジン
				*	@return	0:成功, それ以外:エラー
				*/
				virtual int	init(const MatType &in_type, const LayerParameter &param, MatType &out_type, std::mt19937 &rng) ;
		};

		/**
		*	出力層のベースクラス
		*/
		class OutputLayerBase : public LayerBase
		{
			public:
				/**
				*	層の種別を取得する
				*	@return	層の種別
				*/
				virtual	MlpOutputLayerType	type() const = 0;

				/**
				*	初期化
				*	@param	train_set	学習データ
				*	@param	in_type		入力行列のヘッダー
				*	@param	param		パラメータ
				*	@param	rng			乱数エンジン
				*	@return	0:成功, それ以外:エラー
				*/
				virtual int	init(const TrainingSet &train_set, const MatType &in_type, const LayerParameter &param, std::mt19937 &rng) ;
		};
			
		typedef std::shared_ptr<LayerBase>			PtrLayerBase;
		typedef std::shared_ptr<HiddenLayerBase>	PtrHiddenLayerBase;
		typedef std::shared_ptr<OutputLayerBase>	PtrOutputLayerBase;
	};
}