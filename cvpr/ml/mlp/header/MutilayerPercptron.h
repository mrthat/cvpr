#pragma once
#include "..\..\base\header\StaticalModel.h"
#include "MlpLayerBase.h"

namespace cvpr
{

	namespace mlp {

		class MlpParameterbase : public StaticalModelParameter
		{
			public:
				int		num_hidden_layers;	//< 隠れ層の数

				int		max_iter;			//< 最適化の最大繰り返し回数

				double	min_delta;			//< 最適化の最小更新幅 下回った場合終了する

				double	decay_rate;			//< 学習率の減衰率 η_new = η_old / (1 + decay_rate * iter)

				double	resample_rate;		//< mini batch のサンプル率

				MlpOutputLayerType	output_layer_type;	//< 出力層の活性化関数の種類

				unsigned long	rnd_seed;	//< 乱数のシード

				MlpParameterbase()
					: num_hidden_layers(1), max_iter(100), min_delta(0.01),
					output_layer_type(LAYER_TYPE_SIGMOID), rnd_seed(19861124),
					decay_rate(0.005), resample_rate(0.1) {};

				#pragma region override methods

				virtual int		save(const std::string &save_path) const;

				virtual int		load(const std::string &load_path);

				#pragma endregion

			protected:
		};

		/**
		*	MLP用パラメータ
		*/
		class MultilayerPerceptronParameter : public virtual LayerParameter, public MlpParameterbase
		{
			public:
				
				#pragma region override methods

				virtual int		save(const std::string &save_path) const;

				virtual int		load(const std::string &load_path);

				#pragma endregion

			protected:
		};

		/**
		*	多層パーセプトロン
		*	特徴ベクトル,教師データは列ベクトルを想定
		*/
		class MultilayerPerceptron : public StaticalModel
		{
			public:
			
				#pragma region	override methods

				virtual int		save(const std::string &save_path) const ;

				virtual int		load(const std::string &load_path) ;

				virtual int		predict(const cv::Mat &feature, PredictionResult *result, const PredictionParameter *param = nullptr);

				virtual int		train(const TrainingSet &train_set, const StaticalModelParameter *param) ;

				#pragma endregion

				/**
				*	MLP学習メソッド
				*	@param	train_set	学習セット
				*	@param	param		学習パラメータ
				*	@return	学習の成否. 0:成功, それ以外:エラー
				*/
				int	train(const TrainingSet &train_set, const MultilayerPerceptronParameter &param) ;
			
				/**
				*	予測メソッド
				*	@param	feature	特徴ベクトル
				*	@param	result	予測結果
				*	@return	0:成功, other:エラー
				*/
				int	predict(const cv::Mat &feature, ClassificationResult &result) ;
			
				/**
				*	予測メソッド
				*	@param	feature	特徴ベクトル
				*	@param	result	予測結果
				*	@return	0:成功, other:エラー
				*/
				int	predict(const cv::Mat &feature, RegressionResult &result) ;

			protected:

				/**
				*	隠れ層の種別を取得する
				*	@return	隠れ層の種別
				*/
				virtual MlpHiddenLayerType	get_hidden_layer_type() { return LAYER_TYPE_TANH; };
			
				/**
				*	パラメータの値が有効化検査する
				*	@param	param	検査対象のパラメータ
				*	@return	true:有効, false:無効値あり
				*/
				static bool	is_valid_parameter(const MultilayerPerceptronParameter &param) ;
	
				/**
				*	学習データが有効か検査する．
				*	(= 列ベクトルかどうか)
				*	@param	train_set	検査対象の学習データ
				*	@return	true:有効, false:無効
				*/
				static bool	is_valid_train_set(const TrainingSet &train_set) ;

				/**
				*	学習の最初に各層を初期化する
				*	事前に学習データとパラメータは検査済み想定
				*	@param	train_set	学習データ <= is_valid~で検査済み
				*	@param	param		パラメータ <= is_valid~で検査済み
				*	@param	rng			乱数エンジン
				*/
				virtual void	init_layers(const TrainingSet &train_set, const MultilayerPerceptronParameter &param, std::mt19937 &rng) ;

				/**
				*	層毎のactivationを計算する
				*	@param	feature		特徴ベクトル
				*	@param	activations	activationの計算結果．layers_.size()+1だけある．0はfeature．idxが若いほうが入力側.
				*/
				void	calc_all_a_z(const cv::Mat &feature, std::vector<cv::Mat> &a_j, std::vector<cv::Mat> &z_j) const ;

				/**
				*	誤差を逆伝搬させる
				*	@param	layer_k	k層目のレイヤー
				*	@param	layer_j	j層目(=k-1)のレイヤー
				*	@param	err_k	k層目の誤差
				*	@param	z_j		j層目のactivation
				*	@param	a_j		j層目の重みと入力の行列積
				*	@param	err_j	逆伝搬させた誤差の行列
				*/
				void	backprop_error(const PtrLayerBase layer_k, const PtrLayerBase layer_j, const cv::Mat &err_k, const cv::Mat &z_j, const cv::Mat &a_j, cv::Mat &err_j) ;

				/**
				*	層の配列取得
				*	outとhiddenで分けたので一緒くたの配列がほしいときよう
				*	@return	層の配列
				*/
				std::vector<mlp::PtrLayerBase>	layers() const
				{
					std::vector<mlp::PtrLayerBase>	dst;
				
					dst.insert(dst.end(), hidden_layers_.begin(), hidden_layers_.end());
					dst.push_back(out_layer_);

					return dst;
				}

				static const std::string	FNAME_MLP_CFG;			//< MLPの構成情報のファイル名
				static const std::string	CFG_TAG_LAYER_TYPES;	//< MLPの各層の種別情報のタグ名 
				static const std::string	FNAME_LEYER_CFG;

				std::vector<mlp::PtrHiddenLayerBase>	hidden_layers_;		//< 隠れ層

				PtrOutputLayerBase	out_layer_;						//< 出力層
		};

	};
};