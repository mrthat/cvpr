#pragma once
#include "MlpLayerBase.h"

namespace cvpr
{
	namespace mlp
	{
		/**
		*	畳み込み層のパラメータ
		*/
		class CnnLayerParameter : public virtual LayerParameter
		{
			public:

				cv::Size	kernel_size;	//< カーネルサイズ
				
				#pragma region override methods

				virtual int		save(const std::string &save_path) const;

				virtual int		load(const std::string &load_path);

				#pragma endregion

			protected:
		};
		
		/**
		*	活性化関数がシグモイドのレイヤー
		*	2クラス識別時の出力層に使用
		*/
		class SigmoidLayer : public OutputLayerBase
		{
			public:

				MlpOutputLayerType	type() const {
					return LAYER_TYPE_SIGMOID;
				};

				virtual void	calc_activation(const cv::Mat &a_j, cv::Mat &dst) const ;

				virtual void	calc_derivative(const cv::Mat &z_j, const cv::Mat &a_j, cv::Mat &dst) const ;
		};

		/**
		*	活性化関数が線形のレイヤー
		*	回帰時の出力層に使用
		*/
		class LinearLayer:  public OutputLayerBase
		{
			public:

				MlpOutputLayerType	type() const {
					return LAYER_TYPE_LINEAR;
				};

				virtual void	calc_activation(const cv::Mat &a_j, cv::Mat &dst) const ;

				virtual void	calc_derivative(const cv::Mat &z_j, const cv::Mat &a_j, cv::Mat &dst) const ;
		};

		/**
		*	活性化関数がソフトマックスのレイヤー
		*	多クラス識別の出力層に使用
		*/
		class SoftMaxLayer : public OutputLayerBase
		{
			public:

				MlpOutputLayerType	type() const {
					return LAYER_TYPE_SOFTMAX;
				};

				virtual void	calc_activation(const cv::Mat &a_j, cv::Mat &dst) const ;

				virtual void	calc_derivative(const cv::Mat &z_j, const cv::Mat &a_j, cv::Mat &dst) const ;
		};

		/**
		*	活性化関数がtanhのレイヤー
		*	隠れ層に使用
		*/
		class TanhLayer : public HiddenLayerBase
		{
			public:

				MlpHiddenLayerType	type() const {
					return LAYER_TYPE_TANH;
				};

				virtual void	calc_activation(const cv::Mat &a_j, cv::Mat &dst) const ;

				virtual void	calc_derivative(const cv::Mat &z_j, const cv::Mat &a_j, cv::Mat &dst) const ;
		};
			
		class ConvolutionLayer : public HiddenLayerBase
		{
			public:

				virtual MlpHiddenLayerType	type() const { return LAYER_TYPE_CONVOLUTION ; } ;

				virtual int		save(const std::string &fname) const;

				virtual int		load(const std::string &fname) ;

				virtual void	calc_a_j(const cv::Mat &feature, cv::Mat &dst) const;

				virtual void	calc_de_da(const cv::Mat &err, cv::Mat &dst) const ;

				virtual void	calc_activation(const cv::Mat &a_j, cv::Mat &dst) const ;

				virtual void	calc_derivative(const cv::Mat &z_j, const cv::Mat &a_j, cv::Mat &dst) const ;

				virtual int		init(const MatType &in_type, const LayerParameter &param, MatType &out_type, std::mt19937 &rng) ;

				virtual void	calc_param_delta(const cv::Mat &err, const cv::Mat &activation, cv::Mat &weight_delta, double &bias_delta) const;

				virtual void	update(const LayerParameter &param, const cv::Mat &weight_delta, double bias_delta) ;

			protected:

				/**
				*	カーネルサイズを取得
				*	@param	scale	出力のサイズにかける倍率
				*	@return	カーネルサイズ
				*/
				cv::Size	kernel_size(double scale = 1.0) const
				{
					cv::Size	sz;

					if (kernels_.empty()) {
						return sz;
					}

					sz	=	kernels_[0].size();
					sz.height	=	(int)(sz.height * scale);
					sz.width	=	(int)(sz.width * scale);

					return sz;
				}

				MatType	get_dst_type() const
				{
					int					dst_type	=	CV_64FC(in_type_.channels() * (int)kernels_.size());
					cv::Size			map_sz		=	in_type_.size() - (kernel_size() - cv::Size(1,1));
	
					return MatType(map_sz, dst_type);
				}

				std::vector<cv::Mat>	kernels_;	//< フィルタカーネル

				MatType		in_type_;	//< 入力の行列ヘッダー
		};

	};
};