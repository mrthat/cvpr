#pragma once
#include "..\..\base\header\StaticalModel.h"

namespace cvpr
{
	class LinearModelParameterBase : public StaticalModelParameter
	{
		public:
			unsigned long	rnd_seed;			/*< 乱数のシード */
			std::size_t		max_iter;			/*< 最適化の最大繰り返し回数 */
			double			min_delta;			/*< 最適化の繰り返し間の最小評価値変化 */
			double			update_rate;		/*< パラメータ更新率 */
			double			lambda;				/*< 正則化項の係数 */
			RegularizeType	regularize_type;	/*< 正則化のタイプ */

			LinearModelParameterBase()
				: rnd_seed(19861124), max_iter(1000), min_delta(0.01),
				  update_rate(0.02), lambda(0.01), regularize_type(REGULARIZE_L2)
				{};

			int	save(const std::string &save_path) const;
			int	load(const std::string &load_path) ;
	};

	class LinearModelBase : public StaticalModel
	{

		public:

			virtual			~LinearModelBase() {};

			virtual int		save(const std::string &save_path) const ;

			virtual int		load(const std::string &load_path);

			virtual int		predict(const cv::Mat &feature, PredictionResult *result, const PredictionParameter *param = nullptr) = 0;

			virtual int		train(const TrainingSet &train_set, const StaticalModelParameter *param);

			/**
			*	線形回帰の学習メソッド
			*	@param	train_set	学習データ.特徴ベクトルは列ベクトル
			*	@param	param		全項目埋まっていること．デフォルト値の使用や値のチェックはしてない．
			*/
			int				train(const TrainingSet &train_set, const LinearModelParameterBase *param);

		protected:

			/**
			*	重みベクトル
			*/
			cv::Mat			weight_;

			/**
			*	バイアス
			*/
			double			w0_;

			/**
			*	パラメータ更新量を計算する(普通のgd)
			*	@param	train_set	更新量計算に使用する学習データ
			*	@param	delta_w		特徴ベクトルに対する重みベクトルの更新量
			*	@param	delta_w0	バイアスの更新量
			*/
			virtual void	calc_param_delta(const TrainingSet &train_set, cv::Mat &delta_w, double &delta_w0) = 0;
			
			/**
			*	学習の最初に重みをランダムに初期化する
			*	@param	train_set	学習データ
			*	@param	rnd			乱数エンジン
			*/
			virtual void			init_weight(const TrainingSet &train_set, std::mt19937 &rnd) ;
			
			/**
			*	L1正則化
			*	@param[in,out]	weight	重みベクトル.内容を正則化して書き換える
			*	@param[in,out]	w0		バイアス.(使わんけど一応引数にしとく)
			*	@param			labmda	どのくらい正則化するか(大きいとよりスパースになる)
			*/
			virtual void	regularize_l1(cv::Mat &weight/*I/O*/, double &w0/*I/O*/, double lambda) const;

			/**
			*	L2正則化
			*	@param[in,out]	weight	重みベクトル.内容を正則化して書き換える
			*	@param[in,out]	w0		バイアス.(使わんけど一応引数にしとく)
			*	@param			labmda	どのくらい正則化するか
			*/
			virtual void	regularize_l2(cv::Mat &weight/*I/O*/, double &w0/*I/O*/, double lambda) const;
		private:

	};

	class LinearRegressionParameter : public LinearModelParameterBase
	{
	};

	class LinearRegression : public LinearModelBase
	{
		public:
			
			virtual int		predict(const cv::Mat &feature, PredictionResult *result, const PredictionParameter *param = nullptr);

			/**
			*	回帰用のpredict
			*	@param	feature	列ベクトル
			*	@param	result	予測値
			*	@return	実行結果 0:成功, else: error
			*/
			virtual int		predict(const cv::Mat &feature, RegressionResult &result);

			/**
			*	線形回帰を学習する
			*	@param	train_set	学習データ．特徴ベクトルは列ベクトル．ラベルは列ベクトル
			*	@param	param		パラメータ．全項目適正値なこと．内容のチェックはしていない
			*/
			int	train(const TrainingSet &train_set, const LinearRegressionParameter &param)
			{
				return LinearModelBase::train(train_set, &param);
			}

		protected:
			
			/**
			*	パラメータ更新量を計算する(普通のgd)
			*	@param	train_set	更新量計算に使用する学習データ
			*	@param	delta_w		特徴ベクトルに対する重みベクトルの更新量
			*	@param	delta_w0	バイアスの更新量
			*/
			virtual void	calc_param_delta(const TrainingSet &train_set, cv::Mat &delta_w, double &delta_w0);
			
		private:
			/**
			*	重みとバイアスから応答値を求める
			*	@param	feature	列ベクトル
			*/
			cv::Mat	calc_activation(const cv::Mat &feature) const ;
	};
};