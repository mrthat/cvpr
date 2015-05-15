#pragma		once
#define		_USE_MATH_DEFINES
#include	<math.h>
#include	<random>
#include	<numeric>
#include	<array>
#include	<memory>
#include	<opencv2\core\core.hpp>
#include	"PredictionParameter.h"
#include	"utils.h"

namespace cvpr
{
	/**
	*	訓練サンプル
	*	ラベルと特徴ベクトルのペア
	*/
	class TrainingExample
	{
		public:
			/**
			*	デフォルトコンストラクタ
			*/
			TrainingExample() {};

			/**
			*	要素指定初期化
			*	@param	feature	特徴ベクトル
			*	@param	label	ラベル
			*/
			TrainingExample(cv::Mat feature, cv::Mat target, PtrPredictionParameter param_ = nullptr)
				: feature(feature), target(target), param(param_) {};

			/**
			*	デストラクタ
			*/
			virtual	~TrainingExample() {};

			/**
			*	特徴ベクトル
			*/
			cv::Mat	feature;

			/**
			*	ラベル
			*/
			cv::Mat	target;

			/**
			*	サンプルに紐づく予測用パラメータがアレば入れる(nullもOKとする)
			*/
			PtrPredictionParameter	param;

			/**
			*	label内の最大要素のidxを取得
			*	@return	label内の最大要素のidx
			*/
			int	label() const
			{
				return max_idx(target);
			}

		protected:
	};

	typedef std::shared_ptr<TrainingExample> PtrTrainingExample;

	/**
	*	行列のメタデータクラス
	*/
	class MatType
	{
		public:
			/** 行列の各次元の要素数 */
			std::vector<int>	sizes;
			/** 行列のデータのタイプ(open cv形式) */
			int					data_type;

			/**
			*	デフォルトコンストラクタ
			*/
			MatType()
				: data_type(0) {};

			/**
			*	2dのサイズとデータ型指定して構築
			*	@param	size		2dの行列のサイズ
			*	@param	data_type	データのタイプ
			*/
			MatType(const cv::Size &size, int data_type)
				: data_type(data_type)
			{
				sizes.assign(2, 0);
				sizes[0]	=	size.height;
				sizes[1]	=	size.width;
			}
			
			/**
			*	2dのサイズとデータ型指定して構築
			*	@param	rows		行数
			*	@param	cols		列数
			*	@param	data_type	データのタイプ
			*/
			MatType(int rows, int cols, int data_type)
				: data_type(data_type)
			{
				sizes.assign(2, 0);
				sizes[0]	=	rows;
				sizes[1]	=	cols;
			}

			/**
			*	mdのサイズとデータ型指定して構築
			*	@param	sizes		mdのサイズ
			*	@param	data_type	データのタイプ
			*/
			MatType(const std::vector<int> &sizes, int data_type)
				: sizes(sizes), data_type(data_type) {};

			/**
			*	行列のヘッダーで初期化
			*	@param	mat	サイズ等取得元の行列
			*/
			MatType(const cv::Mat &mat)
				: data_type(mat.type())
			{
				sizes.assign(mat.dims, 0);
				for (int ii = 0; ii < mat.dims; ++ii) {
					sizes[ii]	=	mat.size[ii];
				}
			}

			/**
			*	行列タイプ同士の一致比較
			*	@param	type	比較対象の行列タイプ
			*	@return	true:一致, false:異なる
			*/
			bool	equals(const MatType &type) const
			{
				if (this->data_type != type.data_type) {
					return false;
				}

				if (this->sizes.size() != type.sizes.size()) {
					return false;
				}

				for (std::size_t ii = 0; ii < sizes.size(); ++ii) {
					if (this->sizes[ii] != type.sizes[ii]) {
						return false;
					}
				}

				return true;
			}

			/**
			*	行列タイプ同士の一致比較
			*	@param	mat	対象の行列
			*	@return	true:一致, false:異なる
			*/
			bool	equals(const cv::Mat &mat) const
			{
				MatType	lhs(mat);

				return equals(lhs);
			}

			/**
			*	行列タイプで行列作った時の総要素数を返す
			*	@return 総要素数
			*/
			std::size_t	total() const
			{
				std::size_t	num	=	1;
				for (std::size_t ii = 0; ii < sizes.size(); ++ii) {
					num	*=	sizes[ii];
				}

				num	*=	CV_MAT_CN(data_type);

				return num;
			}

			/**
			*	0,1次元目を使用してcv::Sizeを作成して返す
			*	@return	height=0次元目,width=1次元目の要素数のcv::Size
			*/
			cv::Size	size() const
			{
				return cv::Size(sizes[1], sizes[0]);
			}

			/**
			*	シングルチャンネルかどうかを返す
			*	@return	true:シングルチャンネル, false:シングルじゃない
			*/
			bool	is_single_channel() const
			{
				return CV_MAT_CN(data_type) == 1;
			}

			/**
			*	列ベクトルの条件を満たすサイズになってるかを返す
			*	@return	true:列ベクトル, false:列ベクトルじゃない
			*/
			bool	is_col_vector() const
			{
				if (sizes.size() != 2) {
					return false;
				}

				if (sizes[1] != 1) {
					return false;
				}

				if (sizes[0] <= 0) {
					return false;
				}

				return true;
			}

			/**
			*	チャンネル数取得
			*	@return	チャンネル数
			*/
			int	channels() const
			{
				return CV_MAT_CN(data_type);
			}

			/**
			*	行列の(配列的な)次元数を取得
			*	@return	次元数
			*/
			size_t	dims() const
			{
				return sizes.size();
			}

			/**
			*	データのビット深度コード取得 (CV_64Fとかのアレ)
			*	@return	ビット深度コード
			*/
			int	depth() const
			{
				return CV_MAT_DEPTH(data_type);
			}

		protected:
	};

	/**
	*	学習データセットクラス
	*/
	class TrainingSet
	{
		public:

			/**
			*	コンストラクタ
			*	@param	feature_type	特徴ベクトルの行列タイプ
			*	@param	label_type		教師データの行列タイプ
			*/
			TrainingSet(const MatType &feature_type, const MatType &label_type)
				: feature_type_(feature_type), label_type_(label_type) {};

			TrainingSet(const TrainingSet &ts)
				: feature_type_(ts.get_feature_type()), label_type_(ts.get_label_type()),
				examples_(ts.examples_)
			{};

			TrainingSet() {};

			virtual	~TrainingSet() {};

			/**
			*	格納されている訓練サンプルを取得する
			*	@param	idx	サンプルのインデックス
			*/
			const PtrTrainingExample	operator[](std::size_t idx) const 
			{
				return examples_[idx]; 
			}

			/**
			*	訓練サンプルを追加する
			*	@param	example	追加する訓練サンプル
			*	@return	true:成功, false:失敗
			*/
			bool	push_back(const PtrTrainingExample &example);

			/**
			*	サンプル総数取得
			*	@return	サンプル総数
			*/
			std::size_t	size() const { return examples_.size(); };
			
			/**
			*	保持する全サンプルの削除
			*/
			void	clear() { examples_.clear(); };

			/**
			*	サンプルの削除
			*	@param	idx	削除するサンプルのインデックス
			*/
			void	erase(int idx)
			{
				examples_.erase(examples_.begin() + idx);
			}

			/**
			*	ラベルのエントロピー計算する
			*	@return	ラベルエントロピー  計算できなかったりした場合0
			*/
			double	compute_label_entropy() const ;

			/**
			*	全サンプルでラベルの和を取る
			*	@return	和の計算結果
			*/
			cv::Mat	calc_label_sum()const;
		
			/**
			*	特徴ベクトルのタイプを取得
			*/
			MatType	get_feature_type() const
			{
				return feature_type_;
			}

			/**
			*	ラベルのタイプを取得
			*/
			MatType	get_label_type() const
			{
				return label_type_;
			}

			/**
			*	特徴ベクトル及び教師データをそれぞれL2正規化した新しいデータセットを返す
			*	新しく領域確保するのでメモリ注意
			*	@return	L2正規化したデータセット
			*/
			TrainingSet	calc_l2_normalized_set() const ;

			/**
			*	特徴ベクトルの各次元について，最小値を求める
			*	サンプルはすべてdoubleにコンバートして算出する．
			*	= 出力もdouble行列
			*	@param	min_val	最小値算出結果
			*/
			void	find_feature_min(cv::Mat &min_val) const ;

			/**
			*	特徴ベクトルの各次元について，最大値を求める
			*	サンプルはすべてdoubleにコンバートして算出する．
			*	= 出力もdouble行列
			*	@param	max_val	最大値算出結果
			*/
			void	find_feature_max(cv::Mat &max_val) const ;
			
			/**
			*	行列配列から学習データセットを作る．
			*	@param	features	特徴ベクトル
			*	@param	labels		教師データ
			*	@param	copy_data	行列のコピー生成フラグ
			*	@param	vectorize	feature,labelを列ベクトルにする処理の有無フラグ
			*	@return	生成したデータセット
			*/
			static TrainingSet	mat_arr_to_train_set(const std::vector<cv::Mat> &features, const std::vector<cv::Mat> &labels, bool copy_data = false, bool vectorize = false) ;

			/**
			*	データセット内のサンプルから，指定サンプル率でサンプリングして
			*	新しいデータセット作って返す
			*	@param	sample_rate	サンプル率
			*	@param	rng			乱数エンジン
			*	@return	サンプリングした新しいデータセット
			*/
			TrainingSet	random_sample(double sample_rate, std::mt19937 &rng) const ;

			/**
			*	保持するデータの中で，bagに含まれないものを返す
			*	(=ポインタ一致しないもの)
			*	@param	bag	入力セット
			*	@return	not(this ∧ bag)
			*/
			TrainingSet	get_out_of_bag(const TrainingSet &bag) const ;

			/**
			*	教師の行列の平均を求める
			*	@param	dst	出力の行列
			*/
			void compute_target_mean(cv::Mat &dst) const;

			/**
			*	教師の行列の分散を求める
			*	@return	分散
			*/
			double compute_target_var() const;

		protected:

			/** 特徴ベクトルのタイプ */
			MatType		feature_type_;

			/** 教師データのタイプ */
			MatType		label_type_;

			/** 訓練サンプルのコンテナ */
			std::vector<const PtrTrainingExample>	examples_;

			/**
			*	有効な訓練サンプルかどうか判定する
			*	@param	example	判定対象の訓練サンプル
			*	@return	true:有効, false: 無効
			*/
			bool	is_valid_example(const PtrTrainingExample &example) const;

			/**
			*	教師の二乗の平均
			*	@param	dst	出力
			*/
			void compute_target_mean2(cv::Mat &dst) const;
	};

};
