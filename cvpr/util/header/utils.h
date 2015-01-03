#pragma once
#include <vector>
#include <random>
#include <opencv2\core\core.hpp>

#define	STRINGIZE(var) (#var)

namespace cvpr
{
	/**
	*	元配列からランダムサンプルしてbagとout of bagに入れる
	*	入れる数は適当で乱数の具合によってはかたよる
	*	@param	tp				配列の要素の型
	*	@param	src				入力配列
	*	@param	bag				出力のbag
	*	@param	oob				出力のout of bag
	*	@param	sample_rate		src内のbagに入れる確立 <=> [0..1.0]
	*	@param	rng				乱数生成器
	*/
	template<class tp> void	random_sample(const std::vector<tp> &src,
		std::vector<tp> &bag, std::vector<tp> &oob, double sample_rate, std::mt19937 &rng);

	/**
	*	入力配列から一様分布にしたがって1つをサンプルして返す
	*	@param	tp	配列要素の型
	*	@param	src	入力配列
	*	@param	rng	乱数エンジン
	*	@return	サンプルした結果
	*/
	template<class tp> tp	random_sample(const std::vector<tp> &src, std::mt19937 &rng)
	{
		std::uniform_real_distribution<std::size_t>	real_dist(0, src.size());

		return src[real_dist(rng)];
	};

	/**
	*	ディガンマ関数
	*	@param	hk	関数値を計算したい正の整数
	*	@return	0<hk:hkの時の関数値,else:qnan
	*/
	double	digamma(int hk);

	/**
	*	半整数の時のディガンマ関数
	*	@param	hk	関数値を計算したい正の整数
	*	@return	0<hk:hkの時の関数値,else:qnan
	*/
	double	digamma_half(int hk);

	/**
	*	文字列を区切り文字を指定して分割する
	*	@param	src		分割する文字列
	*	@param	delim	区切り文字
	*	@param	分割後の文字列配列
	*/
	void	split(const std::string &src, char delim, std::vector<std::string> &dst) ;

	/**
	*	行列内の最大要素のインデックスを返す
	*	@param	mat	要素の算出元行列
	*	@return	最大要素のインデックス
	*/
	int		max_idx(const cv::Mat &mat);

	/**
	*	m + log(sum(exp(mat[i]-m)))を計算する
	*	m = max(mat[i])
	*	matは2D,double,1channel
	*	@param	mat	2d,double,1channelの行列
	*	@return	m + log(sum(exp(mat[i]-m)))
	*/
	double	log_sum_exp(const cv::Mat &mat);

	/**
	*	入力行列を1cn,1列にした行列のヘッダーを取得
	*	@param	mat	入力行列
	*	@return	matを1ch1colにしたヘッダー
	*/
	cv::Mat	get_colvec_header(const cv::Mat &mat) ;

	/**
	*	入力行列が列ベクトルか調べる
	*	@param	mat	入力行列
	*	@return	true:列ベクトル, false:それ以外
	*/
	bool	is_column_vector(const cv::Mat &mat);

	/**
	*	一様分布 [0..1)でmatの要素を埋める
	*	@param	ty		出力行列の要素の型(=基本型)
	*	@param	rng		乱数エンジン
	*	@param	mat		要素うめ先の行列
	*/
	template<typename ty>
	void	rand_init(std::mt19937 &rng, cv::Mat &mat) 
	{
		cv::Mat	tmp	=	mat.reshape(1, get_total1(mat));
		std::uniform_real_distribution<>	dst;

		for (int ii = 0; ii <tmp.rows; ++ii) {
			tmp.at<ty>(ii)	=	dst(rng);
		}
	};

	/**
	*	vectorが{}で初期化できないのでそれっぽいことをするもの
	*	@param	v0	0番目の要素
	*	@param	v1	1番目の要素
	*	@param	v2	2番目の要素
	*	@return	{v0,v1,v2}のvector
	*/
	template<typename ty>
	std::vector<ty>	get_vector(ty v0, ty v1, ty v2)
	{
		std::array<ty, 3>	arr = {v0, v1, v2};

		return std::vector<ty>(arr.begin(), arr.end());
	}

	/**
	*	matの配列から，(配列的に)1次元多い行列を作る．
	*	メモリは新しく確保される
	*	配列がからなら空行列が変えるが，配列内の行列形式が違うとかはopencvの例外でる
	*	@param	arr	作成元の行列の配列
	*	@param	dst	出力行列
	*	@return	true:成功, false失敗
	*/
	bool	mat_arr_to_hdim_mat(const std::vector<cv::Mat> &arr, cv::Mat &dst) ;

	/**
	*	(配列的に)一番外側の要素数が1の場合に
	*	1次元少ない行列を返す
	*	入力が2dの場合はコピーを返す
	*	@param	mat			入力行列
	*	@param	dst			出力行列
	*	@param	copy_data	新しくメモリ確保するかどうかのフラグ
	*	@return	0: 成功, 1:入力が2Dなのでコピーした, それ以外:エラー
	*/
	int	reduce_mat_dim(const cv::Mat &mat, cv::Mat &dst, bool copy_data = false) ;

	/**
	*	行列の(配列的な)次元を1つ減らしてvectorに詰める
	*	@param	mat			入力行列
	*	@param	dst			出力配列
	*	@param	copy_data	新しくメモリ確保するかどうかのフラグ
	*	@return	0: 成功, 1:入力が2Dなのでコピーした, それ以外:エラー
	*/
	int	reduce_mat_dim(const cv::Mat &mat, std::vector<cv::Mat> &dst, bool copy_data = false) ;

	/**
	*	total * channelsを返す
	*	⇔チャンネル数考慮した要素数
	*	@param	mat	入力行列
	*	@return	要素数
	*/
	int	get_total1(const cv::Mat &mat) ;

	/**
	*	座標がmat中にあるか調べる
	*	@param	mat	検査対象の行列
	*	@param	pt	mat中の点座標
	*/
	bool contains(const cv::Mat &mat, const cv::Point pt, int margin = 0) ;
};