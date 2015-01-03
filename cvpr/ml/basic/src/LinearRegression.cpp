#include "..\header\LinearRegression.h"

using namespace cvpr;

/*
ファイル入出力用のパラメーデフォルト値
*/
static const unsigned long	DEFAULT_SEED		=	19861124;
static const std::size_t	DEFAULT_ITER		=	100;
static const double			DEFAULT_MIN_DELTA	=	0.01;
static const double			DEFAULT_UPDATE_RATE	=	0.01;
static const double			DEFAULT_LAMBDA		=	0.01;
static const StaticalModelParameter::RegularizeType	DEFAULT_REGTYPE	=	StaticalModelParameter::REGULARIZE_NONE;

int	
LinearModelParameterBase::save(const std::string &save_path) const
{
	cv::FileStorage	fs(save_path, cv::FileStorage::WRITE);

	if (!fs.isOpened()) {
		return -1;
	}

	cv::write(fs, "rnd_seed", (int)rnd_seed);
	cv::write(fs, "max_iter", (int)max_iter);
	cv::write(fs, "min_delta", min_delta);
	cv::write(fs, "update_rate", update_rate);
	cv::write(fs, "lambda", lambda);
	cv::write(fs, "regularize_type", (int)regularize_type);

	return 0;
}

int	
LinearModelParameterBase::load(const std::string &load_path) 
{
	cv::FileStorage	fs(load_path, cv::FileStorage::READ);

	if (!fs.isOpened()) {
		return -1;
	}

	int	tmp	=	0;

	tmp	=	fs["rnd_seed"];
	rnd_seed	=	tmp <= 0 ? DEFAULT_SEED : tmp;
	
	tmp	=	fs["max_iter"];
	max_iter	=	tmp <= 0 ? DEFAULT_ITER : tmp;

	cv::read(fs["min_delta"], min_delta, DEFAULT_MIN_DELTA);

	cv::read(fs["update_rate"], update_rate, DEFAULT_UPDATE_RATE);

	cv::read(fs["lambda"], lambda, DEFAULT_LAMBDA);

	tmp	=fs["regularize_type"];

	regularize_type	=	static_cast<RegularizeType>(tmp);

	return 0;
}

int
LinearModelBase::save(const std::string &save_path) const
{
	cv::FileStorage	cvfs(save_path, cv::FileStorage::WRITE);
	
	if (!cvfs.isOpened()) {
		return -1;
	}

	cv::write(cvfs, "weight", this->weight_);
	cv::write(cvfs, "w0", this->w0_);
	return 0;
}

int
LinearModelBase::load(const std::string &load_path)
{
	cv::FileStorage	cvfs(load_path, cv::FileStorage::READ);

	if (!cvfs.isOpened()) {
		return -1;
	}

	cv::read(cvfs["weight"], this->weight_);
	cv::read(cvfs["w0"], this->w0_, 0);
	return 0;
}

int
LinearModelBase::train(const TrainingSet &train_set, const StaticalModelParameter *param)
{
	const LinearModelParameterBase	*regression_param	=	dynamic_cast<const LinearModelParameterBase*>(param);

	if (nullptr == regression_param) {
		return -1;
	}
	
	MatType	ftype	=	train_set.get_feature_type();
	MatType	ltype	=	train_set.get_label_type();
	
	if (1 != CV_MAT_CN(ftype.data_type)) {
		return -1;
	}

	if (1 != CV_MAT_CN(ltype.data_type)) {
		return -1;
	}
	
	if (2 != ftype.sizes.size()) {
		return -1;
	}

	if (2 != ltype.sizes.size()) {
		return -1;
	}
	
	if (1 != ftype.sizes[1]) {
		return -1;
	}

	if (1 != ltype.sizes[1]) {
		return -1;
	}

	return train(train_set, regression_param);
}

int
LinearModelBase::train(const TrainingSet &train_set, const LinearModelParameterBase *param)
{
	std::mt19937	rnd(param->rnd_seed);
	double			rate	=	param->update_rate;

	puts("init weight");
	init_weight(train_set, rnd);

	for (std::size_t ii = 1; ii <= param->max_iter; ++ii) {
		double	delta;
		cv::Mat	grad_w;
		double	grad_w0;

		printf("%d th iter\n", ii);

		calc_param_delta(train_set, grad_w, grad_w0);
		
		this->weight_	-=	grad_w * rate;
		this->w0_		-=	grad_w0 * rate;

		switch (param->regularize_type) {
			default:
			case StaticalModelParameter::REGULARIZE_NONE:
				break;
			case StaticalModelParameter::REGULARIZE_L2:
				regularize_l2(this->weight_, this->w0_, param->lambda);
				break;
			case StaticalModelParameter::REGULARIZE_L1:
				regularize_l1(this->weight_, this->w0_, param->lambda);
				break;
		}

		// 終了判定する
		delta	=	grad_w.dot(grad_w) + grad_w0 * grad_w0;
		delta	*=	rate;

		printf("delta = %f\n", delta);

		if (delta < param->min_delta) {
			break;
		}
		
	}
	return 0;
}

void
LinearModelBase::regularize_l1(cv::Mat &weight, double &w0, double lambda) const
{
	// 注:w0は普通正則化しないとか読んだ

	//weightは行ベクトル
	for (int hh = 0; hh < weight.rows; ++hh) {
		for (int ww = 0; ww < weight.cols; ++ww) {
			double	val	=	weight.at<double>(hh, ww);

			if (std::abs(val) < lambda) {
				val	=	0;
			} else {
				if (val < 0) {
					val	+=	lambda;
				} else {
					val	-=	lambda;
				}
			}

			weight.at<double>(hh, ww)	=	val;
		}
	}
}

void
LinearModelBase::regularize_l2(cv::Mat &weight, double &w0, double lambda) const
{
	weight	-=	lambda * weight;
}

void	
LinearModelBase::init_weight(const TrainingSet &train_set, std::mt19937 &rnd) 
{
	MatType	ftype	=	train_set.get_feature_type();
	MatType	ltype	=	train_set.get_label_type();
	std::uniform_real_distribution<double>	uni_dist;
	
	weight_.create((int)ftype.total(), (int)ltype.total(), CV_64FC1);

	for (int ii = 0; ii < weight_.rows; ++ii) {
		for (int jj = 0; jj < weight_.cols; ++jj) {
			weight_.at<double>(ii, jj)	=	uni_dist(rnd);
		}
	}

	w0_	=	uni_dist(rnd);
}

int		LinearRegression::predict(const cv::Mat &feature, PredictionResult *result) 
{
	RegressionResult	*rresult	=	dynamic_cast<RegressionResult*>(result);

	if (nullptr == rresult) {
		return -1;
	}

	return predict(feature, *rresult);
}

int		LinearRegression::predict(const cv::Mat &feature, RegressionResult &result) 
{
	cv::Mat	posterior	=	calc_activation(feature);

	result.set_posterior(posterior);

	return 0;
}

cv::Mat	LinearRegression::calc_activation(const cv::Mat &feature) const 
{
	return (feature * weight_) + w0_;
}

void	LinearRegression::calc_param_delta(const TrainingSet &train_set, cv::Mat &delta_w, double &delta_w0)
{
	delta_w		=	weight_.clone();
	delta_w		=	0;
	delta_w0	=	0;

	
	// 全データで損失計算して，更新量に加算する
	for (std::size_t ii = 0; ii < train_set.size(); ++ii) {
		cv::Mat	loss	=	train_set[ii]->label - calc_activation(train_set[ii]->feature);
		
		delta_w		+=	loss.t() * train_set[ii]->feature;
		delta_w0	+=	loss.dot(loss);
	}

	delta_w		/=	(double)train_set.size();
	delta_w0	/=	(double)train_set.size();
}