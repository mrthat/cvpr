#include "..\header\LogisticRegression.h"

using namespace cvpr;

cv::Mat
LogisticRegression::calc_activation(const cv::Mat &feature) const
{
	cv::Mat	dotp	=	weight_ * feature + w0_;
	cv::Mat	exp_;

	cv::exp(-dotp, exp_);

	return 1.0 / (1.0 + exp_);
}

cv::Mat
LogisticRegression::calc_loss(const PtrTrainingExample data) const
{
	cv::Mat	activation	=	calc_activation(data->feature);

	return (data->target - activation);
}

int
LogisticRegression::predict(const cv::Mat &feature, PredictionResult *result, const PredictionParameter *param)
{
	ClassificationResult	*res	=	dynamic_cast<ClassificationResult*>(result);

	if (nullptr == res) {
		return -1;
	}

	return	predict(feature, *res);
}

int
LogisticRegression::predict(const cv::Mat &feature, ClassificationResult &result) const
{
	// 2クラス識別器なので2行1列
	cv::Mat	posterior	=	calc_activation(feature);

	result.set_posterior(posterior);

	return 0;
}

void
LogisticRegression::calc_param_delta(const TrainingSet &train_set, cv::Mat &delta_w, double &delta_w0)
{
	delta_w		=	weight_.clone();
	delta_w		=	0;
	delta_w0	=	0;

	double	tp	=	0.0;

	// 全データで損失計算して，更新量に加算する
	for (std::size_t ii = 0; ii < train_set.size(); ++ii) {
		cv::Mat	data_64f;
		cv::Mat	loss	=	calc_loss(train_set[ii]);
		
		delta_w		+=	loss * train_set[ii]->feature.t();
		delta_w0	+=	loss.dot(loss);
	}

	delta_w		=	-1.0/train_set.size() * delta_w;
	delta_w0	=	-1.0/train_set.size() * delta_w0;
}

void	
LogisticRegression::init_weight(const TrainingSet &train_set, std::mt19937 &rnd) 
{
	MatType	ftype	=	train_set.get_feature_type();
	MatType	ltype	=	train_set.get_label_type();
	std::uniform_real_distribution<double>	uni_dist;
	
	// K行M列	K:=クラス数,M:=特徴次元数
	weight_.create(ltype.size().height, ftype.size().height, CV_64FC1);

	for (int ii = 0; ii < weight_.rows; ++ii) {
		for (int jj = 0; jj < weight_.cols; ++jj) {
			weight_.at<double>(ii, jj)	=	uni_dist(rnd);
		}
	}

	w0_	=	uni_dist(rnd);
}

cv::Mat	SoftMaxRegression::calc_activation(const cv::Mat &feature) const
{
	
	cv::Mat	dotp	=	weight_ * feature + w0_;
	cv::Mat	exp_;
	cv::Mat	tmp;
	double	log_sum	=	log_sum_exp(dotp);

	tmp	=	dotp - log_sum;

	cv::exp(tmp, exp_);

	return exp_ ;
}

cv::Mat	SoftMaxRegression::calc_loss(const PtrTrainingExample data) const
{
	cv::Mat	activation	=	calc_activation(data->feature);

	return data->target.mul(data->target - activation);
}