#include "..\header\MlpLayerBase.h"

using namespace cvpr;
using namespace mlp;

int		LayerParameter::save(const std::string &save_path) const
{
	cv::FileStorage	cvfs(save_path, cv::FileStorage::WRITE);
	
	if (!cvfs.isOpened()) {
		return -1;
	}

	cv::write(cvfs, STRINGIZE(num_hidden_units), num_hidden_units);
	cv::write(cvfs, STRINGIZE(update_rate), update_rate);
	cv::write(cvfs, STRINGIZE(lambda), lambda);
	cv::write(cvfs, STRINGIZE(regularize_type), regularize_type);

	return 0;
}

int		LayerParameter::load(const std::string &load_path)
{
	cv::FileStorage	cvfs(load_path, cv::FileStorage::READ);
	int	i_tmp	=	0;

	if (!cvfs.isOpened()) {
		return -1;
	}
	
	cv::read(cvfs[STRINGIZE(num_hidden_units)], num_hidden_units, 10);
	cv::read(cvfs[STRINGIZE(update_rate)], update_rate, 0.01);
	cv::read(cvfs[STRINGIZE(lambda)], lambda, 0.001);

	cv::read(cvfs[STRINGIZE(regularize_type)], i_tmp, REGULARIZE_L2);
	regularize_type	=	(RegularizeType)i_tmp;

	return 0;
}

int	LayerBase::save(const std::string &path) const
{
	cv::FileStorage	cvfs(path, cv::FileStorage::WRITE);

	if (!cvfs.isOpened()) {
		return -1;
	}

	cv::write(cvfs, STRINGIZE(weight_), weight_);

	cv::write(cvfs, STRINGIZE(bias_), bias_);

	return 0;
}

int	LayerBase::load(const std::string &path) 
{
	cv::FileStorage	cvfs(path, cv::FileStorage::READ);

	if (!cvfs.isOpened()) {
		return -1;
	}

	cv::read(cvfs[STRINGIZE(weight_)], weight_);

	if (weight_.empty()) {
		return -1;
	}

	cv::read(cvfs[STRINGIZE(bias_)], bias_, DBL_MAX);

	if (DBL_MAX == bias_) {
		return -1;
	}

	return 0;
}

void	LayerBase::foward_prop(const cv::Mat &feature, cv::Mat &dst) const 
{
	cv::Mat	a_j;
	
	calc_a_j(feature, a_j);

	calc_activation(a_j, dst);
}

void	LayerBase::calc_param_delta(const cv::Mat &err, const cv::Mat &activation, cv::Mat &weight_delta, double &bias_delta) const
{
	cv::Mat	tmp	=	get_colvec_header(activation);

	weight_delta	=	err * tmp.t();
	bias_delta		=	cv::sum(err)[0];
}

void	LayerBase::update(const LayerParameter &param, const cv::Mat &weight_delta, double bias_delta) 
{
	weight_	+=	param.update_rate * weight_delta;
	bias_	+=	param.update_rate * bias_delta;

	switch (param.regularize_type) {
		default:
		case StaticalModelParameter::REGULARIZE_NONE:
			break;
		case StaticalModelParameter::REGULARIZE_L2:
			regularize_l2(param);
			break;
		case StaticalModelParameter::REGULARIZE_L1:
			regularize_l1(param);
			break;
	}

}

void	LayerBase::regularize_l2(const LayerParameter &param) 
{
	weight_	-=	param.lambda * weight_;
}

void	LayerBase::regularize_l1(const LayerParameter &param) 
{
	// 注:w0は普通正則化しないとか読んだ

	//weightは行ベクトル
	for (int hh = 0; hh < weight_.rows; ++hh) {
		for (int ww = 0; ww < weight_.cols; ++ww) {
			double	val	=	weight_.at<double>(hh, ww);

			if (std::abs(val) < param.lambda) {
				val	=	0;
			} else {
				if (val < 0) {
					val	+=	param.lambda;
				} else {
					val	-=	param.lambda;
				}
			}

			weight_.at<double>(hh, ww)	=	val;
		}
	}
}

int	HiddenLayerBase::init(const MatType &in_type, const LayerParameter &param, MatType &out_type, std::mt19937 &rng) 
{
	std::uniform_real_distribution<>	uni_dst(-1.0, 1.0) ;
	int	in_dims		=	(int)in_type.total();
	int	out_dims	=	param.num_hidden_units;

	weight_.create(out_dims, in_dims, CV_64FC1);

	rand_init<double>(rng, weight_);

	weight_	/=	in_dims;

	bias_	=	uni_dst(rng) / in_dims;

	out_type	=	MatType(cv::Size(1, out_dims), CV_64FC1);

	return 0;
}

int	OutputLayerBase::init(const TrainingSet &train_set, const MatType &in_type, const LayerParameter &param, std::mt19937 &rng) 
{
	std::uniform_real_distribution<>	uni_dst(-1.0, 1.0) ;
	int	in_dims		=	(int)in_type.total();
	int	out_dims	=	(int)train_set.get_label_type().total();

	weight_.create(out_dims, in_dims, CV_64FC1);

	rand_init<double>(rng, weight_);

	weight_	/=	in_dims;

	bias_	=	uni_dst(rng) / in_dims;

	return 0;
}
