#include "..\header\MlpLayer.h"
#include <opencv2\imgproc\imgproc.hpp>

using namespace cvpr;
using namespace mlp;

int	CnnLayerParameter::save(const std::string &save_path) const
{
	if (0 != LayerParameter::save(save_path)) {
		return -1;
	}

	cv::FileStorage	cvfs(save_path, cv::FileStorage::APPEND);
	
	if (!cvfs.isOpened()) {
		return -1;
	}

	cv::write(cvfs, STRINGIZE(kernel_size), kernel_size);

	return 0;
}

int	CnnLayerParameter::load(const std::string &load_path)
{
	std::vector<int>	tmp;

	if (0 != LayerParameter::load(load_path)) {
		return -1;
	}

	cv::FileStorage	cvfs(load_path, cv::FileStorage::READ);
	
	if (!cvfs.isOpened()) {
		return -1;
	}

	cv::read(cvfs[STRINGIZE(kernel_size)], tmp);

	if (2 != tmp.size()) {
		return -1;
	}

	kernel_size	=	cv::Size(tmp[1], tmp[0]);

	return 0;
}

void	SigmoidLayer::calc_activation(const cv::Mat &a_j, cv::Mat &dst) const 
{
	cv::Mat	exp_;

	cv::exp(-a_j, exp_);

	cv::divide(1.0, 1.0 + exp_, dst);
}

void	SigmoidLayer::calc_derivative(const cv::Mat &z_j, const cv::Mat &a_j, cv::Mat &dst) const 
{
	cv::Mat	h_aj;
	
	if (z_j.empty()) {
		calc_activation(a_j, h_aj);
	} else {
		h_aj	=	z_j;
	}

	cv::multiply(h_aj, 1.0-h_aj, dst);
}

void	LinearLayer::calc_activation(const cv::Mat &a_j, cv::Mat &dst) const 
{
	a_j.copyTo(dst);
}

void	LinearLayer::calc_derivative(const cv::Mat &z_j, const cv::Mat &a_j, cv::Mat &dst) const 
{
	dst.create(a_j.size(), a_j.type());
	dst	=	1.0;
}

void	SoftMaxLayer::calc_activation(const cv::Mat &a_j, cv::Mat &dst) const 
{
	double	log_sum	=	log_sum_exp(a_j);

	cv::exp(a_j - log_sum, dst) ;
}

void	SoftMaxLayer::calc_derivative(const cv::Mat &z_j, const cv::Mat &a_j, cv::Mat &dst) const 
{
	cv::Mat	h_aj;

	if (z_j.empty()) {
		calc_activation(a_j, h_aj);
	} else {
		h_aj	=	z_j;
	}

	cv::multiply(h_aj, 1.0-h_aj, dst);
}


void	TanhLayer::calc_activation(const cv::Mat &a_j, cv::Mat &dst) const 
{
	cv::Mat	exp_;
	cv::Mat	one_	=	cv::Mat::ones(a_j.size(), CV_64FC1);
	
	cv::exp(-2*a_j, exp_);
	
	cv::divide(one_-exp_, one_+exp_, dst);
}

void	TanhLayer::calc_derivative(const cv::Mat &z_j, const cv::Mat &a_j, cv::Mat &dst) const 
{
	cv::Mat	h_aj;
	
	if (z_j.empty()) {
		calc_activation(a_j, h_aj);
	} else {
		h_aj	=	z_j;
	}

	dst	=	1 - h_aj.mul(h_aj);
}

int		ConvolutionLayer::save(const std::string &fname) const
{
	cv::FileStorage	cvfs(fname, cv::FileStorage::WRITE);

	if (!cvfs.isOpened()) {
		return -1;
	}

	cv::write(cvfs, STRINGIZE(bias_), bias_);
	cv::write(cvfs, STRINGIZE(in_type_.sizes), in_type_.sizes);
	cv::write(cvfs, STRINGIZE(in_type_.data_type), in_type_.data_type);
	cv::write(cvfs, STRINGIZE(kernels_), kernels_);
	/*
	{
		cv::WriteStructContext	ws(cvfs, STRINGIZE(kernels_), cv::FileNode::SEQ);
		for (std::size_t ii = 0; ii < kernels_.size(); ++ii) {
			cv::write(*ws.fs, "", kernels_[ii]);
		}
	}
	*/
	return 0;
}

int		ConvolutionLayer::load(const std::string &fname) 
{
	cv::FileStorage	cvfs(fname, cv::FileStorage::READ);
	cv::FileNode	fn;

	if (!cvfs.isOpened()) {
		return -1;
	}

	kernels_.empty();

	cv::read(cvfs[STRINGIZE(bias_)], bias_, DBL_MAX);
	cv::read(cvfs[STRINGIZE(in_type_.sizes)], in_type_.sizes);
	cv::read(cvfs[STRINGIZE(in_type_.data_type)], in_type_.data_type, -1);

	fn	=	cvfs[STRINGIZE(kernels_)];

	if (fn.empty()) {
		return -1;
	}

	for (auto ii = fn.begin(); ii != fn.end(); ++ii) {
		cv::Mat	kernel;

		cv::read(*ii, kernel);
		
		if (kernel.empty()) {
			return -1;
		}

		kernels_.push_back(kernel);
	}

	if (DBL_MAX == bias_) {
		return -1;
	}

	if (in_type_.sizes.empty()) {
		return -1;
	}

	if (-1 == in_type_.data_type) {
		return -1;
	}

	return 0;
}

void	ConvolutionLayer::calc_a_j(const cv::Mat &feature, cv::Mat &dst) const
{
	cv::Size				ksize_half;
	cv::Size				ksize;
	std::vector<cv::Mat>	maps;

	ksize	=	kernels_[0].size();
	ksize_half.height	=	ksize.height / 2;
	ksize_half.width	=	ksize.width / 2;

	for (std::size_t ii = 0; ii < kernels_.size(); ++ii) {
		cv::Mat	map;
		cv::Point	tl(ksize_half.width, ksize_half.height);
		cv::Size	sz	=	feature.size() - (ksize - cv::Size(1, 1));
		cv::Rect	roi(tl, sz);

		cv::filter2D(feature, map, CV_64F, kernels_[ii]);
		maps.push_back(map(roi).clone());
	}

	cv::merge(maps, dst);
}

void	ConvolutionLayer::calc_de_da(const cv::Mat &err, cv::Mat &dst) const 
{
	// 逆伝搬されたエラーは列ベクトルなので，出力と同じ形式にreshapeする
	int		input_channels_	=	in_type_.channels();
	cv::Size	input_size_	=	in_type_.size();
	int		map_cn		=	input_channels_ * (int)kernels_.size();
	int		map_rows	=	input_size_.height - (kernels_[0].rows - 1);
	cv::Mat	tmp			=	err.reshape(map_cn, map_rows);
	std::vector<cv::Mat>	re_maps(input_channels_);
	std::vector<cv::Mat>	err_maps;

	cv::split(tmp, err_maps);


	for (std::size_t ii = 0; ii < re_maps.size(); ++ii) {
		re_maps[ii].create(input_size_, CV_64FC1);
		re_maps[ii]	=	0;
	}

	// エラーをカーネル倍して足して入力を復元する
	for (int ii = 0; ii < (int)kernels_.size(); ++ii) {
		for (int cn = 0; cn < input_channels_; ++cn) {
			int	idx	=	ii * input_channels_ + cn;

			for (int hh = 0; hh < kernels_[ii].rows; ++hh) {
				for (int ww = 0; ww < kernels_[ii].cols; ++ww) {
					double		scale	=	kernels_[ii].at<double>(hh, ww);
					cv::Mat		remap	=	scale * err_maps[idx];
					cv::Rect	roi(ww, hh, err_maps[idx].cols, err_maps[idx].rows);
					
					re_maps[cn](roi)	+=	remap;
				}
			}
		}
	}

	cv::merge(re_maps, dst);

	dst	=	get_colvec_header(dst);
}

void	ConvolutionLayer::calc_activation(const cv::Mat &a_j, cv::Mat &dst) const 
{
	cv::Mat	tmp	=	a_j.reshape(1);
	cv::Mat	exp_;
	
	cv::exp(-2*tmp, exp_);
	
	cv::divide(1.0 - exp_, 1.0 + exp_, dst);
	dst	=	dst.reshape(a_j.channels());
}

void	ConvolutionLayer::calc_derivative(const cv::Mat &z_j, const cv::Mat &a_j, cv::Mat &dst) const 
{
	cv::Mat	h_aj;
	
	if (z_j.empty()) {
		calc_activation(a_j, h_aj);
	} else {
		h_aj	=	z_j;
	}

	cv::Mat	tmp	=	h_aj.reshape(1);

	dst	=	1 -tmp.mul(tmp);

	dst	=	dst.reshape(h_aj.channels());
}

void	ConvolutionLayer::calc_param_delta(const cv::Mat &err, const cv::Mat &activation, cv::Mat &weight_delta, double &bias_delta) const
{
	// エラーは列ベクトルだがここでは気にしない
	std::vector<cv::Mat>	deltas(kernels_.size());
	int	map_total	=	(int)(get_dst_type().total() / kernels_.size());


	for (std::size_t ii = 0; ii < deltas.size(); ++ii) {
		deltas[ii]	=	kernels_[ii].clone();
		deltas[ii]	=	0;
	}

	bias_delta	=	0;

	for (int ii = 0; ii < (int)kernels_.size(); ++ii) {
		int	num_delta	=	0;

		for (int jj = 0; jj < map_total; ++jj) {
			int	idx	=	ii * map_total + jj;
			double	scale	=	err.at<double>(idx);
			cv::Mat	delta	=	scale * kernels_[ii];

			if (std::abs(scale) < std::numeric_limits<double>::min()) {
				continue;
			}

			deltas[ii]	+=	delta;
			++num_delta;
		}

		deltas[ii]	/=	std::max((double)num_delta, 1.0);
		bias_delta	+=	cv::sum(deltas[ii])[0] / deltas[ii].total();
	}

	bias_delta	/=	deltas.size();

	cv::merge(deltas, weight_delta);
}
#include <opencv2\highgui\highgui.hpp>
void	ConvolutionLayer::update(const LayerParameter &param, const cv::Mat &weight_delta, double bias_delta) 
{
	std::vector<cv::Mat>	deltas;

	cv::split(weight_delta, deltas);

	bias_	+=	bias_delta * param.update_rate;

	for (std::size_t ii = 0; ii < deltas.size(); ++ii) {
		kernels_[ii]	+=	deltas[ii] * param.update_rate;
	}

	{
		cv::Mat	flts(kernel_size().height, kernel_size().width * (int)kernels_.size(), CV_64FC1);
		cv::Mat	dlts(kernel_size().height, kernel_size().width * (int)kernels_.size(), CV_64FC1);

		for (int ii = 0; ii < (int)kernels_.size(); ++ii) {
			double	min_val, max_val;
			cv::Mat	tmp,tmp2;
			
			cv::minMaxIdx(kernels_[ii], &min_val, &max_val);
			tmp	=	(kernels_[ii]-min_val) / (max_val-min_val);
			tmp2	=	flts(cv::Rect(ii*kernel_size().width, 0, kernel_size().width, kernel_size().height));
			tmp.copyTo(tmp2);


			cv::minMaxIdx(deltas[ii], &min_val, &max_val);
			tmp	=	(deltas[ii]-min_val) / (max_val-min_val);
			tmp2	=	dlts(cv::Rect(ii*kernel_size().width, 0, kernel_size().width, kernel_size().height));
			tmp.copyTo(tmp2);
		}
		cv::namedWindow("dlt");
		cv::namedWindow("flt");
		cv::imshow("dlt",dlts);
		cv::imshow("flt",flts);
		cv::waitKey(1);
	}
}

int	ConvolutionLayer::init(const MatType &in_type, const LayerParameter &param, MatType &out_type, std::mt19937 &rng) 
{
	const CnnLayerParameter	&cnn_param	=	dynamic_cast<const CnnLayerParameter&>(param);
	std::uniform_real_distribution<>	dist;

	if (in_type.dims() != 2) {
		return -1;
	}

	in_type_	=	in_type;

	kernels_.resize(cnn_param.num_hidden_units);

	for (std::size_t ii = 0; ii < kernels_.size(); ++ii) {
		kernels_[ii].create(cnn_param.kernel_size, CV_64FC1);
		rand_init<double>(rng, kernels_[ii]);
		kernels_[ii]	/=	cv::norm(kernels_[ii]);
	}

	bias_	=	dist(rng);

	out_type	=	get_dst_type();

	return 0;
}
