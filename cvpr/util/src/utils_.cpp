#define		_USE_MATH_DEFINES
#include	<cmath>
#include	<limits>
#include	<sstream>
#include	"..\header\utils_.h"

const static double EULER_CONSTANT	=	0.577215664901532860606512090082;

template<class tp> void	
	cvpr::random_sample(const std::vector<tp> &src,
	std::vector<tp> &bag, std::vector<tp> &oob, double sample_rate, std::mt19937 &rng)
{
	std::uniform_real_distribution<>	real_dist;

	for (auto ii = src.begin(); ii != src.end(); ++ii) {
		if (real_dist(rng) <= rate) {
			bag.push_back(ii);
		} else {
			oob.push_back(ii);
		}
	}
}

double	
	cvpr::digamma(int hk)
{
	double	sum	=	0;

	if (hk < 0) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	for (int ii = 1; ii < hk-1; ++ii) {
		sum	+=	1.0 / (double)(ii);
	}

	return - EULER_CONSTANT + sum;
};

double	
	cvpr::digamma_half(int hk)
{
	double	sum	=	0;

	if (hk < 0) {
		return std::numeric_limits<double>::quiet_NaN();
	}

	for (int ii = 0; ii < hk -1; ++ii) {
		sum	+=	1.0 / (2.0 * ii + 1);
	}

	return - EULER_CONSTANT - 2.0 * M_LN2 + 2.0 * sum;
};

void
	cvpr::split(const std::string &src, char delim, std::vector<std::string> &dst) 
{
	std::istringstream	iss(src);
	std::string			tmp;

	while (std::getline(iss, tmp, delim)) {
		dst.push_back(tmp);
	}
}

int	
	cvpr::max_idx(const cv::Mat &mat)
{
	double	min_val	=	0;
	double	max_val	=	0;
	std::vector<int>	max_idx(mat.dims, 0);
	int	idx	=	0;

	cv::minMaxIdx(mat, nullptr, nullptr, nullptr, &max_idx[0]);

	for (int ii = 0; ii < mat.dims-1; ++ii) {
		idx += max_idx[ii] * mat.size[ii+1];
	}

	idx	+=	max_idx.back();

	return idx;
}

double	
	cvpr::log_sum_exp(const cv::Mat &mat)
{
	cv::Mat	normalized_mat;
	cv::Mat	exp_;
	double	log_sum;
	double	max_val	=	0;

	cv::minMaxLoc(mat, nullptr, &max_val);

	normalized_mat	=	mat - max_val;

	cv::exp(normalized_mat, exp_);

	log_sum	=	cv::log(cv::sum(exp_)[0]);

	return max_val + log_sum;
}


cv::Mat	
	cvpr::get_colvec_header(const cv::Mat &mat) 
{
	if (is_column_vector(mat)) {
		return mat;
	}

	return mat.reshape(1, (int)mat.total() * mat.channels());
}

bool	
	cvpr::is_column_vector(const cv::Mat &mat)
{
	if (1 != mat.channels()) {
		return false;
	}

	if (2 != mat.dims) {
		return false;
	}

	if (1 != mat.cols) {
		return false;
	}

	return true;
}

bool	cvpr::mat_arr_to_hdim_mat(const std::vector<cv::Mat> &arr, cv::Mat &dst) 
{
	std::vector<int>	sizes(1, (int)arr.size());

	if (arr.empty()) {
		return false;
	}

	for (int ii = 0; ii < arr[0].dims; ++ii) {
		sizes.push_back(arr[0].size[ii]);
	}

	dst.create((int)sizes.size(), &sizes[0], arr[0].type());

	sizes[0]	=	1;

	for (int ii = 0; ii < (int)arr.size(); ++ii) {
		cv::Mat	tmp((int)sizes.size(), &sizes[0], arr[ii].type(), arr[ii].data);
		cv::Mat	to_cpy	=	dst.row(ii);

		tmp.copyTo(to_cpy);
	}

	return true;
}

int	cvpr::reduce_mat_dim(const cv::Mat &mat, cv::Mat &dst, bool copy_data) 
{
	if (2 == mat.dims) {
		if (copy_data) {
			dst	=	mat.clone();
		} else {
			dst	=	mat;
		}
		return 1;
	}

	if (1 != mat.size[0]) {
		return -1;
	}

	std::vector<int>	sizes(mat.dims - 1);

	for (int ii = 0; ii < mat.dims; ++ii) {
		sizes[ii]	=	mat.size[ii+1];
	}

	if (!copy_data) {
		dst	=	cv::Mat((int)sizes.size(), &sizes[0], mat.type(), mat.data);

	} else {
		dst.create((int)sizes.size(), &sizes[0], mat.type());

		memcpy(dst.data, mat.data, dst.total() * dst.elemSize());
	}

	return 0;
}

int	cvpr::reduce_mat_dim(const cv::Mat &mat, std::vector<cv::Mat> &dst, bool copy_data) 
{
	dst.clear();

	if (2 == mat.dims) {
		if (copy_data) {
			dst.push_back(mat.clone());
		} else {
			dst.push_back(mat);
		}

		return 1;
	}

	for (int ii = 0; ii < mat.size[0]; ++ii) {
		cv::Mat	row_mat	=	mat.row(ii);
		cv::Mat	tmp;

		reduce_mat_dim(row_mat, tmp, copy_data);

		dst.push_back(row_mat);
	}

	return 0;
}

int	cvpr::get_total1(const cv::Mat &mat) 
{
	return (int)mat.total() * mat.channels();
}

bool cvpr::contains(const cv::Mat &mat, const cv::Point &pt, int margin) 
{
	if (pt.x < margin) {
		return false;
	}

	if (pt.y < margin) {
		return false;
	}

	if ((mat.cols - margin) <= pt.x) {
		return false;
	}

	if ((mat.rows - margin) <= pt.y) {
		return false;
	}

	return true;
}

cv::Rect cvpr::get_rect(const cv::Mat &mat)
{
	return cv::Rect(0, 0, mat.cols, mat.rows);
}