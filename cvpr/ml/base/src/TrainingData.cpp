#include "..\header\TrainingData.h"

using namespace cvpr;

bool	TrainingSet::push_back(const PtrTrainingExample &example)
{
	if (!is_valid_example(example)) {
		return false;
	}

	examples_.push_back(example);

	return true;
}

double	TrainingSet::compute_label_entropy() const 
{
	cv::Mat		label_sum	=	calc_label_sum();
	int			num_label_elem	=	get_total1(label_sum);
	double		total_votes	=	0.0;
	double		*ptr_data	=	label_sum.ptr<double>();
	double		entropy		=	0.0;
	
	// ラベルの値の総数を求める
	for (std::size_t ii = 0; ii < num_label_elem; ++ii) {
		total_votes	+=	ptr_data[ii];
	}

	// 総数が十分小さかったら計算出来ないので0
	if (total_votes < std::numeric_limits<double>::min()) {
		return 0;
	}

	// ラベルの各次元についてエントロピーを求めて足す
	for (std::size_t ii = 0; ii < num_label_elem; ++ii) {
		double	label_prob	=	ptr_data[ii] / total_votes;

		// 生起確率が十分小さかったらエントロピー0
		if (label_prob < std::numeric_limits<double>::min()) {
			continue;
		}

		entropy	+=	- label_prob * std::log(label_prob);
	}

	return entropy;
}

bool	TrainingSet::is_valid_example(const PtrTrainingExample &example) const
{
	// 特徴ベクトルとラベルのサイズと型があってるか調べる
	if (!feature_type_.equals(example->feature)) {
		return false;
	}

	if (!label_type_.equals(example->target)) {
		return false;
	}

	return true;
}

cv::Mat	TrainingSet::calc_label_sum()const
{
	cv::Mat	label_sum((int)label_type_.sizes.size(), &label_type_.sizes[0], label_type_.data_type);
	cv::Mat	tmp;

	label_sum	=	0;

	for (auto ii = examples_.begin(); ii != examples_.end(); ++ii) {
		(*ii)->target.convertTo(tmp, CV_64F);
		label_sum	+=	tmp;
	}

	return label_sum;
}

TrainingSet	TrainingSet::calc_l2_normalized_set() const 
{
	TrainingSet	dst(get_feature_type(), get_label_type());

	for (std::size_t ii = 0; ii < size(); ++ii) {
		PtrTrainingExample	ex(new TrainingExample) ;

		cv::normalize(this->operator[](ii)->feature, ex->feature, 1.0, 0, cv::NORM_L2);
		cv::normalize(this->operator[](ii)->target, ex->target, 1.0, 0, cv::NORM_L2);

		dst.push_back(ex);
	}

	return dst;
}

void	TrainingSet::find_feature_min(cv::Mat &min_val) const 
{
	MatType	ftype	=	get_feature_type();
	int		cn		=	CV_MAT_CN(ftype.data_type);

	min_val.create((int)ftype.sizes.size(), &ftype.sizes[0], CV_64FC(cn));
	min_val	=	std::numeric_limits<double>::max();

	for (std::size_t ii = 0; ii < size(); ++ii) {
		cv::Mat	feature;

		this->operator[](ii)->feature.convertTo(feature, CV_64F);

		for (std::size_t jj = 0; jj < ftype.total(); ++jj) {
			min_val.ptr<double>()[jj]	=	std::min(feature.ptr<double>()[jj], min_val.ptr<double>()[jj]);
		}
	}
}

void	TrainingSet::find_feature_max(cv::Mat &max_val) const 
{
	MatType	ftype	=	get_feature_type();
	int		cn		=	CV_MAT_CN(ftype.data_type);

	max_val.create((int)ftype.sizes.size(), &ftype.sizes[0], CV_64FC(cn));
	max_val	=	-std::numeric_limits<double>::max();

	for (std::size_t ii = 0; ii < size(); ++ii) {
		cv::Mat	feature;

		this->operator[](ii)->feature.convertTo(feature, CV_64F);

		for (std::size_t jj = 0; jj < ftype.total(); ++jj) {
			max_val.ptr<double>()[jj]	=	std::max(feature.ptr<double>()[jj], max_val.ptr<double>()[jj]);
		}
	}
}

TrainingSet	TrainingSet::mat_arr_to_train_set(const std::vector<cv::Mat> &features, const std::vector<cv::Mat> &labels, bool copy_data, bool vectorize) 
{
	if (features.empty()) {
		return TrainingSet();
	}

	if (features.size() != labels.size()) {
		return TrainingSet();
	}

	MatType	ftype, ltype;

	if (vectorize) {
		ftype	=	MatType(get_colvec_header(features[0]));
		ltype	=	MatType(get_colvec_header(labels[0]));
	} else {
		ftype	=	MatType(features[0]);
		ltype	=	MatType(labels[0]);
	}

	TrainingSet	dst(ftype, ltype);

	for (std::size_t ii = 0; ii < features.size(); ++ii) {
		cv::Mat	ftmp;
		cv::Mat	ltmp;
		PtrTrainingExample	ex(new TrainingExample);
		
		if (copy_data) {
			ftmp	=	features[ii].clone();
			ltmp	=	labels[ii].clone();
		} else {
			ftmp	=	features[ii];
			ltmp	=	labels[ii];
		}

		if (vectorize) {
			ftmp	=	get_colvec_header(ftmp);
			ltmp	=	get_colvec_header(ltmp);
		}

		ex->feature	=	ftmp;
		ex->target	=	ltmp;

		dst.push_back(ex);
	}

	return dst;
}

TrainingSet	TrainingSet::random_sample(double sample_rate, std::mt19937 &rng) const 
{
	TrainingSet	dst_set(get_feature_type(), get_label_type());
	std::uniform_real_distribution<>	uni_dst;

	for (std::size_t ii = 0; ii < size(); ++ii) {
		double	prob	=	uni_dst(rng);

		if (sample_rate <= prob) {
			continue;
		}

		dst_set.push_back(this->operator[](ii));
	}

	return dst_set;
}

TrainingSet	TrainingSet::get_out_of_bag(const TrainingSet &bag) const 
{
	TrainingSet	dst_set(get_feature_type(), get_label_type());

	for (std::size_t ithis = 0; ithis < this->size(); ++ithis) {
		bool	is_found	=	false;
		
		for (std::size_t ibag = 0; ibag < bag.size(); ++ibag) {

			if (bag[ibag] == this->operator[](ithis)) {
				is_found	=	true;
				break;
			}
		}

		if (!is_found) {
			dst_set.push_back(this->operator[](ithis));
		}

	}

	return dst_set;
}

/**
*	ラベルの行列の平均を求める
*	@param	dst	出力の行列
*/
void TrainingSet::compute_target_mean(cv::Mat &dst) const
{
	if (examples_.empty())
		return;

	dst	=	calc_label_sum() / size();
}

/**
*	ラベルの行列の分散を求める
*	@return	分散
*/
double TrainingSet::compute_target_var() const
{
	cv::Mat	mean;
	double	var	=	0;

	if (examples_.empty())
		return 0;

	compute_target_mean(mean);

	for (std::size_t ii = 0; ii < examples_.size(); ++ii) {
		cv::Mat	tmp;
		double	*pdata	=	nullptr;

		examples_[ii]->target.convertTo(tmp, CV_64F);

		// 若干桁が怖い
		tmp	=	tmp - mean;
		
		pdata	=	tmp.ptr<double>();

		for (int jj = 0; jj < cvpr::get_total1(tmp); ++jj) {
			var	+=	(*pdata)*(*pdata);
			++pdata;
		}
	}

	return var;
}