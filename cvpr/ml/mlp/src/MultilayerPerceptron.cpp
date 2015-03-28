#define _CRT_SECURE_NO_WARNINGS 1
#include "..\header\MutilayerPercptron.h"
#include "..\header\MlpLayerFactory.h"
#include "utils.h"
#include <direct.h>

using namespace cvpr ;
using namespace mlp;

const std::string	MultilayerPerceptron::FNAME_MLP_CFG			=	"mlp_cfg.txt";
const std::string	MultilayerPerceptron::CFG_TAG_LAYER_TYPES	=	"layer_types";
const std::string	MultilayerPerceptron::FNAME_LEYER_CFG		=	"layer";

static const std::string SUFFIX_MLP_FILE = "mlp";
static const std::string SUFFIX_LAYER_FILE = "layer";

int		MlpParameterbase::save(const std::string &save_path) const
{
	cv::FileStorage	cvfs(save_path, cv::FileStorage::WRITE);

	if (!cvfs.isOpened()) {
		return -1;
	}

	cv::write(cvfs, STRINGIZE(num_hidden_layers), num_hidden_layers);
	cv::write(cvfs, STRINGIZE(max_iter), max_iter);
	cv::write(cvfs, STRINGIZE(min_delta), min_delta);
	cv::write(cvfs, STRINGIZE(decay_rate), decay_rate);
	cv::write(cvfs, STRINGIZE(resample_rate), resample_rate);
	cv::write(cvfs, STRINGIZE(output_layer_type), (int)output_layer_type);
	cv::write(cvfs, STRINGIZE(rnd_seed), (int)rnd_seed);

	return 0;
}

int		MlpParameterbase::load(const std::string &load_path)
{
	cv::FileStorage	cvfs(load_path, cv::FileStorage::READ);
	int	i_tmp	=	0;

	if (!cvfs.isOpened()) {
		return -1;
	}
	
	cv::read(cvfs[STRINGIZE(num_hidden_layers)], num_hidden_layers, 1);
	cv::read(cvfs[STRINGIZE(max_iter)], max_iter, 100);
	cv::read(cvfs[STRINGIZE(min_delta)], min_delta, 0.01);
	cv::read(cvfs[STRINGIZE(decay_rate)], decay_rate, 0.005);
	cv::read(cvfs[STRINGIZE(resample_rate)], resample_rate, 0.1);

	cv::read(cvfs[STRINGIZE(output_layer_type)], i_tmp, LAYER_TYPE_SIGMOID);
	output_layer_type	=	(MlpOutputLayerType)i_tmp;

	cv::read(cvfs[STRINGIZE(rnd_seed)], i_tmp, 19861124);
	rnd_seed	=	i_tmp;
	
	return 0;
}

int	MultilayerPerceptronParameter::save(const std::string &save_path) const
{
	if (0 != MlpParameterbase::save(save_path + SUFFIX_MLP_FILE)) {
		return -1;
	}

	if (0 != LayerParameter::save(save_path + SUFFIX_LAYER_FILE)) {
		return -1;
	}

	return 0;
}

int	MultilayerPerceptronParameter::load(const std::string &load_path)
{
	if (0 != MlpParameterbase::load(load_path + SUFFIX_MLP_FILE)) {
		return -1;
	}

	if (0 != LayerParameter::load(load_path + SUFFIX_LAYER_FILE)) {
		return -1;
	}

	return 0;
}

int		MultilayerPerceptron::save(const std::string &save_path) const 
{
	// 出力先にフォルダ掘る
	if (ENOENT == _mkdir(save_path.c_str())) {
		return -1;
	}

	// メタデータファイルを作って，層の種別を書き込む
	cv::FileStorage	cvfs(save_path + "\\" + FNAME_MLP_CFG, cv::FileStorage::WRITE);
	std::vector<int>	layer_types(layers().size());

	if (!cvfs.isOpened()) {
		return -1;
	}

	for (std::size_t ii = 0; ii < hidden_layers_.size(); ++ii) {
		layer_types[ii]	=	(int)hidden_layers_[ii]->type();
	}

	layer_types.back()	=	(int)out_layer_->type();

	cv::write<int>(cvfs, CFG_TAG_LAYER_TYPES, layer_types);
	
	// 各層のパラメータを書き込む
	for (std::size_t ii = 0; ii < layers().size(); ++ii) {
		char	str_ii[256];
		std::string	fname	=	save_path + "\\" + FNAME_LEYER_CFG;

		sprintf(str_ii, "%03d", ii);
		
		fname.append(str_ii);
		
		if (0 != layers()[ii]->save(fname)) {
			return -1;
		}
	}

	return 0;
}

int		MultilayerPerceptron::load(const std::string &load_path) 
{
	cv::FileStorage	cvfs(load_path + "\\" + FNAME_MLP_CFG, cv::FileStorage::READ);
	std::vector<int>	layer_types;

	if (!cvfs.isOpened()) {
		return -1;
	}

	cv::read<int>(cvfs[CFG_TAG_LAYER_TYPES], layer_types, std::vector<int>());

	if (layer_types.empty()) {
		return -1;
	}

	hidden_layers_.clear();
	out_layer_.reset();

	for (std::size_t ii = 0; ii < layer_types.size() - 1; ++ii) {
		PtrHiddenLayerBase	layer	=	LayerFactory::create((MlpHiddenLayerType)layer_types[ii]);
		std::string		fname	=	load_path + "\\" + FNAME_LEYER_CFG;
		char			str_ii[256];

		if (nullptr == layer) {
			return -1;
		}

		sprintf(str_ii, "%03d", ii);
		
		fname.append(str_ii);

		if (0 != layer->load(fname)) {
			return -1;
		}

		hidden_layers_.push_back(layer);
	}

	{
		out_layer_	=	LayerFactory::create((MlpOutputLayerType)layer_types.back());
		std::string		fname	=	load_path + "\\" + FNAME_LEYER_CFG;
		char			str_ii[256];

		if (nullptr == out_layer_) {
			return -1;
		}

		sprintf(str_ii, "%03d", layer_types.size() - 1);
		
		fname.append(str_ii);

		if (0 != out_layer_->load(fname)) {
			return -1;
		}
	}

	return 0;
}

int		MultilayerPerceptron::predict(const cv::Mat &feature, PredictionResult *result) 
{
	cv::Mat	z_i	=	feature;
	cv::Mat	z_j;
	std::vector<PtrLayerBase>	layers_	=	layers();

	for (std::size_t ii = 0; ii < layers_.size(); ++ii) {
		layers_[ii]->foward_prop(z_i, z_j);

		z_i	=	z_j;
	}

	result->set_posterior(z_j);

	return 0;
}

int		MultilayerPerceptron::train(const TrainingSet &train_set, const StaticalModelParameter *param) 
{
	auto *mlp_param	=	dynamic_cast<const MultilayerPerceptronParameter*>(param);

	if (nullptr == mlp_param) {
		return -1;
	}

	return train(train_set, *mlp_param);
}

int	MultilayerPerceptron::train(const TrainingSet &train_set, const MultilayerPerceptronParameter &param) 
{
	std::mt19937	rng(param.rnd_seed);
	std::vector<PtrLayerBase>	layers_;

	if (!is_valid_parameter(param)) {
		return -1;
	}

	if (!is_valid_train_set(train_set)) {
		return -1;
	}

	// 層配列の初期化
	init_layers(train_set, param, rng);

	layers_	=	layers();

	for (int ii = 0; ii < param.max_iter; ++ii) {
		TrainingSet				mini_batch	=	train_set.random_sample(param.resample_rate, rng);
		std::vector<cv::Mat>	weight_deltas(param.num_hidden_layers + 1);
		std::vector<cv::Mat>	a_j;
		std::vector<cv::Mat>	z_j;
		std::vector<double>		bias_deltas(param.num_hidden_layers + 1, 0);
		std::vector<cv::Mat>	err_k(param.num_hidden_layers + 1);
		std::vector<cv::Mat>	weight_delta(param.num_hidden_layers + 1);
		double					err_sum			=	0;
		double					bias_delta		=	0;

		printf("mini_batch_size=%d\n", mini_batch.size());

		if (mini_batch.size() <= 0) {
			continue;
		}

		for (std::size_t jj = 0; jj < mini_batch.size(); ++jj) { 

			// 層毎にactivationを得る
			PtrTrainingExample		example	=	mini_batch[jj];

			calc_all_a_z(example->feature, a_j, z_j);
			
			// 出力層で誤差を得る
			err_k.back()	=	example->target - z_j.back();

			// 前の層の誤差を用いて，今の層の誤差を得る
			// 今の層の誤差と今の層の入力からパラメータ増分を得る
			for (int kk = (int)layers_.size() - 1; 0 <= kk; --kk) {

				layers_[kk]->calc_param_delta(err_k[kk], z_j[kk], weight_delta[kk], bias_delta);
				
				err_sum	+=	cv::norm(err_k[kk]);

				if (weight_deltas[kk].empty()) {
					weight_deltas[kk]	=	weight_delta[kk].clone();
				} else {
					weight_deltas[kk]	+=	weight_delta[kk];
				}

				bias_deltas[kk]		+=	bias_delta;
				
				if (0 < kk) {
					backprop_error(layers_[kk], layers_[kk-1], err_k[kk], z_j[kk], a_j[kk-1], err_k[kk-1]);
				}
			}
			
		}

		// パラメータ増分を用いてパラメータを更新する
		for (std::size_t jj = 0; jj < layers_.size(); ++jj) {

			double	scale	=	1.0 / (1.0 + param.decay_rate * ii);

			weight_deltas[jj]	*=	scale/mini_batch.size();
			bias_deltas[jj]		*=	scale/mini_batch.size();


			layers_[jj]->update(param, weight_deltas[jj], bias_deltas[jj]);

		}

		err_sum	/=	mini_batch.size();

		// 学習経過を標準出力する
		printf("\niter = %d, err = %f\n", ii, err_sum);
		
		// 終了条件を満たしていたら終了する
		if (err_sum < param.min_delta) {
			break;
		}
	}

	return 0;
}

int	MultilayerPerceptron::predict(const cv::Mat &feature, ClassificationResult &result) 
{
	return predict(feature, &result);
}

int	MultilayerPerceptron::predict(const cv::Mat &feature, RegressionResult &result) 
{
	return predict(feature, &result);
}

bool	MultilayerPerceptron::is_valid_parameter(const MultilayerPerceptronParameter &param) 
{
	if (param.max_iter <= 0) {
		return false;
	}
	/*
	if (param.num_hidden_layers <= 0) {
		return false;
	}
	*/
	if (param.num_hidden_units <= 0) {
		return false;
	}

	if (param.update_rate <= std::numeric_limits<double>::min()) {
		return false;
	}

	switch (param.output_layer_type) {
		default:
			return false;
		case LAYER_TYPE_LINEAR:
		case LAYER_TYPE_SIGMOID:
		case LAYER_TYPE_SOFTMAX:
			break;
	}

	return true;
}

bool	MultilayerPerceptron::is_valid_train_set(const TrainingSet &train_set) 
{
	MatType	ftype	=	train_set.get_feature_type();
	MatType	ltype	=	train_set.get_label_type();

	/*
	if (!ftype.is_single_channel()) {
		return false;
	}

	if (!ftype.is_col_vector()) {
		return false;
	}
	*/
	if (!ltype.is_single_channel()) {
		return false;
	}

	if (!ltype.is_col_vector()) {
		return false;
	}

	return true;
}

void	MultilayerPerceptron::init_layers(const TrainingSet &train_set, const MultilayerPerceptronParameter &param, std::mt19937 &rng) 
{
	// 隠れ層+出力層分配列確保
	hidden_layers_.resize(param.num_hidden_layers);

	// 隠れ層はsigmoidで作る
	for (std::size_t ii = 0; ii < hidden_layers_.size(); ++ii) {
		hidden_layers_[ii]	=	LayerFactory::create(get_hidden_layer_type());
	}

	// 出力層はパラメータで指定して作る
	out_layer_	=	LayerFactory::create(param.output_layer_type);

	MatType	in_type	=	train_set.get_feature_type();
	MatType	out_type;
	
	for (std::size_t ii = 0; ii < hidden_layers_.size(); ++ii) {
		hidden_layers_[ii]->init(in_type, param, out_type, rng);

		in_type	=	out_type;
	}

	out_layer_->init(train_set, in_type, param, rng);

}

void	MultilayerPerceptron::calc_all_a_z(const cv::Mat &feature, std::vector<cv::Mat> &a_j, std::vector<cv::Mat> &z_j) const 
{
	std::vector<PtrLayerBase>	layers_	=	layers();

	a_j.resize(layers_.size());
	z_j.resize(layers_.size()+1);

	feature.copyTo(z_j.front());

	for (std::size_t ii = 0; ii < layers_.size(); ++ii) {

		layers_[ii]->calc_a_j(z_j[ii], a_j[ii]);

		layers_[ii]->calc_activation(a_j[ii], z_j[ii+1]);
	}

}

void	MultilayerPerceptron::backprop_error(const PtrLayerBase layer_k, const PtrLayerBase layer_j, const cv::Mat &err_k, const cv::Mat &z_j, const cv::Mat &a_j, cv::Mat &err_j) 
{
	cv::Mat	err_sum;
	cv::Mat	h_prime;
	
	layer_k->calc_de_da(err_k, err_sum);

	layer_j->calc_derivative(z_j, a_j, h_prime);

	err_sum	=	err_sum.reshape(1, get_total1(err_sum));

	h_prime	=	h_prime.reshape(1, get_total1(h_prime));

	cv::multiply(h_prime, err_sum, err_j);
}
