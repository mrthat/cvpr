#include <direct.h>
#include "..\header\GradientBoosth.h"

using namespace cvpr;

int		GradientBoost::save(const std::string &save_path) const
{
	_mkdir(save_path.c_str());
	cv::FileStorage		cvfs(get_data_path(save_path), cv::FileStorage::WRITE);
	std::vector<int>	learner_types;

	if (!cvfs.isOpened())
		return -1;

	cv::write(cvfs, STRINGIZE(shrinkage), shrinkage);

	cv::write(cvfs, STRINGIZE(f0), f0);

	for (std::size_t ii = 0; ii < weak_learner.size(); ++ii)
		learner_types.push_back(weak_learner[ii]->get_type());

	cv::write(cvfs, STRINGIZE(learner_types), learner_types);

	for (std::size_t ii = 0; ii < weak_learner.size(); ++ii) {
		std::string	model_path	=	save_path + "\\model" + std::to_string(ii);
		int	state	=	weak_learner[ii]->save(model_path);

		if (0 != state)
			return state;
	}

	return 0;
}

int		GradientBoost::load(const std::string &load_path)
{
	cv::FileStorage	cvfs(get_data_path(load_path), cv::FileStorage::READ);
	std::vector<int>	learner_types;

	release();

	if (!cvfs.isOpened())
		return -1;

	cv::read(cvfs[STRINGIZE(shrinkage)], shrinkage, -1);
	cv::read(cvfs[STRINGIZE(f0)], f0);
	cv::read(cvfs[STRINGIZE(learner_types)], learner_types);

	for (std::size_t ii = 0; ii < learner_types.size(); ++ii) {
		PtrWeakLearner	model	=	WeakLearnerFactory::create((StatModelType)learner_types[ii]);

		if (nullptr == model)
			return -1;

		std::string	model_path	=	load_path + "\\model" + std::to_string(ii);

		int	state	=	model->load(model_path);

		if (0 != state)
			return state;

		weak_learner.push_back(model);
	}

	return 0;
}

int		GradientBoost::predict(const cv::Mat &feature, PredictionResult *result, const PredictionParameter *param)
{
	RegressionResult	*result_	=	dynamic_cast<RegressionResult*>(result);
	cv::Mat	posterior	=	f0.clone();

	if (nullptr == result_)
		return -1;

	for (std::size_t ii = 0; ii < weak_learner.size(); ++ii) {
		RegressionResult	tmp;

		int	state	=	weak_learner[ii]->predict(feature, &tmp, param);

		if (0 != state) {
			return state;
		}

		posterior	+=	tmp.get_posterior() * shrinkage;
	}

	result_->set_posterior(posterior);

	return 0;
}

int		GradientBoost::train(const TrainingSet &train_set, const StaticalModelParameter *param)
{
	const GradientBoostParameter	*param_	=	dynamic_cast<const GradientBoostParameter*>(param);

	release();

	if (train_set.size() == 0)
		return -1;

	if (nullptr == param_)
		return -1;

	return train(train_set, *param_);
}

int GradientBoost::train(const TrainingSet &datas, const GradientBoostParameter &param)
{
	if (!is_valid_param(datas, param))
		return -1;

	shrinkage	=	param.shrinkage;

	if (0 != find_initial_model(datas, param))
		return -1;

	for (unsigned ii = 0; ii < param.nr_rounds; ++ii) {
		TrainingSet		curr_set	=	calc_next_target(datas, param);
		PtrWeakLearner	fk			=	nullptr;

		if (curr_set.size() == 0) {
			return -1;
		}

		// 残差を監視して収束判定した方がいい気もするがとりあえず放置


		// 学習もする想定で重いので注意
		fk	=	param.factory->next(curr_set);

		if (nullptr == fk) {
			return -1;
		}

		weak_learner.push_back(fk);

	}

	return 0;
}
bool	GradientBoost::is_valid_param(const TrainingSet &datas, const GradientBoostParameter &param) const
{
	if (param.shrinkage < std::numeric_limits<double>::min())
		return false;

	if (1.0 < param.shrinkage)
		return false;

	if (param.nr_rounds <= 0)
		return false;

	return true;
}


int GradientBoost::find_initial_model(const TrainingSet &datas, const GradientBoostParameter &param)
{
	// L2ロスで最適なのは平均なので，とりあえず平均を設定しておく．
	datas.compute_target_mean(f0);

	if (f0.empty())
		return -1;

	return 0;
}

void GradientBoost::release()
{
	f0.release();
	weak_learner.clear();
	shrinkage	=	0;
}

TrainingSet GradientBoost::calc_next_target(const TrainingSet &datas, const GradientBoostParameter &param)
{
	TrainingSet	dst(datas.get_feature_type(), datas.get_label_type());

	for (std::size_t ii = 0; ii < datas.size(); ++ii) {
		PtrTrainingExample	data(new TrainingExample);
		RegressionResult	result;

		if (0 != predict(datas[ii]->feature, &result, datas[ii]->param.get()))
			return TrainingSet();

		// 目標ゲクトルだけ更新して他のはヘッダーだけ
		data->feature	=	datas[ii]->feature;
		data->param		=	datas[ii]->param;
		data->target	=	datas[ii]->target - result.get_posterior();

		dst.push_back(data);
	}

	return dst;
}

std::string GradientBoost::get_data_path(const std::string &save_path) const
{
	return save_path + "\\gb_data.txt";
}