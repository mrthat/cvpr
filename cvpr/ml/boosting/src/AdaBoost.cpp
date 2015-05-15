//#include <tbb/task_scheduler_init.h>
//#include <tbb/parallel_for.h>
#include "..\header\AdaBoost.h"
#include <numeric>
#include <direct.h>

using namespace cvpr;

void AdaBoost::clear()
{
	this->sum_alpha_ = 0;
	this->alpha_t_.clear();
	this->weak_classifiers_.clear();
}

int AdaBoost::train(const TrainingSet &train_set, const StaticalModelParameter *param)
{
	const AdaboostParameter *abc_param = static_cast<const AdaboostParameter*>(param);

	if (abc_param == nullptr) {
		return -1;
	}
	
	clear();

	cv::RNG rng(abc_param->seed);

	return train(train_set, *abc_param, rng);
}

double AdaBoost::evaluate(const TrainingSet &datas, const cv::Mat &weight, PtrWeakLearner &model) 
{
	cv::Mat	errs(datas.size(), 1, CV_64FC1);

	errs	=	0.0;

	for (std::size_t ii = 0; ii < datas.size(); ++ii) {
		ClassificationResult	result;
		int	tlabel	=	datas[ii]->label();
		double	err	=	0.0;

		model->predict(datas[ii]->feature, &result);

		errs.at<double>(ii)	=	1.0 - result.get_posterior().at<double>(tlabel);
	}

	return errs.dot(weight);
}

int AdaBoost::train(const TrainingSet &datas, const AdaboostParameter &param, cv::RNG &rng)
{
	cv::Mat	sample_weights(datas.size(), 1, CV_64FC1, 1.0);

	for (unsigned int round = 0; round < param.nr_rounds; ++round) {
		double	alpha	=	0;
		std::vector<PtrWeakLearner>		pool;
		std::vector<WeakLearnerentry>	eval_results;
		WeakLearnerentry				*best_learner	=	nullptr;
		cv::Mat							tmp_weights;

		// 弱識別器のプールを作る．
		// ラウンド毎に毎回プールを作って選ぶストラテジー
		// 最初に作って使いまわす場合には多少修正がいる
		// 毎回作るならサンプル重みなど入力できるようにした方がいいかもしれん
		pool	=	param.factory->create_trained_pool(datas, param.nr_weak_learners);
		eval_results.resize(pool.size());

		sample_weights	=	sample_weights / cv::sum(sample_weights)[0];

		// 弱識別器を評価 -> 評価値の最も高いものを取得
		for (std::size_t ii = 0; ii < pool.size(); ++ii) {
			eval_results[ii].learner	=	pool[ii];
			// 派生ブースティングを作る場合はここのevaluateと下のupdate_sample_weightを変えればいけそう(?)
			eval_results[ii].eval		=	evaluate(datas, sample_weights, pool[ii]);
		}

		std::sort(eval_results.begin(), eval_results.end(), [](const WeakLearnerentry &lhs, const WeakLearnerentry &rhs) {
			return lhs.eval < rhs.eval;
		});

		best_learner	=	&eval_results.front();

		printf("%dth round best_error = %f\n", round, best_learner->eval);

		// 識別エラーが十分小さいなら帰る
		if (param.min_error_rate < best_learner->eval)
			break;

		// 弱識別器の重みを計算 -> サンプル重みを更新
		alpha	=	calc_alpha(best_learner->eval);
		update_sample_weights(datas, sample_weights, best_learner->learner, alpha, tmp_weights);

		sample_weights	=	tmp_weights;

		// メンバに追加
		this->alpha_t_.push_back(alpha);
		this->weak_classifiers_.push_back(best_learner->learner);
	}

	this->sum_alpha_ = std::accumulate(this->alpha_t_.begin(), this->alpha_t_.end(), 0.0);

	return 0;
}


void AdaBoost::update_sample_weights(const TrainingSet &datas, const cv::Mat &src_weight,
	PtrWeakLearner &model, double alpha, cv::Mat &dst_weight) 
{
	dst_weight.create(src_weight.size(), CV_64FC1);

	for (std::size_t ii = 0; ii < datas.size(); ++ii) {
		ClassificationResult	result;
		int	label	=	-1;

		model->predict(datas[ii]->feature, &result);

		label	=	result.label();

		// おそらく派生ブースト作る場合はここの式が変わる
		if (label == datas[ii]->label()) {
			dst_weight.at<double>(ii)	=	src_weight.at<double>(ii) * std::exp(-alpha);
		} else {
			dst_weight.at<double>(ii)	=	src_weight.at<double>(ii) * std::exp(alpha);
		}
	}
}

int AdaBoost::save(const std::string &save_path) const
{
	std::string		dname(save_path);
	cv::FileStorage	cvfs;
	std::vector<int>	model_types;

	if (dname.back() != '\\') {
		dname += '\\';
	}

	_mkdir(save_path.c_str());

	cvfs.open(dname + "model", cv::FileStorage::WRITE);

	if (!cvfs.isOpened())
		return -1;

	cv::write(cvfs, STRINGIZE(alpha_t_), alpha_t_);

	for (std::size_t ii = 0; ii < weak_classifiers_.size(); ++ii) {
		char	sub_path[_MAX_PATH];
		int		code	=	0;

		sprintf(sub_path, "%sweak%d", dname.c_str(), ii);
		code	=	weak_classifiers_[ii]->save(sub_path);
		
		if (0 != code)
			return code;

		model_types.push_back(weak_classifiers_[ii]->get_type());
	}

	cv::write(cvfs, "types", model_types);

	return 0;
}

int AdaBoost::load(const std::string &load_path) 
{
	std::string	dname(load_path);
	std::vector<int>	model_types;

	weak_classifiers_.clear();

	if (dname.back() != '\\') {
		dname += '\\';
	}

	cv::FileStorage cvfs(dname + "model", cv::FileStorage::READ);

	cv::read<double>(cvfs[STRINGIZE(alpha_t_)], alpha_t_, std::vector<double>());
	cv::read(cvfs["types"], model_types);

	sum_alpha_ = std::accumulate(this->alpha_t_.begin(), this->alpha_t_.end(), 0.0);

	for (std::size_t ii = 0; ii < model_types.size(); ++ii) {
		char sub_dir[256];
		PtrWeakLearner	wl	=	WeakLearnerFactory::create((StatModelType)model_types[ii]);

		if (nullptr == wl) {
			return -1;
		}

		sprintf(sub_dir, "%sweak%d", dname.c_str(), ii);
		
		int	stat	=	wl->load(sub_dir);
		if (0 != stat)
			return stat;
		weak_classifiers_.push_back(wl);
	}

	return 0;
}


int AdaBoost::predict(const cv::Mat &feature, PredictionResult *result, const PredictionParameter *param)
{
	ClassificationResult	*cresult	=	dynamic_cast<ClassificationResult*>(result);

	if (nullptr == cresult)
		return -1;

	if (weak_classifiers_.empty())
		return -1;

	std::vector<cv::Mat>	pr;
	cv::Mat	avg;

	// 弱識別器全部で識別する
	for (std::size_t ii = 0; ii < weak_classifiers_.size(); ++ii) {
		ClassificationResult	tmp;
		
		weak_classifiers_[ii]->predict(feature, &tmp);
		pr.push_back(tmp.get_posterior());
	}

	// 加重平均して結果に詰める
	avg	=	pr.front() * alpha_t_.front() / sum_alpha_;

	for (std::size_t ii = 1; ii < pr.size(); ++ii) {
		avg	+=	pr[ii] * alpha_t_[ii] / sum_alpha_;
	}

	cresult->set_posterior(avg);

	return 0;
}