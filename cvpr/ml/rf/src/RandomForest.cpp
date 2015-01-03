#define _CRT_SECURE_NO_WARNINGS 1
#include "..\header\RandomForest.h"

using namespace cvpr;

int	RandomForest::train(const cvpr::TrainingSet &train_set, const cvpr::RandomForestParameter &param)
{
	std::mt19937	rng(param.rng_seed);

	trees_.resize(param.num_trees);

	for (unsigned ii = 0; ii < param.num_trees; ++ii) {

		this->trees_[ii]	=	RandomizedTreeFactory::Create(tree_type());
		if (nullptr == this->trees_[ii]) {
			return -1;
		}
	}

	std::vector<const TrainingSet>	train_sets;
	std::vector<const TrainingSet>	oob_sets;
	std::vector<const StaticalModelParameter*>	params;
	std::vector<int>	train_results(param.num_trees, 0);
#ifdef PARALLEL_TRAIN
	for (unsigned ii = 0; ii < param.num_trees; ++ii) {
		TrainingSet	train_subset	=	train_set.RandomSample(param.resample_rate, rng);
		TrainingSet	oob_set			=	train_set.GetOutOfBag(train_subset);
		RandomForestParameter	*param_subset	=	new RandomForestParameter(param);

		param_subset->rng_seed		=	rng();

		train_sets.push_back(train_subset);
		oob_sets.push_back(oob_set);
		params.push_back(param_subset);
	}

	tbb::task_scheduler_init	init;

	ParallelTrain	para_train(train_sets, oob_sets,
								params, this->trees_, train_results,
								param.num_retrain, param.min_oob_error);
				
	tbb::parallel_for(tbb::blocked_range<unsigned>(0, param.num_trees),
		para_train, tbb::auto_partitioner());
					
	init.terminate();

	for (unsigned ii = 0; ii < params.size(); ++ii) {
		delete params[ii];
	}
#else
				
	for (unsigned ii = 0; ii < this->trees_.size(); ++ii) {
		TrainingSet	train_subset	=	train_set.random_sample(param.resample_rate, rng);
		TrainingSet	oob_set			=	train_set.get_out_of_bag(train_subset);
		
		RandomForestParameter	param_subset(param);
		param_subset.rng_seed		=	rng();

		int training_result	=	this->trees_[ii]->train(train_subset, &param_subset);
		
		if (0 != training_result) {
			return -1;
		}
	}
				
#endif
	return 0;
}

int	RandomForest::train(const cvpr::TrainingSet &train_set, const cvpr::StaticalModelParameter *param)
{
	const RandomForestParameter	*random_forest_param	=	dynamic_cast<const RandomForestParameter*>(param);
	
	if (nullptr == random_forest_param) {
		return -1;
	}

	return train(train_set, *random_forest_param);
}

int	RandomForest::save(const std::string &save_path) const
{
	puts("random forest:save");
	
	_mkdir(save_path.c_str());

	std::string	tree_type_path	=	save_path;
	
	tree_type_path.append("\\forest_data.txt");
	
	save_tree_type(tree_type_path);

	for (unsigned ii = 0; ii < this->trees_.size(); ++ii) {
	
		char	tree_file_path[FILENAME_MAX];
		int		save_result;
		
		sprintf(tree_file_path, "%s\\tree%d", save_path.c_str(), ii);
		
		save_result	=	this->trees_[ii]->save(tree_file_path);
		
		if (save_result != 0) {
			return save_result;
		}
	}

	return 0;
}

int	RandomForest::load(const std::string &load_path)
{
	
	puts("random forest:load");

	trees_.clear();

	std::vector<TreeType>	tree_types;
	std::string				tree_type_path	=	load_path;
	
	tree_type_path.append("\\forest_data.txt");
	
	load_tree_type(tree_type_path, tree_types);
	
	this->trees_.assign(tree_types.size(), nullptr);

	for (unsigned ii = 0; ii < this->trees_.size(); ++ii) {
		
		char	tree_file_path[FILENAME_MAX];
		int		load_result;

		sprintf(tree_file_path, "%s\\tree%d", load_path.c_str(), ii);
		
		this->trees_[ii]	=	RandomizedTreeFactory::Create(tree_types[ii]);
		
		load_result	=	this->trees_[ii]->load(tree_file_path);

		if (load_result != 0) {
			return load_result;
		}

	}

	return 0;
}

int	RandomForest::predict(const cv::Mat &feature, cvpr::PredictionResult *result)
{
	//‘Stree‚Ìpredict‚ðŒÄ‚Ô
	std::vector<PtrPredictionResult>	results(this->trees_.size(), nullptr);
	bool	is_prediction_succeed	=	true;

	for (std::size_t ii = 0; ii < this->trees_.size(); ++ii) {
		results[ii]	=	PredictionResultFactory::create(result_type());
		int	predict_result	=	this->trees_[ii]->predict(feature, results[ii].get());
		if (predict_result < 0) {
			is_prediction_succeed	=	false;
		}
	}

	if (!is_prediction_succeed) {
		return -1;
	}

	//•½‹Ï
	merge_results(results, result);

	return 0;
}

int	RandomForest::add_tree(PtrRandomizedTree tree)
{
	if (nullptr == tree) {
		return -1;
	}

	if (tree->tree_type() != tree_type()) {
		return -1;
	}

	this->trees_.push_back(tree);
	
	return 0;
}

int	RandomForest::merge(const RandomForest *forest)
{
	
	if (this->tree_type() != forest->tree_type()) {
		return -1;
	}

	for (std::size_t ii = 0; ii < forest->trees_.size(); ++ii) {
		this->trees_.push_back(forest->trees_[ii]);
	}

	return 0;
}

int	RandomForest::save_tree_type(const std::string &data_path) const
{
	std::ofstream	data_file(data_path);

	if (data_file.bad()) {
		return -1;
	}

	data_file << "num_trees=" << this->trees_.size() << std::endl;
	for (unsigned ii = 0; ii < this->trees_.size(); ++ii) {
		data_file << "tree_type=" << tree_type() << std::endl;
	}

	return 0;
}

int	RandomForest::load_tree_type(const std::string &data_path, std::vector<cvpr::TreeType> &tree_types) const
{

	std::ifstream	data_file(data_path);
	std::string		line_buff;
	unsigned		num_trees	=	0;
	
	tree_types.clear();

	if (data_file.bad()) {
		return -1;
	}

	std::getline(data_file, line_buff);

	if (1 != sscanf(line_buff.c_str(), "num_trees=%d", &num_trees)) {
		return -1;
	}

	tree_types.assign(num_trees, TREE_TYPE_UNKNOWN);

	for (unsigned ii = 0; ii < num_trees && (!data_file.eof()); ++ii) {
		std::getline(data_file, line_buff);

		if (1 != sscanf(line_buff.c_str(), "tree_type=%d", &tree_types[ii])) {
			return -1;
		}
	}

	return 0;
}

void	ClassificationForest::merge_results(const std::vector<PtrPredictionResult> &results, PredictionResult *dst) 
{
	cv::Mat	average;

	if (results.empty()) {
		return ;
	}

	average	=	results.front()->get_posterior().clone();

	for (std::size_t ii = 1; ii < results.size(); ++ii) {
		average	+=	results[ii]->get_posterior();
	}

	average	/=	(double)results.size();

	dst->set_posterior(average);
};

