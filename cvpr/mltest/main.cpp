#include "..\ml\rf\header\RandomForest.h"
#include "..\util\header\DataSetLoader.h"
#include "..\ml\boosting\header\GradientBoosth.h"
#include "..\ml\factory\header\RandomizedTreeFactory.h"
//#include "..\util\header\PathUtil.h"
#include <random>
#include "DataReader.h"

int main()
{
	std::mt19937 rng(19861124);
	cvpr::TrainingSet datas = DataReader::wine("C:\\git\\cvpr\\datasets\\UCI\\wine\\winequality-white.csv");
	cvpr::TrainingSet tr = datas.random_sample(0.7, rng);
	cvpr::TrainingSet ts = datas.get_out_of_bag(tr);
	/*
	cvpr::RegressionForest rf;
	cvpr::RandomForestParameter	param;

	param.split_type_list = cvpr::RandomForestParameter::default_split_list();

	rf.train(tr, &param);
	*/

	cvpr::GradientBoost	rf;
	cvpr::GradientBoostParameter	param;
	cvpr::StageWiseRegressionTreeFactory	factory;
	cvpr::RegressionTreeParameter	wparam;

	param.factory	=	&factory;
	param.nr_rounds	=	5;
	param.shrinkage	=	0.6;
	
	wparam.split_type_list	=	cvpr::RandomizedTreeParameter::default_split_list();
	factory.set_param(wparam);
	
	if (0 != rf.train(tr, &param))
		puts("ugaaaa");

	double err = 0;

	for (size_t ii = 0; ii < ts.size(); ++ii) {
		cvpr::RegressionResult rr;

		rf.predict(ts[ii]->feature, &rr);

		printf("gr=%f,pr=%f\n", ts[ii]->target.at<double>(0), rr.get_posterior().at<double>(0));
		err += std::abs(rr.get_posterior().at<double>(0, 0) - ts[ii]->target.at<double>(0, 0));
	}

	err /= (double)ts.size();

	printf("err=%f", err);

	int row = 9;

	/*
	cvpr::TrainingSet tr = libSVMDatasetLoader::Load("..\\..\\datasets\\lib_svm\\svm_guide1\\svmguide1.txt");
	cvpr::TrainingSet ts = libSVMDatasetLoader::Load("..\\..\\datasets\\lib_svm\\svm_guide1\\svmguide1.t");

	cvpr::ClassificationForest cf;
	cvpr::RandomForestParameter param;

	param.split_type_list = cvpr::RandomForestParameter::default_split_list();

	cf.train(tr, param);

	cf.save("hoge");
	
	cvpr::ClassificationForest cf2;
	
	cf2.load("hoge");
	
	double cnt = 0;
	double cnt2 = 0;

	for (int ii = 0; ii < ts.size(); ++ii) {
		cvpr::ClassificationResult cr;

		cf.predict(ts[ii]->feature, &cr);

		if (cr.get_max_posterior_idx() == ts[ii]->target_max_idx()) {
			cnt++;
		}
		
		cf2.predict(ts[ii]->feature, &cr);
		
		if (cr.get_max_posterior_idx() == ts[ii]->target_max_idx()) {
			cnt2++;
		}
	}

	FILE *fp = fopen("huga.txt", "w");
	fprintf (fp,"%f\n", cnt / ts.size());
	fprintf (fp,"%f\n", cnt2 / ts.size());
	*/
	return 0;
}