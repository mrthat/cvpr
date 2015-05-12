#include "..\ml\rf\header\RandomForest.h"
#include "..\util\header\DataSetLoader.h"
//#include "..\util\header\PathUtil.h"



int main()
{


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