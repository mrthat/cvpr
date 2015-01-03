#define _CRT_SECURE_NO_WARNINGS 1

#include <fstream>
#include <sstream>
#include "..\..\..\util\header\utils.h"
#include "..\header\DataSetLoader.h"

using namespace cvpr;

TrainingSet
LibSVMDatasetLoader::load_classification_data(const std::string &fname) 
{
	std::vector<Sample>	samples;

	load(fname, samples);

	return samples_to_data(samples);
}

TrainingSet
LibSVMDatasetLoader::load_regression_data(const std::string &fname) 
{
	std::vector<Sample>	samples;

	load(fname, samples);

	return samples_to_data(samples, CVT_REGRESSION);
}

TrainingSet	
LibSVMDatasetLoader::samples_to_data(const std::vector<Sample> &samples, int cvt_code)
{
	int	dim_max		=	0;
	int	label_max	=	0;
	MatType	ftype;
	MatType	ltype;

	// �T���v���z�񂩂�C�����x�N�g����index�̍ő�l�ƃ��x���̍ő�l��������
	for (auto ii = samples.begin(); ii != samples.end(); ++ii) {
		dim_max	=	std::max(dim_max, ii->feature.back().dim);
		label_max	=	std::max<int>(label_max, (int)ii->label);
	}

	// �ő�l���ُ�Ȃ��ŕԂ�
	if (dim_max <= 0) {
		return TrainingSet(ftype, ltype);
	} 

	if (label_max < 0 && CVT_REGRESSION == cvt_code) {
		return TrainingSet(ftype, ltype);
	}

	// �ő�l����w�K�Z�b�g�̃w�b�_�[���
	ftype	=	MatType(cv::Size(1, dim_max), CV_64FC1);

	if (CVT_CLASSIFICATION == cvt_code) {
		ltype	=	MatType(cv::Size(1, label_max+1), CV_64FC1);
	} else {
		ltype	=	MatType(cv::Size(1, 1), CV_64FC1);
	}
	
	TrainingSet	tset(ftype, ltype);

	// �T���v���z����Ȃ߂ČP���T���v���������ăZ�b�g�ɋl�߂�
	for (auto ii = samples.begin(); ii != samples.end(); ++ii) {
		PtrTrainingExample	example(new TrainingExample);

		example->label.create(ltype.size(), ltype.data_type);
		example->feature.create(ftype.size(), ftype.data_type);

		example->label		=	0;
		example->feature	=	0;

		if (CVT_CLASSIFICATION == cvt_code) {
			example->label.at<double>((int)(ii->label), 0)	=	1;
		} else {
			example->label.at<double>(0, 0)	=	ii->label;
		}

		for (auto jj = ii->feature.begin(); jj != ii->feature.end(); ++jj) {
			example->feature.at<double>(jj->dim-1, 0)	=	jj->val;
		}

		tset.push_back(example);
	}

	return tset;
}

bool
LibSVMDatasetLoader::load(const std::string &fname, std::vector<Sample> &samples) 
{
	std::ifstream	in_file(fname);

	samples.clear();

	if (!in_file) {
		return false;
	}

	// �����܂ōs���ɓǂ�
	// �e�s�T���v���ɕϊ����ăX�^�b�N�ɓ����
	while (!in_file.eof()) {
		std::string	line_buff;
		Sample	sample ;

		std::getline(in_file, line_buff);
		
		if (!line_buff_to_sample(line_buff, sample)) {
			continue;
		}

		samples.push_back(sample);
	}

	return false;
}

bool	
LibSVMDatasetLoader::line_buff_to_sample(const std::string &buff, Sample &sample) 
{
	std::vector<std::string>	splitted_buff;

	// �X�y�[�X��split���ėv�f���Ƃɕ�����
	split(buff, ' ', splitted_buff);

	// �v�f��2�ȏ�(�Œ�ł����x��+����1��)
	if (splitted_buff.size() < 2) {
		return false;
	}

	// ���x���𐔒l�ɂ���
	if (1 != sscanf(splitted_buff[0].c_str(), "%lf", &(sample.label))) {
		return false;
	}

	// �����𐔒l�ɂ���
	for (auto ii = splitted_buff.begin() + 1; ii != splitted_buff.end(); ++ii) {
		FeatureElem	elem;
		if (2 != sscanf(ii->c_str(), "%d:%lf", &(elem.dim), &(elem.val))) {
			continue;
		}
		sample.feature.push_back(elem);
	}

	return true;
}
