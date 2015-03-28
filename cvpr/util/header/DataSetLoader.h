#pragma once
#include <fstream>
#include <string>
#include <opencv2\highgui\highgui.hpp>
#if 0
#include "..\..\ml\base\header\TrainingData.h"
/*
class MNISTDigitImagLoader
{
	public:
		enum {DIGIT_HEIGHT = 28, DIGIT_WIDTH = 28};
		enum {NUM_DIGIT_CATEGORY = 10};
		std::vector<cv::Mat>	GetImgs(unsigned digit_id) const
		{
			return this->imgs[digit_id];
		}

		cvpr::TrainingSet	GetTrainSet() const
		{
			cvpr::TrainingSet	train_set(cv::Size(DIGIT_WIDTH, DIGIT_HEIGHT), CV_8UC1, cv::Size(1, NUM_DIGIT_CATEGORY), CV_64FC1);
			for (unsigned ii = 0; ii < NUM_DIGIT_CATEGORY; ++ii) {
				std::vector<cv::Mat>	target_digit	=	GetImgs(ii);
				for (unsigned jj = 0; jj < target_digit.size(); ++jj) {
					cvpr::TrainingExample	*example	=	new cvpr::TrainingExample;
					target_digit[jj].copyTo(example->feature);
					example->target.create(NUM_DIGIT_CATEGORY, 1, CV_64FC1);
					example->target = 0;
					example->target.at<double>(ii, 0)	=	1.0;

					if (train_set.AddExample(example) != 0) {
						delete example;
					}
				}
			}

			return train_set;
		}

		std::vector<cv::Mat>	GetImgs() const
		{
			std::vector<cv::Mat>	all_imgs;

			for (unsigned ii = 0; ii < this->imgs.size(); ++ii) {
				all_imgs.insert(all_imgs.end(), this->imgs[ii].begin(), this->imgs[ii].end());
			}

			return all_imgs;
		}
		static cv::Size	GetDigitSize()
		{
			return cv::Size(DIGIT_WIDTH, DIGIT_HEIGHT);
		}
		int	Load(const std::string &list_path)
		{
			std::ifstream	list(list_path);
		

			if (list.bad()) {
				return -1;
			}

			this->imgs.resize(NUM_DIGIT_CATEGORY);
			
			for (int ii = 0; ii < NUM_DIGIT_CATEGORY; ++ii) {
				std::string					line_buff;
				std::vector<std::string>	splitted;
				cv::Mat	big_img;
				std::string::size_type	pos;
				
				std::getline(list, line_buff);

				pos	=	line_buff.find("\n", 0);

				if (pos != std::string::npos) {
					line_buff[pos]	=	'\0';
				}

				big_img	=	cv::imread(line_buff, 0);

				this->imgs[ii]	=	SplitBigImg(big_img);

			}			

			return 0;
		}
	protected:
		
		std::vector<std::vector<cv::Mat> >	imgs;
		
		std::vector<cv::Mat>	SplitBigImg(const cv::Mat &big)
		{
			std::vector<cv::Mat>	imgs;
			unsigned	num_row		=	big.rows / DIGIT_HEIGHT;
			unsigned	num_cols	=	big.cols / DIGIT_WIDTH;

			for (unsigned hh = 0; hh < big.rows; hh += DIGIT_HEIGHT) {
				for (unsigned ww = 0; ww < big.cols; ww += DIGIT_WIDTH) {
					cv::Rect	roi(ww, hh, DIGIT_WIDTH, DIGIT_HEIGHT);
					cv::Mat		img	=	big(roi).clone();

					imgs.push_back(img);
				}
			}

			return imgs;
		}
};
*/
class libSVMDatasetLoader
{
	public:
		static cvpr::TrainingSet	Load(std::string fname)
		{
			cvpr::MatType	src_type, dst_type;
			std::ifstream	data_file(fname);
			int				din		=	0;
			int				dout	=	0;

			if (data_file.bad()) {
				return cvpr::TrainingSet(src_type, dst_type);
			}

			din		=	SeekNumInputDims(fname);
			dout	=	SeekNumOutDims(fname);

			printf("%d,%d", din, dout);

			if (din	<= 0 || dout <= 0) {
				return cvpr::TrainingSet(src_type, dst_type);
			}

			src_type	=	cvpr::MatType(din, 1, CV_64FC1);
			dst_type	=	cvpr::MatType(dout, 1, CV_64FC1);
			

			cvpr::TrainingSet loaded_set(src_type, dst_type);
	
			while (!data_file.eof()) {
				std::string					line_buff;
				std::vector<std::string>	splitted;
				cvpr::PtrTrainingExample	example	=	cvpr::PtrTrainingExample(new cvpr::TrainingExample);
				int							cvt_result;

				std::getline(data_file, line_buff);

				cvt_result = DataLine2TrainingExample(line_buff, loaded_set.get_feature_type().total(), loaded_set.get_label_type().total(), *example);

				if ( -1 != cvt_result) {
					loaded_set.push_back(example);
				}
						
			}

			return loaded_set;

		}
		static int					SeekNumInputDims(std::string fname);
		static int					SeekNumOutDims(std::string fname);
	protected:
		static int					DataLine2TrainingExample(const std::string &data_str, unsigned dim_in, unsigned dim_out, cvpr::TrainingExample &example)
		{
			std::vector<std::string>	splitted	=	Split(data_str, ' ');
			unsigned					temp_idx;
			float						temp_val;

			if (splitted.empty()) {
				return -1;
			}

			example.feature.create(dim_in, 1, CV_64FC1);

			for (unsigned ii = 1; ii < splitted.size(); ++ii) {
				sscanf(splitted[ii].c_str(), "%d:%f", &temp_idx, &temp_val);
				if (dim_in < temp_idx) {
					continue;
				}
				example.feature.at<double>(temp_idx - 1)	=	temp_val;
			}	

			example.target.create(dim_out, 1, CV_64FC1);
			example.target	=	0;

			sscanf(splitted[0].c_str(), "%d", &temp_idx);
			if (dim_out <= temp_idx) {
				return 1;
			}

			example.target.at<double>(temp_idx)	=	1;

			return 0;
		}


		static std::vector<std::string>	Split(const std::string &str, char delim);
};
#endif