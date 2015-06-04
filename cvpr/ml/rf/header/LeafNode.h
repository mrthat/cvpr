#pragma once
#include <sstream>
#include "TreeNode.h"

namespace cvpr
{
	namespace TreeNode
	{
		/**
		* �N���X���ʖ؂̗t�m�[�h�N���X
		*/
		class LeafNodeClassification : public LeafNodeBase
		{
			public:

				LeafNodeType	leaf_type() const { return LEAF_TYPE_CLASSIFICATION; };

				int				operator()(const cv::Mat &feature, PredictionResult *result)
				{
					// �����o�Ɏ����Ă鎖��m����Ԃ�
					
					result->set_posterior(posterior_);

					return 0;
				}

				int				train(const TrainingSet &train_set)
				{
					train_set.compute_target_mean(posterior_);
					
					this->posterior_	/=	(double)train_set.size();

					return 0;
				}

				virtual int	save(cv::FileStorage &cvfs) const
				{
					LeafNodeBase::save(cvfs);
					
					cv::internal::WriteStructContext	wsc(cvfs, "LNC", cv::FileNode::MAP);

					cv::write(cvfs, "posterior", this->posterior_);

					return 0;
				}

				virtual int	load(cv::FileStorage &cvfs)
				{
					LeafNodeBase::load(cvfs);
					cv::FileNode	target_node	=	cvfs["LNC"];

					cv::read(target_node["posterior"], this->posterior_);

					return 0;
				}

			protected:
				cv::Mat_<double>	posterior_;

		};

		class LeafNodeRegression : public LeafNodeBase
		{
			public:

				LeafNodeType	leaf_type() const { return LEAF_TYPE_REGRESSION; }

				int				operator()(const cv::Mat &feature, PredictionResult *result)
				{
					// �����o�Ɏ����Ă鎖��m����Ԃ�
					RegressionResult	*dst	=	dynamic_cast<RegressionResult*>(result);
					if (nullptr == dst) {
						return -1;
					}

					dst->set_posterior(posterior_);

					return 0;
				}

				int				train(const TrainingSet &train_set)
				{
					// �t�m�[�h�ɓ��B�����f�[�^�̕��ς����Ƃ��Ă���
					// �K�E�X�Ő��蓙����Ȃ�g������
					train_set.compute_target_mean(posterior_);

					this->posterior_	/=	(double)train_set.size();

					return 0;
				}

				virtual int	save(cv::FileStorage &cvfs) const
				{
					LeafNodeBase::save(cvfs);

					cv::internal::WriteStructContext	wsc(cvfs, "LNR", cv::FileNode::MAP);

					cv::write(cvfs, "posterior", this->posterior_);

					return 0;
				}

				virtual int	load(cv::FileStorage &cvfs)
				{
					LeafNodeBase::load(cvfs);
					cv::FileNode	target_node	=	cvfs["LNR"];

					cv::read(target_node["posterior"], this->posterior_);

					return 0;
				}

			protected:

				cv::Mat_<double>	posterior_;
		};
	};
};