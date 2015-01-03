#pragma once
#include <sstream>
#include "TreeNode.h"

namespace cvpr
{
	namespace TreeNode
	{
		/**
		* クラス識別木の葉ノードクラス
		*/
		class LeafNodeClassification : public LeafNodeBase
		{
			public:

				LeafNodeType	leaf_type() const { return LEAF_TYPE_CLASSIFICATION; };

				int				operator()(const cv::Mat &feature, PredictionResult *result)
				{
					// メンバに持ってる事後確率を返す
					ClassificationResult	*dst	=	dynamic_cast<ClassificationResult*>(result);
					if (nullptr == dst) {
						return -1;
					}

					dst->set_posterior(posterior_);

					return 0;
				}

				int				train(const TrainingSet &train_set)
				{
					this->posterior_	=	train_set.calc_label_sum();

					this->posterior_	/=	(double)train_set.size();

					return 0;
				}

				virtual int	save(cv::FileStorage &cvfs) const
				{
					LeafNodeBase::save(cvfs);
					
					cv::internal::WriteStructContext	wsc(cvfs, "LNC", cv::FileNode::MAP);

					//cv::write(*wsc.fs, "posterior", this->posterior_);

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

				cv::Mat	GetPosteriror() const
				{
					return this->posterior_.clone();
				}

				void	SetPosterior(const cv::Mat &posterior) 
				{
					posterior.copyTo(this->posterior_);
				}

			protected:
				cv::Mat_<double>	posterior_;

		};
	};
};