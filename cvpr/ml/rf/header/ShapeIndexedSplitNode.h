#pragma once
#include <sstream>
#include "TreeNode.h"
#include "..\..\..\util\include\utils.h"

namespace cvpr
{
	namespace TreeNode
	{	
		//! �`��Q�ƕ����m�[�h�̃p�����[�^
		class ShapeIndexedSplitParameter : public SplitNodeParameterBase
		{
			public:
			//! �`��
			std::vector<cv::Point2f>	shape;

			/**
			*	�����ʒu�ɓK�p����ϊ�
			*	���K�����ꂽ���ψʒu�𐄒�ʒu�ɕϊ�����
			*	2*3�̃A�t�B���ϊ��s��
			*/
			cv::Mat	transform;

			protected:
		};

		//! �`��Q�ƕ����m�[�h�̊w�K�p�����[�^
		class ShapeIndexedTrainParameter : public virtual StaticalModelParameter
		{
			public:

			/**
			*	�f�t�H���g�R���X�g���N�^
			*/
			ShapeIndexedTrainParameter()
				: radius(0.05) {};

			/**
			*	�`��_����̓����_�̔z�u�͈͔��a(pix)
			*/
			double	radius;

			/**
			*	�`��_��
			*/
			std::size_t	num_shape;

			protected:

			virtual int		save(const std::string &save_path) const { return 0; };

			virtual int		load(const std::string &load_path) { return 0; };
		};

		/**
		*	�p�����[�^�œn���ꂽ�`��_����̑��Έʒu���g�p����
		*	split����N���X�D
		*/
		template<typename ty>
		class SplitNodeShapeIndexed : public TreeNode::SplitNodeBase
		{
			public:

			SplitNodeType get_split_type() const { return SPLIT_TYPE_SHAPE_INDEXED; };

			int	save(cv::FileStorage &cvfs) const
			{
				int	ret	=	SplitNodeBase::save(cvfs);

				if (0 != ret)
					return ret;

				cv::write(cvfs, STRINGIZE(shape_index), (int)shape_index);

				cv::write(cvfs, STRINGIZE(offsets), offsets);

				return 0;
			}

			int	load(cv::FileStorage &cvfs)
			{
				int	ret	=	SplitNodeBase::load(cvfs);
				int	tmp	=	0;

				if (0 != ret)
					return ret;

				cv::read(cvfs[STRINGIZE(shape_index)], tmp, 0);

				shape_index	=	(std::size_t)tmp;

				cv::read(cvfs[STRINGIZE(offsets)], offsets);

				return 0;
			}

			protected:

			enum {
				//! �`��X���̓����_��
				NUM_FEATURE_POS	=	2,
			};

			unsigned get_num_attributes() const { return 0; }

			void init_params(const TrainingSet &train_set, const StaticalModelParameter *param, std::mt19937 &rnd)
			{
				// �`��p�����[�^�͑S�f�[�^����������z��Ō����l�l�����Ȃ�

				const ShapeIndexedTrainParameter	*param_	=	dynamic_cast<const ShapeIndexedTrainParameter*>(param);

				assert(nullptr != param_);
				
				std::uniform_int_distribution<int>		distr_idx((int)0, (int)param_->num_shape - 1);
				std::uniform_real_distribution<double>	distr_pos(-param_->radius, param_->radius);

				// �Ώی`��_���߂�
				shape_index	=	distr_idx(rnd);

				// ���Γ����_�ʒu�����߂�
				for (int ii = 0; ii < NUM_FEATURE_POS; ++ii) {
					offsets.push_back(cv::Point2f(distr_pos(rnd), distr_pos(rnd)));
				}
			}

			double kernel_function(const cv::Mat &feature, const SplitNodeParameterBase *param) const
			{
				std::vector<cv::Point2f>	warped_offsets;
				std::vector<cv::Point2f>	feature_pos;
				cv::Mat	transform;
				const ShapeIndexedSplitParameter	*param_	=	dynamic_cast<const ShapeIndexedSplitParameter*>(param);
				std::vector<double>	val;

				assert(nullptr != param_);

				cv::transform(offsets, warped_offsets, param_->transform);

				for (std::size_t ii = 0; ii < warped_offsets.size(); ++ii) {
					cv::Point2f	pos	=	param_->shape[shape_index] + warped_offsets[ii];

					pos	=	cvpr::round<float>(feature, pos); // �͈͊O�Ȃ��O�ł����C������?

					val.push_back((double)feature.at<ty>((int)pos.y, (int)pos.x));
				}

				return val[0] - val[1];
			}

			//! �g�p����`��_�̃C���f�b�N�X
			std::size_t	shape_index;

			//! �`�󂩂�̃I�t�Z�b�g �`��f�[�^��2d��z��
			std::vector<cv::Point2f>	offsets;

		};
	};
};