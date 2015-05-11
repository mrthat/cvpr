#pragma once
#include <random>
#include "LinearRegression.h"

namespace cvpr
{
	/**
	*	���W�X�e�B�b�N��A�̃p�����[�^�N���X
	*/
	class LogisticRegressionParameter : public LinearModelParameterBase
	{		
	};

	/**
	*	2�N���X���ʗp���W�X�e�B�b�N��A
	*/
	class LogisticRegression : public LinearModelBase
	{

		public:

			virtual int		predict(const cv::Mat &feature, PredictionResult *result, const PredictionParameter *param = nullptr);

			/**
			*	���W�X�e�B�b�N��A�̗\�����\�b�h
			*	@param	feature	��x�N�g���̓����x�N�g����z��(=> ����cvt�͂��Ȃ�)
			*	@param	result	�\������ �e�v�f�ɂ��̃N���X�ł���m����������k����x�N�g��
			*/
			int				predict(const cv::Mat &feature, ClassificationResult &result) const;

			/**
			*	���W�X�e�B�b�N��A�̊w�K���\�b�h
			*	2�N���X���ʗp
			*	���t�f�[�^��0,1�̓�l��z��D���x�����������̏ꍇ�́C�e�����Ɨ�����2�N���X���ʂŉ���
			*	@param	train_set	�w�K�f�[�^.�����x�N�g���͗�x�N�g��,���t�f�[�^��2�l
			*	@param	param		�S���ږ��܂��Ă��邱�ƁD�f�t�H���g�l�̎g�p��l�̃`�F�b�N�͂��ĂȂ��D
			*/
			int				train(const TrainingSet &train_set, const LogisticRegressionParameter &param)
			{
				return LinearModelBase::train(train_set, &param);
			};

		protected:

			/**
			*	�P���T���v���ɂ��āC�\�����ʂƃ��x�����瑹�������߂�
			*	@param	data	�����v�Z�Ώۂ̃T���v��
			*	@return	�����l
			*/
			virtual cv::Mat	calc_loss(const PtrTrainingExample data) const;

			/**
			*	�p�����[�^�X�V�ʂ��v�Z����(���ʂ�gd)
			*	@param	train_set	�X�V�ʌv�Z�Ɏg�p����w�K�f�[�^
			*	@param	delta_w		�����x�N�g���ɑ΂���d�݃x�N�g���̍X�V��
			*	@param	delta_w0	�o�C�A�X�̍X�V��
			*/
			virtual void	calc_param_delta(const TrainingSet &train_set, cv::Mat &delta_w, double &delta_w0);
			
			/**
			*	�w�K�̍ŏ��ɏd�݂������_���ɏ���������
			*	@param	train_set	�w�K�f�[�^
			*	@param	rnd			�����G���W��
			*/
			void			init_weight(const TrainingSet &train_set, std::mt19937 &rnd) ;
			
			/**
			*	�����x�N�g������d�݂ƃo�C�A�X�����ăX�J���[�̃��X�|���X�����߂�
			*	@param	feature	�����x�N�g��
			*	@return	�����l
			*/
			virtual cv::Mat	calc_activation(const cv::Mat &feature) const;

		private:

	};

	class SoftMaxRegression : public LogisticRegression
	{
		public:
		protected:

			virtual cv::Mat	calc_activation(const cv::Mat &feature) const;
			
			virtual cv::Mat	calc_loss(const PtrTrainingExample data) const;
	};
};