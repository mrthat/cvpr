#pragma once
#include "..\..\base\header\StaticalModel.h"

namespace cvpr
{
	class LinearModelParameterBase : public StaticalModelParameter
	{
		public:
			unsigned long	rnd_seed;			/*< �����̃V�[�h */
			std::size_t		max_iter;			/*< �œK���̍ő�J��Ԃ��� */
			double			min_delta;			/*< �œK���̌J��Ԃ��Ԃ̍ŏ��]���l�ω� */
			double			update_rate;		/*< �p�����[�^�X�V�� */
			double			lambda;				/*< ���������̌W�� */
			RegularizeType	regularize_type;	/*< �������̃^�C�v */

			LinearModelParameterBase()
				: rnd_seed(19861124), max_iter(1000), min_delta(0.01),
				  update_rate(0.02), lambda(0.01), regularize_type(REGULARIZE_L2)
				{};

			int	save(const std::string &save_path) const;
			int	load(const std::string &load_path) ;
	};

	class LinearModelBase : public StaticalModel
	{

		public:

			virtual			~LinearModelBase() {};

			virtual int		save(const std::string &save_path) const ;

			virtual int		load(const std::string &load_path);

			virtual int		predict(const cv::Mat &feature, PredictionResult *result, const PredictionParameter *param = nullptr) = 0;

			virtual int		train(const TrainingSet &train_set, const StaticalModelParameter *param);

			/**
			*	���`��A�̊w�K���\�b�h
			*	@param	train_set	�w�K�f�[�^.�����x�N�g���͗�x�N�g��
			*	@param	param		�S���ږ��܂��Ă��邱�ƁD�f�t�H���g�l�̎g�p��l�̃`�F�b�N�͂��ĂȂ��D
			*/
			int				train(const TrainingSet &train_set, const LinearModelParameterBase *param);

		protected:

			/**
			*	�d�݃x�N�g��
			*/
			cv::Mat			weight_;

			/**
			*	�o�C�A�X
			*/
			double			w0_;

			/**
			*	�p�����[�^�X�V�ʂ��v�Z����(���ʂ�gd)
			*	@param	train_set	�X�V�ʌv�Z�Ɏg�p����w�K�f�[�^
			*	@param	delta_w		�����x�N�g���ɑ΂���d�݃x�N�g���̍X�V��
			*	@param	delta_w0	�o�C�A�X�̍X�V��
			*/
			virtual void	calc_param_delta(const TrainingSet &train_set, cv::Mat &delta_w, double &delta_w0) = 0;
			
			/**
			*	�w�K�̍ŏ��ɏd�݂������_���ɏ���������
			*	@param	train_set	�w�K�f�[�^
			*	@param	rnd			�����G���W��
			*/
			virtual void			init_weight(const TrainingSet &train_set, std::mt19937 &rnd) ;
			
			/**
			*	L1������
			*	@param[in,out]	weight	�d�݃x�N�g��.���e�𐳑������ď���������
			*	@param[in,out]	w0		�o�C�A�X.(�g��񂯂ǈꉞ�����ɂ��Ƃ�)
			*	@param			labmda	�ǂ̂��炢���������邩(�傫���Ƃ��X�p�[�X�ɂȂ�)
			*/
			virtual void	regularize_l1(cv::Mat &weight/*I/O*/, double &w0/*I/O*/, double lambda) const;

			/**
			*	L2������
			*	@param[in,out]	weight	�d�݃x�N�g��.���e�𐳑������ď���������
			*	@param[in,out]	w0		�o�C�A�X.(�g��񂯂ǈꉞ�����ɂ��Ƃ�)
			*	@param			labmda	�ǂ̂��炢���������邩
			*/
			virtual void	regularize_l2(cv::Mat &weight/*I/O*/, double &w0/*I/O*/, double lambda) const;
		private:

	};

	class LinearRegressionParameter : public LinearModelParameterBase
	{
	};

	class LinearRegression : public LinearModelBase
	{
		public:
			
			virtual int		predict(const cv::Mat &feature, PredictionResult *result, const PredictionParameter *param = nullptr);

			/**
			*	��A�p��predict
			*	@param	feature	��x�N�g��
			*	@param	result	�\���l
			*	@return	���s���� 0:����, else: error
			*/
			virtual int		predict(const cv::Mat &feature, RegressionResult &result);

			/**
			*	���`��A���w�K����
			*	@param	train_set	�w�K�f�[�^�D�����x�N�g���͗�x�N�g���D���x���͗�x�N�g��
			*	@param	param		�p�����[�^�D�S���ړK���l�Ȃ��ƁD���e�̃`�F�b�N�͂��Ă��Ȃ�
			*/
			int	train(const TrainingSet &train_set, const LinearRegressionParameter &param)
			{
				return LinearModelBase::train(train_set, &param);
			}

		protected:
			
			/**
			*	�p�����[�^�X�V�ʂ��v�Z����(���ʂ�gd)
			*	@param	train_set	�X�V�ʌv�Z�Ɏg�p����w�K�f�[�^
			*	@param	delta_w		�����x�N�g���ɑ΂���d�݃x�N�g���̍X�V��
			*	@param	delta_w0	�o�C�A�X�̍X�V��
			*/
			virtual void	calc_param_delta(const TrainingSet &train_set, cv::Mat &delta_w, double &delta_w0);
			
		private:
			/**
			*	�d�݂ƃo�C�A�X���牞���l�����߂�
			*	@param	feature	��x�N�g��
			*/
			cv::Mat	calc_activation(const cv::Mat &feature) const ;
	};
};