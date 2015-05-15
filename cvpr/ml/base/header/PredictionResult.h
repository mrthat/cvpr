#pragma once
#include <memory>
#include <vector>
#include <cmath>
#include <opencv2\core\core.hpp>
#include "utils.h"


namespace cvpr
{
	/**
	*	���ʌ��ʂ̃^�C�v
	*/
	typedef enum {
		RESULT_TYPE_CLASSIFICATION,			//< �N���X����
		RESULT_TYPE_REGRESSION,				//< ��A
	} ResultType;

	/**
	*	���ʌ��ʃx�[�X�N���X
	*/
	class PredictionResult
	{
		public:

			/**
			*	�\�����ʍs���setter
			*/
			void	set_posterior(const cv::Mat &posterior) 
			{
				this->posterior_	=	posterior;
			}

			/**
			*	�\�����ʍs���getter
			*/
			const cv::Mat	get_posterior() const
			{
				return this->posterior_;
			}

			/**
			*	���ʌ��ʂ̃^�C�v���擾
			*/
			virtual ResultType	type() const = 0;

			virtual				~PredictionResult() {};

		protected:
			/** �\�����ʍs�� */
			cv::Mat_<double>	posterior_;
	};

	/**
	*	�N���X���ʌ��ʃN���X
	*/
	class ClassificationResult : public PredictionResult
	{
		public:

			/**
			*	���ʌ��ʂ̃^�C�v���擾
			*/
			virtual cvpr::ResultType	type() const { return RESULT_TYPE_CLASSIFICATION; };
			
			/**
			*	posterior_���̍ő�v�f��idx���擾
			*	@return	label���̍ő�v�f��idx
			*/
			int	label() const 
			{
				return max_idx(posterior_);
			}

		protected:
	};
	
	/**
	*	��A�\�����ʃN���X
	*/
	class RegressionResult : public PredictionResult
	{
		public:
			
			/**
			*	���ʌ��ʂ̃^�C�v���擾
			*/
			virtual cvpr::ResultType	type() const { return RESULT_TYPE_REGRESSION; };

		protected:
	};
	
	typedef std::shared_ptr<PredictionResult> PtrPredictionResult ;

	/**
	*	�\�����ʂ�factory�N���X
	*/
	class PredictionResultFactory
	{
		public:
			/**
			*	�\�����ʎ�ʂ���\�����ʂ𐶐����ĕԂ�
			*	@param	type	�\�����ʎ��
			*	@return	null:�m��Ȃ�type, ���̑�:�Ή�����N���X�̃C���X�^���X
			*/
			static PtrPredictionResult	create(ResultType type) ;

		protected:
	};
};