#pragma once
#include "PredictionResult.h"
#include "TrainingData.h"
#include "MlDefines.h"

namespace cvpr
{

	/** �p�����[�^�x�[�X�N���X */
	class StaticalModelParameter
	{
		public:
			/** �������̃^�C�v */
			typedef enum { REGULARIZE_L2, REGULARIZE_L1, REGULARIZE_NONE } RegularizeType;
			
			virtual	~StaticalModelParameter() {};
			
			/**
			*	�p�����[�^�̃t�@�C���o��
			*	@param	save_path	�o�͐�̃p�X
			*	@return	�G���[�R�[�h��
			*/
			virtual int		save(const std::string &save_path) const	=	0;

			/**
			*	�p�����[�^�̃t�@�C������
			*	@param	load_path	���͌��̃p�X
			*	@return	�G���[�R�[�h��
			*/
			virtual int		load(const std::string &load_path)	=	0;

		protected:
	};

	/** ���ʊ�x�[�X�N���X */
	class StaticalModel
	{
		public:
			virtual			~StaticalModel(){};
			
			/**
			*	�p�����[�^�̃t�@�C���o��
			*	@param	save_path	�o�͐�̃p�X
			*	@return	�G���[�R�[�h��
			*/
			virtual int		save(const std::string &save_path) const	=	0;
			
			/**
			*	�p�����[�^�̃t�@�C������
			*	@param	load_path	���͌��̃p�X
			*	@return	�G���[�R�[�h��
			*/
			virtual int		load(const std::string &load_path)	=	0;

			/**
			*	�����x�N�g������\�����ʂ��o��
			*	@param	feature	���͂̓����x�N�g��
			*	@param	result	�o�̗͂\������
			*/
			virtual int		predict(const cv::Mat &feature, PredictionResult *result)	=	0;

			/**
			*	�w�K�����s����
			*	@param	train_set	�w�K�f�[�^
			*	@param	param		�w�K�p�����[�^
			*/
			virtual int		train(const TrainingSet &train_set, const StaticalModelParameter *param)	=	0;

			/**
			*	�N���X�̌^����Ԃ�
			*	@return	���v���f���̎��ʎq
			*/
			virtual StatModelType get_type() const { return StatModelType::STAT_MODEL; };

		protected:

		private:
	};

	//! �㎯�ʊ�p�����[�^�N���X (�s�v������)
	class WeakLearnerParameter : public StaticalModelParameter
	{
	};

	//! �㎯�ʊ�x�[�X�N���X
	class WeakLearner : public StaticalModel
	{
	};

	typedef std::shared_ptr<StaticalModel> PtrStaticalModel;
	typedef std::shared_ptr<WeakLearner> PtrWeakLearner;
	typedef std::shared_ptr<WeakLearnerParameter> PtrWeakLearnerParam;
};