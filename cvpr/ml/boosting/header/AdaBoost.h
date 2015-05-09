#pragma once
//#include <tbb\blocked_range.h>
#include "..\..\base\header\StaticalModel.h"
#include "..\..\factory\header\StatModelFactory.h"
namespace cvpr
{
	//! adaboost�p�����[�^�N���X
	class AdaboostParameter : public StaticalModelParameter
	{
		public:

			int		save(const std::string &save_path) const;

			int		load(const std::string &load_path) ;

			//! �����ł��؂�p�덷�ŏ��l
			double min_error_rate;

			//! ���E���h�� => �I������㎯�ʊ�̐�
			unsigned int nr_rounds;

			//! �I�����̎㎯�ʃv�[���̑���
			unsigned int nr_weak_learners;

			//! �����̃V�[�h
			uint64 seed;
			
			//! �㎯�ʊ�t�@�N�g���[
			mutable WeakLearnerPoolFactoryBase *factory;

		protected:
	};

	//! adaboost�N���X
	class AdaBoost : public StaticalModel
	{
		public:

			/**
			*	�p�����[�^�̃t�@�C���o��
			*	@param	save_path	�o�͐�̃p�X
			*	@return	�G���[�R�[�h��
			*/
			virtual int		save(const std::string &save_path) const ;
			
			/**
			*	�p�����[�^�̃t�@�C������
			*	@param	load_path	���͌��̃p�X
			*	@return	�G���[�R�[�h��
			*/
			virtual int		load(const std::string &load_path) ;

			/**
			*	�����x�N�g������\�����ʂ��o��
			*	@param	feature	���͂̓����x�N�g��
			*	@param	result	�o�̗͂\������
			*/
			virtual int		predict(const cv::Mat &feature, PredictionResult *result);

			/**
			*	�w�K�����s����
			*	@param	train_set	�w�K�f�[�^
			*	@param	param		�w�K�p�����[�^
			*/
			virtual int		train(const TrainingSet &train_set, const StaticalModelParameter *param);

			/**
			*	�N���X�̌^����Ԃ�
			*	@return	���v���f���̎��ʎq
			*/
			virtual StatModelType get_type() const { return StatModelType::ADABOOST; };

		protected:
			
			/**
			*	�㎯�ʊ�̌��
			*/
			struct WeakLearnerentry {
				//! �㎯�ʊ�
				PtrWeakLearner	learner;

				//! �]���l
				double			eval;
			};

			/**
			*	���݂̃��f����j������
			*/
			void clear();

			/**
			*	�w�K�f�[�^�Əd�݂��g�p���Ď㎯�ʊ��]������
			*	@param	datas	�w�K�f�[�^
			*	@param	weight	�f�[�^�̏d��
			*	@param	model	�]���Ώۂ̃��f��
			*	@return	�]���l
			*/
			double evaluate(const TrainingSet &datas, const cv::Mat &weight, PtrWeakLearner &model) ;

			/**
			*	�w�K�T���v���̏d�݂��X�V����
			*	@param	datas		�w�K�T���v��
			*	@param	src_weight	���݃T���v���̏d��
			*	@param	model		���̃��f���̎��ʌ��ʂōX�V�l���v�Z����
			*	@param	alpha		���f���̏d��
			*	@param	dst_weight	�X�V��̃T���v���d��
			*/
			void update_sample_weights(const TrainingSet &datas, const cv::Mat &src_weight,
				PtrWeakLearner &model, double alpha, cv::Mat &dst_weight) ;

			/**
			*	
			*/
			double calc_alpha(double error) const { return 0.5 * log( (1.0 - error) / error); };
			
			int train(const TrainingSet &datas, const AdaboostParameter &param, cv::RNG &rng);

			std::vector<PtrWeakLearner> weak_classifiers_;
			
			std::vector<double> alpha_t_;
			
			double sum_alpha_;

	};
};