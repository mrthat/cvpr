#pragma once

#include "..\..\base\header\StaticalModel.h"
#include "..\..\factory\header\StatModelFactory.h"

namespace cvpr
{
	/**
	*	gradient boost�p�p�����[�^
	*/
	class GradientBoostParameter : public StaticalModelParameter
	{
		public:

			int save(const std::string &save_path) const { return 0; };

			int load(const std::string &load_path) { return 0; };

			//! �I������㎯�ʊ퐔
			unsigned nr_rounds;

			//! �����̃V�[�h
			//uint64 seed;

			//! �X�e�[�W���Ɏ㎯�ʊ퐶�����邽�߂̃t�@�N�g���[
			StageWiseStatModelFactoryBase *factory;

			//! �w�K��
			double shrinkage;

		protected:
	};

	class GradientBoost : public StaticalModel
	{
		public:
			/**
			*	�p�����[�^�̃t�@�C���o��
			*	@param	save_path	�o�͐�̃p�X
			*	@return	�G���[�R�[�h��
			*/
			virtual int		save(const std::string &save_path) const;
			
			/**
			*	�p�����[�^�̃t�@�C������
			*	@param	load_path	���͌��̃p�X
			*	@return	�G���[�R�[�h��
			*/
			virtual int		load(const std::string &load_path);

			/**
			*	�����x�N�g������\�����ʂ��o��
			*	@param	feature	���͂̓����x�N�g��
			*	@param	result	�o�̗͂\������
			*/
			virtual int		predict(const cv::Mat &feature, PredictionResult *result, const PredictionParameter *param = nullptr);

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
			*	�w�K�{����
			*	@param	datas	�w�K�f�[�^
			*	@param	param	�p�����[�^
			*/
			int train(const TrainingSet &datas, const GradientBoostParameter &param);

			/**
			*	0�Ԗڂ̎㎯�ʊ�(-> ��A�̏ꍇ�T���v���̕���(L2�ŏ���))
			*/
			cv::Mat	f0;

			//! �㎯�ʊ�
			std::vector<PtrWeakLearner> weak_laerner;

			//! �w�K�� (�e���ʊ퓝�����Ɏg�p)
			double	shrinkage;
	};


};