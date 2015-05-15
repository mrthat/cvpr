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
			*	�p�����[�^���s���łȂ������ׂ�
			*	�w�K�Z�b�g�ɑ΂��đÓ����ǂ����𒲂ׂ邩������Ȃ��̂ňꉞ�f�[�^���n��
			*	@param	datas	�f�[�^
			*	@param	param	�����Ώۃp�����[�^
			*	@param	false�Ȃ�s��
			*/
			virtual bool	is_valid_param(const TrainingSet &datas, const GradientBoostParameter &param) const;

			/**
			*	�������f��(f0)���v�Z����
			*	@param	datas	�w�K�f�[�^
			*	@param	parama	�p�����[�^
			*	@return	����
			*/
			int find_initial_model(const TrainingSet &datas, const GradientBoostParameter &param);

			/**
			*	�ڕW�x�N�g�����X�V�����w�K�Z�b�g���擾
			*	�e�X�e�[�W�Ŏ㎯�ʊ�͊w�K�ςݕ��ƖڕW�Ƃ̎c���Ƀt�B�b�e�B���O���邽��,
			*	���X�e�[�W�V�����w�K�Z�b�g�����K�v������D
			*	�����x�N�g����shallow copy
			*	@param	datas	�X�V���f�[�^
			*	@param	param	�p�����[�^
			*/
			TrainingSet calc_next_target(const TrainingSet &datas, const GradientBoostParameter &param);

			/**
			*	�F�X�J�����čĊw�K�ł���悤�ɂ���
			*/
			void release();

			/**
			*	GradientBoost�̃f�[�^�t�@�C���p�X�擾
			*	@param	�ۑ���f�B���N�g���p�X
			*	@return	�f�[�^�t�@�C���p�X
			*/
			std::string get_data_path(const std::string &save_path) const;

			/**
			*	0�Ԗڂ̎㎯�ʊ�(-> ��A�̏ꍇ�T���v���̕���(L2�ŏ���))
			*/
			cv::Mat	f0;

			//! �㎯�ʊ�
			std::vector<PtrWeakLearner> weak_learner;

			//! �w�K�� (�e���ʊ퓝�����Ɏg�p)
			double	shrinkage;

	};


};