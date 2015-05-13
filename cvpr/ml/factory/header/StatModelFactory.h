#pragma once

#include "..\..\base\header\StaticalModel.h"

namespace cvpr
{
	//! �㎯�ʊ�t�@�N�g���[ �Ȃ�ł����邷�������
	class WeakLearnerFactory
	{
		public:

			/**
			*	�������\�b�h
			*	@param	type	�������郂�f���̌^
			*	@return	�V�������f��
			*/
			static PtrWeakLearner create(const StatModelType &type);

		protected:
	};

	/**
	*	greedy�Ȃ�Ŏg���悤�ɁC�X�e�[�W���Ɏ��ʊ퐶�����ĕԂ��N���X
	*/
	class StageWiseStatModelFactoryBase
	{
		public:

			/**
			*	<�w�K�ς�>�̎��ʊ��Ԃ�. �p�����[�^�͓����Ŏw�����邱��
			*	(�O�̏󋵂��p�����[�^�œn����悤�ɂ��ׂ����E�E�E�H)
			*	@param	datas	�w�K�f�[�^
			*	@return	�w�K�ς݃��f��
			*/
			virtual PtrWeakLearner next(const TrainingSet &datas) = 0;
	};

	//! �㎯�ʊ�̊w�K�ς݃v�[�����쐬����N���X
	class WeakLearnerPoolFactoryBase
	{
		public:
			/**
			*	�w�K�ς݂̎㎯�ʊ�v�[���𐶐�����
			*/
			virtual std::vector<PtrWeakLearner> create_trained_pool(const TrainingSet &datas, std::size_t pool_size) ;

		protected:
			
			/**
			*	���̃��f���p�p�����[�^���擾����
			*	@return	���̃��f���p�p�����[�^
			*/
			virtual PtrWeakLearnerParam next_param() = 0;

			/**
			*	���̃��f���𐶐�����
			*	@return	���Ƀv�[���ɉ����郂�f��
			*/
			virtual PtrWeakLearner next_model() = 0;
	};
};