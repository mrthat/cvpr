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