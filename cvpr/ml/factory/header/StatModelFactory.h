#pragma once

#include "..\..\base\header\StaticalModel.h"
#include "..\..\rf\header\RandomForest.h"

namespace cvpr
{
	//! ���v���f���t�@�N�g���[��{�N���X
	class StatModelFactoryBase
	{
		public:

			/**
			*	�������\�b�h
			*	@param	type	�������郂�f���̌^
			*	@return	�V�������f��
			*/
			virtual PtrStaticalModel create(const StatModelType &type) ;

		protected:
	};

	//! �㎯�ʊ�t�@�N�g���[��{�N���X
	class WeakLearnerFactoryBase
	{
		public:

			/**
			*	�������\�b�h
			*	@param	type	�������郂�f���̌^
			*	@return	�V�������f��
			*/
			virtual PtrWeakLearner create(const StatModelType &type) ;

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

	//! �㎯�ʊ�Ƃ���RandomizedTree�������ăv�[���𐶐�����N���X
	class ClassificationTreePoolFactory : public WeakLearnerFactoryBase
	{
		
		public:

			ClassificationTreeParameter	param;

		protected:
			
			virtual PtrWeakLearnerParam next_param() ;

			virtual PtrWeakLearner next_model() ;
	};
};