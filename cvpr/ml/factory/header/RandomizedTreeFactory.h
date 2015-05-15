#include "StatModelFactory.h"
#include "..\..\rf\header\RandomizedTree.h"
#include "..\..\rf\header\ClassificationTree.h"
#include "..\..\rf\header\RegressionTree.h"

namespace cvpr
{
	/**
	*	randomized tree�̃t�@�N�g���[�N���X
	*/
	class RandomizedTreeFactory
	{
		public:
		virtual ~RandomizedTreeFactory() {};

		/**
		*	�؂𐶐�����
		*	@param	tree_type	�؂̎��
		*	@return	�m��Ȃ����:nullptr, ���̑�:�L���Ȃۂ���
		*/
		static PtrRandomizedTree	Create(TreeType tree_type);

		protected:
	};

	//! �㎯�ʊ�Ƃ���RandomizedTree�������ăv�[���𐶐�����N���X
	class ClassificationTreePoolFactory : public WeakLearnerPoolFactoryBase
	{

		public:

		/**
		*	�p�����[�^�ݒ�
		*	@param	param	�ݒ肷��p�����[�^
		*/
		void set_param(const ClassificationTreeParameter &param_)
		{
			this->param	=	param_;
			rng	=	std::mt19937(param.rng_seed);
		}

		protected:

		ClassificationTreeParameter	param;

		std::mt19937 rng;

		virtual PtrWeakLearnerParam next_param();

		virtual PtrWeakLearner next_model();
	};

	//! GradientBoost�����X�e�[�W������
	class StageWiseRegressionTreeFactory : public StageWiseStatModelFactoryBase
	{
		public:

			virtual PtrWeakLearner next(const TrainingSet &datas) ;

			/**
			*	�V�����؂̊w�K���Ɏg�p�����p�����[�^��ݒ肷��
			*/
			void set_param(const RegressionTreeParameter &param_)
			{
				param	=	param_;
				rng		=	std::mt19937(param.rng_seed);
			}

		protected:

			//! �V�����؂̐������Ɋw�K�Ɏg�p�����p�����[�^
			//! �����̃V�[�h�͏�����������
			RegressionTreeParameter param;

			//! �����̃V�[�h�𐶐����闐��(�z���g�͂����g���̂͗ǂ��Ȃ���)
			std::mt19937 rng;
	};
};