#include "StatModelFactory.h"
#include "..\..\rf\header\RandomizedTree.h"
#include "..\..\rf\header\ClassificationTree.h"

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

		ClassificationTreeParameter	param;

		protected:

		virtual PtrWeakLearnerParam next_param();

		virtual PtrWeakLearner next_model();
	};
};