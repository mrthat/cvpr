#pragma once
#include <random>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <direct.h>
#include "..\..\base\header\StaticalModel.h"
#include "TreeNode.h"
#include "TreeNodeFactory.h"

namespace cvpr
{
	/**
	*	tree�̎��
	*/
	typedef enum {
		TREE_TYPE_CLASSIFICATION,	//< �N���X���ʗptree
		TREE_TYPE_UNKNOWN			//< �����l
	} TreeType;

	/**
	*	randomized tree�̃p�����[�^�N���X
	*/
	class RandomizedTreeParameter : public StaticalModelParameter
	{
		public:
			RandomizedTreeParameter()
				: max_height(10), min_samples(10), num_splits(1000), rng_seed(19861124), min_info_grain(0.001f)
			{};
			virtual			~RandomizedTreeParameter() {};
			
			unsigned		max_height;		//< �؂̍ő卂�� �z����O�ɐ�����ł��؂�
			
			unsigned		min_samples;	//< �t�m�[�h�̍ŏ��T���v���� ����������ɐ�����ł��؂�

			unsigned		num_splits;		//< �ŗǕ�����T������ۂ̃v�[���̐�

			unsigned long	rng_seed;		//< �����̃V�[�h

			float			min_info_grain;	//< �����ɂ���ē���ꂽ��񗘓��̍ŏ��l ��������ꍇ������ł��؂�

			/**
			*	�����m�[�h�̎�ʂ̃��X�g
			*	���X�g���̎�ʂ̕����m�[�h�����w�K�Ɏg��
			*/
			std::vector<TreeNode::SplitNodeType>	split_type_list;
			
			/**
			*	�f�t�H���g�̕����m�[�h��ʃ��X�g��Ԃ�
			*/
			static	std::vector<TreeNode::SplitNodeType>	default_split_list() ;

		protected:

			virtual int		save(const std::string &save_path) const { return 0; };

			virtual int		load(const std::string &load_path) { return 0; };
	};

	/**
	*	randomized tree�̊�{�N���X
	*/
	class RandomizedTree : public StaticalModel
	{
		public:
			/**
			*	tree�̎�ʂ��擾
			*	@return	tree�̎��
			*/
			virtual TreeType	tree_type() const = 0;

			virtual		~RandomizedTree() { };

			virtual int	save(const std::string &save_path) const ;

			virtual int	load(const std::string &load_path) ;

			virtual int	predict(const cv::Mat &feature, PredictionResult *result) ;

			virtual int	train(const TrainingSet &train_set, const StaticalModelParameter *param) ;

			int			train(const TrainingSet &train_set, const RandomizedTreeParameter &param) ;

		protected:
			
			/**
			*	�ؓ��̊e�m�[�h�̎�ʂ��t�@�C���o�͂���
			*	@param	file_path	�o�̓t�@�C���p�X
			*	@return 0:����, -1:���s
			*/
			int	save_node_type(const std::string &file_path) const ;

			/**
			*	�ؓ��̊e�m�[�h�̎�ʂ��t�@�C��������͂���
			*	@param	file_path		���̓t�@�C���p�X
			*	@param	node_types		split or leaf�̎��
			*	@param	detail_types	split����leaf���̏ڍׂȎ��
			*/
			int	load_node_type(const std::string &file_path, std::vector<int> &node_types, std::vector<int> &detail_types) ;

			/**
			*	leaf node�̎�ʂ��擾����
			*	@return leaf node�̎��
			*/
			virtual TreeNode::LeafNodeType	leaf_type() const	=	0;

			/**
			*	�؂̐����̏I�������𖞂��������f����
			*	@param	train_set	�w�K�Z�b�g
			*	@param	param		�p�����[�^
			*	@param	tree_height	�w�K���̃m�[�h�̍����ʒu
			*	@return	true:�I�������𖞂���, false:�I�������𖞂����Ȃ�
			*/
			virtual bool	is_end_growth(const TrainingSet &train_set, const RandomizedTreeParameter &param, unsigned tree_height) const = 0;
			
			/**
			*	�؂𐬒������� = ���ڈʒu�ɐV�����m�[�h�ǉ����ăp�����[�^�œK��
			*	@param	train_set	�w�K�Z�b�g(���̃m�[�h�ɗ����T�u�Z�b�g)
			*	@param	param		�w�K�p�����[�^
			*	@param	tree_height	�w�K���̃m�[�h�̍����ʒu
			*	@param	node_id		�V�����ǉ�����
			*/
			virtual int		grow_tree(const TrainingSet &train_set, const RandomizedTreeParameter &param, unsigned tree_height, unsigned node_id, std::mt19937 &rng) ;

			/**
			*	�w�K�o�߂�W���o�͂���
			*	@param	split		�w�K�ς݂̕����m�[�h
			*	@param	train_set	�w�K�Z�b�g
			*/
			virtual void	print_train_log(const TreeNode::PtrSplitNodeBase split, const TrainingSet &train_set) const = 0;

			/**
			*	�w�K�o�߂�W���o�͂���
			*	@param	leaf		�w�K�ς݂̗t�m�[�h
			*	@param	train_set	�w�K�Z�b�g
			*/
			virtual void	print_train_log(const TreeNode::PtrLeafNodeBase leaf, const TrainingSet &train_set) const = 0;

			/**
			*	�����O��̏�񗘓����v�Z����
			*	@param	train_set	�����O�̊w�K�Z�b�g
			*	@param	left_set	������̊w�K�Z�b�g(��)
			*	@param	right_set	������̊w�K�Z�b�g(�E)
			*	@param	param		�w�K�p�����[�^
			*	@return	��񗘓�
			*/
			virtual double	calc_entropy_gain(const TrainingSet &train_set, const TrainingSet &left_set, const TrainingSet &right_set, const RandomizedTreeParameter &param) const = 0;

			std::vector<TreeNode::PtrNodeBase> nodes_;	//< �ؓ��̃m�[�h�̔z��D�w�K���id�Ń\�[�g���āCid=�z���idx�ɂȂ�悤�ɂ���

			int feature_depth_;	//< ���͓����x�N�g���̌^(CV_64F�Ƃ�)
	};

	typedef std::shared_ptr<RandomizedTree>	PtrRandomizedTree ;

	/**
	*	�N���X���ʗp��randomized tree
	*/
	class ClassificationTree : public RandomizedTree
	{
		public:
			
			virtual TreeType	tree_type() const { return TREE_TYPE_CLASSIFICATION; };

			virtual int	train(const TrainingSet &train_set, const StaticalModelParameter *param)
			{
				return RandomizedTree::train(train_set, param);
			}

		protected:

			virtual void	print_train_log(const TreeNode::PtrSplitNodeBase split, const TrainingSet &train_set) const ;

			TreeNode::LeafNodeType	leaf_type() const { return TreeNode::LEAF_TYPE_CLASSIFICATION; };

			virtual double	calc_entropy_gain(const TrainingSet &train_set, const TrainingSet &left_set, const TrainingSet &right_set, const RandomizedTreeParameter &param) const ;

			virtual void	print_train_log(const TreeNode::PtrLeafNodeBase leaf, const TrainingSet &train_set) const ;

			virtual bool	is_end_growth(const TrainingSet &train_set, const cvpr::RandomizedTreeParameter &param, unsigned tree_height) const ;
	};

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
			static PtrRandomizedTree	Create(TreeType tree_type)
			{
				switch (tree_type) {
					default:
						return PtrRandomizedTree(nullptr);
					case TREE_TYPE_CLASSIFICATION:
						return PtrRandomizedTree(new ClassificationTree());
				}
			}
		protected:
	};
};