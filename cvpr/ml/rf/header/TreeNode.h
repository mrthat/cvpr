#pragma once
#include <random>
#include <iostream>
#include <limits>
#include <sstream>
#include <vector>
#include "..\..\base\header\TrainingData.h"
#include "..\..\base\header\PredictionResult.h"
namespace cvpr
{
	namespace TreeNode
	{
		/**
		*	�m�[�h�̎��
		*/
		typedef enum {
			NODE_TYPE_BASE,		//< �x�[�X�N���X
			NODE_TYPE_SPLIT,	//< �����m�[�h
			NODE_TYPE_LEAF		//< �t�m�[�h
		} NodeType;

		/**
		*	�����m�[�h�̏ڍׂȎ��
		*/
		typedef enum {
			SPLIT_TYPE_BASE,		//< �x�[�X�N���X
			SPLIT_TYPE_AXISALIGNED,	//< �����s�J�b�g
			SPLIT_TYPE_LINE,		//< �����ʃJ�b�g
			SPLIT_TYPE_CONIC,		//< �~���J�b�g
			SPLIT_TYPE_ATAN,		//< 2�_�Ƃ���atan
			SPLIT_TYPE_MIN,			//< 2�_�Ƃ���min
			SPLIT_TYPE_HAAR,		//< haar�̏o�͂ŃJ�b�g
			SPLIT_TYPE_HAAR_INT,	//< ��Ɠ��������C���͂�integral image�ɂȂ��Ă�ꍇ
			SPLIT_TYPE_LAST			//< ��������`�O�O
		} SplitNodeType;

		/**
		*	�t�m�[�h�̏ڍׂȎ��
		*/
		typedef enum {
			LEAF_TYPE_BASE,				//< �x�[�X�N���X
			LEAF_TYPE_CLASSIFICATION,	//< �N���X���ʗp
			//LEAF_TYPE_REGRESSION,		//< ��A�p
			//LEAF_TYPE_DENSITY,			//< ���x����p
			//LEAF_TYPE_MANIFOLD			//< ���l�̗p
		} LeafNodeType;

		/**
		*	�m�[�h�̃x�[�X�N���X
		*/
		class NodeBase
		{
			public:

				/**
				*	�f�t�H���g�R���X�g���N�^
				*/
				NodeBase()
						: left_node_id(0), right_node_id(0), node_id(0) {};

				virtual				~NodeBase() {};

				unsigned			left_node_id;	//< ���q�m�[�h��id
				
				unsigned			right_node_id;	//< �E�q�m�[�h��id

				unsigned			node_id;		//< ������id

				bool	operator<(const NodeBase &rhs) const
				{
					return node_id < rhs.node_id;
				}

				/**
				*	�m�[�h�̎�ʂ��擾
				*	@return	�m�[�h�̎��
				*/
				virtual NodeType	get_node_type() const	=	0;

				/**
				*	�p�����[�^�t�@�C���o��
				*	@param	cvfs	�o�͐�t�@�C��
				*	@return	0:����, -1:���s
				*/
				virtual int	save(cv::FileStorage &cvfs) const ;

				/**
				*	�p�����[�^�t�@�C������
				*	@param	cvfs	���̓t�@�C��
				*	@return	0:����, -1:���s
				*/
				virtual int	load(cv::FileStorage &cvfs) ;

			protected:
		};

		/**
		*	�����m�[�h�x�[�X�N���X
		*/
		class SplitNodeBase
			: public NodeBase
		{
			public:

				/**
				*	�������ʂ�\���萔
				*/
				enum {
					LEFT = -1,	//< �����ɍs��^^
					RIGHT = 1	//< �E���ɍs��^^
				};

				/**
				*	�����ɕK�v�ȓ_�̐�
				*/
				enum {
					NUM_CUTPOINT = 2
				};

				SplitNodeBase()
					: cut_points_(NUM_CUTPOINT, 0) {};

				virtual					~SplitNodeBase() {};
				
				NodeType				get_node_type() const { return NODE_TYPE_SPLIT; }
				
				/**
				*	�����m�[�h�̏ڍ׎�ʂ��擾
				*	@return	�����m�[�h�̎��
				*/
				virtual SplitNodeType	get_split_type() const	=	0;

				/**
				*	��������
				*	@return	LEFT or RIGHT
				*/
				virtual int				operator()(const cv::Mat &feature) const ;

				/**
				*	�f�[�^�Z�b�g�܂Ƃ߂ĕ���
				*	@param	train_set	�����������Z�b�g
				*	@param	left_set	��������LEFT�̃T���v��������set
				*	@param	right_set	��������RIGHT�̃T���v�������set
				*/
				virtual void			operator()(const TrainingSet &train_set, TrainingSet &left_set, TrainingSet &right_set) const ;

				/**
				*	�p�����[�^�w�K����
				*	@param	train_set	�w�K�Z�b�g
				*	@param	rnd			�����G���W��
				*	@param	left		�w�K�㎞��LEFT�ɕ������ꂽ�T���v��������(�I�v�V����)
				*	@param	right		�w�K�㎞��RIGHT�ɕ������ꂽ�T���v��������(�I�v�V����)
				*	@return	0:����, -1:���s
				*/
				virtual int				train(const TrainingSet &train_set, std::mt19937 &rnd, TrainingSet *left = nullptr, TrainingSet *right = nullptr) ;

				virtual int	save(cv::FileStorage &cvfs) const ;

				virtual int	load(cv::FileStorage &cvfs) ;

			protected:
				
				/**
				*	
				*/
				enum {
					IDX_UNDER_CUTPOINT = 0,
					IDX_UPPER_CUTPOINT = 1
				};

				/**
				*	�J�[�l���v�Z�Ɏg��attribute�̐����擾
				*	@return	attribute�̐�
				*/
				virtual unsigned		get_num_attributes() const	=	0;
				
				/**
				*	attributes_�̏���������
				*	@param	train_set	�w�K������
				*	@param	rnd			�����G���W��
				*/
				void					init_attributes(const TrainingSet &train_set, std::mt19937 &rnd) ;

				/**
				*	�p�����[�^����������
				*	@param	train_set	�w�K�Z�b�g
				*	@param	rnd			�����G���W��
				*/
				virtual void			init_params(const TrainingSet &train_set, std::mt19937 &rnd)	=	0;

				/**
				*	�J�[�l���֐��v�Z����
				*	���͂́C���T���v������attribute�̓_�����T���v��������x�N�g���Ȃ��Ƃɒ���
				*	@param	feature	�����x�N�g��
				*	@return	�J�[�l���l
				*/
				virtual double			kernel_function(const cv::Mat &feature) const	=	0;

				/**
				*	�J�[�l���֐��̌v�Z���ʂ��獶or�E�𔻒肵�ĕԂ�
				*	@param	kernel_value	�J�[�l���l
				*	@return	LEFT or RIGHT
				*/
				virtual int		split(double kernel_value) const
				{
					if (this->cut_points_[IDX_UNDER_CUTPOINT] < kernel_value
						&& kernel_value < this->cut_points_[IDX_UPPER_CUTPOINT]) {
							return RIGHT;
					} else {
							return LEFT;
					}
				}

				std::vector<int>	attributes_;	//< �����x�N�g������attribute�Ƃ��ė��p����_��idx

				std::vector<double>	cut_points_;	//< �� or �E�𔻒肷�邽�߂̒l�͈̔͂�\���z��
		};

		/**
		*	�t�m�[�h�̃x�[�X�N���X
		*/
		class LeafNodeBase
			: public NodeBase
		{
			public:

				virtual					~LeafNodeBase() {};
				
				NodeType				get_node_type() const { return NODE_TYPE_LEAF; };

				/**
				*	�t�m�[�h�̎�ʂ��擾
				*	@return	�t�m�[�h�̎��
				*/
				virtual LeafNodeType	leaf_type() const	=	0;

				/**
				*	�t�m�[�h�̏����擾����
				*	@param	feature	�����x�N�g��
				*	@param	result	�t�m�[�h�̏�����锠
				*	@return	0:����, -1:���s
				*/
				virtual int				operator()(const cv::Mat &feature, PredictionResult *result)	=	0;

				/**
				*	�t�m�[�h�ɏ�񒙂߂�
				*	@param	train_set	�w�K�Z�b�g
				*	@return	0;����, -1:���s
				*/
				virtual int				train(const TrainingSet &train_set)	=	0;
				
				using	NodeBase::save ;

				using	NodeBase::load ;

			protected:
		};
		
		typedef std::shared_ptr<NodeBase> PtrNodeBase ;
		typedef std::shared_ptr<SplitNodeBase> PtrSplitNodeBase ;
		typedef std::shared_ptr<LeafNodeBase> PtrLeafNodeBase ;
	};
};