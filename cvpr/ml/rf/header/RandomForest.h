#pragma once
//#include <tbb/blocked_range.h>
#include "..\..\base\header\TrainingData.h"
#include "..\..\base\header\StaticalModel.h"
#include "..\header\RandomizedTree.h"
#include <direct.h>
#include <fstream>
#include <sstream>
//#include <tbb\tbb.h>
//#include "ParallelTrain.h"

#ifdef min
#undef min
#endif

#ifdef max
#undef max
#endif

//#define PARALLEL_TRAIN

namespace cvpr
{
	/**
	*	Random Forest�̊w�K�p�����[�^�N���X
	*/
	class RandomForestParameter : public cvpr::RandomizedTreeParameter
	{
		public:
			RandomForestParameter()
				: num_trees(5), resample_rate(0.4)
			{};

			virtual		~RandomForestParameter() {};

			unsigned	num_trees;			//< tree�̐�
			
			double		resample_rate;		//< �etree�w�K���Ƀf�[�^�Z�b�g����T���v�����銄��

		protected:
	};

	/**
	*	random forest�n�̃x�[�X�N���X
	*/
	class RandomForest : public WeakLearner
	{
		public:
			
			RandomForest() {};
			
			virtual			~RandomForest() {};

			/**
			*	�w�K���s���\�b�h
			*	@param	train_set	�w�K�f�[�^
			*	@param	param		�w�K�p�����[�^
			*	@return	0:����, -1:���s
			*/
			virtual int		train(const cvpr::TrainingSet &train_set, const cvpr::RandomForestParameter &param) ;
			
			/**
			*	�w�K�ς݂�tree��forest�ɒǉ�����.
			*	@param	tree	�ǉ�����tree
			*	@return	0:����, -1:���s
			*/
			virtual int		add_tree(PtrRandomizedTree tree) ;

			/**
			*	�w�K�ς݂�forest(����tree)�������ɒǉ�����
			*	@param	forest	�ǉ�����forest
			*	@return	0:����, -1:���s
			*/
			virtual int		merge(const RandomForest *forest) ;

			/**
			*	���ʂ̎�ʂ��擾
			*	@return	���ʂ̎��
			*/
			virtual ResultType	result_type() const = 0 ;

#pragma region override methods

			virtual int		save(const std::string &save_path) const ;

			virtual int		load(const std::string &load_path) ;

			virtual int		predict(const cv::Mat &feature, cvpr::PredictionResult *result, const PredictionParameter *param = nullptr);

			virtual int		train(const cvpr::TrainingSet &train_set, const cvpr::StaticalModelParameter *param) ;

#pragma endregion

		protected:
			
			/**
			*	���ݕێ�����tree�̎�ʂ��t�@�C���ɕۑ�����
			*	@param	data_path	�o�͐�t�@�C���p�X
			*	@return	0:����, -1:���s
			*/
			int	save_tree_type(const std::string &data_path) const ;

			/**
			*	(save_tree_type��)�t�@�C���ɕۑ�����tree��ʂ�ǂݍ���
			*	@param	data_path,	���̓t�@�C���p�X
			*	@param	tree_types	�ǂݍ���tree���
			*	@return	0:����, -1:���s
			*/
			int	load_tree_type(const std::string &data_path, std::vector<cvpr::TreeType> &tree_types) const ;

			/**
			*	�w�K����tree�̎�ʂ��擾����
			*/
			virtual	cvpr::TreeType	tree_type() const = 0;

			/**
			*	�etree�̗\�����ʂ��}�[�W����
			*	@param	results	�etree�̗\������
			*	@param	dst		�}�[�W��̌���
			*/
			virtual void	merge_results(const std::vector<PtrPredictionResult> &results, PredictionResult *dst) = 0;
			
			std::vector<PtrRandomizedTree>	trees_;	//<�v�f��tree���i�[����R���e�i
	};
	
	/**
	*	�N���X���ʗp��RandomForest�N���X
	*/
	class ClassificationForest : public RandomForest
	{
		public:

			//using	RandomForest::predict;

			virtual ResultType	result_type() const{ return RESULT_TYPE_CLASSIFICATION; };

		protected:
			
			virtual	cvpr::TreeType	tree_type() const { return TREE_TYPE_CLASSIFICATION; };

			virtual void	merge_results(const std::vector<PtrPredictionResult> &results, PredictionResult *dst) ;
	};
};
