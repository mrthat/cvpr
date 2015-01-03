#pragma once
#include <fstream>
#include <string>
#include <opencv2\highgui\highgui.hpp>
#include "..\..\base\header\TrainingData.h"

namespace cvpr {
	class LibSVMDatasetLoader
	{
		public:
			/**
			*	�N���X���ʗp�̃f�[�^���t�@�C������ǂݍ���
			*	@param	fname	�t�@�C���p�X
			*	@return	�ǂݍ��񂾃f�[�^
			*/
			static TrainingSet	load_classification_data(const std::string &fname) ;

			/**
			*	��A�p�̃f�[�^���t�@�C������ǂݍ���
			*	@param	fname	�t�@�C���p�X
			*	@return	�ǂݍ��񂾃f�[�^
			*/
			static TrainingSet	load_regression_data(const std::string &fname) ;

		protected:

			enum {
				CVT_CLASSIFICATION,
				CVT_REGRESSION
			};

			/**
			*	�����x�N�g���̗v�f
			*/
			struct	FeatureElem
			{
				int		dim;	//< �������ڂ̗v�f��\�����̃C���f�b�N�X
				double	val;	//< �v�f�l
			};

			/**
			*	LibSVM�̃f�[�^�Z�b�g�t�@�C����1�s = 1�T���v��
			*/
			struct	Sample
			{
				double						label;		//< ���t�f�[�^
				std::vector<FeatureElem>	feature;	//< �����x�N�g��
			};

			/**
			*	�t�@�C���ǂݍ���
			*	@param	fname	�t�@�C���p�X
			*	@param	samples	�ǂݍ��񂾃T���v��
			*	@return	true:����,false:���s
			*/
			static bool	load(const std::string &fname, std::vector<Sample> &samples) ;

			/**
			*	1�T���v�����̃f�[�^�����񂩂�T���v���ɕϊ�����
			*	@param	buff	�f�[�^������
			*	@param	sample	�ԊҌ�̃T���v��
			*	@return	true:����,false:���s
			*/
			static bool	line_buff_to_sample(const std::string &buff, Sample &sample) ;

			/**
			*	�T���v���z�񂩂�f�[�^�Z�b�g���쐬����
			*	�N���X���ʗp �� sample�̃��x���͏����؂�̂ĂāC�f�[�^�Z�b�g�̃��x���̊Y����������1�ɂ���
			*	��A�p �� �f�[�^�Z�b�g���x����1x1�̍s��ō쐬���Csample�̃��x�������̂܂ܓ���
			*	@param	samples		���͂̃T���v���z��
			*	@param	cvt_code	sample���R���o�[�g���鎞�̃R�[�h
			*	@return	�f�[�^�Z�b�g�D�쐬���s�̏ꍇ�͋�
			*/
			static TrainingSet	samples_to_data(const std::vector<Sample> &samples, int cvt_code = CVT_CLASSIFICATION);
			
	};
};