#pragma once

#include <vector>
#include <string>
#include <opencv2\core\core.hpp>

#include "..\..\ml\base\header\TrainingImage.h"

namespace cvpr{

	//! iBUG���ăO���[�v���o���Ă�face point annotation�̃N���X
	//! http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
	class IbugFaceAnnotation
	{
		public:

		/**
		*	�f�t�H���g�R���X�g���N�^
		*/
		IbugFaceAnnotation()
			: no(-1) {};

		/**
		*	�摜�t�@�C���ƃZ�b�g�ɂȂ��Ă���pts�t�@�C�����J��
		*	@param	img_path	�摜�t�@�C���p�X
		*	@return	����
		*/
		int	open(const std::string &img_path);

		/**
		*	�摜�ƃ����h�}�[�N��ۑ�����D
		*	�摜�p�X�����w��D�����h�}�[�N�͊g���q��pts�ɕς��ēf��
		*	@param	img_path	�摜�p�X
		*	@return	����
		*/
		int	write(const std::string &img_path);

		/**
		* .pts�t�@�C�����J��
		* @param	path_pts	�t�@�C���p�X
		* @return	����
		*/
		int	open_pts(const std::string &path_pts);

		/**
		*	�摜�����J��
		*	@param	img_path	�摜�p�X
		*	@return	����
		*/
		int	open_img(const std::string &img_path);

		/**
		*	landmark�̊O�ڋ�` + ��`�̎w�芄���̃}�[�W����
		*	�摜��؂�o���čĐݒ肷��D
		*	landmark�ʒu���ړ�����
		*	�摜�[�ɒB���ė]�������Ȃ��ꍇ�߂�l�ł킩�邪�C
		*	�Đݒ�͍s���D
		*
		*	@param	margin_rate	�O�ڋ�`�̉����̗]�������邩
		*	@return	1:�[����������, 0:����
		*/
		int	trim(double margin_rate);

		/**
		*	�摜�t�@�C���p�X����pts�t�@�C���p�X�𓾂�
		*	(�g���q�ς��邾��)
		*	@param	img_path	�摜�p�X
		*	@return	pts�t�@�C���p�X
		*/
		static std::string	get_pts_path(const std::string &img_path);

		//! �ǂݍ��񂾉摜
		cv::Mat image;

		//! �ǂݍ��񂾃t�@�C����"���O"
		std::string name;

		//! �ǂݍ��񂾃t�@�C���̔ԍ�(�����)
		int	no;

		//! �ǂݍ��񂾓_��
		std::vector<cv::Point2f> pts;

		//! �o�[�W�����ԍ�
		int version;

		protected:
	};

	class IbugFaceAnnotationos
	{
		public:

		/**
		*	�f�[�^�̃A�N�Z�T
		*	@param	idx	�f�[�^�̃C���f�b�N�X
		*	@return	idx�Ԗڂ̃f�[�^
		*/
		IbugFaceAnnotation& operator[](const std::size_t &idx)
		{
			return annotations[idx];
		}

		/**
		*	�f�[�^�̃A�N�Z�T
		*	@param	idx	�f�[�^�̃C���f�b�N�X
		*	@return	idx�Ԗڂ̃f�[�^
		*/
		const IbugFaceAnnotation& operator[](const std::size_t &idx) const
		{
			return annotations[idx];
		}

		/**
		*	�f�[�^�����擾
		*	@return	�f�[�^��
		*/
		std::size_t	size() const
		{
			return annotations.size();
		}

		/**
		*	�摜�t�@�C�����X�g�p�X���w�肵�āC
		*	�摜�t�@�C���ƃA�m�e�[�V������ǂݍ���
		*	@param	list_path	�摜�t�@�C�����X�g�̃p�X
		*	@param	need_trim	�ǂݍ��݌��trim���邩�ǂ���
		*	@param	margin_rate	trim����ꍇ�Ɏg���}�[�W����
		*	@return	����
		*/
		int	open(const std::string &list_path, bool need_trim = false, double margin_rate = 0.1);

		/**
		*	�w�K�Z�b�g�𐶐�����D
		*	�����x�N�g���͉摜���̂܂܁D�o�b�t�@�̓R�s�[����Ȃ�
		*	@return	�w�K�Z�b�g
		*/
		TrainingImage create_train_set() const;

		//protected:

		//!	�A�m�e�[�V����
		std::vector<IbugFaceAnnotation>	annotations;

		/**
		*	�摜�t�@�C�����X�g�ǂݍ���
		*	@param	path	�摜�t�@�C�����X�g�p�X
		*	@param	dst		�ǂݎ�茋��
		*	@return	����
		*/
		int	read_list(const std::string &path, std::vector<std::string> &dst) const;

		/**
		*	�摜�p�X���摜���Ɖ摜�ԍ��ɕ�������
		*	c:\\hoge_2.jpg => 2, hoge
		*	@param	path	��������p�X
		*	@param	no		�摜�ԍ�(�ԍ����Ȃ����-1)
		*	@param	name	�摜��
		*/
		void split_img_path(const std::string &path, int &no, std::string &name);

		/**
		*	�摜���ŃA�m�e�[�V��������������D
		*/
		int	find(const std::string &name);

		/**
		*	���ό`����Z�o����
		*	@param	dst	���ό`��
		*/
		void	get_average_shape(std::vector<cv::Point2f> &dst) const;
	};

};