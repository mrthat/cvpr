#pragma once

#include <vector>
#include <string>
#include <opencv2\core\core.hpp>

//! iBUG���ăO���[�v���o���Ă�face point annotation�̃N���X
//! http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
class IbugFaceAnnotation
{
	public:

	/**
	*	�摜�t�@�C���ƃZ�b�g�ɂȂ��Ă���pts�t�@�C�����J��
	*	@param	img_path	�摜�t�@�C���p�X
	*	@return	����
	*/
	int open(const std::string &img_path);

	protected:

	/**
	* .pts�t�@�C�����J��
	* @param	path_pts	�t�@�C���p�X
	* @return	����
	*/
	int	open_pts(const std::string &path_pts);

	cv::Mat image;

	//! �ǂݍ��񂾃t�@�C���̃t�@�C�����������������o��������
	std::string file_name;

	//! �ǂݍ��񂾓_��
	std::vector<cv::Point2d> pts;

	//! �o�[�W�����ԍ�
	int version;
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
		*	@return	����
		*/
		int	open(const std::string &list_path);

	protected:

		//!	�A�m�e�[�V����
		std::vector<IbugFaceAnnotation>	annotations;

		/**
		*	�摜�t�@�C�����X�g�ǂݍ���
		*	@param	path	�摜�t�@�C�����X�g�p�X
		*	@param	dst		�ǂݎ�茋��
		*	@return	����
		*/
		int	read_list(const std::string &path, std::vector<std::string> &dst) const;
};