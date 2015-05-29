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

	//! �ǂݍ��񂾓_��
	std::vector<cv::Point2d> pts;

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
};