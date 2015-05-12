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