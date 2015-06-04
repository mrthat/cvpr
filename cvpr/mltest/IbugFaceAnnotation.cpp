#include "IbugFaceAnnotation.h"
#include "..\header\PathUtil.h"
#include "..\header\StringUtil.h"
#include "..\..\ml\rf\header\ShapeIndexedTree.h"
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\video\tracking.hpp>
#include <fstream>

using namespace cvpr;

int	IbugFaceAnnotation::trim(double margin_rate)
{
	cv::Rect bbox	=	cv::boundingRect(pts);
	int	bbox_state	=	0;

	// �}�[�W�����ǉ������O�ڋ�`�����߂�D
	bbox.x	-=	(int)std::floor(bbox.width * margin_rate);
	bbox.y	-=	(int)std::floor(bbox.height * margin_rate);
	bbox.width	*=	1.0 + 2 * margin_rate;
	bbox.height	*=	1.0 + 2 * margin_rate;

	// �摜�ɑ΂��ĊO�ڋ�`���傫������ꍇ��
	// �摜�ɍ��킹�Ęg��Ԃ�ς���
	if (bbox.x < 0) {
		bbox.x		=	0;
		bbox_state	=	1;
	}

	if (bbox.y < 0) {
		bbox.y		=	0;
		bbox_state	=	1;
	}

	if (image.rows <= bbox.br().y) {
		bbox.height	=	image.rows - bbox.y - 1;
		bbox_state	=	1;
	}

	if (image.cols <= bbox.br().x) {
		bbox.width	=	image.cols - bbox.x - 1;
		bbox_state	=	1;
	}

	// �摜�؂�o���ƃ����h�}�[�N�ʒu���킹
	image	=	image(bbox).clone();

	for (std::size_t ii = 0; ii < pts.size(); ++ii) {
		pts[ii].x	-=	bbox.x;
		pts[ii].y	-=	bbox.y;
	}

	return bbox_state;
}

int	IbugFaceAnnotation::open(const std::string &img_path)
{
	if (0 != open_img(img_path))
		return -1;

	// �g���q��pts�ɕύX
	std::string	pts_path	=	get_pts_path(img_path);

	return open_pts(pts_path);
}

int	IbugFaceAnnotation::write(const std::string &img_path)
{
	if (!cv::imwrite(img_path, image))
		return -1;

	std::string	pts_path	=	get_pts_path(img_path);
	std::ofstream	file(pts_path, std::ios::out | std::ios::trunc);

	if (!file)
		return -1;

	file << "version: " << version << std::endl;
	file << "n_points: " << pts.size() << std::endl;	
	file << "{" << std::endl;
	
	for (std::size_t ii = 0; ii < pts.size(); ++ii) {
		file << pts[ii].x << " " << pts[ii].y << std::endl;
	}

	file << "}";

	return 0;
}

int IbugFaceAnnotation::open_pts(const std::string &path_pts)
{
	std::ifstream	file(path_pts);
	std::string		buff;
	int	num	=	0;

	if (!file)
		return -1;

	// version�s
	if (!std::getline(file, buff))
		return -1;

	if (1 != sscanf(buff.c_str(), "version: %d", &version))
		return -1;

	// �_���s
	if (!std::getline(file, buff))
		return -1;

	if (1 != sscanf(buff.c_str(), "n_points: %d", &num))
		return -1;

	// {��ǂݔ�΂�
	std::getline(file, buff);

	for (int ii = 0; ii < num; ++ii) {
		if (!std::getline(file, buff))
			return -1;

		cv::Point2d	pt;

		if (2 != sscanf(buff.c_str(), "%lf %lf", &pt.x, &pt.y))
			return -1;

		pts.push_back(pt);
	}

	//file_name	=	cvpr::get_file_name(path_pts);

	return 0;
}

int	IbugFaceAnnotation::open_img(const std::string &img_path)
{
	image	=	cv::imread(img_path);

	if (image.empty())
		return -1;

	std::string	fname	=	cvpr::get_file_name(img_path);
	std::string::size_type	pos	=	fname.find('_');

	if (std::string::npos == pos) {
		name	=	fname;
		return 0;
	}

	name	=	fname.substr(0, pos);

	sscanf(fname.c_str(), "%*[^_]_%d*", &no);

	return 0;
}

std::string	IbugFaceAnnotation::get_pts_path(const std::string &img_path)
{
	// �g���q��pts�ɕύX
	std::string::size_type	dot_pos	=	img_path.find_last_of('.');

	if (dot_pos == std::string::npos)
		return "";

	return img_path.substr(0, dot_pos) + ".pts";
}

int	IbugFaceAnnotationos::open(const std::string &list_path, bool need_trim, double margin_rate)
{
	std::vector<std::string>	list;

	int	ret	=	read_list(list_path, list);

	if (0 != ret)
		return ret;

	for (std::size_t ii = 0; ii < list.size(); ++ii) {
		IbugFaceAnnotation	ann;
		std::string	name;
		int	imgno	=	0;
		std::string	pts_path	=	IbugFaceAnnotation::get_pts_path(list[ii]);
		int	ret	=	0;

		split_img_path(list[ii], imgno, name);

		ret	=	ann.open_pts(pts_path);

		if (0 != ret)
			return ret;

		ann.name	=	name;
		ann.no		=	imgno;

		ret	=	ann.open_img(list[ii]);

		if (0 != ret)
			return ret;

		if (need_trim)
			ann.trim(margin_rate);

		annotations.push_back(ann);
	}

	return 0;
}

int	IbugFaceAnnotationos::read_list(const std::string &path, std::vector<std::string> &dst) const
{
	std::ifstream	ifs(path);
	std::string		buff;

	dst.clear();

	if (!ifs)
		return -1;

	while (std::getline(ifs, buff)) {
		std::string line	=	cvpr::trim_right(buff, "\r\n ");

		if (line.empty())
			continue;

		dst.push_back(line);
	}

	return 0;
}

void	IbugFaceAnnotationos::split_img_path(const std::string &path, int &no, std::string &name)
{
	std::string	fname	=	cvpr::get_file_name(path);
	std::string::size_type	pos	=	fname.find_last_of("_");

	if (1 != sscanf(fname.c_str(), "%*[^_]_%d", &no)) {
		no	=	-1;
	}

	name	=	fname.substr(0, pos);
}

int	IbugFaceAnnotationos::find(const std::string &name)
{
	for (std::size_t ii = 0; ii < annotations.size(); ++ii) {
		if (annotations[ii].name.compare(name) == 0)
			return (int)ii;
	}

	return -1;
}

void	IbugFaceAnnotationos::get_average_shape(std::vector<cv::Point2f> &dst) const
{
	if (annotations.empty())
		return;

	dst.assign(annotations.front().pts.size(), cv::Point2f(0, 0));

	for (auto ann = annotations.begin(); ann != annotations.end(); ++ann) {
		for (std::size_t ii = 0; ii < ann->pts.size(); ++ii) {
			dst[ii].x	+=	ann->pts[ii].x / ann->image.cols;
			dst[ii].y	+=	ann->pts[ii].y / ann->image.rows;
		}
	}

	for (auto pt = dst.begin(); pt != dst.end(); ++pt) {
		pt->x	/=	annotations.size();
		pt->y	/=	annotations.size();
	}
}

TrainingImage IbugFaceAnnotationos::create_train_set() const
{
	if (annotations.empty())
		return TrainingImage(MatType());

	// �A�m�e�[�V�����̓_��͑S������������Ƃ������Ƃɂ���
	TrainingImage	dst(MatType(annotations.front().pts.size(), 2, CV_32FC1));
	std::vector<cv::Point2f>	avg_shape;

	for (auto ii = annotations.begin(); ii != annotations.end(); ++ii) {
		PtrTrainingExample	sample(new TrainingExample);
		cv::Mat	trans;
		ShapeIndexedPredictionParameter	*param	=	new ShapeIndexedPredictionParameter;

		// �����Ƌ��t�����
		sample->feature	=	ii->image;
		sample->target.create(ii->pts.size(), 2, CV_32FC1);
		for (std::size_t row = 0; row < ii->pts.size(); ++row) {
			sample->target.at<float>(row, 0)	=	ii->pts[row].x;
			sample->target.at<float>(row, 1)	=	ii->pts[row].y;
		}

		sample->param.reset(param);

		// ���ʗp�p�����[�^�����
		param->shape		=	ii->pts;
		param->transform	=	cv::estimateRigidTransform(avg_shape, ii->pts, false);

		if (!dst.push_back(sample)) {
			// �Ȃ񂩂���Ȃ�G���[����
			continue;
		}
	}

	return dst;
}