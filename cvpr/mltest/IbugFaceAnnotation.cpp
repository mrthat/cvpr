#include "IbugFaceAnnotation.h"
#include "..\..\util\include\utils.h"
#include <opencv2\imgproc\imgproc.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <fstream>

int	IbugFaceAnnotation::trim(double margin_rate)
{
	cv::Rect bbox	=	cv::boundingRect(pts);
	int	bbox_state	=	0;

	// マージン分追加した外接矩形を求める．
	bbox.x	-=	bbox.width * margin_rate;
	bbox.y	-=	bbox.height * margin_rate;
	bbox.width	*=	1.0 + 2 * margin_rate;
	bbox.height	*=	1.0 + 2 * margin_rate;

	// 画像に対して外接矩形が大きすぎる場合は
	// 画像に合わせて枠状態を変える
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

	// 画像切り出しとランドマーク位置合わせ
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

	// 拡張子をptsに変更
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
}

int IbugFaceAnnotation::open_pts(const std::string &path_pts)
{
	std::ifstream	file(path_pts);
	std::string		buff;
	int	num	=	0;

	if (!file)
		return -1;

	// version行
	if (!std::getline(file, buff))
		return -1;

	if (1 != sscanf(buff.c_str(), "version: %d", &version))
		return -1;

	// 点数行
	if (!std::getline(file, buff))
		return -1;

	if (1 != sscanf(buff.c_str(), "n_points: %d", &num))
		return -1;

	// {を読み飛ばし
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
	// 拡張子をptsに変更
	std::string::size_type	dot_pos	=	img_path.find_last_of('.');

	if (dot_pos == std::string::npos)
		return "";

	return img_path.substr(0, dot_pos) + ".pts";
}

int	IbugFaceAnnotationos::open(const std::string &list_path)
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
		int	idx	=	0;

		split_img_path(list[ii], imgno, name);

		ret	=	ann.open_pts(pts_path);

		if (0 != ret)
			return ret;

		idx	=	find(name);
		ann.name	=	name;
		ann.no		=	imgno;

		if (-1 != idx) {
			ann.image	=	annotations[idx].image;
		} else {
			ret	=	ann.open_img(list[ii]);

			if (0 != ret)
				return ret;
		}

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
			return ii;
	}

	return -1;
}