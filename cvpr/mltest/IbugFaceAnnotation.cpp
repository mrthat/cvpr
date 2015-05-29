#include "IbugFaceAnnotation.h"
#include "..\..\util\include\utils.h"
#include <opencv2\highgui\highgui.hpp>
#include <fstream>


int	IbugFaceAnnotation::open(const std::string &img_path)
{
	image	=	cv::imread(img_path);

	if (image.empty())
		return -1;

	// 拡張子をptsに変更
	std::string::size_type	dot_pos	=	img_path.find_last_of('.');

	if (dot_pos == std::string::npos)
		return -1;

	std::string	pts_path	=	img_path.substr(0, dot_pos) + ".pts";

	return open_pts(pts_path);

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

		if (-1 != idx) {
			ann.image	=	annotations[idx].image;
			ann.name	=	name;
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