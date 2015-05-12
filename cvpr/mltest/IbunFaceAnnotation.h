#pragma once

#include <vector>
#include <string>
#include <opencv2\core\core.hpp>

//! iBUGってグループが出してるface point annotationのクラス
//! http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
class IbugFaceAnnotation
{
	public:

	/**
	*	画像ファイルとセットになっているptsファイルを開く
	*	@param	img_path	画像ファイルパス
	*	@return	成否
	*/
	int open(const std::string &img_path);

	protected:

	/**
	* .ptsファイルを開く
	* @param	path_pts	ファイルパス
	* @return	成否
	*/
	int	open_pts(const std::string &path_pts);

	cv::Mat image;

	//! 読み込んだファイルのファイル名部分だけ抜き出したもの
	std::string file_name;

	//! 読み込んだ点列
	std::vector<cv::Point2d> pts;

	//! バージョン番号
	int version;
};