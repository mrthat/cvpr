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

class IbugFaceAnnotationos
{
	public:

		/**
		*	データのアクセサ
		*	@param	idx	データのインデックス
		*	@return	idx番目のデータ
		*/
		IbugFaceAnnotation& operator[](const std::size_t &idx)
		{
			return annotations[idx];
		}

		/**
		*	データのアクセサ
		*	@param	idx	データのインデックス
		*	@return	idx番目のデータ
		*/
		const IbugFaceAnnotation& operator[](const std::size_t &idx) const
		{
			return annotations[idx];
		}

		/**
		*	データ数を取得
		*	@return	データ数
		*/
		std::size_t	size() const
		{
			return annotations.size();
		}

		/**
		*	画像ファイルリストパスを指定して，
		*	画像ファイルとアノテーションを読み込む
		*	@param	list_path	画像ファイルリストのパス
		*	@return	成否
		*/
		int	open(const std::string &list_path);

	protected:

		//!	アノテーション
		std::vector<IbugFaceAnnotation>	annotations;

		/**
		*	画像ファイルリスト読み込み
		*	@param	path	画像ファイルリストパス
		*	@param	dst		読み取り結果
		*	@return	成否
		*/
		int	read_list(const std::string &path, std::vector<std::string> &dst) const;
};