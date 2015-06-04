#pragma once

#include <vector>
#include <string>
#include <opencv2\core\core.hpp>

#include "..\..\ml\base\header\TrainingImage.h"

namespace cvpr{

	//! iBUGってグループが出してるface point annotationのクラス
	//! http://ibug.doc.ic.ac.uk/resources/facial-point-annotations/
	class IbugFaceAnnotation
	{
		public:

		/**
		*	デフォルトコンストラクタ
		*/
		IbugFaceAnnotation()
			: no(-1) {};

		/**
		*	画像ファイルとセットになっているptsファイルを開く
		*	@param	img_path	画像ファイルパス
		*	@return	成否
		*/
		int	open(const std::string &img_path);

		/**
		*	画像とランドマークを保存する．
		*	画像パスだけ指定．ランドマークは拡張子をptsに変えて吐く
		*	@param	img_path	画像パス
		*	@return	成否
		*/
		int	write(const std::string &img_path);

		/**
		* .ptsファイルを開く
		* @param	path_pts	ファイルパス
		* @return	成否
		*/
		int	open_pts(const std::string &path_pts);

		/**
		*	画像だけ開く
		*	@param	img_path	画像パス
		*	@return	成否
		*/
		int	open_img(const std::string &img_path);

		/**
		*	landmarkの外接矩形 + 矩形の指定割合のマージンで
		*	画像を切り出して再設定する．
		*	landmark位置も移動する
		*	画像端に達して余白が取れない場合戻り値でわかるが，
		*	再設定は行う．
		*
		*	@param	margin_rate	外接矩形の何割の余白をつけるか
		*	@return	1:端っこいった, 0:正常
		*/
		int	trim(double margin_rate);

		/**
		*	画像ファイルパスからptsファイルパスを得る
		*	(拡張子変えるだけ)
		*	@param	img_path	画像パス
		*	@return	ptsファイルパス
		*/
		static std::string	get_pts_path(const std::string &img_path);

		//! 読み込んだ画像
		cv::Mat image;

		//! 読み込んだファイルの"名前"
		std::string name;

		//! 読み込んだファイルの番号(あれば)
		int	no;

		//! 読み込んだ点列
		std::vector<cv::Point2f> pts;

		//! バージョン番号
		int version;

		protected:
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
		*	@param	need_trim	読み込み後にtrimするかどうか
		*	@param	margin_rate	trimする場合に使うマージン率
		*	@return	成否
		*/
		int	open(const std::string &list_path, bool need_trim = false, double margin_rate = 0.1);

		/**
		*	学習セットを生成する．
		*	特徴ベクトルは画像そのまま．バッファはコピーされない
		*	@return	学習セット
		*/
		TrainingImage create_train_set() const;

		//protected:

		//!	アノテーション
		std::vector<IbugFaceAnnotation>	annotations;

		/**
		*	画像ファイルリスト読み込み
		*	@param	path	画像ファイルリストパス
		*	@param	dst		読み取り結果
		*	@return	成否
		*/
		int	read_list(const std::string &path, std::vector<std::string> &dst) const;

		/**
		*	画像パスを画像名と画像番号に分解する
		*	c:\\hoge_2.jpg => 2, hoge
		*	@param	path	分解するパス
		*	@param	no		画像番号(番号がなければ-1)
		*	@param	name	画像名
		*/
		void split_img_path(const std::string &path, int &no, std::string &name);

		/**
		*	画像名でアノテーションを検索する．
		*/
		int	find(const std::string &name);

		/**
		*	平均形状を算出する
		*	@param	dst	平均形状
		*/
		void	get_average_shape(std::vector<cv::Point2f> &dst) const;
	};

};