#pragma once
#include <fstream>
#include <string>
#include <opencv2\highgui\highgui.hpp>
#include "..\..\base\header\TrainingData.h"

namespace cvpr {
	class LibSVMDatasetLoader
	{
		public:
			/**
			*	クラス識別用のデータをファイルから読み込む
			*	@param	fname	ファイルパス
			*	@return	読み込んだデータ
			*/
			static TrainingSet	load_classification_data(const std::string &fname) ;

			/**
			*	回帰用のデータをファイルから読み込む
			*	@param	fname	ファイルパス
			*	@return	読み込んだデータ
			*/
			static TrainingSet	load_regression_data(const std::string &fname) ;

		protected:

			enum {
				CVT_CLASSIFICATION,
				CVT_REGRESSION
			};

			/**
			*	特徴ベクトルの要素
			*/
			struct	FeatureElem
			{
				int		dim;	//< 何次元目の要素を表すかのインデックス
				double	val;	//< 要素値
			};

			/**
			*	LibSVMのデータセットファイルの1行 = 1サンプル
			*/
			struct	Sample
			{
				double						label;		//< 教師データ
				std::vector<FeatureElem>	feature;	//< 特徴ベクトル
			};

			/**
			*	ファイル読み込み
			*	@param	fname	ファイルパス
			*	@param	samples	読み込んだサンプル
			*	@return	true:成功,false:失敗
			*/
			static bool	load(const std::string &fname, std::vector<Sample> &samples) ;

			/**
			*	1サンプル分のデータ文字列からサンプルに変換する
			*	@param	buff	データ文字列
			*	@param	sample	返還後のサンプル
			*	@return	true:成功,false:失敗
			*/
			static bool	line_buff_to_sample(const std::string &buff, Sample &sample) ;

			/**
			*	サンプル配列からデータセットを作成する
			*	クラス識別用 ⇔ sampleのラベルは小数切り捨てて，データセットのラベルの該当次元だけ1にする
			*	回帰用 ⇔ データセットラベルは1x1の行列で作成し，sampleのラベルがそのまま入る
			*	@param	samples		入力のサンプル配列
			*	@param	cvt_code	sampleをコンバートする時のコード
			*	@return	データセット．作成失敗の場合は空
			*/
			static TrainingSet	samples_to_data(const std::vector<Sample> &samples, int cvt_code = CVT_CLASSIFICATION);
			
	};
};