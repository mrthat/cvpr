#include "..\ml\rf\header\RandomForest.h"
#include "..\util\header\DataSetLoader.h"
//#include "..\util\header\PathUtil.h"

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
		int open(const std::string &img_path) ;

	protected:

		/**
		* .ptsファイルを開く
		* @param	path_pts	ファイルパス
		* @return	成否
		*/
		int	open_pts(const std::string &path_pts) ;

		cv::Mat image;

		//! 読み込んだファイルのファイル名部分だけ抜き出したもの
		std::string file_name;

		//! 読み込んだ点列
		std::vector<cv::Point2d> pts;

		//! バージョン番号
		int version;
};

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

int main()
{
	IbugFaceAnnotation	ifa;

	ifa.open("C:\\git_work\\cvpr\\datasets\\face_alignment\\afw\\134212_1.jpg");

	int g = 9;

	/*
	cvpr::TrainingSet tr = libSVMDatasetLoader::Load("..\\..\\datasets\\lib_svm\\svm_guide1\\svmguide1.txt");
	cvpr::TrainingSet ts = libSVMDatasetLoader::Load("..\\..\\datasets\\lib_svm\\svm_guide1\\svmguide1.t");

	cvpr::ClassificationForest cf;
	cvpr::RandomForestParameter param;

	param.split_type_list = cvpr::RandomForestParameter::default_split_list();

	cf.train(tr, param);

	cf.save("hoge");
	
	cvpr::ClassificationForest cf2;
	
	cf2.load("hoge");
	
	double cnt = 0;
	double cnt2 = 0;

	for (int ii = 0; ii < ts.size(); ++ii) {
		cvpr::ClassificationResult cr;

		cf.predict(ts[ii]->feature, &cr);

		if (cr.get_max_posterior_idx() == ts[ii]->target_max_idx()) {
			cnt++;
		}
		
		cf2.predict(ts[ii]->feature, &cr);
		
		if (cr.get_max_posterior_idx() == ts[ii]->target_max_idx()) {
			cnt2++;
		}
	}

	FILE *fp = fopen("huga.txt", "w");
	fprintf (fp,"%f\n", cnt / ts.size());
	fprintf (fp,"%f\n", cnt2 / ts.size());
	*/
	return 0;
}