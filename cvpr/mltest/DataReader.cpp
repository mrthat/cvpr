#include "DataReader.h"

using namespace cvpr;

TrainingSet DataReader::wine(const std::string &path)
{
	TrainingSet	dst(MatType(11, 1, CV_64FC1), MatType(1, 1, CV_64FC1));
	FILE *fp = fopen(path.c_str(), "r");
	char	buff[2048];

	if (nullptr == fp)
		return dst;

	fgets(buff, 2048, fp); // ƒwƒbƒ_[‹Æ“Ç‚İ”ò‚Î‚µ

	while (0 == feof(fp) && 0 == ferror(fp)) {
		double data[12];

		fgets(buff, 2048, fp);

		int num = sscanf(buff, "%lf;%lf;%lf;%lf;%lf;%lf;%lf;%lf;%lf;%lf;%lf;%lf", &data[0], &data[1], &data[2], &data[3], &data[4], &data[5], &data[6], &data[7], &data[8], &data[9], &data[10], &data[11]);

		if (num != 12)
			continue;

		PtrTrainingExample pdata(new TrainingExample);

		pdata->target.create(1, 1, CV_64FC1);
		pdata->target.at<double>(0, 0)	=	data[11];
		pdata->feature.create(11, 1, CV_64FC1);

		for (int ii = 0; ii < 11; ++ii) {
			pdata->feature.at<double>(ii, 0)	=	data[ii];
		}

		dst.push_back(pdata);
	}

	fclose(fp);

	return dst;
}