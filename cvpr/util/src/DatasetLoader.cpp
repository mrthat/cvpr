#include <fstream>
#include <sstream>
#if 0
#include "..\header\DatasetLoader.h"

using namespace cvpr;

std::vector<std::string>	libSVMDatasetLoader::Split(const std::string &str, char delim)
{
	std::vector<std::string>	splitted;
	std::string					temp;
	std::istringstream			str_stream(str);

	while (std::getline(str_stream, temp, delim)) {
		splitted.push_back(temp);
	}

	return splitted;
}

int	libSVMDatasetLoader::SeekNumInputDims(std::string fname)
{
	std::ifstream	data_file(fname);
	int				max_dim_idx	=	-1;

	if (data_file.bad()) {
		return max_dim_idx;
	}

	while (!data_file.eof()) {
		std::string					line_buff;
		std::vector<std::string>	splitted;

		std::getline(data_file, line_buff);
		
		splitted	=	Split(line_buff, ' ');
		
		if (splitted.empty()) {
			continue;
		}
		
		for (unsigned ii = 1; ii < splitted.size(); ++ii) {
			int	dim_idx	=	0;
		
			sscanf(splitted[ii].c_str(), "%d:%*f", &dim_idx);
			max_dim_idx	=	std::max(max_dim_idx, dim_idx);
		}
	}

	return max_dim_idx;
}

int	libSVMDatasetLoader::SeekNumOutDims(std::string fname)
{
	std::ifstream	data_file(fname);
	int				max_label_idx	=	-1;

	if (data_file.bad()) {
		return max_label_idx;
	}

	while (!data_file.eof()) {
		int							label_idx;
		std::string					line_buff;
		std::vector<std::string>	splitted;

		std::getline(data_file, line_buff);
		
		if (line_buff.empty()) {
			continue;
		}
		
		sscanf(line_buff.c_str(), "%d", &label_idx);
		max_label_idx	=	std::max(max_label_idx, label_idx);
	}

	return max_label_idx + 1;

}


#endif