#include "..\header\PathUtil.h"

std::string cvpr::get_file_name(const std::string &path)
{
	std::string				dst		=	path;
	std::string::size_type	sep_pos	=	dst.find_last_of("\\");
	std::string::size_type	ext_pos	=	std::string::npos;
	
	if (std::string::npos != sep_pos) {
		dst	=	dst.substr(sep_pos + 1);
	}

	ext_pos	=	dst.find_last_of(".");

	if (std::string::npos != ext_pos) {
		dst	=	dst.substr(0, ext_pos);
	}

	return dst;
}