#include "..\header\StringUtil.h"

std::string cvpr::trim_right(const std::string &str, const std::string &trim)
{
	std::string::size_type	pos	=	str.find_last_not_of(trim);

	if (std::string::npos == pos) {
		return str;
	}

	return str.substr(0, pos);
}