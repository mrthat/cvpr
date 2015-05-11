#pragma once

#include <string>

namespace cvpr
{
	/**
	*	ファイルパスを分割してファイル名だけ取り出す
	*	@param	path	ファイルパス
	*	@return	ファイル名
	*/
	std::string get_file_name(const std::string &path);
};