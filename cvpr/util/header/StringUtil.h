#pragma once

#include <string>

namespace cvpr
{
	/**
	*	指定文字群で文字列の右側をtrimする．
	*	@param	str		trimされる文字列
	*	@param	trim	strの末尾からtrimしたい文字群
	*	@return	trimされたstr
	*/
	std::string trim_right(const std::string &str, const std::string &trim);

}