#pragma once

#include <string>

namespace cvpr
{
	/**
	*	�w�蕶���Q�ŕ�����̉E����trim����D
	*	@param	str		trim����镶����
	*	@param	trim	str�̖�������trim�����������Q
	*	@return	trim���ꂽstr
	*/
	std::string trim_right(const std::string &str, const std::string &trim);

}