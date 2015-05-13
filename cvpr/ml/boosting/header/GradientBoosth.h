#pragma once

#include "..\..\base\header\StaticalModel.h"
#include "..\..\factory\header\StatModelFactory.h"

namespace cvpr
{
	/**
	*	gradient boost�p�p�����[�^
	*/
	class GradientBoostParameter : public StaticalModelParameter
	{
		public:

			int save(const std::string &save_path) const { return 0; };

			int load(const std::string &load_path) { return 0; };

			//! �I������㎯�ʊ퐔
			unsigned nr_rounds;

			//! �����̃V�[�h
			//uint64 seed;

			//! �X�e�[�W���Ɏ㎯�ʊ퐶�����邽�߂̃t�@�N�g���[
			StageWiseStatModelFactoryBase *factory;

			//! �w�K��
			double shrinkage;

		protected:
	};

	class GradientBoost : public StaticalModel
	{
		public:
		protected:

			/**
			*	0�Ԗڂ̎㎯�ʊ�(-> ��A�̏ꍇ�T���v���̕���(L2�ŏ���))
			*/
			cv::Mat	f0;

			//! �㎯�ʊ�
			std::vector<PtrWeakLearner> weak_laerner;

			//! �w�K�� (�e���ʊ퓝�����Ɏg�p)
			double	shrinkage;
	};
};