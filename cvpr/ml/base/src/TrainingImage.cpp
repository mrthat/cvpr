#include "..\header\TrainingImage.h"

using namespace cvpr;

bool	TrainingImage::is_valid_example(const PtrTrainingExample &example) const
{
	// ���x���̃T�C�Y�ƌ^�������Ă邩���ׂ�
	if (!label_type_.equals(example->target)) {
		return false;
	}

	return true;
}
