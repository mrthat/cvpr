#pragma once
#include "TrainingData.h"

namespace cvpr
{
	class TrainingImage : public TrainingSet
	{
		public:

			TrainingImage(const MatType &label_type)
			{
				label_type_	=	label_type;
			};

		protected:

			using TrainingSet::get_feature_type;
			using TrainingSet::find_feature_min;
			using TrainingSet::find_feature_max;

			bool	is_valid_example(const PtrTrainingExample &example) const;

	};
};