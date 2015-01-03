#include "..\header\PredictionResult.h"
#include "..\..\..\util\header\utils.h"

using namespace cvpr;

int	
ClassificationResult::get_max_posterior_idx() const
{
	return	max_idx(posterior_);
}

PtrPredictionResult	PredictionResultFactory::create(ResultType type) 
{
	switch (type) {
		default:
			return PtrPredictionResult(nullptr);
		case RESULT_TYPE_CLASSIFICATION:
			return PtrPredictionResult(new ClassificationResult());
		case RESULT_TYPE_REGRESSION:
			return PtrPredictionResult(new RegressionResult());
	}
	
	return PtrPredictionResult(nullptr);
}