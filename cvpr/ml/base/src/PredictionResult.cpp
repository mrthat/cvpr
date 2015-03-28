#include "..\header\PredictionResult.h"
#include "utils.h"

using namespace cvpr;

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