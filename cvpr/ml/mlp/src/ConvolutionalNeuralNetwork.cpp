#include "..\header\ConvolutionalNeuralNetwork.h"

using namespace cvpr;
using namespace mlp;

static const std::string SUFFIX_MLP_FILE = "mlp";
static const std::string SUFFIX_LAYER_FILE = "layer";

int	CnnParameter::save(const std::string &save_path) const
{
	if (0 != MlpParameterbase::save(save_path + SUFFIX_MLP_FILE)) {
		return -1;
	}

	if (0 != CnnLayerParameter::save(save_path + SUFFIX_LAYER_FILE)) {
		return -1;
	}

	return 0;
}

int	CnnParameter::load(const std::string &load_path)
{
	if (0 != MlpParameterbase::load(load_path + SUFFIX_MLP_FILE)) {
		return -1;
	}

	if (0 != CnnLayerParameter::load(load_path + SUFFIX_LAYER_FILE)) {
		return -1;
	}

	return 0;
}