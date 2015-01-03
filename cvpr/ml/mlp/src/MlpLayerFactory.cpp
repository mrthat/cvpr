#include "..\header\MlpLayerFactory.h"
#include "..\header\MlpLayer.h"

using namespace cvpr;
using namespace mlp;

PtrOutputLayerBase	LayerFactory::create(MlpOutputLayerType layer_type) 
{
	switch (layer_type) {
		default:
			break;

		case LAYER_TYPE_LINEAR:

			return PtrOutputLayerBase(new LinearLayer());

		case LAYER_TYPE_SIGMOID:

			return PtrOutputLayerBase(new SigmoidLayer());

		case LAYER_TYPE_SOFTMAX:

			 return PtrOutputLayerBase(new SoftMaxLayer());
	}

	return PtrOutputLayerBase();
}

PtrHiddenLayerBase	LayerFactory::create(MlpHiddenLayerType layer_type) 
{
	switch (layer_type) {
		default:
			break;

		case LAYER_TYPE_TANH:

			return PtrHiddenLayerBase(new TanhLayer());

		case LAYER_TYPE_CONVOLUTION:

			return PtrHiddenLayerBase(new ConvolutionLayer());
	}

	return PtrHiddenLayerBase();
}