#pragma once
#include "MutilayerPercptron.h"
#include "MlpLayer.h"

namespace cvpr {

	namespace mlp
	{
		/**
		*	畳み込みニューラルネットワークのパラメータ
		*/
		class CnnParameter : public MultilayerPerceptronParameter, public CnnLayerParameter
		{
			public:
				
				#pragma region override methods

				virtual int		save(const std::string &save_path) const;

				virtual int		load(const std::string &load_path);

				#pragma endregion

			protected:
		};

		/**
		*	畳み込みニューラルネットワーク
		*/
		class ConvolutionalNeuralNetwork : public MultilayerPerceptron
		{
			public:
				int	train(const TrainingSet &train_set, const CnnLayerParameter &param) 
				{
					return MultilayerPerceptron::train(train_set, &param);
				};
			protected:
				virtual MlpHiddenLayerType	get_hidden_layer_type() { return LAYER_TYPE_CONVOLUTION; };
		};

	};
}
