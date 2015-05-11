#pragma once
#include "..\..\base\header\StaticalModel.h"
#include "MlpLayerBase.h"

namespace cvpr
{

	namespace mlp {

		class MlpParameterbase : public StaticalModelParameter
		{
			public:
				int		num_hidden_layers;	//< �B��w�̐�

				int		max_iter;			//< �œK���̍ő�J��Ԃ���

				double	min_delta;			//< �œK���̍ŏ��X�V�� ��������ꍇ�I������

				double	decay_rate;			//< �w�K���̌����� ��_new = ��_old / (1 + decay_rate * iter)

				double	resample_rate;		//< mini batch �̃T���v����

				MlpOutputLayerType	output_layer_type;	//< �o�͑w�̊������֐��̎��

				unsigned long	rnd_seed;	//< �����̃V�[�h

				MlpParameterbase()
					: num_hidden_layers(1), max_iter(100), min_delta(0.01),
					output_layer_type(LAYER_TYPE_SIGMOID), rnd_seed(19861124),
					decay_rate(0.005), resample_rate(0.1) {};

				#pragma region override methods

				virtual int		save(const std::string &save_path) const;

				virtual int		load(const std::string &load_path);

				#pragma endregion

			protected:
		};

		/**
		*	MLP�p�p�����[�^
		*/
		class MultilayerPerceptronParameter : public virtual LayerParameter, public MlpParameterbase
		{
			public:
				
				#pragma region override methods

				virtual int		save(const std::string &save_path) const;

				virtual int		load(const std::string &load_path);

				#pragma endregion

			protected:
		};

		/**
		*	���w�p�[�Z�v�g����
		*	�����x�N�g��,���t�f�[�^�͗�x�N�g����z��
		*/
		class MultilayerPerceptron : public StaticalModel
		{
			public:
			
				#pragma region	override methods

				virtual int		save(const std::string &save_path) const ;

				virtual int		load(const std::string &load_path) ;

				virtual int		predict(const cv::Mat &feature, PredictionResult *result, const PredictionParameter *param = nullptr);

				virtual int		train(const TrainingSet &train_set, const StaticalModelParameter *param) ;

				#pragma endregion

				/**
				*	MLP�w�K���\�b�h
				*	@param	train_set	�w�K�Z�b�g
				*	@param	param		�w�K�p�����[�^
				*	@return	�w�K�̐���. 0:����, ����ȊO:�G���[
				*/
				int	train(const TrainingSet &train_set, const MultilayerPerceptronParameter &param) ;
			
				/**
				*	�\�����\�b�h
				*	@param	feature	�����x�N�g��
				*	@param	result	�\������
				*	@return	0:����, other:�G���[
				*/
				int	predict(const cv::Mat &feature, ClassificationResult &result) ;
			
				/**
				*	�\�����\�b�h
				*	@param	feature	�����x�N�g��
				*	@param	result	�\������
				*	@return	0:����, other:�G���[
				*/
				int	predict(const cv::Mat &feature, RegressionResult &result) ;

			protected:

				/**
				*	�B��w�̎�ʂ��擾����
				*	@return	�B��w�̎��
				*/
				virtual MlpHiddenLayerType	get_hidden_layer_type() { return LAYER_TYPE_TANH; };
			
				/**
				*	�p�����[�^�̒l���L������������
				*	@param	param	�����Ώۂ̃p�����[�^
				*	@return	true:�L��, false:�����l����
				*/
				static bool	is_valid_parameter(const MultilayerPerceptronParameter &param) ;
	
				/**
				*	�w�K�f�[�^���L������������D
				*	(= ��x�N�g�����ǂ���)
				*	@param	train_set	�����Ώۂ̊w�K�f�[�^
				*	@return	true:�L��, false:����
				*/
				static bool	is_valid_train_set(const TrainingSet &train_set) ;

				/**
				*	�w�K�̍ŏ��Ɋe�w������������
				*	���O�Ɋw�K�f�[�^�ƃp�����[�^�͌����ςݑz��
				*	@param	train_set	�w�K�f�[�^ <= is_valid~�Ō����ς�
				*	@param	param		�p�����[�^ <= is_valid~�Ō����ς�
				*	@param	rng			�����G���W��
				*/
				virtual void	init_layers(const TrainingSet &train_set, const MultilayerPerceptronParameter &param, std::mt19937 &rng) ;

				/**
				*	�w����activation���v�Z����
				*	@param	feature		�����x�N�g��
				*	@param	activations	activation�̌v�Z���ʁDlayers_.size()+1��������D0��feature�Didx���Ⴂ�ق������͑�.
				*/
				void	calc_all_a_z(const cv::Mat &feature, std::vector<cv::Mat> &a_j, std::vector<cv::Mat> &z_j) const ;

				/**
				*	�덷���t�`��������
				*	@param	layer_k	k�w�ڂ̃��C���[
				*	@param	layer_j	j�w��(=k-1)�̃��C���[
				*	@param	err_k	k�w�ڂ̌덷
				*	@param	z_j		j�w�ڂ�activation
				*	@param	a_j		j�w�ڂ̏d�݂Ɠ��͂̍s���
				*	@param	err_j	�t�`���������덷�̍s��
				*/
				void	backprop_error(const PtrLayerBase layer_k, const PtrLayerBase layer_j, const cv::Mat &err_k, const cv::Mat &z_j, const cv::Mat &a_j, cv::Mat &err_j) ;

				/**
				*	�w�̔z��擾
				*	out��hidden�ŕ������̂ňꏏ�����̔z�񂪂ق����Ƃ��悤
				*	@return	�w�̔z��
				*/
				std::vector<mlp::PtrLayerBase>	layers() const
				{
					std::vector<mlp::PtrLayerBase>	dst;
				
					dst.insert(dst.end(), hidden_layers_.begin(), hidden_layers_.end());
					dst.push_back(out_layer_);

					return dst;
				}

				static const std::string	FNAME_MLP_CFG;			//< MLP�̍\�����̃t�@�C����
				static const std::string	CFG_TAG_LAYER_TYPES;	//< MLP�̊e�w�̎�ʏ��̃^�O�� 
				static const std::string	FNAME_LEYER_CFG;

				std::vector<mlp::PtrHiddenLayerBase>	hidden_layers_;		//< �B��w

				PtrOutputLayerBase	out_layer_;						//< �o�͑w
		};

	};
};