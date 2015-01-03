#pragma once
#include "..\..\base\header\StaticalModel.h"

namespace cvpr
{
	/**
	*	�o�͑w�̊������֐��̃^�C�v
	*/
	enum MlpOutputLayerType { 
		LAYER_TYPE_LINEAR,		//<	���` ��A���Ɏg��
		LAYER_TYPE_SIGMOID,		//< �V�O���C�h 2�N���X���ʂɎg��
		LAYER_TYPE_SOFTMAX,		//< �\�t�g�}�b�N�X ���N���X���ʂɎg��
	};
	
	/**
	*	�B��w�̎��
	*/
	enum MlpHiddenLayerType
	{
		LAYER_TYPE_TANH,		//< tanh
		LAYER_TYPE_CONVOLUTION,	//< ��ݍ���
	};

	namespace mlp
	{

		class LayerParameter : public cvpr::StaticalModelParameter
		{
			public:
				
			int		num_hidden_units;	//< �B��w���̃��j�b�g��
			
			double	update_rate;		//< �w�K��
			
			double	lambda;				//< ���������̔{��
			
			RegularizeType	regularize_type;	//< �������̎��
			
			LayerParameter()
				:num_hidden_units(10), update_rate(0.01),
				lambda(0.001), regularize_type(REGULARIZE_L2) {};

			#pragma region override methods

			virtual int		save(const std::string &save_path) const;

			virtual int		load(const std::string &load_path);

			#pragma endregion

			protected:
				
		};
		
		/**
		*	MLP���̃��C���[�̃x�[�X�N���X
		*/
		class LayerBase
		{
			public:

				/**
				*	�p�����[�^�̓��e���t�@�C���ɕۑ�����
				*	@param	fname	�o�͐�t�@�C���p�X
				*	@return	0:����, -1:���s
				*/
				virtual int	save(const std::string &fname) const;

				/**
				*	�t�@�C������p�����[�^�̓��e��ǂݍ���
				*	@param	fname	���̓t�@�C���ς�
				*	@return	0:����, -1:���s
				*/
				virtual int	load(const std::string &fname) ;

				/**
				*	a_j=w_ji.z_i���v�Z����
				*	@param	feature	���̓x�N�g��
				*	@param	dst		�v�Z����(a_j)
				*/
				virtual void	calc_a_j(const cv::Mat &feature, cv::Mat &dst) const
				{
					cv::Mat	tmp	=	get_colvec_header(feature);

					cv::gemm(weight_, tmp, 1.0, cv::Mat(), 0, dst);

					dst	+=	bias_;
				}

				/**
				*	sum_k (dE_n / da_k) ���v�Z����
				*	@param	err	delta_k
				*	@param	dst	�o�͔z��
				*/
				virtual void	calc_de_da(const cv::Mat &err, cv::Mat &dst) const 
				{
					dst	=	weight_.t() * err;
				};
					
				/**
				*	�������֐�
				*	
				*	@param	a_j		w_ji * zi��������
				*	@param	dst		�������֐��̌v�Z����
				*/
				virtual void	calc_activation(const cv::Mat &a_j, cv::Mat &dst) const = 0;

				/**
				*	���͂��������ɓ`��������
				*	@param	feature	���̓x�N�g��
				*	@param	dst		���`����̃x�N�g��
				*/
				virtual void	foward_prop(const cv::Mat &feature, cv::Mat &dst) const ;

				/**
				*	�������֐��̓��֐�
				*	@param	z_j	�������֐��̌���
				*	@param	a_j	w_ji*z_i�̌���
				*	@param	dst	���֐��̌v�Z����
				*/
				virtual void	calc_derivative(const cv::Mat &z_j, const cv::Mat &a_j, cv::Mat &dst) const = 0;

				/**
				*	�G���[����p�����[�^�X�V�������߂�
				*	@param	err	�G���[
				*	@param	weight_delta	�d�݂̍X�V��
				*	@param	bias_delta		�o�C�A�X�X�V��
				*/
				virtual void	calc_param_delta(const cv::Mat &err, const cv::Mat &activation, cv::Mat &weight_delta, double &bias_delta) const;

				/**
				*	�p�����[�^�X�V����K�p����
				*	@param	weight_delta	�d�݂̍X�V��
				*	@param	bias_delta		�o�C�A�X�X�V��
				*/
				virtual void	update(const LayerParameter &param, const cv::Mat &weight_delta, double bias_delta) ;
					
			protected:
					
				/**
				*	l2������������
				*	@param	param	�p�����[�^
				*/
				void	regularize_l2(const LayerParameter &param) ;

				/**
				*	l1������������
				*	@param	param	�p�����[�^
				*/
				void	regularize_l1(const LayerParameter &param) ;

				cv::Mat	weight_;	//<	�d�ݍs��

				double	bias_;		//< �o�C�A�X
		};

		/**
		*	�B��w�̃x�[�X�N���X
		*/
		class HiddenLayerBase : public LayerBase
		{
			public:
				/**
				*	�w�̎�ʂ��擾����
				*	@return	�w�̎��
				*/
				virtual MlpHiddenLayerType	type() const = 0;

				/**
				*	�w�̏�����
				*	@param	in_type		���͍s��̃w�b�_�[
				*	@param	param		�p�����[�^
				*	@param	out_type	�o�͍s��̃w�b�_�[
				*	@param	rng			�����G���W��
				*	@return	0:����, ����ȊO:�G���[
				*/
				virtual int	init(const MatType &in_type, const LayerParameter &param, MatType &out_type, std::mt19937 &rng) ;
		};

		/**
		*	�o�͑w�̃x�[�X�N���X
		*/
		class OutputLayerBase : public LayerBase
		{
			public:
				/**
				*	�w�̎�ʂ��擾����
				*	@return	�w�̎��
				*/
				virtual	MlpOutputLayerType	type() const = 0;

				/**
				*	������
				*	@param	train_set	�w�K�f�[�^
				*	@param	in_type		���͍s��̃w�b�_�[
				*	@param	param		�p�����[�^
				*	@param	rng			�����G���W��
				*	@return	0:����, ����ȊO:�G���[
				*/
				virtual int	init(const TrainingSet &train_set, const MatType &in_type, const LayerParameter &param, std::mt19937 &rng) ;
		};
			
		typedef std::shared_ptr<LayerBase>			PtrLayerBase;
		typedef std::shared_ptr<HiddenLayerBase>	PtrHiddenLayerBase;
		typedef std::shared_ptr<OutputLayerBase>	PtrOutputLayerBase;
	};
}