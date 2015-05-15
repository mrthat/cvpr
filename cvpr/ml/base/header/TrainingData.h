#pragma		once
#define		_USE_MATH_DEFINES
#include	<math.h>
#include	<random>
#include	<numeric>
#include	<array>
#include	<memory>
#include	<opencv2\core\core.hpp>
#include	"PredictionParameter.h"
#include	"utils.h"

namespace cvpr
{
	/**
	*	�P���T���v��
	*	���x���Ɠ����x�N�g���̃y�A
	*/
	class TrainingExample
	{
		public:
			/**
			*	�f�t�H���g�R���X�g���N�^
			*/
			TrainingExample() {};

			/**
			*	�v�f�w�菉����
			*	@param	feature	�����x�N�g��
			*	@param	label	���x��
			*/
			TrainingExample(cv::Mat feature, cv::Mat target, PtrPredictionParameter param_ = nullptr)
				: feature(feature), target(target), param(param_) {};

			/**
			*	�f�X�g���N�^
			*/
			virtual	~TrainingExample() {};

			/**
			*	�����x�N�g��
			*/
			cv::Mat	feature;

			/**
			*	���x��
			*/
			cv::Mat	target;

			/**
			*	�T���v���ɕR�Â��\���p�p�����[�^���A���Γ����(null��OK�Ƃ���)
			*/
			PtrPredictionParameter	param;

			/**
			*	label���̍ő�v�f��idx���擾
			*	@return	label���̍ő�v�f��idx
			*/
			int	label() const
			{
				return max_idx(target);
			}

		protected:
	};

	typedef std::shared_ptr<TrainingExample> PtrTrainingExample;

	/**
	*	�s��̃��^�f�[�^�N���X
	*/
	class MatType
	{
		public:
			/** �s��̊e�����̗v�f�� */
			std::vector<int>	sizes;
			/** �s��̃f�[�^�̃^�C�v(open cv�`��) */
			int					data_type;

			/**
			*	�f�t�H���g�R���X�g���N�^
			*/
			MatType()
				: data_type(0) {};

			/**
			*	2d�̃T�C�Y�ƃf�[�^�^�w�肵�č\�z
			*	@param	size		2d�̍s��̃T�C�Y
			*	@param	data_type	�f�[�^�̃^�C�v
			*/
			MatType(const cv::Size &size, int data_type)
				: data_type(data_type)
			{
				sizes.assign(2, 0);
				sizes[0]	=	size.height;
				sizes[1]	=	size.width;
			}
			
			/**
			*	2d�̃T�C�Y�ƃf�[�^�^�w�肵�č\�z
			*	@param	rows		�s��
			*	@param	cols		��
			*	@param	data_type	�f�[�^�̃^�C�v
			*/
			MatType(int rows, int cols, int data_type)
				: data_type(data_type)
			{
				sizes.assign(2, 0);
				sizes[0]	=	rows;
				sizes[1]	=	cols;
			}

			/**
			*	md�̃T�C�Y�ƃf�[�^�^�w�肵�č\�z
			*	@param	sizes		md�̃T�C�Y
			*	@param	data_type	�f�[�^�̃^�C�v
			*/
			MatType(const std::vector<int> &sizes, int data_type)
				: sizes(sizes), data_type(data_type) {};

			/**
			*	�s��̃w�b�_�[�ŏ�����
			*	@param	mat	�T�C�Y���擾���̍s��
			*/
			MatType(const cv::Mat &mat)
				: data_type(mat.type())
			{
				sizes.assign(mat.dims, 0);
				for (int ii = 0; ii < mat.dims; ++ii) {
					sizes[ii]	=	mat.size[ii];
				}
			}

			/**
			*	�s��^�C�v���m�̈�v��r
			*	@param	type	��r�Ώۂ̍s��^�C�v
			*	@return	true:��v, false:�قȂ�
			*/
			bool	equals(const MatType &type) const
			{
				if (this->data_type != type.data_type) {
					return false;
				}

				if (this->sizes.size() != type.sizes.size()) {
					return false;
				}

				for (std::size_t ii = 0; ii < sizes.size(); ++ii) {
					if (this->sizes[ii] != type.sizes[ii]) {
						return false;
					}
				}

				return true;
			}

			/**
			*	�s��^�C�v���m�̈�v��r
			*	@param	mat	�Ώۂ̍s��
			*	@return	true:��v, false:�قȂ�
			*/
			bool	equals(const cv::Mat &mat) const
			{
				MatType	lhs(mat);

				return equals(lhs);
			}

			/**
			*	�s��^�C�v�ōs���������̑��v�f����Ԃ�
			*	@return ���v�f��
			*/
			std::size_t	total() const
			{
				std::size_t	num	=	1;
				for (std::size_t ii = 0; ii < sizes.size(); ++ii) {
					num	*=	sizes[ii];
				}

				num	*=	CV_MAT_CN(data_type);

				return num;
			}

			/**
			*	0,1�����ڂ��g�p����cv::Size���쐬���ĕԂ�
			*	@return	height=0������,width=1�����ڂ̗v�f����cv::Size
			*/
			cv::Size	size() const
			{
				return cv::Size(sizes[1], sizes[0]);
			}

			/**
			*	�V���O���`�����l�����ǂ�����Ԃ�
			*	@return	true:�V���O���`�����l��, false:�V���O������Ȃ�
			*/
			bool	is_single_channel() const
			{
				return CV_MAT_CN(data_type) == 1;
			}

			/**
			*	��x�N�g���̏����𖞂����T�C�Y�ɂȂ��Ă邩��Ԃ�
			*	@return	true:��x�N�g��, false:��x�N�g������Ȃ�
			*/
			bool	is_col_vector() const
			{
				if (sizes.size() != 2) {
					return false;
				}

				if (sizes[1] != 1) {
					return false;
				}

				if (sizes[0] <= 0) {
					return false;
				}

				return true;
			}

			/**
			*	�`�����l�����擾
			*	@return	�`�����l����
			*/
			int	channels() const
			{
				return CV_MAT_CN(data_type);
			}

			/**
			*	�s���(�z��I��)���������擾
			*	@return	������
			*/
			size_t	dims() const
			{
				return sizes.size();
			}

			/**
			*	�f�[�^�̃r�b�g�[�x�R�[�h�擾 (CV_64F�Ƃ��̃A��)
			*	@return	�r�b�g�[�x�R�[�h
			*/
			int	depth() const
			{
				return CV_MAT_DEPTH(data_type);
			}

		protected:
	};

	/**
	*	�w�K�f�[�^�Z�b�g�N���X
	*/
	class TrainingSet
	{
		public:

			/**
			*	�R���X�g���N�^
			*	@param	feature_type	�����x�N�g���̍s��^�C�v
			*	@param	label_type		���t�f�[�^�̍s��^�C�v
			*/
			TrainingSet(const MatType &feature_type, const MatType &label_type)
				: feature_type_(feature_type), label_type_(label_type) {};

			TrainingSet(const TrainingSet &ts)
				: feature_type_(ts.get_feature_type()), label_type_(ts.get_label_type()),
				examples_(ts.examples_)
			{};

			TrainingSet() {};

			virtual	~TrainingSet() {};

			/**
			*	�i�[����Ă���P���T���v�����擾����
			*	@param	idx	�T���v���̃C���f�b�N�X
			*/
			const PtrTrainingExample	operator[](std::size_t idx) const 
			{
				return examples_[idx]; 
			}

			/**
			*	�P���T���v����ǉ�����
			*	@param	example	�ǉ�����P���T���v��
			*	@return	true:����, false:���s
			*/
			bool	push_back(const PtrTrainingExample &example);

			/**
			*	�T���v�������擾
			*	@return	�T���v������
			*/
			std::size_t	size() const { return examples_.size(); };
			
			/**
			*	�ێ�����S�T���v���̍폜
			*/
			void	clear() { examples_.clear(); };

			/**
			*	�T���v���̍폜
			*	@param	idx	�폜����T���v���̃C���f�b�N�X
			*/
			void	erase(int idx)
			{
				examples_.erase(examples_.begin() + idx);
			}

			/**
			*	���x���̃G���g���s�[�v�Z����
			*	@return	���x���G���g���s�[  �v�Z�ł��Ȃ������肵���ꍇ0
			*/
			double	compute_label_entropy() const ;

			/**
			*	�S�T���v���Ń��x���̘a�����
			*	@return	�a�̌v�Z����
			*/
			cv::Mat	calc_label_sum()const;
		
			/**
			*	�����x�N�g���̃^�C�v���擾
			*/
			MatType	get_feature_type() const
			{
				return feature_type_;
			}

			/**
			*	���x���̃^�C�v���擾
			*/
			MatType	get_label_type() const
			{
				return label_type_;
			}

			/**
			*	�����x�N�g���y�ы��t�f�[�^�����ꂼ��L2���K�������V�����f�[�^�Z�b�g��Ԃ�
			*	�V�����̈�m�ۂ���̂Ń���������
			*	@return	L2���K�������f�[�^�Z�b�g
			*/
			TrainingSet	calc_l2_normalized_set() const ;

			/**
			*	�����x�N�g���̊e�����ɂ��āC�ŏ��l�����߂�
			*	�T���v���͂��ׂ�double�ɃR���o�[�g���ĎZ�o����D
			*	= �o�͂�double�s��
			*	@param	min_val	�ŏ��l�Z�o����
			*/
			void	find_feature_min(cv::Mat &min_val) const ;

			/**
			*	�����x�N�g���̊e�����ɂ��āC�ő�l�����߂�
			*	�T���v���͂��ׂ�double�ɃR���o�[�g���ĎZ�o����D
			*	= �o�͂�double�s��
			*	@param	max_val	�ő�l�Z�o����
			*/
			void	find_feature_max(cv::Mat &max_val) const ;
			
			/**
			*	�s��z�񂩂�w�K�f�[�^�Z�b�g�����D
			*	@param	features	�����x�N�g��
			*	@param	labels		���t�f�[�^
			*	@param	copy_data	�s��̃R�s�[�����t���O
			*	@param	vectorize	feature,label���x�N�g���ɂ��鏈���̗L���t���O
			*	@return	���������f�[�^�Z�b�g
			*/
			static TrainingSet	mat_arr_to_train_set(const std::vector<cv::Mat> &features, const std::vector<cv::Mat> &labels, bool copy_data = false, bool vectorize = false) ;

			/**
			*	�f�[�^�Z�b�g���̃T���v������C�w��T���v�����ŃT���v�����O����
			*	�V�����f�[�^�Z�b�g����ĕԂ�
			*	@param	sample_rate	�T���v����
			*	@param	rng			�����G���W��
			*	@return	�T���v�����O�����V�����f�[�^�Z�b�g
			*/
			TrainingSet	random_sample(double sample_rate, std::mt19937 &rng) const ;

			/**
			*	�ێ�����f�[�^�̒��ŁCbag�Ɋ܂܂�Ȃ����̂�Ԃ�
			*	(=�|�C���^��v���Ȃ�����)
			*	@param	bag	���̓Z�b�g
			*	@return	not(this �� bag)
			*/
			TrainingSet	get_out_of_bag(const TrainingSet &bag) const ;

			/**
			*	���t�̍s��̕��ς����߂�
			*	@param	dst	�o�͂̍s��
			*/
			void compute_target_mean(cv::Mat &dst) const;

			/**
			*	���t�̍s��̕��U�����߂�
			*	@return	���U
			*/
			double compute_target_var() const;

		protected:

			/** �����x�N�g���̃^�C�v */
			MatType		feature_type_;

			/** ���t�f�[�^�̃^�C�v */
			MatType		label_type_;

			/** �P���T���v���̃R���e�i */
			std::vector<const PtrTrainingExample>	examples_;

			/**
			*	�L���ȌP���T���v�����ǂ������肷��
			*	@param	example	����Ώۂ̌P���T���v��
			*	@return	true:�L��, false: ����
			*/
			bool	is_valid_example(const PtrTrainingExample &example) const;

			/**
			*	���t�̓��̕���
			*	@param	dst	�o��
			*/
			void compute_target_mean2(cv::Mat &dst) const;
	};

};
