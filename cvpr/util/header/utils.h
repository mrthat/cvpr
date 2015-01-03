#pragma once
#include <vector>
#include <random>
#include <opencv2\core\core.hpp>

#define	STRINGIZE(var) (#var)

namespace cvpr
{
	/**
	*	���z�񂩂烉���_���T���v������bag��out of bag�ɓ����
	*	����鐔�͓K���ŗ����̋�ɂ���Ă͂������
	*	@param	tp				�z��̗v�f�̌^
	*	@param	src				���͔z��
	*	@param	bag				�o�͂�bag
	*	@param	oob				�o�͂�out of bag
	*	@param	sample_rate		src����bag�ɓ����m�� <=> [0..1.0]
	*	@param	rng				����������
	*/
	template<class tp> void	random_sample(const std::vector<tp> &src,
		std::vector<tp> &bag, std::vector<tp> &oob, double sample_rate, std::mt19937 &rng);

	/**
	*	���͔z�񂩂��l���z�ɂ���������1���T���v�����ĕԂ�
	*	@param	tp	�z��v�f�̌^
	*	@param	src	���͔z��
	*	@param	rng	�����G���W��
	*	@return	�T���v����������
	*/
	template<class tp> tp	random_sample(const std::vector<tp> &src, std::mt19937 &rng)
	{
		std::uniform_real_distribution<std::size_t>	real_dist(0, src.size());

		return src[real_dist(rng)];
	};

	/**
	*	�f�B�K���}�֐�
	*	@param	hk	�֐��l���v�Z���������̐���
	*	@return	0<hk:hk�̎��̊֐��l,else:qnan
	*/
	double	digamma(int hk);

	/**
	*	�������̎��̃f�B�K���}�֐�
	*	@param	hk	�֐��l���v�Z���������̐���
	*	@return	0<hk:hk�̎��̊֐��l,else:qnan
	*/
	double	digamma_half(int hk);

	/**
	*	���������؂蕶�����w�肵�ĕ�������
	*	@param	src		�������镶����
	*	@param	delim	��؂蕶��
	*	@param	������̕�����z��
	*/
	void	split(const std::string &src, char delim, std::vector<std::string> &dst) ;

	/**
	*	�s����̍ő�v�f�̃C���f�b�N�X��Ԃ�
	*	@param	mat	�v�f�̎Z�o���s��
	*	@return	�ő�v�f�̃C���f�b�N�X
	*/
	int		max_idx(const cv::Mat &mat);

	/**
	*	m + log(sum(exp(mat[i]-m)))���v�Z����
	*	m = max(mat[i])
	*	mat��2D,double,1channel
	*	@param	mat	2d,double,1channel�̍s��
	*	@return	m + log(sum(exp(mat[i]-m)))
	*/
	double	log_sum_exp(const cv::Mat &mat);

	/**
	*	���͍s���1cn,1��ɂ����s��̃w�b�_�[���擾
	*	@param	mat	���͍s��
	*	@return	mat��1ch1col�ɂ����w�b�_�[
	*/
	cv::Mat	get_colvec_header(const cv::Mat &mat) ;

	/**
	*	���͍s�񂪗�x�N�g�������ׂ�
	*	@param	mat	���͍s��
	*	@return	true:��x�N�g��, false:����ȊO
	*/
	bool	is_column_vector(const cv::Mat &mat);

	/**
	*	��l���z [0..1)��mat�̗v�f�𖄂߂�
	*	@param	ty		�o�͍s��̗v�f�̌^(=��{�^)
	*	@param	rng		�����G���W��
	*	@param	mat		�v�f���ߐ�̍s��
	*/
	template<typename ty>
	void	rand_init(std::mt19937 &rng, cv::Mat &mat) 
	{
		cv::Mat	tmp	=	mat.reshape(1, get_total1(mat));
		std::uniform_real_distribution<>	dst;

		for (int ii = 0; ii <tmp.rows; ++ii) {
			tmp.at<ty>(ii)	=	dst(rng);
		}
	};

	/**
	*	vector��{}�ŏ������ł��Ȃ��̂ł�����ۂ����Ƃ��������
	*	@param	v0	0�Ԗڂ̗v�f
	*	@param	v1	1�Ԗڂ̗v�f
	*	@param	v2	2�Ԗڂ̗v�f
	*	@return	{v0,v1,v2}��vector
	*/
	template<typename ty>
	std::vector<ty>	get_vector(ty v0, ty v1, ty v2)
	{
		std::array<ty, 3>	arr = {v0, v1, v2};

		return std::vector<ty>(arr.begin(), arr.end());
	}

	/**
	*	mat�̔z�񂩂�C(�z��I��)1���������s������D
	*	�������͐V�����m�ۂ����
	*	�z�񂪂���Ȃ��s�񂪕ς��邪�C�z����̍s��`�����Ⴄ�Ƃ���opencv�̗�O�ł�
	*	@param	arr	�쐬���̍s��̔z��
	*	@param	dst	�o�͍s��
	*	@return	true:����, false���s
	*/
	bool	mat_arr_to_hdim_mat(const std::vector<cv::Mat> &arr, cv::Mat &dst) ;

	/**
	*	(�z��I��)��ԊO���̗v�f����1�̏ꍇ��
	*	1�������Ȃ��s���Ԃ�
	*	���͂�2d�̏ꍇ�̓R�s�[��Ԃ�
	*	@param	mat			���͍s��
	*	@param	dst			�o�͍s��
	*	@param	copy_data	�V�����������m�ۂ��邩�ǂ����̃t���O
	*	@return	0: ����, 1:���͂�2D�Ȃ̂ŃR�s�[����, ����ȊO:�G���[
	*/
	int	reduce_mat_dim(const cv::Mat &mat, cv::Mat &dst, bool copy_data = false) ;

	/**
	*	�s���(�z��I��)������1���炵��vector�ɋl�߂�
	*	@param	mat			���͍s��
	*	@param	dst			�o�͔z��
	*	@param	copy_data	�V�����������m�ۂ��邩�ǂ����̃t���O
	*	@return	0: ����, 1:���͂�2D�Ȃ̂ŃR�s�[����, ����ȊO:�G���[
	*/
	int	reduce_mat_dim(const cv::Mat &mat, std::vector<cv::Mat> &dst, bool copy_data = false) ;

	/**
	*	total * channels��Ԃ�
	*	�̃`�����l�����l�������v�f��
	*	@param	mat	���͍s��
	*	@return	�v�f��
	*/
	int	get_total1(const cv::Mat &mat) ;

	/**
	*	���W��mat���ɂ��邩���ׂ�
	*	@param	mat	�����Ώۂ̍s��
	*	@param	pt	mat���̓_���W
	*/
	bool contains(const cv::Mat &mat, const cv::Point pt, int margin = 0) ;
};