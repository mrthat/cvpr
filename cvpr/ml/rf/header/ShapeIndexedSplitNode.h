#pragma once
#include <sstream>
#include "TreeNode.h"
#include "..\..\..\util\include\utils.h"

namespace cvpr
{
	namespace TreeNode
	{	
		//! 形状参照分割ノードのパラメータ
		class ShapeIndexedSplitParameter : public SplitNodeParameterBase
		{
			public:
			//! 形状
			std::vector<cv::Point2f>	shape;

			/**
			*	特徴位置に適用する変換
			*	正規化された平均位置を推定位置に変換する
			*	2*3のアフィン変換行列
			*/
			cv::Mat	transform;

			protected:
		};

		//! 形状参照分割ノードの学習パラメータ
		class ShapeIndexedTrainParameter : public virtual StaticalModelParameter
		{
			public:

			/**
			*	デフォルトコンストラクタ
			*/
			ShapeIndexedTrainParameter()
				: radius(0.05) {};

			/**
			*	形状点周りの特徴点の配置範囲半径(pix)
			*/
			double	radius;

			/**
			*	形状点個数
			*/
			std::size_t	num_shape;

			protected:

			virtual int		save(const std::string &save_path) const { return 0; };

			virtual int		load(const std::string &load_path) { return 0; };
		};

		/**
		*	パラメータで渡された形状点からの相対位置を使用して
		*	splitするクラス．
		*/
		template<typename ty>
		class SplitNodeShapeIndexed : public TreeNode::SplitNodeBase
		{
			public:

			SplitNodeType get_split_type() const { return SPLIT_TYPE_SHAPE_INDEXED; };

			int	save(cv::FileStorage &cvfs) const
			{
				int	ret	=	SplitNodeBase::save(cvfs);

				if (0 != ret)
					return ret;

				cv::write(cvfs, STRINGIZE(shape_index), (int)shape_index);

				cv::write(cvfs, STRINGIZE(offsets), offsets);

				return 0;
			}

			int	load(cv::FileStorage &cvfs)
			{
				int	ret	=	SplitNodeBase::load(cvfs);
				int	tmp	=	0;

				if (0 != ret)
					return ret;

				cv::read(cvfs[STRINGIZE(shape_index)], tmp, 0);

				shape_index	=	(std::size_t)tmp;

				cv::read(cvfs[STRINGIZE(offsets)], offsets);

				return 0;
			}

			protected:

			enum {
				//! 形状店回りの特徴点数
				NUM_FEATURE_POS	=	2,
			};

			unsigned get_num_attributes() const { return 0; }

			void init_params(const TrainingSet &train_set, const StaticalModelParameter *param, std::mt19937 &rnd)
			{
				// 形状パラメータは全データ同じ数ある想定で欠損値考慮しない

				const ShapeIndexedTrainParameter	*param_	=	dynamic_cast<const ShapeIndexedTrainParameter*>(param);

				assert(nullptr != param_);
				
				std::uniform_int_distribution<int>		distr_idx((int)0, (int)param_->num_shape - 1);
				std::uniform_real_distribution<double>	distr_pos(-param_->radius, param_->radius);

				// 対象形状点決める
				shape_index	=	distr_idx(rnd);

				// 相対特徴点位置を決める
				for (int ii = 0; ii < NUM_FEATURE_POS; ++ii) {
					offsets.push_back(cv::Point2f(distr_pos(rnd), distr_pos(rnd)));
				}
			}

			double kernel_function(const cv::Mat &feature, const SplitNodeParameterBase *param) const
			{
				std::vector<cv::Point2f>	warped_offsets;
				std::vector<cv::Point2f>	feature_pos;
				cv::Mat	transform;
				const ShapeIndexedSplitParameter	*param_	=	dynamic_cast<const ShapeIndexedSplitParameter*>(param);
				std::vector<double>	val;

				assert(nullptr != param_);

				cv::transform(offsets, warped_offsets, param_->transform);

				for (std::size_t ii = 0; ii < warped_offsets.size(); ++ii) {
					cv::Point2f	pos	=	param_->shape[shape_index] + warped_offsets[ii];

					pos	=	cvpr::round<float>(feature, pos); // 範囲外なら例外でいい気もする?

					val.push_back((double)feature.at<ty>((int)pos.y, (int)pos.x));
				}

				return val[0] - val[1];
			}

			//! 使用する形状点のインデックス
			std::size_t	shape_index;

			//! 形状からのオフセット 形状データは2dを想定
			std::vector<cv::Point2f>	offsets;

		};
	};
};