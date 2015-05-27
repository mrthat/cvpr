#pragma once
#include <sstream>
#include "TreeNode.h"

namespace cvpr
{
	namespace TreeNode
	{
		//! 乱数が非常に小さかった場合に代わりに使う値
		const double DELTA	=	0.0001f;

		/**
		*	軸並行で分割する
		*/
		template<typename ty>
		class SplitNodeAxisAligned : public TreeNode::SplitNodeBase
		{
			public:

				SplitNodeAxisAligned() {};

				SplitNodeType	get_split_type() const { return SPLIT_TYPE_AXISALIGNED; };
				
				virtual int	save(cv::FileStorage &cvfs) const
				{
					return SplitNodeBase::save(cvfs);
				}

				virtual int	load(cv::FileStorage &cvfs)
				{
					return SplitNodeBase::load(cvfs);
				}

			protected:
				enum {NUM_SELECT_FEATURES = 1};

				unsigned	get_num_attributes() const { return NUM_SELECT_FEATURES; };

				void		init_params(const TrainingSet &train_set, const StaticalModelParameter *param, std::mt19937 &rnd) {};

				double		kernel_function(const cv::Mat &feature, const SplitNodeParameterBase *param) const
				{					
					const ty	*data_head	=	reinterpret_cast<const ty*>(feature.data);
					return *(data_head + this->attributes_[0]);
				};
		};
		
		template<typename ty>
		class SplitNodeOrientedLine : public TreeNode::SplitNodeBase
		{
			public:
				SplitNodeOrientedLine() : line_params_(NUM_PARAMS, 0) {};

				SplitNodeType	get_split_type() const { return SPLIT_TYPE_LINE; };
				
				virtual int	save(cv::FileStorage &cvfs) const
				{
					SplitNodeBase::save(cvfs);
					cv::internal::WriteStructContext	wsc(cvfs, "SNOL", cv::FileNode::MAP);

					cv::write<double>(cvfs, "line_param", this->line_params_);

					return 0;
				}

				virtual int	load(cv::FileStorage &cvfs)
				{
					SplitNodeBase::load(cvfs);
					cv::FileNode	target_node	=	cvfs["SNOL"];

					cv::read<double>(target_node["line_param"], this->line_params_);

					return 0;
				}

			protected:

				enum {NUM_SELECT_FEATURES = 2};

				enum {NUM_PARAMS = 3};

				unsigned	get_num_attributes() const { return NUM_SELECT_FEATURES; };

				void		init_params(const TrainingSet &train_set, const StaticalModelParameter *param, std::mt19937 &rnd)
				{
					for (unsigned ii = 0; ii < this->line_params_.size(); ++ii) {
		
						//[0..1]
						this->line_params_[ii] = (rnd() - rnd.min()) / (double)(rnd.max()-rnd.min()); 

						//[-1..1]
						this->line_params_[ii] = this->line_params_[ii] * 2 - 1;

					}
				}

				double		kernel_function(const cv::Mat &feature, const SplitNodeParameterBase *param) const
				{
					const ty	*data_head	=	reinterpret_cast<const ty*>(feature.data);
					ty	val0	=	*(data_head + this->attributes_[0]);
					ty	val1	=	*(data_head + this->attributes_[1]);
					//a * x + b * y + c
					return	val0 * this->line_params_[0] + val1 * this->line_params_[1] + this->line_params_[2];
				}

				std::vector<double>	line_params_;
		};
		
		template<typename ty>
		class SplitNodeConicSection : public cvpr::TreeNode::SplitNodeBase
		{
			public:

				SplitNodeConicSection() : conic_params_(NUM_PARAMS, 0.0f) {};

				SplitNodeType	get_split_type() const { return SPLIT_TYPE_CONIC; };

				virtual int	save(cv::FileStorage &cvfs) const
				{
					SplitNodeBase::save(cvfs);
					cv::internal::WriteStructContext	wsc(cvfs, "SNCS", cv::FileNode::MAP);

					cv::write<double>(cvfs, "cs_param", this->conic_params_);

					return 0;
				}

				virtual int	load(cv::FileStorage &cvfs)
				{
					SplitNodeBase::load(cvfs);
					cv::FileNode	target_node	=	cvfs["SNCS"];

					cv::read<double>(target_node["cs_param"], this->conic_params_);

					return 0;
				}

			protected:

				enum {NUM_SELECT_FEATURES = 2};

				enum {NUM_PARAMS = 6};

				unsigned	get_num_attributes() const { return NUM_SELECT_FEATURES; };

				void		init_params(const TrainingSet &train_set, const StaticalModelParameter *param, std::mt19937 &rnd)
				{
					for (unsigned ii = 0; ii < this->conic_params_.size(); ++ii) {
						//[0..1]
						this->conic_params_[ii] = (rnd() - rnd.min()) / (float)(rnd.max()-rnd.min()); 

						//[-1..1]
						this->conic_params_[ii] = this->conic_params_[ii] * 2 - 1;

						//A, B, C not zero
						if (ii < 3) {
							if (fabs(this->conic_params_[ii]) < std::numeric_limits<double>::min()) {
								this->conic_params_[ii]	=	DELTA;
							}
						}
					}
				}

				double		kernel_function(const cv::Mat &feature, const SplitNodeParameterBase *param) const
				{
					const ty	*data_head	=	reinterpret_cast<const ty*>(feature.data);
					double	xx	=	*(data_head + this->attributes_[0]);
					double	yy	=	*(data_head + this->attributes_[1]);

					//Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0

					return xx * xx * this->conic_params_[0] + xx * yy * this->conic_params_[1]
						 + yy * yy * this->conic_params_[2] + xx * this->conic_params_[3]
						 + yy * this->conic_params_[4]		+ this->conic_params_[5];
				}

				std::vector<double>	conic_params_;
		};
		
		template<typename ty>
		class SplitNodeAtan : public TreeNode::SplitNodeBase
		{
			public:

				SplitNodeAtan() {};

				SplitNodeType	get_split_type() const { return SPLIT_TYPE_ATAN; };

			protected:

				enum {NUM_SELECT_FEATURES = 2};

				unsigned	get_num_attributes() const { return NUM_SELECT_FEATURES; };

				void		init_params(const TrainingSet &train_set, const StaticalModelParameter *param, std::mt19937 &rnd) {};

				double		kernel_function(const cv::Mat &feature, const SplitNodeParameterBase *param) const
				{
					const ty	*data_head	=	reinterpret_cast<const ty*>(feature.data);
					double	xx	=	*(data_head + this->attributes_[0]);
					double	yy	=	*(data_head + this->attributes_[1]);
					return std::atan2(xx, yy);
				}
		};
		
		template<typename ty>
		class SplitNodeMin : public TreeNode::SplitNodeBase
		{
			public:

				SplitNodeMin() {};

				SplitNodeType	get_split_type() const { return SPLIT_TYPE_MIN; };

			protected:

				enum {NUM_SELECT_FEATURES = 2};

				unsigned	get_num_attributes() const { return NUM_SELECT_FEATURES; };

				void		init_params(const TrainingSet &train_set, const StaticalModelParameter *param, std::mt19937 &rnd) {};

				double		kernel_function(const cv::Mat &feature, const SplitNodeParameterBase *param) const
				{
					const ty	*data_head	=	reinterpret_cast<const ty*>(feature.data);
					double	xx	=	*(data_head + this->attributes_[0]);
					double	yy	=	*(data_head + this->attributes_[1]);
					return std::min(xx, yy);
				}
		};

		//2Dマルチチャネル想定
		template<typename ty>
		class SplitNodeHaar : public SplitNodeBase
		{
			public:
				
				SplitNodeHaar()
				{
					this->roi_.assign(NUM_SELECT_FEATURES, cv::Rect());
					this->coi_.assign(NUM_SELECT_FEATURES, 0);
				};
				
				virtual int	save(cv::FileStorage &cvfs) const
				{
					SplitNodeBase::save(cvfs);
					cv::internal::WriteStructContext	wsc(cvfs, "SNHAAR", cv::FileNode::MAP);

					for (unsigned ii = 0; ii < NUM_SELECT_FEATURES; ++ii) {
						char	node_name[256];
						sprintf(node_name, "roi%d", ii);
						cv::write<int>(cvfs, node_name, this->roi_[ii]);
						sprintf(node_name, "col%d", ii);
						cv::write(cvfs, node_name, this->coi_[ii]);
					}

					return 0;
				}

				virtual int	load(cv::FileStorage &cvfs)
				{
					SplitNodeBase::load(cvfs);
					cv::FileNode	top_node	=	cvfs["SNHAAR"];

					for (unsigned ii = 0; ii < NUM_SELECT_FEATURES; ++ii) {
						std::vector<int>	roi;
						char	node_name[256];
						
						sprintf(node_name, "roi%d", ii);
						cv::read<int>(top_node[node_name], roi, std::vector<int>(4, 0));
						this->roi_[ii].x		=	roi[0];
						this->roi_[ii].y		=	roi[1];
						this->roi_[ii].width	=	roi[2];
						this->roi_[ii].height	=	roi[3];
						sprintf(node_name, "col%d", ii);
						cv::read(top_node[node_name], this->coi_[ii], 0);
					}

					return 0;
				}

				virtual SplitNodeType	get_split_type() const { return SPLIT_TYPE_HAAR; };

			protected:
				enum { NUM_SELECT_FEATURES = 2 };
				std::vector<cv::Rect>	roi_;
				std::vector<int>		coi_;

				unsigned	get_num_attributes() const { return NUM_SELECT_FEATURES; };

				virtual double	CalcIntegral(const cv::Mat &feature, const cv::Rect &roi, int coi) const
				{
#ifdef _DEBUG
					if (feature.dims != 2) {
						assert(!"dims!=2");
					}

					if (roi.x < 0) {
						assert(!"roi.x < 0");
					}

					if (roi.y < 0) {
						assert(!"roi.y < 0");
					}

					if ((feature.rows - roi.y) < roi.height) {
						assert(!"(feature.rows - roi.y) < roi.height");
					}

					if ((feature.cols - roi.x) < roi.width) {
						assert(!"(feature.cols - roi.x) < roi.width");
					}
#endif

					int	col_step	=	(int)feature.step1(0);
					int	ch_step		=	(int)feature.step1(1);
					const ty	*raw_data	=	(const ty*)feature.data;
					double	acc	=	0;

					for (int dh = 0; dh < roi.height; ++dh) {
						for (int dw = 0; dw < roi.width; ++dw) {
							acc	+=	*(raw_data + dw * ch_step + coi);
						}
						raw_data	+=	col_step;
					}
					acc	/=	roi.area();
					return acc;
				}
				
				virtual void			init_params(const TrainingSet &train_set, const StaticalModelParameter *param, std::mt19937 &rnd)
				{
					std::uniform_real_distribution<>	uni_dist(0.0, 1.0);
					MatType				ftype			=	train_set.get_feature_type();
					const int			num_ch			=	ftype.channels();
					std::vector<int>	feature_size	=	ftype.sizes;
					cv::Size			img_size;

					if (feature_size.size() != 2) {
						return ;
					}

					img_size	=	cv::Size(feature_size[0], feature_size[1]);

					for (int ii = 0; ii < NUM_SELECT_FEATURES; ++ii) {
						this->coi_[ii]		=	(int)(num_ch * uni_dist(rnd));
						this->roi_[ii].x	=	(int)((img_size.width -1) * uni_dist(rnd));
						this->roi_[ii].y	=	(int)((img_size.height -1) * uni_dist(rnd));

						this->roi_[ii].x	-=	this->roi_[ii].x % 4;
						this->roi_[ii].y	-=	this->roi_[ii].x % 4;

						this->roi_[ii].width	=	(int)((img_size.width - this->roi_[ii].x) * uni_dist(rnd));
						this->roi_[ii].height	=	(int)((img_size.height - this->roi_[ii].y) * uni_dist(rnd));
						
						this->roi_[ii].width	-=	this->roi_[ii].x % 4;
						this->roi_[ii].height	-=	this->roi_[ii].x % 4;

						this->roi_[ii].width	=	std::max(this->roi_[ii].width, 1);
						this->roi_[ii].height	=	std::max(this->roi_[ii].height, 1);
					}

				};
				virtual double			kernel_function(const cv::Mat &feature, const SplitNodeParameterBase *param) const
				{
					return CalcIntegral(feature, this->roi_[0], this->coi_[0]) - CalcIntegral(feature, this->roi_[1], this->coi_[1]);
					//return feature.at<ty>(0, 0) - feature.at<ty>(1, 0);
				}

		};

		template<typename ty>
		class SplitNodeHaarIntegral : public SplitNodeHaar<ty>
		{
			public:

				virtual SplitNodeType	get_split_type() const { return SPLIT_TYPE_HAAR_INT; };

			protected:

				virtual double	CalcIntegral(const cv::Mat &feature, const cv::Rect &roi, int coi) const
				{
#ifdef _DEBUG
					if (feature.dims != 2) {
						assert(!"dims!=2");
					}

					if (roi.x < 0) {
						assert(!"roi.x < 0");
					}

					if (roi.y < 0) {
						assert(!"roi.y < 0");
					}

					if ((feature.rows - roi.y) < roi.height) {
						assert(!"(feature.rows - roi.y) < roi.height");
					}

					if ((feature.cols - roi.x) < roi.width) {
						assert(!"(feature.cols - roi.x) < roi.width");
					}
#endif
					
					int	col_step	=	(int)feature.step1(0);
					int	ch_step		=	(int)feature.step1(1);
					cv::Point	tmp	=	roi.br();
					cv::Point	tl	=	cv::Point(roi.x * ch_step + coi, roi.y * col_step);
					cv::Point	br	=	cv::Point(tmp.x * ch_step + coi, tmp.y * col_step);
					const ty	*raw_data	=	(const ty*)feature.data;
					ty	tl_val	=	*(raw_data + tl.y + tl.x);
					ty	tr_val	=	*(raw_data + tl.y + br.x);
					ty	bl_val	=	*(raw_data + br.y + tl.x);
					ty	br_val	=	*(raw_data + br.y + br.x);
					double	acc	=	br_val - bl_val - tr_val + tl_val;
					acc	/=	roi.area();
					return acc;
				}
		};

		//! 形状参照分割ノードのパラメータ
		class ShapeIndexedSplitParameter : public SplitNodeParameterBase
		{
			public:
				//! 形状
				std::vector<cv::Point2d>	shape;

				/**
				*	特徴位置に適用する変換
				*	推定位置を平均位置に変換した時の逆変換を入れる
				*/
				cv::Mat	transform;

			protected:
		};

		//! 形状参照分割ノードの学習パラメータ
		class ShapeIndexedTrainParameter : public StaticalModel
		{
			public:

			/**
			*	形状点周りの特徴点の配置範囲半径(pix)
			*/
			int	radius;

			/**
			*	形状点個数
			*/
			std::size_t	num_shape;
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

			protected:

				enum {
					//! 形状店回りの特徴点数
					NUM_FEATURE_POS	=	2,
				};

				unsigned get_num_attributes() const { return 0; }

				void init_params(const TrainingSet &train_set, const StaticalModelParameter *param, std::mt19937 &rnd);

				double kernel_function(const cv::Mat &feature, const SplitNodeParameterBase *param) const;

				//! 使用する形状点のインデックス
				std::size_t	shape_index;

				//! 形状からのオフセット 形状データは2dを想定
				std::vector<cv::Point2d>	offsets;

		};
	};
};