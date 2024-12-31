#pragma once

#include "yolo/pubDef.h"

class InferInterface{
public:
    virtual std::shared_future<std::vector<Result>> forward(Input* inputs, int& n_images, float conf_thre, float nms_thre, bool inferLog=false) = 0;
	virtual std::vector<std::vector<float>> get_records() = 0;
	//-------------------------------------------------------------------------------------------------------------
	virtual void warmup() = 0;
	virtual bool add_images(Input* inputs, int& n_images, float conf_thre, float nms_thre, std::string channel_id) = 0;
	virtual int get_qsize() = 0;
	//-------------------------------------------------------------------------------------------------------------
};

std::shared_ptr<InferInterface> create_infer(std::string &file, int max_det, std::string& device, bool modelLog, bool multi_label=true);
//-------------------------------------------------------------------------------------------------------------
std::shared_ptr<InferInterface> create_infer(std::string nickName, PushResult callback, void *userP, std::string &file, int max_det, int max_qsize, std::string& device, bool modelLog, bool multi_label=true);
//-------------------------------------------------------------------------------------------------------------