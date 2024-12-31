#pragma once

#include <map>
#include <string>
#include <vector>
#include <memory>
#include <future>
#include <cstring>

enum TASKTYPE {
	Detection,
	Pose
};

static std::map<std::string, int> TASKMAP = {
	{"detect", 	TASKTYPE::Detection},
	{"pose", 	TASKTYPE::Pose}
};

struct Input{
	long timestamp;									// 时间戳, 单位ms
	std::string unique_id;
	unsigned char* data;
	int height;
	int width;
};

struct KeyPoint_t {
    float x = 0.0f;     
    float y = 0.0f;     
    float score = 0.0f; 
};

struct Box{
    float left = -1.;
	float top = -1.;
	float right = -1.;
	float bottom = -1.;
	float score = -1.;
    int label = -1;
	int track_id = -1;
	float intrude_ratio = 0.;
	std::vector<KeyPoint_t> keypoints; 
	/* ------------------------------------------------------------------ */
	bool sure = false;
};

struct Result{
	/* ------------------------------------------------------------------ */
	std::string channel_id = "";					// 摄像头通道id
	std::string unique_id = "";						// 图像id
	long timestamp = 0;								// 时间戳, 单位ms
	std::string msg = "";
	std::string event_id = "";						// 事件id
	std::string event_type = "";					// 事件类型
	unsigned char* data = nullptr;
	int height = -1;
	int width = -1;
	int naviStatus = 0;
	int data_ref_count = 0;
	long proof_timestamp = 0;
	/* ------------------------------------------------------------------ */
	std::vector<Box> bboxes;
	/* ------------------------------------------------------------------ */
	bool provide_result = false;
	bool checked = false;
	std::vector<std::tuple<int, float>> pdf;
	bool result = false;
	std::string nick_name = "NULL";
};

typedef void (*PushResult)(std::vector<Result> results, void *userP);
// class name到class index的映射: 不同模型的推理结果产生后, 通过该映射表, 将类别名映射到统一的类别索引 
static std::map<std::string, int> UNIFIED_CLS2IDX = {
    {"person",              0},    
    {"dog",      			1},    
    {"cat",    				2}   
}; 