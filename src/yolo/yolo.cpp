#include <stdio.h>
#include <fstream> 
#include <map>
#include <string>
#include <vector>
#include <math.h>
#include <functional>                                   // std::ref()需要用这个库
#include <unistd.h>
#include <thread>                                       // 线程
#include <queue>                                        // 队列
#include <mutex>                                        // 线程锁
#include <chrono>                                       // 时间库
#include <memory>                                       // 智能指针
#include <future>                                       // future和promise都在这个库里，实现线程间数据传输
#include <condition_variable>                           // 线程通信库
#include <filesystem> 
#include <unistd.h>
#include <dirent.h>                                     // opendir和readdir包含在这里
#include <sys/stat.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferRuntime.h>

#include "spdlog/logger.h"                              // spdlog日志相关
#include "spdlog/spdlog.h"                              // spdlog日志相关
#include "spdlog/sinks/basic_file_sink.h"               // spdlog日志相关
#include "opencv2/opencv.hpp"
#include "yolo/yolo.h"
#include "yolo/model-utils.h"

using namespace std;
using namespace cv;
using namespace nvinfer1;

using time_point = chrono::high_resolution_clock;
template <typename Rep, typename Period>
float micros_cast(const std::chrono::duration<Rep, Period>& d) {return static_cast<float>(std::chrono::duration_cast<std::chrono::microseconds>(d).count()) / 1000.;};

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)
bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line){
    if(code != cudaSuccess){    
        const char* err_name = cudaGetErrorName(code);    
        const char* err_message = cudaGetErrorString(code);  
        printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);   
        return false;
    }
    return true;
}

template<typename _T>
shared_ptr<_T> make_nvshared(_T* ptr){
    return shared_ptr<_T>(ptr, [](_T* p){delete p;});
}

struct Job{
    shared_ptr<promise<vector<Result>>> pro;        //为了实现线程间数据的传输，需要定义一个promise，由智能指针托管, pro负责将消费者消费完的数据传递给生产者
    vector<Mat> input_images;                       // 输入图像, 多batch 
    vector<int> heights;
    vector<int> widths; 
    string channel_id;
    vector<long> timestamps;
    vector<unsigned char*> input_images_data; 
    vector<string> unique_ids;
    bool inferLog{false};                                  // 是否打印日志
};

void preprocess_kernel_invoker(
    int src_width, int src_height, int src_line_size,
    int dst_width, int dst_height, int dst_line_size,
    uint8_t* src_device, uint8_t* intermediate_device, 
    float* dst_device, uint8_t fill_value, int dst_img_area, size_t offset
);

void postprocess_kernel_invoker(
    float* predict, int num_bboxes, int num_classes, float conf_thre_, 
    float nms_thre_, float* invert_affine_matrix, float* parray, int max_objects, 
    int NUM_BOX_ELEMENT
);

inline const char* severity_string(nvinfer1::ILogger::Severity t){
    switch(t){
        case nvinfer1::ILogger::Severity::kINTERNAL_ERROR: return "internal_error";
        case nvinfer1::ILogger::Severity::kERROR:   return "error";
        case nvinfer1::ILogger::Severity::kWARNING: return "warning";
        case nvinfer1::ILogger::Severity::kINFO:    return "info";
        case nvinfer1::ILogger::Severity::kVERBOSE: return "verbose";
        default: return "unknow";
    }
}

class TRTLogger: public nvinfer1::ILogger{
public:
    virtual void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override{
        if(severity <= Severity::kWARNING){
            // 打印带颜色的字符，格式如下：
            // printf("\033[47;33m打印的文本\033[0m");
            // 其中 \033[ 是起始标记
            //      47    是背景颜色
            //      ;     分隔符
            //      33    文字颜色
            //      m     开始标记结束
            //      \033[0m 是终止标记
            // 其中背景颜色或者文字颜色可不写
            // 部分颜色代码 https://blog.csdn.net/ericbar/article/details/79652086
            if(severity == Severity::kWARNING){
                printf("\033[33m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else if(severity <= Severity::kERROR){
                printf("\033[31m%s: %s\033[0m\n", severity_string(severity), msg);
            }
            else{
                printf("%s: %s\n", severity_string(severity), msg);
            }
        }
    }
};

class InferImpl : public InferInterface{                                        
public:
    virtual ~InferImpl(){
        stop();
        spdlog::warn("Destruct instance done!");
    }

    void stop(){
        if(running_){
            running_ = false;
            cv_.notify_one();                                                   // 通知worker给break掉        
        }
        if(worker_thread_.joinable())                                           // 子线程加入     
            worker_thread_.join();
    }

    bool startup(
        const string& file, 
        const size_t& md,
        string& device,
        bool modelLog=false,
        bool multi_label=true
    ){
        
        multi_label_ = multi_label;
        modelPath_ = file;
        modelLog_ = modelLog;
        running_ = true;                                                        // 启动后，运行状态设置为true
        string modelName = getfileName(modelPath_, true);
        vector<string> splits = splitString(modelName, "-");
        if (splits[0] == "v8" || splits[0] == "v11") // todo: 这里改了
            is_v5_ = false;
        string task_name = splitString(splits[splits.size() - 1], ".")[0];  // todo: 这里改了
        task_ = TASKMAP[task_name];
        promise<bool> pro;
        CUdevice _device = 0;
        shared_ptr<cudaDeviceProp> properties(new cudaDeviceProp);
        cudaGetDeviceProperties(properties.get(), _device); 
        auto device_name = properties->name; auto global_memory = properties->totalGlobalMem / (1 << 30);
        if (modelLog_) spdlog::info("Device {} name is {}, total memory {}GB", _device, device_name, global_memory);
        worker_thread_ = thread(&InferImpl::worker, this, std::ref(pro));       // 为什么要加std::ref, 因为创建新的线程的时候希望新的线程操作原始的pro,而不是复制一份新的pro
        return pro.get_future().get();	
    }

    // --------------------------------------------------
    bool startup(
        string nickName,
        PushResult callback,
        void* userP,
        const string& file, 
        const size_t& md,
        int& mq,
        string& device,
        bool modelLog=false,
        bool multi_label=true
    ){
        nick_name_ = nickName;
        callback_ = callback;
        userP_ = userP;
        multi_label_ = multi_label;
        // is_qsize_set_和max_qsize_为静态成员变量
        if (!is_qsize_set_){
            InferImpl::max_qsize_ = mq;
            is_qsize_set_ = true;
        }
        modelPath_ = file;
        modelLog_ = modelLog;
        running_ = true;                                                        // 启动后，运行状态设置为true
        string modelName = getfileName(modelPath_, true);
        vector<string> splits = splitString(modelName, "-");
        if (splits[0] == "v8" || splits[0] == "v11") // todo: 这里改了
            is_v5_ = false;
        string task_name = splitString(splits[splits.size() - 1], ".")[0];  // todo: 这里改了
        task_ = TASKMAP[task_name];
        promise<bool> pro;
        CUdevice _device = 0;
        shared_ptr<cudaDeviceProp> properties(new cudaDeviceProp);
        cudaGetDeviceProperties(properties.get(), _device); 
        auto device_name = properties->name; auto global_memory = properties->totalGlobalMem / (1 << 30);
        if (modelLog_) spdlog::info("Device {} name is {}, total memory {}GB", _device, device_name, global_memory);       
        worker_thread_ = thread(&InferImpl::worker, this, std::ref(pro));       // 为什么要加std::ref, 因为创建新的线程的时候希望新的线程操作原始的pro,而不是复制一份新的pro
        return pro.get_future().get();	
    }
    // --------------------------------------------

    void worker(promise<bool>& pro){
        // 加载模型
        auto deserializedData = load_file(modelPath_);
        try{
            trtlogger_ = make_shared<TRTLogger>();
            runtime_ = make_nvshared(createInferRuntime(*trtlogger_.get()));
            engine_ = make_nvshared(runtime_->deserializeCudaEngine(deserializedData.data(), deserializedData.size()));
        }catch (const std::exception& e) {
            // failed
            running_ = false;
            spdlog::error("Load model failed from path: {}!", modelPath_);
            pro.set_value(false);                                               // 将start_up中pro.get_future().get()的值设置为false
            return;
        }
        context_ = make_nvshared(engine_->createExecutionContext());
        checkRuntime(cudaStreamCreate(&stream_));
        setInputOutputNames();
        if(context_ == nullptr){
            running_ = false;
            spdlog::error("Create context failed!");
            pro.set_value(false);                                               // 将start_up中pro.get_future().get()的值设置为false
            return;
        }
        string old_suffix = "engine"; string new_suffix = "txt";
        string model_classes_path = replaceSuffix(modelPath_, old_suffix, new_suffix);
        CURRENT_IDX2CLS = readFileToMap(model_classes_path);
        // load success
        pro.set_value(true);  
        if (modelLog_) spdlog::info("Model loaded successfully from {}", modelPath_);
        vector<Job> fetched_jobs;
        while(running_){
            {
                unique_lock<mutex> l(lock_);
                cv_.wait(l, [&](){return !running_ || !jobs_.empty();});        // 一直等着，cv_.wait(lock, predicate):如果 running不在运行状态 或者说 jobs_有东西 而且接收到了notify one的信号
                if(!running_) break;                                            // 如果实例被析构了，那么就结束该线程
                Job job_one = std::move(jobs_.front());
                jobs_.pop();                                                    // 从jobs_任务队列中将当前要推理的job给pop出来 
                l.unlock();                                                     // 注意这里要解锁, 否则调用inference等inference执行完再解锁又变同步了
                inference(job_one);                                             // 调用inference执行推理
            }
        }
    }

    // forward函数是生产者, 异步返回, 在main.cpp中获取结果
    virtual std::shared_future<std::vector<Result>> forward(Input* inputs, int& n_images, float conf_thre, float nms_thre, bool inferLog=false) override{  
        Job job;
        vector<string> unique_ids;                              
        for (int i = 0; i < n_images; ++i){
            job.heights.push_back(inputs[i].height);
            job.widths.push_back(inputs[i].width);
            int numel = inputs[i].height * inputs[i].width * 3;
            unique_ids.push_back(inputs[i].unique_id);
            cv::Mat image_one(inputs[i].height, inputs[i].width, CV_8UC3);
            memcpy(image_one.data, inputs[i].data, numel);
            // string save_path = "images-received/" + inputs[i].unique_id + ".jpg";
            // cv::cvtColor(image_one, image_one, cv::COLOR_RGB2BGR);
            // cv::imwrite(save_path, image_one);
            job.input_images.push_back(image_one);
            job.input_images_data.push_back(inputs[i].data);
            job.timestamps.push_back(inputs[i].timestamp);
        }            

        conf_thre_ = conf_thre;
        nms_thre_ = nms_thre;
        job.pro.reset(new promise<vector<Result>>());
        job.unique_ids = unique_ids;
        job.inferLog = inferLog;

        shared_future<vector<Result>> fut = job.pro->get_future();              // get_future()并不会等待数据返回，get_future().get()才会
        {
            unique_lock<mutex> l(lock_);
            jobs_.emplace(std::move(job));                                      // 向任务队列jobs_中添加任务job
        }
        cv_.notify_one();                                                       // 通知worker线程开始工作了
        return fut;                                                             // 等待模型将推理数据返回fut，然后fut再将数据return出去
    }

    // --------------------------------------------
    // forward函数是生产者, 异步返回, 在main.cpp中获取结果
    std::shared_future<std::vector<Result>> forward(Job& job){             
        shared_future<vector<Result>> fut = job.pro->get_future();              // get_future()并不会等待数据返回，get_future().get()才会
        {
            unique_lock<mutex> l(lock_);
            jobs_.emplace(std::move(job));                                      // 向任务队列jobs_中添加任务job
        }
        cv_.notify_one();                                                       // 通知worker线程开始工作了
        return fut;                                                             // 等待模型将推理数据返回fut，然后fut再将数据return出去
    }
    // --------------------------------------------

    void setInputOutputNames() {
        
    }
    void malloc_data(
        int curr_batch_size,
        float*& input_data_host,
        float*& input_data_device,
        float*& output_data_host,
        float*& output_data_device
    ){
        // Malloc input data
        // int nbindings = engine_->GetNbBindings();
        // for (int i = 0; i < nbindings; ++i){
        //     auto binding_name = engine_->GetBindingName(i);
        //     bool is_input = engine_->BindingIsInput(i);
        // }
        // todo: double check new API getTensorShape
        auto input_dims = engine_->getTensorShape(input_name_);     // 注意: 这里默认为1个输入, 可以使用上面注释的代码获得输入的名字和索引
        input_dims.d[0] = curr_batch_size;                     // attach current batch size to the input_dims
        int input_channels = input_dims.d[1];
        int input_height = input_dims.d[2];
        int input_width = input_dims.d[3];
        context_->setInputShape(input_name_, input_dims);
        int input_numel = curr_batch_size * input_channels * input_height * input_width;     
        checkRuntime(cudaMallocHost(&input_data_host, input_numel * sizeof(float)));
        checkRuntime(cudaMalloc(&input_data_device, input_numel * sizeof(float)));

        // Malloc output data 
        // todo: double check new API getTensorShape
        auto output_dims = engine_->getTensorShape(output_name_);    // 注意: 这里默认为1个输出, 可以使用上面注释的代码获得输出的名字和索引
        output_dims.d[0] = curr_batch_size;
        int output_batch = output_dims.d[0];
        int output_numbox = output_dims.d[1];
        int output_numprob = output_dims.d[2];
        // context_->setInputShape(output_name_, output_dims);
        int output_numel = curr_batch_size * output_numbox * output_numprob;
        checkRuntime(cudaMallocHost(&output_data_host, output_numel * sizeof(float)));
        checkRuntime(cudaMalloc(&output_data_device, output_numel * sizeof(float)));

        if (inferLog_) {
            spdlog::info("Model input shape: {} x {} x {} x {}", curr_batch_size, input_channels, input_height, input_width);
            spdlog::info("Model max output shape: {} x {} x {}", curr_batch_size, output_numbox, output_numprob);
        }
    }

    void free_data(
        float*& input_data_host,
        float*& input_data_device,
        float*& output_data_host,
        float*& output_data_device
    ){
        cudaFree(output_data_device);
        cudaFreeHost(output_data_host);
        cudaFree(input_data_device);
        cudaFreeHost(input_data_host);
    }
    
    void preprocess_cpu(
        float*& input_data_host,
        float*& input_data_device,
        vector<Mat>& batched_imgs, 
        vector<vector<int>>& batched_pad_w, 
        vector<vector<int>>& batched_pad_h, 
        vector<float>& batched_scale_factors, 
        int& curr_batch_size
    ){  // todo: double check new API getTensorShape
        auto input_dims = engine_->getTensorShape(input_name_);         // 注意: 这里默认为1个输入, 可以使用上面注释的代码获得输入的名字和索引
        input_dims.d[0] = curr_batch_size;                         // attach current batch size to the input_dims
        int input_channels = input_dims.d[1];
        int input_height = input_dims.d[2];
        int input_width = input_dims.d[3];
        int input_numel = curr_batch_size * input_channels * input_height * input_width;

        // Resize and pad
        for (int i = 0; i < batched_imgs.size(); ++i){
            Mat& img = batched_imgs[i];
            int img_height = img.rows;
            int img_width = img.cols;
            int img_channels = img.channels();

            float scale_factor = min(static_cast<float>(input_width) / static_cast<float>(img.cols),
                            static_cast<float>(input_height) / static_cast<float>(img.rows));
            int img_new_w_unpad = img.cols * scale_factor;
            int img_new_h_unpad = img.rows * scale_factor;
            int pad_wl = round((input_width - img_new_w_unpad - 0.01) / 2);		                   
            int pad_wr = round((input_width - img_new_w_unpad + 0.01) / 2);
            int pad_ht = round((input_height - img_new_h_unpad - 0.01) / 2);
            int pad_hb = round((input_height - img_new_h_unpad + 0.01) / 2);
            cv::resize(img, img, cv::Size(img_new_w_unpad, img_new_h_unpad));
            cv::copyMakeBorder(img, img, pad_ht, pad_hb, pad_wl, pad_wr, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
            batched_scale_factors.push_back(scale_factor);
            vector<int> pad_w = {pad_wl, pad_wr};
            vector<int> pad_h = {pad_ht, pad_hb};
            batched_pad_w.push_back(pad_w);
            batched_pad_h.push_back(pad_h);
        }

        // HWC-->CHW & /255. & transfer data to input_data_host
        // ! 注意, 由于马哥给我的图片是RGB的, 所以这里没有BGR转到RGB的过程
        float* i_input_data_host;
        size_t img_area = input_height * input_width;
        for (int i = 0; i < batched_imgs.size(); ++i){
            i_input_data_host = input_data_host + img_area * 3 * i;
            unsigned char* pimage = batched_imgs[i].data;
            float* phost_r = i_input_data_host + img_area * 0;
            float* phost_g = i_input_data_host + img_area * 1;
            float* phost_b = i_input_data_host + img_area * 2;
            for(int j = 0; j < img_area; ++j, pimage += 3){
                *phost_r++ = pimage[0] / 255.0f ;
                *phost_g++ = pimage[1] / 255.0f;
                *phost_b++ = pimage[2] / 255.0f;
            }
        }

        // Copy data from host to device
        checkRuntime(cudaMemcpyAsync(input_data_device, input_data_host, input_numel * sizeof(float), cudaMemcpyHostToDevice, stream_));
    }

    void preprocess_gpu_invoker(
        int src_width, int src_height,
        int dst_width, int dst_height,
        uint8_t* src_host, float* dst_device, size_t offset
    ){
        int src_line_size = src_width * 3;
        int dst_line_size = dst_width * 3;
        int dst_img_area = dst_width * dst_height;
        size_t src_size = src_width * src_height * 3 * sizeof(uint8_t);
        size_t intermediate_size = dst_width * dst_height * 3 * sizeof(uint8_t);
        size_t dst_size = dst_width * dst_height * 3 * sizeof(float);
        
        uint8_t* src_device;
        uint8_t* intermediate_device;
        uint8_t fill_value = 114;
        checkRuntime(cudaMalloc(&src_device, src_size));
        checkRuntime(cudaMalloc(&intermediate_device, intermediate_size));
        checkRuntime(cudaMemcpy(src_device, src_host, src_size, cudaMemcpyHostToDevice));

        preprocess_kernel_invoker(
            src_width, src_height, src_line_size,
            dst_width, dst_height, dst_line_size,
            src_device, intermediate_device, 
            dst_device, fill_value, dst_img_area, offset
        );
        this_thread::sleep_for(std::chrono::milliseconds(10));

        checkRuntime(cudaPeekAtLastError());
        checkRuntime(cudaFree(intermediate_device));
        checkRuntime(cudaFree(src_device));
    }

    void preprocess_gpu(
        float*& input_data_host,
        float*& input_data_device,
        vector<Mat>& batched_imgs, 
        vector<vector<int>>& batched_pad_w, 
        vector<vector<int>>& batched_pad_h, 
        vector<float>& batched_scale_factors, 
        int& curr_batch_size
    ){  // todo: double check new API getTensorShape
        auto input_dims = engine_->getTensorShape(input_name_);          // 注意: 这里默认为1个输入, 可以使用上面注释的代码获得输入的名字和索引
        input_dims.d[0] = curr_batch_size;                          // attach current batch size to the input_dims
        int input_channels = input_dims.d[1];
        int input_height = input_dims.d[2];
        int input_width = input_dims.d[3];
        int input_numel = curr_batch_size * input_channels * input_height * input_width;    

        // Resize and pad and transpose and normalize
        for (int i = 0; i < batched_imgs.size(); ++i){
            Mat& img = batched_imgs[i];
            int img_height = img.rows;
            int img_width = img.cols;
            int img_channels = img.channels();

            float scale_factor = min(static_cast<float>(input_width) / static_cast<float>(img.cols),
                            static_cast<float>(input_height) / static_cast<float>(img.rows));
            int img_new_w_unpad = img.cols * scale_factor;
            int img_new_h_unpad = img.rows * scale_factor;
            int pad_wl = round((input_width - img_new_w_unpad - 0.01) / 2);		                   
            int pad_wr = round((input_width - img_new_w_unpad + 0.01) / 2);
            int pad_ht = round((input_height - img_new_h_unpad - 0.01) / 2);
            int pad_hb = round((input_height - img_new_h_unpad + 0.01) / 2);
            batched_scale_factors.push_back(scale_factor);
            vector<int> pad_w = {pad_wl, pad_wr};
            vector<int> pad_h = {pad_ht, pad_hb};
            batched_pad_w.push_back(pad_w);
            batched_pad_h.push_back(pad_h);

            size_t offset = i * input_height * input_width * input_channels;

            preprocess_gpu_invoker(                  
                img_width, img_height,
                input_width, input_height,
                img.data, input_data_device, offset
            );    
        }
    }

    // bool do_infer(int curr_batch_size, void** buffers){
    //     bool success;
    //     cudaStreamSynchronize(stream_);
    //     success = context_->Execute(curr_batch_size, buffers);
    //     if (InferImpl::warmuped_)
    //         while (1){
    //             success = context_->Execute(curr_batch_size, buffers);
    //             spdlog::info("Execute successfully: {}", success);
    //         }
    //     return success;
    // }

    bool do_infer(int curr_batch_size, float** buffers){
        context_->setTensorAddress(input_name_, buffers[0]);
        context_->setTensorAddress(output_name_, buffers[1]);
        bool success = context_->enqueueV3(stream_);
        return success;
    }

    void clip_boxes(
        float& box_left, 
        float& box_right, 
        float& box_top, 
        float& box_bottom, 
        vector<int>& img_org_shape
    ){
        auto clip_value = [](float value, float min_value, float max_value) {
            return (value < min_value) ? min_value : (value > max_value) ? max_value : value;
        };
        int org_height = img_org_shape[0];
        int org_width = img_org_shape[1];
        box_left = clip_value(box_left, 0, org_width);
        box_right = clip_value(box_right, 0, org_width);
        box_top = clip_value(box_top, 0, org_height);
        box_bottom = clip_value(box_bottom, 0, org_height);
    }

    void postprocess_cpu(
        int curr_batch_size,
        float* output_data_host,
        float* output_data_device,
        int& output_shape_size,
        vector<Result>& results, 
        vector<vector<int>>& batched_pad_w, 
        vector<vector<int>>& batched_pad_h, 
        vector<float>& batched_scale_factors,
        vector<vector<int>>& batched_imgs_org_shape
    ){
        for (int i_img = 0; i_img < curr_batch_size; i_img++){
            vector<Box> bboxes;       // 初始化变量bboxes:[[x1, y1, x2, y2, conf, label], [x1, y1, x2, y2, conf, label]...]
            
            if (is_v5_ && (task_ == TASKTYPE::Detection))  // v5 supports detect only
                decode_boxes_1output_v5(output_data_host, i_img, bboxes, batched_pad_w, batched_pad_h, batched_scale_factors, batched_imgs_org_shape[i_img]);
            else if (task_ == TASKTYPE::Detection)         // v8 or v11 detect
                decode_boxes_1output_v8_v11_detect(output_data_host, i_img, bboxes, batched_pad_w, batched_pad_h, batched_scale_factors, batched_imgs_org_shape[i_img]);
            else if (task_ == TASKTYPE::Pose)           // v8 or v11 pose
                decode_boxes_1output_v8_v11_pose(output_data_host, i_img, bboxes, batched_pad_w, batched_pad_h, batched_scale_factors, batched_imgs_org_shape[i_img]);

            if (inferLog_) spdlog::info("Decoded bboxes.size = {}", bboxes.size());

            // nms非极大抑制
            // 通过比较索引为5(confidence)的值来将bboxes所有的框排序
            std::sort(bboxes.begin(), bboxes.end(), [](Box &a, Box &b)
                    { return a.score > b.score; });
            std::vector<bool> remove_flags(bboxes.size()); // 设置一个vector，存储是否保留bbox的flags
            // 定义一个lambda的iou函数
            auto iou = [](const Box &a, const Box &b)
            {
                float cross_left = std::max(a.left, b.left);
                float cross_top = std::max(a.top, b.top);
                float cross_right = std::min(a.right, b.right);
                float cross_bottom = std::min(a.bottom, b.bottom);

                float cross_area = std::max(0.0f, cross_right - cross_left) * std::max(0.0f, cross_bottom - cross_top);
                float union_area = std::max(0.0f, a.right - a.left) * std::max(0.0f, a.bottom - a.top) + std::max(0.0f, b.right - b.left) * std::max(0.0f, b.bottom - b.top) - cross_area;
                if (cross_area == 0 || union_area == 0)
                    return 0.0f;
                return cross_area / union_area;
            };

            for (int i = 0; i < bboxes.size(); ++i){
                if (remove_flags[i])
                    continue;                                       // 如果已经被标记为需要移除，则continue

                auto &ibox = bboxes[i];                             // 获得第i个box
                auto float2int = [] (float x) {return static_cast<int>(round(x));};
                // vector<int> _ibox = {float2int(ibox[0]), float2int(ibox[1]), float2int(ibox[2]), float2int(ibox[3])};
                // results[i_img].boxes.emplace_back(_ibox);           // 将该box放入box_result中，emplace_back和push_back基本一样，区别在于emplace_back是inplace操作
                // results[i_img].labels.emplace_back(int(ibox[4]));
                // results[i_img].scores.emplace_back(ibox[5]);
                Box bbox;
                bbox.left = float2int(ibox.left); 
                bbox.top = float2int(ibox.top);
                bbox.right = float2int(ibox.right);
                bbox.bottom = float2int(ibox.bottom);
                bbox.label = int(ibox.label);
                bbox.score = ibox.score;
                bbox.keypoints = ibox.keypoints;
                results[i_img].bboxes.emplace_back(bbox);
                for (int j = i + 1; j < bboxes.size(); ++j){        // 遍历剩余框，与box_result中的框做iou
                    if (remove_flags[j])
                        continue;                                   // 如果已经被标记为需要移除，则continue

                    auto &jbox = bboxes[j];                         // 获得第j个box
                    if (ibox.label == jbox.label){ 
                        // class matched
                        if (iou(ibox, jbox) >= nms_thre_)       // iou值大于阈值，将该框标记为需要remove
                            remove_flags[j] = true;
                    }
                }
            }
            if (inferLog_) spdlog::info("box_result.size = {}", results[i_img].bboxes.size());
        }
    }

    void synchronize(int curr_batch_size, float*& output_data_host, float*& output_data_device){
        // todo: double check new API getTensorShape
        auto output_dims = engine_->getTensorShape(output_name_);       // 注意: 这里默认为1个输出, 可以使用上面注释的代码获得输出的名字和索引
        int output_numel = curr_batch_size * output_dims.d[1] * output_dims.d[2];
        checkRuntime(cudaMemcpyAsync(output_data_host, output_data_device, sizeof(float) * output_numel, cudaMemcpyDeviceToHost, stream_));
        checkRuntime(cudaStreamSynchronize(stream_));
    }

    void decode_boxes_1output_v5(
        float* output_data_host,
        int& i_img, 
        vector<Box>& bboxes, 
        vector<vector<int>>& batched_pad_w, 
        vector<vector<int>>& batched_pad_h,
        vector<float>& batched_scale_factors,
        vector<int>& img_org_shape
    ){
        // todo: double check new API getTensorShape
        auto output_dims = engine_->getTensorShape(output_name_);        // 注意: 这里默认为1个输出, 可以使用上面注释的代码获得输出的名字和索引
        int output_numbox = output_dims.d[1];
        int output_numprob = output_dims.d[2];
        int num_classes = output_numprob - 5;
        size_t offset_per_image = output_numbox * output_numprob;
        // decode and filter boxes by conf_thre
        multi_label_ = multi_label_ && (num_classes > 1);
        int label; float prob; float confidence;
        vector<int> labels; vector<float> confidences;
        for (int i = 0; i < output_numbox; ++i) {
            Box box_one;
            float *ptr = output_data_host + offset_per_image * i_img + i * output_numprob;  // 每次偏移output_numprob
            float objness = ptr[4];                                                         // 获得置信度
            if (objness < conf_thre_)
                continue;

            float *pclass = ptr + 5;                                                        // 获得类别开始的地址
            label = max_element(pclass, pclass + num_classes) - pclass;                     // 获得概率最大的类别
            prob = pclass[label];                                                           // 获得类别概率最大的概率值
            confidence = prob * objness;                                                    // 计算后验概率
            if (confidence < conf_thre_)
                continue;
            if (multi_label_)
                while (confidence >= conf_thre_){
                    labels.push_back(label);
                    confidences.push_back(confidence);
                    *(pclass + label) = 0.;
                    label = max_element(pclass, pclass + num_classes) - pclass;
                    prob = pclass[label];
                    confidence = prob * objness;
                }                   

            // xywh
            float cx = ptr[0];
            float cy = ptr[1];
            float width = ptr[2];
            float height = ptr[3];

            // xyxy
            float left = cx - width * 0.5;
            float top = cy - height * 0.5;
            float right = cx + width * 0.5;
            float bottom = cy + height * 0.5;

            // the box cords on the origional image
            float image_base_left = (left - batched_pad_w[i_img][0]) / batched_scale_factors[i_img];                             // x1
            float image_base_right = (right - batched_pad_w[i_img][1]) / batched_scale_factors[i_img];                           // x2
            float image_base_top = (top - batched_pad_h[i_img][0]) / batched_scale_factors[i_img];                               // y1
            float image_base_bottom = (bottom - batched_pad_h[i_img][1]) / batched_scale_factors[i_img];                         // y2
            clip_boxes(image_base_left, image_base_right, image_base_top, image_base_bottom, img_org_shape);
            box_one.left = image_base_left;
            box_one.top = image_base_top;
            box_one.right = image_base_right;
            box_one.bottom = image_base_bottom;
            if (multi_label_){
                for (int j = 0; j < labels.size(); ++j){
                    box_one.label = (float)labels[j];
                    box_one.score = confidences[j];
                    bboxes.push_back(box_one);
                }
                labels.clear();
                confidences.clear();
            }else{
                box_one.label = (float)label;
                box_one.score = confidence;
                bboxes.push_back(box_one);  // 放进bboxes中
            }
        }
    }

    void decode_boxes_1output_v8_v11_detect(
        float* output_data_host,
        int& i_img, 
        vector<Box>& bboxes, 
        vector<vector<int>>& batched_pad_w, 
        vector<vector<int>>& batched_pad_h, 
        vector<float>& batched_scale_factors,
        vector<int>& img_org_shape
    ){
        // todo: double check new API getTensorShape
        auto output_dims = engine_->getTensorShape(output_name_);        // 注意: 这里默认为1个输出, 可以使用上面注释的代码获得输出的名字和索引
        int output_numbox = output_dims.d[1];
        int output_numprob = output_dims.d[2];
        int num_classes  = output_numprob - 4;
        size_t offset_per_image = output_numbox * output_numprob;
        // decode and filter boxes by conf_thre
        multi_label_ = multi_label_ && (num_classes > 1);
        int label; float prob; float confidence;
        vector<int> labels; vector<float> confidences;
        for (int i = 0; i < output_numbox; ++i) {
            Box box_one;
            float *ptr = output_data_host + offset_per_image * i_img + i * output_numprob;  // 每次偏移output_numprob
            // float objness = ptr[4];                                                         // 获得置信度
            // if (objness < conf_thre_)
            //     continue;

            float *pclass = ptr + 4;                                                        // 获得类别开始的地址
            label = max_element(pclass, pclass + num_classes) - pclass;                     // 获得概率最大的类别
            prob = pclass[label];                                                           // 获得类别概率最大的概率值
            confidence = prob;                                                    // 计算后验概率
            if (confidence < conf_thre_)
                continue;
            if (multi_label_)
                while (confidence >= conf_thre_){
                    labels.push_back(label);
                    confidences.push_back(confidence);
                    *(pclass + label) = 0.;
                    label = max_element(pclass, pclass + num_classes) - pclass;
                    prob = pclass[label];
                    confidence = prob;
                }                   

            // xywh
            float cx = ptr[0];
            float cy = ptr[1];
            float width = ptr[2];
            float height = ptr[3];

            // xyxy
            float left = cx - width * 0.5;
            float top = cy - height * 0.5;
            float right = cx + width * 0.5;
            float bottom = cy + height * 0.5;

            // the box cords on the origional image
            float image_base_left = (left - batched_pad_w[i_img][0]) / batched_scale_factors[i_img];                             // x1
            float image_base_right = (right - batched_pad_w[i_img][1]) / batched_scale_factors[i_img];                           // x2
            float image_base_top = (top - batched_pad_h[i_img][0]) / batched_scale_factors[i_img];                               // y1
            float image_base_bottom = (bottom - batched_pad_h[i_img][1]) / batched_scale_factors[i_img];                         // y2
            clip_boxes(image_base_left, image_base_right, image_base_top, image_base_bottom, img_org_shape);
            box_one.left = image_base_left;
            box_one.top = image_base_top;
            box_one.right = image_base_right;
            box_one.bottom = image_base_bottom;
            if (multi_label_){
                for (int j = 0; j < labels.size(); ++j){
                    box_one.label = (float)labels[j];
                    box_one.score = confidences[j];
                    bboxes.push_back(box_one);
                }
                labels.clear();
                confidences.clear();
            }else{
                box_one.label = (float)label;
                box_one.score = confidence;
                bboxes.push_back(box_one);  // 放进bboxes中
            }
        }
    }

    void decode_boxes_1output_v8_v11_pose(
        float* output_data_host,
        int& i_img, 
        vector<Box>& bboxes, 
        vector<vector<int>>& batched_pad_w, 
        vector<vector<int>>& batched_pad_h, 
        vector<float>& batched_scale_factors,
        vector<int>& img_org_shape
    ){
        auto output_dims = engine_->getTensorShape(output_name_);  // 注意: 这里默认为1个输出, 可以使用上面注释的代码获得输出的名字和索引
        int output_numbox = output_dims.d[1];                      // 5040
        int output_numprob = output_dims.d[2];                     // 56 = 5(xywh+conf) + 17 * 3 
        int num_keypoints = (output_numprob - 5) / 3;
        int num_classes = 1;
        size_t offset_per_image = output_numbox * output_numprob;
        // decode and filter boxes by conf_thre
        multi_label_ = multi_label_ && (num_classes > 1);
        int label; float prob; float confidence;
        vector<int> labels; vector<float> confidences;
        for (int i = 0; i < output_numbox; ++i) {
            Box box_one;
            float *ptr = output_data_host + offset_per_image * i_img + i * output_numprob;  // 每次偏移output_numprob
            confidence = ptr[4];
            if (confidence < conf_thre_)
                continue;        

            // First, deal with bboxes
            label = 0; multi_label_ = false;
            // xywh
            float b_cx = ptr[0];
            float b_cy = ptr[1];
            float b_width = ptr[2];
            float b_height = ptr[3];

            // xyxy
            float b_left = b_cx - b_width * 0.5;
            float b_top = b_cy - b_height * 0.5;
            float b_right = b_cx + b_width * 0.5;
            float b_bottom = b_cy + b_height * 0.5;

            // the box cords on the origional image
            float b_image_base_left = (b_left - batched_pad_w[i_img][0]) / batched_scale_factors[i_img];                             // x1
            float b_image_base_right = (b_right - batched_pad_w[i_img][1]) / batched_scale_factors[i_img];                           // x2
            float b_image_base_top = (b_top - batched_pad_h[i_img][0]) / batched_scale_factors[i_img];                               // y1
            float b_image_base_bottom = (b_bottom - batched_pad_h[i_img][1]) / batched_scale_factors[i_img];                         // y2
            clip_boxes(b_image_base_left, b_image_base_right, b_image_base_top, b_image_base_bottom, img_org_shape);
            box_one.left = b_image_base_left;
            box_one.top = b_image_base_top;
            box_one.right = b_image_base_right;
            box_one.bottom = b_image_base_bottom;

            // Second, deal with keypoints
            // todo: 需要看看训练代码中对这里的处理方式, 因为置信度大的bbox对应的keypoint的置信度不一定最大
            for (int j = 0; j < num_keypoints; ++j){
                // xy
                KeyPoint_t keypoint_one;
                float* current_ptr = ptr + 5 + j * 3;   // offset xywhconf + 3 * j-keypoints
                float k_cx = current_ptr[0];
                float k_cy = current_ptr[1];
                float k_conf = current_ptr[2];
                float k_image_base_cx = (k_cx - batched_pad_w[i_img][0]) / batched_scale_factors[i_img];
                float k_image_base_cy = (k_cy - batched_pad_h[i_img][0]) / batched_scale_factors[i_img];
                keypoint_one.x = k_image_base_cx;   
                keypoint_one.y = k_image_base_cy;   
                keypoint_one.score = k_conf;        // todo: clip keypoints
                box_one.keypoints.push_back(keypoint_one);
            }

            if (multi_label_){
                for (int j = 0; j < labels.size(); ++j){
                    box_one.label = (float)labels[j];
                    box_one.score = confidences[j];
                    bboxes.push_back(box_one);
                }
                labels.clear();
                confidences.clear();
            }else{
                box_one.label = (float)label;
                box_one.score = confidence;
                bboxes.push_back(box_one); 
            }
        }
    }

    void inference(Job& job){
        // todo: 如果硬解码的话, 可以尝试使用cudagraph对硬解码+前处理+推理建图
        // if (InferImpl::warmuped_)
        //     this_thread::sleep_for(chrono::seconds(100));

        Result dummy;
        int curr_batch_size = job.input_images.size();
        std::vector<Result> results(curr_batch_size, dummy);

        vector<Mat> batched_imgs;
        for (int i = 0; i < job.input_images.size(); ++i){
            batched_imgs.push_back(job.input_images[i]);
            results[i].data = job.input_images_data[i];
        }

        vector<string> unique_ids = job.unique_ids;
        inferLog_ = job.inferLog;
        vector<vector<int>> batched_imgs_org_shape;
        for (int i = 0; i < curr_batch_size; ++i){
            int height = batched_imgs[i].rows;
            int width = batched_imgs[i].cols;
            vector<int> i_shape = {height, width};
            batched_imgs_org_shape.push_back(i_shape);
            results[i].unique_id = unique_ids[i];           // attach each id to results
            results[i].height = height;
            results[i].width =  width;
            results[i].channel_id = job.channel_id;
            results[i].timestamp = job.timestamps[i];
        }

        // prepare data                                     
        float* input_data_host = nullptr;
        float* input_data_device = nullptr;
        float* output_data_host = nullptr;
        float* output_data_device = nullptr;
        malloc_data(curr_batch_size, input_data_host, input_data_device, output_data_host, output_data_device);
        float* buffers[] = {input_data_device, output_data_device};  

    #ifdef WITH_CLOCKING
        auto tiktok = time_point::now();
    #endif     

        // preprocess 
        vector<float> batched_scale_factors;
        vector<vector<int>> batched_pad_w, batched_pad_h;
        // preprocess_cpu(input_data_host, input_data_device, batched_imgs, batched_pad_w, batched_pad_h, batched_scale_factors, curr_batch_size);
        preprocess_gpu(input_data_host, input_data_device, batched_imgs, batched_pad_w, batched_pad_h, batched_scale_factors, curr_batch_size);
        // cudaDeviceSynchronize();         // !注意: 如果要对使用gpu做前处理计时, 需要加这一句
    #ifdef WITH_CLOCKING
        InferImpl::records[0].push_back(micros_cast(time_point::now() - tiktok));
        tiktok = time_point::now();
    #endif

        // infer
        bool success;
        // todo: double check new API setTensorAddress images output 
        success = do_infer(curr_batch_size, buffers);
        if (!success){
            for (int i = 0; i < curr_batch_size; ++i){
                spdlog::error("Model infer failed! The image id is {}", unique_ids[i]);
            }
            job.pro->set_value(results);        // dummy results
            return;
        }
        synchronize(curr_batch_size, output_data_host, output_data_device);
    #ifdef WITH_CLOCKING
        InferImpl::records[1].push_back(micros_cast(time_point::now() - tiktok));
        tiktok = time_point::now();
    #endif

        // postprocess
        // todo: double check new API getTensorShape
        auto output_dims = engine_->getTensorShape(output_name_);     // 注意: 这里默认为1个输出, 可以使用上面注释的代码获得输出的名字和索引
        int output_shape_size = output_dims.nbDims;       
        postprocess_cpu(curr_batch_size, output_data_host, output_data_device, output_shape_size, results, batched_pad_w, batched_pad_h, batched_scale_factors, batched_imgs_org_shape);
    #ifdef WITH_CLOCKING
        InferImpl::records[2].push_back(micros_cast(time_point::now() - tiktok));
        tiktok = time_point::now();
    #endif

        // set results to future
        job.pro->set_value(results);

        // judge warmuped or not 
        if (!InferImpl::warmuped_){
            InferImpl::warmuped_times_++;
            if (InferImpl::warmuped_times_ <= InferImpl::total_warmup_times_){
                free_data(input_data_host, input_data_device, output_data_host, output_data_device);
                return;
            }
        }
        InferImpl::warmuped_ = true;
        
        // callback
        if (!(callback_ == nullptr)){
            reIndexResults(results, CURRENT_IDX2CLS, UNIFIED_CLS2IDX, nick_name_);
            callback_(results, userP_);
        }

        // release data
        free_data(input_data_host, input_data_device, output_data_host, output_data_device);
        // for (int i = 0; i < batched_imgs.size(); ++i){
        //     if (!batched_imgs[i].empty()){
        //         batched_imgs[i].release();
        //         delete [] batched_imgs[i].data;
        //     }
        // }
    }
    
    //------------------------------------------------------------------------------------------------------------------
    virtual void warmup(){
        for (int i = 0; i < InferImpl::total_warmup_times_; ++i){
            int height = 1080;
            int width = 1920;
            cv::Mat dummyImage = cv::Mat::zeros(height, width, CV_8UC3);
            int n_images = 1;
            float dummy_conf_thre = 0.1;
            string dummy_channel_id = "ch01";
            int numel = height * width * 3;
            Input dummy_input;
            dummy_input.height = height;
            dummy_input.width = width;
            dummy_input.data = new unsigned char[numel];
            memcpy(dummy_input.data, dummyImage.data, numel);
            Input dummy_inputs[n_images];
            dummy_inputs[0] = dummy_input;
            auto r = this->forward(dummy_inputs, n_images, 0.4, false).get();
            delete [] dummy_input.data;          
        }
    }
    
    virtual bool add_images(Input* inputs, int& n_images, float conf_thre, float nms_thre, string channel_id) override{
        bool overflow = false;
        // if exceed max queue length, then pop the front job of the queue and delete the image data 
        // finnally push the new one into the queue
        if (InferImpl::warmuped_){              // 只有已经热身过了才会去判断是否超过队列长度
            received_++;
            {
                unique_lock<mutex> l(lock_);
                if (jobs_.size() >= max_qsize_){
                    overflow = true;
                    int n_images_of_qfront = jobs_.front().input_images_data.size();
                    for (int i = 0; i < n_images_of_qfront; ++i){
                        unsigned char* image_data_of_qfront = jobs_.front().input_images_data[i];
                        delete [] image_data_of_qfront;
                        throwed_++;
                    }
                    jobs_.pop();
                    double throwed_ratio = (double)throwed_ / (double(received_) + 1e-5);
                    spdlog::debug("Images receieved: {}, throwed: {}, throwed ratio: {}", received_, throwed_, throwed_ratio);
                }
            }
        }

        Job job;
        job.channel_id = channel_id;                             
        for (int i = 0; i < n_images; ++i){
            job.timestamps.push_back(inputs[i].timestamp);
            job.heights.push_back(inputs[i].height);
            job.widths.push_back(inputs[i].width);
            int numel = inputs[i].height * inputs[i].width * 3;
            job.unique_ids.push_back(inputs[i].unique_id);
            cv::Mat image_one(inputs[i].height, inputs[i].width, CV_8UC3);
            memcpy(image_one.data, inputs[i].data, numel);
            job.input_images.push_back(image_one);
            job.input_images_data.push_back(inputs[i].data);
        }  

        conf_thre_ = conf_thre; 
        nms_thre_ = nms_thre;    
        job.pro.reset(new promise<vector<Result>>());

        forward(job);
        return overflow;
    }

    virtual int get_qsize() override{
        int size; 
        {
            unique_lock<mutex> l(lock_);
            size = jobs_.size();
        }
        return size;
    }
    //------------------------------------------------------------------------------------------------------------------
    virtual vector<vector<float>> get_records() override{       // 计时相关, 可删
        return InferImpl::records;
    }

private:
    // 可调数据
    string nick_name_;
    string modelPath_;                                           // 模型路径
    int max_batch_size_;
    bool multi_label_;
    bool is_v5_{true};                                                // 是使用yolov5模型还是yolov8模型
    int task_;
    float conf_thre_{0.5};
    float nms_thre_{0.6};
    // 多线程有关
    atomic<bool> running_{false};                               // 如果InferImpl类析构，那么开启的子线程也要break
    thread worker_thread_;
    queue<Job> jobs_;                                           // 任务队列
    mutex lock_;                                                // 负责任务队列线程安全的锁
    condition_variable cv_;                                     // 线程通信函数
    // 模型初始化有关   
    const char* input_name_{"images"};
    const char* output_name_{"output"};        
    cudaStream_t stream_;
    shared_ptr<ILogger> trtlogger_;
    shared_ptr<IRuntime> runtime_;
    shared_ptr<ICudaEngine> engine_;
    shared_ptr<IExecutionContext> context_;
    //日志相关
    bool modelLog_;                                             // 模型加载时是否打印日志在控制台
    bool inferLog_;                                             // 模型推理时是否打印日志在控制台
    string logs_dir{"infer-logs"};                              // 日志文件存放的文件夹   
    // 计时相关
    static vector<vector<float>> records;                       // 计时相关: 静态成员变量声明
    //---------------------------------
    // 事件相关
    PushResult callback_;
    void* userP_;
    static int max_qsize_;                                     
    static bool is_qsize_set_;                                  // 队列长度是否已被设置
    static bool warmuped_;                                      // 是否warmup了
    static int total_warmup_times_;                             // 总共需要warmup几次
    static int warmuped_times_;                                 // 当前warmup了几次
    thread boss_thread_;
    std::map<int, string> CURRENT_IDX2CLS;                      // 当前模型class index到class name的映射: 为了保证同一个类别对应多个模型推理出的结果的index保持一致, 设置该变量, 存储index到class name的映射
    long received_{0};                                          // 统计当前接收了多少
    long throwed_{0};                                           // 统计当前扔掉了多少
    //---------------------------------
};

// 在类体外初始化这几个静态变量
bool InferImpl::is_qsize_set_ = false;                          
int InferImpl::max_qsize_ = 100;
bool InferImpl::warmuped_ = false;
int InferImpl::total_warmup_times_ = 5;
int InferImpl::warmuped_times_ = 0;
vector<vector<float>> InferImpl::records(3);                    // 计时相关: 静态成员变量定义, 长度为3

shared_ptr<InferInterface> create_infer(std::string &file, int max_det, std::string& device, bool modelLog, bool multi_label){
    shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->startup(file, max_det, device, modelLog, multi_label)){
        instance.reset();                                                     
    }
    return instance;
};

// --------------------------------------------
shared_ptr<InferInterface> create_infer(std::string nickName, PushResult callback, void *userP, std::string &file, int max_det, int max_qsize, std::string& device, bool modelLog, bool multi_label){
    shared_ptr<InferImpl> instance(new InferImpl());
    if (!instance->startup(nickName, callback, userP, file, max_det, max_qsize, device, modelLog, multi_label)){
        instance.reset();   
        return nullptr;                                                     
    }
    return instance;
};
// --------------------------------------------
