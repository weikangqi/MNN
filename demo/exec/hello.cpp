#include "iostream"

#include <stdio.h>
#include <MNN/ImageProcess.hpp>
#define MNN_OPEN_TIME_TRACE
#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <sstream>
#include <vector>
#include <MNN/expr/Expr.hpp>
#include <MNN/expr/ExprCreator.hpp>
#include <MNN/AutoTime.hpp>
#include <MNN/Interpreter.hpp>
#include <core/TensorUtils.hpp>
#include <MNN/AutoTime.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
using namespace MNN;
using namespace MNN::CV;
using namespace MNN::Express;
int main(int argc, const char* argv[]){

    int originalWidth;
    int originalHeight;
    int originChannel;
    const auto inputImageFileName = argv[2];
    auto inputImage = stbi_load(inputImageFileName, &originalWidth,
                              &originalHeight, &originChannel, 3);
    const auto rgbaPtr = reinterpret_cast<uint8_t *>(inputImage);
    CV::ImageProcess::Config preProcessConfig;
    preProcessConfig.destFormat = CV::RGB;

    preProcessConfig.sourceFormat = CV::RGB;
    // preProcessConfig.destFormat = CV::RGB;
    MNN_PRINT("%d, %d, %d\n",originalWidth,originalHeight,originChannel);
    auto pretreat = std::shared_ptr<CV::ImageProcess>(CV::ImageProcess::create(preProcessConfig));
 
    
    /**
    Interpreter  = cpu_net
    只使用CPU运行，或者异构前的 result，作为真值 。
    **/
    
    std::shared_ptr<Interpreter> cpu_net;
    cpu_net.reset(Interpreter::createFromFile(argv[1]));
    ScheduleConfig cpu_config;
    BackendConfig backendconfig;
    backendconfig.precision = MNN::BackendConfig::Precision_High;

    cpu_config.type = MNN_FORWARD_CPU;
    cpu_config.numThread = 4;
    cpu_config.backendConfig = &backendconfig;
    auto session_cpu = cpu_net->createSession(cpu_config);
    auto input_cpu = cpu_net->getSessionInput(session_cpu, "input.1");
    pretreat->convert(rgbaPtr, originalWidth, originalHeight, 0, input_cpu);

    cpu_net->resizeTensor(input_cpu,input_cpu->shape());
    cpu_net->resizeSession(session_cpu);
    for(int i = 0; i < 30; i++)
    {
        {
            AUTOTIME;
            cpu_net->runSession(session_cpu);
        }

    }
    
    auto output_cpu = cpu_net->getSessionOutput(session_cpu,"66");
    Tensor out_cpu (output_cpu,Tensor::CAFFE);
    output_cpu->copyToHostTensor(&out_cpu);
    /**
    Interpreter  = cpu_net_codl  && gpu_net_codl
    运行codl，创建cpu interpret 和 gpu interpret 。
    **/
    std::shared_ptr<Interpreter> cpu_net_codl;
    cpu_net_codl.reset(Interpreter::createFromFile(argv[1]));
    ScheduleConfig cpu_config_codl;
    BackendConfig backendconfig_codl;
    backendconfig_codl.precision = MNN::BackendConfig::Precision_High;

    cpu_config_codl.type = MNN_FORWARD_CPU;
    cpu_config_codl.numThread = 4;
    cpu_config_codl.backendConfig = &backendconfig;
    auto session_cpu_codl = cpu_net_codl->createSession(cpu_config_codl);
    auto input_cpu_codl = cpu_net_codl->getSessionInput(session_cpu_codl, "input.1");
    pretreat->convert(rgbaPtr, originalWidth, originalHeight, 0, input_cpu_codl);

    cpu_net_codl->resizeTensor(input_cpu_codl,input_cpu_codl->shape());
    cpu_net_codl->resizeSession(session_cpu_codl);


    std::shared_ptr<Interpreter> gpu_net_codl;
    gpu_net_codl.reset(Interpreter::createFromFile(argv[1]));
    ScheduleConfig gpu_config_codl;
    // BackendConfig backendconfig;
    // backendconfig.precision = MNN::BackendConfig::Precision_High;

    gpu_config_codl.type = MNN_FORWARD_OPENCL;
    gpu_config_codl.numThread = 1;
    gpu_config_codl.backendConfig = &backendconfig;
    auto session_gpu_codl = gpu_net_codl->createSession(gpu_config_codl);
    auto input_gpu_codl = gpu_net_codl->getSessionInput(session_gpu_codl, "input.1");
    pretreat->convert(rgbaPtr, originalWidth, originalHeight, 0, input_gpu_codl);

    gpu_net_codl->resizeTensor(input_gpu_codl,input_gpu_codl->shape());
    gpu_net_codl->resizeSession(session_gpu_codl);
    MNN_PRINT("codl\n");
    for(int i = 0; i < 30; i++)
    {
        {
            AUTOTIME;
            gpu_net_codl->runSessionCpuGpu(session_gpu_codl,session_cpu_codl);
        }

    }
    // gpu_net_codl->runSessionCpuGpu(session_gpu_codl,session_cpu_codl);

    // gpu_net->runSession(session_gpu);
    auto output_codl = cpu_net_codl->getSessionOutput(session_cpu_codl,"66");
    // auto output = gpu_net->getSessionOutput(session_gpu,"66");
    Tensor out_codl (output_codl,Tensor::CAFFE);
    output_codl->copyToHostTensor(&out_codl);
    // MNN_PRINT("\n out_cpu \n");
    // out_cpu.print();
    // MNN_PRINT("\n out_gpu \n");
    // output_codl->print();
    // auto success = TensorUtils::compareTensors(&out_codl,&out_cpu,0.01);
    // if(success)
    // {
    //     MNN_PRINT("codl success");
    // }
    // else {
    //     MNN_PRINT("codl faild");
    // }





    return 0;
}