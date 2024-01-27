#ifndef Singal_hpp
#define Singal_hpp
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <MNN/ErrorCode.hpp>
#include <MNN/MNNForwardType.h>
#include <MNN/Tensor.hpp>
#include <utility>
namespace MNN
{
    struct syn_data
    {
        //<input output>;
        std::pair<std::vector<Tensor *>, std::vector<Tensor *>> data;
        int index;
    };
}


#endif /* Session_hpp */