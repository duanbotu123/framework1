#ifndef REGISTRATION_H
#define REGISTRATION_H
#include <string>
//包括哪些头文件？
#include <nanoflann.hpp>
#include <Eigen/Dense>
#include <vector>
#include <utility>
#include <iostream>   // 如果需要调试输出


class REG
{
public:
    // 构造和析构
    REG() = default;
    virtual ~REG() = default;
    
    virtual void Reg(const std::string& file_target,
                       const std::string& file_source,
                       const std::string& out_path)=0; 
    

};




#endif // REGISTRATION_H


