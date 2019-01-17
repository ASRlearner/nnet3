//
// Created by smz on 19-1-2.
//

#ifndef NNET_TEST_ONLINEVECTORINPUT_H
#define NNET_TEST_ONLINEVECTORINPUT_H

#endif //NNET_TEST_ONLINEVECTORINPUT_H


#include "online/online-feat-input.h"
#include "feat/feature-functions.h"
#include <queue>
//#include <vector>

using namespace kaldi;


//通过portaudio采集音频到缓冲区
// 然后通过缓冲区读取定长数据到矩阵中 矩阵为1行n列的矩阵 方便后续处理
class OnlineAudioMatrix{
public:
    OnlineAudioMatrix(OnlineAudioSourceItf *input,int32 nsamples);
    //将定长音频数据点存储到矩阵中输出
    bool compute(Matrix<BaseFloat > &output);
private:
    //portaudio对象
    OnlineAudioSourceItf *input;
    //需要采样的音频数据点数
    int32  nsamples;
};

OnlineAudioMatrix::OnlineAudioMatrix(OnlineAudioSourceItf *input,
                                      int32 nsamples):input(input),
                                      nsamples(nsamples){}

bool OnlineAudioMatrix::compute(Matrix<BaseFloat> &output) {
    Vector<BaseFloat > buf(nsamples);
    //从缓冲区中读取定长音频数据到vector
    bool ans=input->Read(&buf);
    std::cout<< buf.Dim()<<std::endl;
    //定义输出矩阵的规格为行向量 长度为得到的音频数据点数
    output.Resize(1,buf.Dim());
    //把vector中数据全部拷贝到矩阵中
    for(int i=0;i<buf.Dim();i++)
        (output)(0,i)=buf(i);

    return ans;
}