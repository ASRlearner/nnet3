//
// Created by smz on 19-1-11.
//

#ifndef NNET_TEST_VAD_DETECT_H
#define NNET_TEST_VAD_DETECT_H

#endif //NNET_TEST_VAD_DETECT_H


#include <queue>
#include "online/online-feat-input.h"
#include "feat/feature-functions.h"
#include "math.h"

using  namespace kaldi;

//清空队列
void clear(std::queue<Vector<BaseFloat >> &q){
    std::queue<Vector<BaseFloat >> empty;
    swap(empty,q);
}

class OnlineMatrixOutput{
public:
    OnlineMatrixOutput(OnlineAudioSourceItf *input,int32 sample_freq,bool state);
    void Compute(Matrix<BaseFloat > &output);
    //返回一句话是否说完了的判断结果
    bool is_finished() { return finished_;}
    //通过状态返回截断临界值的大小
    BaseFloat cutoff(bool state);
    //这是基于当前语音发生截断以后的情况,也就是说由于出现了静音帧
    //所以需要判断接下来0.5秒的静音情况
    //用于解码完后取0.5秒的音频数据判断静音情况
    void read();
private:
    OnlineAudioSourceItf *input_;
    //每次测试的采样点数
    int32 nsample;
    //判断是否第一次采集
    bool start;
    //判断一句话是否说完了 即是否有足够长的静音
    bool finished_;
    //当前环境的状态
    bool state_;
    //read函数判断中末尾静音帧数
    int32 end_frames;
    //截断的值
    BaseFloat jieduan;
    //存放有效的音频数据块
    std::queue<Vector<BaseFloat >> frame_;
    //用于判断断句时的有效音频数据
    std::queue<Vector<BaseFloat >> next_;
};

OnlineMatrixOutput::OnlineMatrixOutput(OnlineAudioSourceItf *input, int32 sample_freq,bool state):
                                       input_(input),state_(state),finished_(false),end_frames(0),
                                       start(false){
    //得到每次判断的采样点数
    nsample = static_cast<int32>(sample_freq);
    //根据状态得到截断阈值
    jieduan = cutoff(state_);
}

//判断0.5秒内是否有新的语音
//对于出现静音后的这0.5秒内的有效帧 保存在next_中
//与下一波解码中的音频数据拼接到一起
void OnlineMatrixOutput::read() {
    //std::cerr<<"-";
    //如果已经换行 则重置finished_的值
    if(finished_)
        finished_=false;
    //末尾静音帧数清零
    end_frames=0;
    //开始下一波0.5秒的判断
    //这里8千次循环
    for(int i=0;i<20;i++)
    {
        Vector<BaseFloat >buf(nsample);
        bool ans=input_->Read(&buf);
        BaseFloat sum=0;
        for(int j=0;j<buf.Dim();j++)
            sum += abs(buf(j));
        //把该帧所有数据点求和
        //如果有一帧的能量超过了阈值
        if (sum / 400 >= jieduan) {
            next_.push(buf);
            //末尾静音帧数清零
            end_frames=0;
            //std::cerr<<"1";
        }else{
            //静音帧数加1
            //std::cerr<<"0";
            end_frames++;
        }
    }
}

//阈值(需要调整)
// 由于这种写法对阈值极其敏感所以阈值可能要设置的更为保守一些
BaseFloat OnlineMatrixOutput::cutoff(bool state) {
    if(state)
        return 1500;
    else
        return 2000;
}

//获取超过阈值的有效帧至解码块中
void OnlineMatrixOutput::Compute(Matrix<BaseFloat > &output) {
    //初始化静音的帧数为0
    int32 silence_frame=0;
    //在进行新的解码之前清空解码块
    //clear(frame_);
    //如果之前以静音帧结束或者是第一次开始取音频数据
    if(end_frames>0||((end_frames==0)&&next_.empty())){
    //在出现有效帧之前一直循环
      while(!finished_) {
        //一帧的采样点个数
        Vector<BaseFloat > buf(nsample);
        bool ans = input_->Read(&buf);
        //把该帧所有数据点求和
        BaseFloat sum=0;
        for(int i=0;i<buf.Dim();i++)
            sum += abs(buf(i));
        //如果有一帧的能量超过了阈值
        if (sum / 400 >= jieduan) {
            if(!start)
                //认为已经开始采集了
                start=true;
            //保留原0.5秒内的有效帧
            while(!next_.empty())
            {
                frame_.push(next_.front());
                next_.pop();
            }
           frame_.push(buf);
           //认为语音已经开始，退出循环
           //std::cerr<<"一";
            break;
        }else{
            //静音帧递增
            silence_frame++;
            //std::cerr<<"二";
        }
        //如果不是开始阶段且连续静音的帧数大于等于40(1s)则认为需要断句
        if(start&&((silence_frame+end_frames) > 39)) {
            //std::cerr << "+";
            finished_ = true;
        }
      }
    }
    //如果一直是静音 把next中有效帧取出
    while(!next_.empty()) {
        frame_.push(next_.front());
        next_.pop();
    }
    //这里不满足要求 在跳出前一个循环后不再有新的特征
    if(!finished_){
      while(1){
          //std::cerr<<"三";
          Vector<BaseFloat > data(nsample);
          bool more=input_->Read(&data);
          BaseFloat sum=0;
          for(int i=0;i<data.Dim();i++)
              sum += abs(data(i));
          if(sum / 400 >= jieduan)
            frame_.push(data);
          else//如果出现低于阈值的帧则退出循环
            break;
      }
    }
    auto nlength= static_cast<int32 >(frame_.size()*400);
    //偏置值初始为0
    int32 offset=0;
    //如果矩阵不为空
    if(nlength!=0)
    output.Resize(1,nlength);
    //如果无有效帧 则初始化矩阵为特殊值
    else
        output.Resize(2,400);
    //只要队列不为空 将队列中的有效帧存入矩阵中
      while(!frame_.empty()){
        Vector<BaseFloat > frames;
        //取出队首的帧
        frames=frame_.front();
        for(int j=0;j<frames.Dim();j++)
            (output)(0,j+offset)=frames(j);
        //加上当前帧的长度
        offset+=frames.Dim();
        //队首出队
        frame_.pop();
      }
    for(int i=0;i<offset/400;i++)
    std::cerr<< "|";
}