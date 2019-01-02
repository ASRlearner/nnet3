//
// Created by smz on 19-1-2.
//

#ifndef NNET_TEST_ONLINEVECTORINPUT_H
#define NNET_TEST_ONLINEVECTORINPUT_H

#endif //NNET_TEST_ONLINEVECTORINPUT_H

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>


#include "online/online-feat-input.h"
#include "feat/feature-functions.h"
#include <queue>
//#include <vector>

using namespace kaldi;


class OnlineVectorInput : public OnlineFeatInputItf{
public:
    OnlineVectorInput(int32 udp_port,int32 nsample);

    virtual bool Compute(Matrix<BaseFloat > *output);
    //返回音频数据的长度
    virtual int32 Dim() const { return read_nsamples;}
    //返回客户端网络地址结构
    const sockaddr_in& client_addr() const { return client_addr_;}
    //返回套接字描述符
    const int32 socket_desc() const { return  socket_desc_;}
private:
    std::queue<Vector<BaseFloat >> buf;
    int32 read_nsamples;
    int32 socket_desc_;
    sockaddr_in client_addr_;
    sockaddr_in server_addr_;
};