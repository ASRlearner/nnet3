//
// Created by smz on 19-1-9.
//

#ifndef NNET_TEST_ONLINECLIENTRECIEVE_H
#define NNET_TEST_ONLINECLIENTRECIEVE_H

#endif //NNET_TEST_ONLINECLIENTRECIEVE_H

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>


#include "online/online-feat-input.h"
#include "feat/feature-functions.h"
#include <queue>
//#include <vector>

using namespace kaldi;

class OnlineVectorInput{
public:
    OnlineVectorInput(int32 udp_port,int32 nsample);

    //以矩阵的形式输出音频特征
    virtual bool Compute(Matrix<BaseFloat > &output);
    //返回音频数据的长度
    virtual int32 Dim() const { return read_nsamples;}
    //通过udp传输接收来自客户端的音频数据 存入queue中
    bool Read();
    //返回客户端网络地址结构
    const sockaddr_in& client_addr() const { return client_addr_;}
    //返回套接字描述符
    const int32 socket_desc() const { return  socket_desc_;}
private:
    //队列作为缓冲区 存放音频数据的向量 先进先出
    //std::queue<Vector<BaseFloat >> *buffer;
    //音频数据长度
    int32 read_nsamples;
    int32 socket_desc_;
    sockaddr_in client_addr_;
    sockaddr_in server_addr_;
};

OnlineVectorInput ::OnlineVectorInput(int32 udp_port, int32 nsample):
        read_nsamples(nsample){
    server_addr_.sin_family=AF_INET;
    server_addr_.sin_port=htons(udp_port);
    server_addr_.sin_addr.s_addr=INADDR_ANY;

    socket_desc_=socket(AF_INET,SOCK_DGRAM,IPPROTO_UDP);

    if(socket_desc_==-1)
        KALDI_ERR << "socket()函数调用失败";
    int32 recvbuf_size=30000;
    if(setsockopt(socket_desc_,SOL_SOCKET,SO_RCVBUF,&recvbuf_size, sizeof(recvbuf_size))==-1)
        KALDI_ERR << "setsockopt()设置接收缓冲区大小失败";
    if(bind(socket_desc_, reinterpret_cast<sockaddr*>(&server_addr_), sizeof(server_addr_)) == -1)
        KALDI_ERR << "bind()调用失败";
}

//这个函数应当在其返回true的情况下不断调用
bool OnlineVectorInput ::Read() {
    //设定一个缓冲区
    char buf[65535];
    //客户端网络地址结构的长度
    socklen_t caddr_len= sizeof(&client_addr_);
    //接收到的音频数据的字节数
    ssize_t nrecv=recvfrom(socket_desc_,buf, sizeof(buf),0,
                           reinterpret_cast<sockaddr*>(&client_addr_),
                           &caddr_len);
    if(nrecv==-1){
        KALDI_ERR << "recvfrom()调用失败";
        return false;
    }
    std::stringstream ss(std::stringstream::in  | std::stringstream::out );
    //把buf中的二进制字节流写入到ss中
    ss.write(buf,nrecv);
    Vector<BaseFloat > audiodata(read_nsamples);

    audiodata.Read(ss,true);
    //把接收到的音频数据存入缓冲区
    //buffer->push(audiodata);
    return true;
}

//直接从客户端读取int16类型的数据
bool OnlineVectorInput::Compute(Matrix<BaseFloat> &output) {
    //以向量的形式接收客户端传来的int16数据
    Vector<int16 > buf(output.NumCols());
    socklen_t addrlen_= sizeof(client_addr_);
    //已接收到的字节数
    int32 receive=0;
    char buffer[6553000];
    //实际请求的字节数
    size_t request_byte=buf.Dim()* sizeof(int16);
    //如果接收到的字节数小于实际需要的 则不断接收
    while(receive<request_byte){
        //单次接收到的字节数
        ssize_t nrecv=recvfrom(socket_desc_,buffer+receive,request_byte-receive,
                               0, reinterpret_cast<sockaddr*>(&client_addr_),&addrlen_);
        //排除recvfrom调用失败的情况
        if(nrecv > 0)
            receive+=nrecv;
    }
    //从buffer中拷贝请求的字节数的数据到buf中
    memcpy(buf.Data(),&buffer,request_byte);

    //将buf中接收到的int16数据强制转换成float型 后存入矩阵output中
    for (int i=0;i<buf.Dim();i++) {
        (output)(0,i)= static_cast<BaseFloat >(buf(i));
    }
    return true;
}