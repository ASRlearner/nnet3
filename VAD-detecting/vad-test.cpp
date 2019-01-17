//
// Created by smz on 19-1-11.
//

#include "vad-detect.h"


using namespace kaldi;

int main(){
    typedef kaldi::int32 int32;

    int32 nsamples=400;

    OnlinePaSource au_src(500,16000,32768,4);

    OnlineMatrixOutput data(&au_src,nsamples,false);

    bool end=false;
    int valid=0;

    while(1){
        Matrix<BaseFloat > audio;

        data.Compute(audio);
        //如果有有效的帧出现
        if(audio.NumRows()==1)
            valid++;
        //如果矩阵为空或者累计静音帧至40
        if(data.is_finished()||(audio.NumRows()==2)){
            //如果当前有有效帧或者之前出现了有效帧则断句
            if(valid)
            std::cerr << "断句" <<std::endl;
            //如果当前因为长时间静音导致断句 则重置有效次数
            //if(audio.NumRows()==2)
                valid=0;
        }
//        if(pinjun>2000)
//        std::cerr<< pinjun <<std::endl;
        //read用来判断当前截断点后0.5秒内的静音情况
        data.read();
    }
    return 1;
}