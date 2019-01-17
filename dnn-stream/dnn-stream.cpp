//
// Created by smz on 19-1-10.
//

#include "online/online-audio-source.h"
#include "online/online-feat-input.h"
#include "online/online-faster-decoder.h"
#include "online/onlinebin-util.h"
#include "feat/feature-mfcc.h"
#include "nnet3/nnet-utils.h"
#include "nnet3/decodable-online-looped.h"
#include "nnet3/decodable-simple-looped.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online-nnet3-decode/OnlineVectorInput.h"

namespace kaldi{

}

int main(int argc,char *argv[]){
    try {
        using namespace kaldi;
        using namespace fst;
        typedef kaldi::int32 int32;
        typedef kaldi::int64 int64;

        const int32 KTimeout = 500;

        const int32 KSample_fre = 16000;

        const int32 KBufferSize = 327680;

        const int32 KReportInt = 4;

        const char *usage="nnet3/chain模型(不带ivector)的流式识别 不同与nnet3-local中的打印方式 \n\n"
                          "会判断语音是否结束来判断是否需要换行\n\n"
                          "用法 dnn-stream [options] model fst silence-id \n\n";
        ParseOptions po(usage);

        nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
        OnlineFasterDecoderOpts decoder_opts;
        OnlineNnet2FeaturePipelineConfig feature_opts;

        std::string symbol_table;
        //每次解码的采样音频数据时间长度
        BaseFloat chunk_second=1.5;

        po.Register("word-symbol-table",&symbol_table,"读取的words.txt文件");

        decodable_opts.Register(&po);
        feature_opts.Register(&po);

        po.Read(argc,argv);

        if(po.NumArgs()!=3){
            po.PrintUsage();
            return 1;
        }

        std::string model_filename=po.GetArg(1),
                    fst_filename=po.GetArg(2),
                    silence_phones=po.GetArg(3);

        //进一步配置特征管道参数
        OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);

        //读取声学模型
        TransitionModel trans_model;
        nnet3::AmNnetSimple am_nnet;
        {
            bool binary;
            Input ki(model_filename,&binary);
            trans_model.Read(ki.Stream(),&binary);
            am_nnet.Read(ki.Stream(),&binary);
            nnet3::SetBatchnormTestMode(true,&(am_nnet.GetNnet()));
            nnet3::SetDropoutTestMode(true,&(am_nnet.GetNnet()));
            nnet3::CollapseModel(nnet3::CollapseModelConfig(),&(am_nnet.GetNnet()));
        }

        //读取静音音素id
        std::vector<int32> silence;
        if(!SplitStringToIntegers(silence_phones ,":" ,false ,&silence ))
            KALDI_ERR << " 无效的静音音素字符串 " << silence_phones;
        if(silence.empty())
            KALDI_ERR << "没有给出静音音素！";

        //读取符号表
        fst::SymbolTable *sym_table=NULL;
        if(!symbol_table.empty())
            if(!(sym_table=fst::SymbolTable::ReadText(symbol_table)))
                KALDI_ERR << "words.txt读取失败";

        //读取解码图
        fst::Fst<fst::StdArc > *decode_fst=ReadFstKaldiGeneric(fst_filename);

        //可解码对象的参数配置
        nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts,
                                                            &am_nnet);

        //初始化音频数据采集存储对象
        OnlinePaSource au_arc(KTimeout,KSample_fre,KBufferSize,KReportInt);

        //每次解码音频数据采样点个数
        auto nsamples= static_cast<int32 >(KSample_fre*chunk_second);

        //初始化特征矩阵
        OnlineAudioMatrix audiodata(&au_arc,nsamples);

        //存储解码得到的词图
        VectorFst<LatticeArc> out_fst;

        //初始化在线快速解码器
        OnlineFasterDecoder decoder(*decode_fst,decoder_opts,
                                    silence,trans_model);

        OnlineFeatureInterface *ivector=NULL;

        std::cout<< "开始采集音频数据 提取定长数据进行解码" <<std::endl;

        while(1){

            Matrix<BaseFloat > data;
            bool more_feats=audiodata.compute(data);

            if(more_feats){

                SubVector<BaseFloat > audata(data,0);

                OnlineNnet2FeaturePipeline feature_pipeline(feature_info);

                //初始化可解码对象
                nnet3::DecodableAmNnetLoopedOnline decodable(trans_model,decodable_info,
                                                             &feature_pipeline,ivector);
                //接收音频数据
                feature_pipeline.AcceptWaveform(KSample_fre,audata);
                //接收完成
                feature_pipeline.InputFinished();

                bool result_res=false;


            }else{
                KALDI_WARN << "缓冲区中没有更多音频数据了！！";
                break;
            }
        }

        delete decode_fst;
        return 1;
    }catch(std::exception &e){
       std::cerr << e.what() <<std::endl;
        return -1;
    }
}