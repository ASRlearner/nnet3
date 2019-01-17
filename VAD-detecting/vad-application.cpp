//
// Created by smz on 19-1-17.
//

#include "feat/feature-mfcc.h"
#include "online/online-audio-source.h"
#include "online/online-feat-input.h"
#include "online/onlinebin-util.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"
#include "OnlineVectorInput.h"

namespace kaldi {

    void GetResultAndPrintout(const fst::SymbolTable *word_sym,
                              const CompactLattice &lattice){
        if(lattice.NumStates() == 0){
            KALDI_WARN << "这是空的词图";
            return;
        }

        CompactLattice best_path_lat;
        //生成最短路径的压缩词图
        CompactLatticeShortestPath(lattice,&best_path_lat);

        Lattice best_path_clat;
        //把最佳路径的压缩词图转化到词图
        ConvertLattice(best_path_lat,&best_path_clat);
        //初始化词图弧权重 对齐 词id表等等
        LatticeWeight weight;
        std::vector<int32 > alignment;
        std::vector<int32 > words;
        //获取线性符号序列到words中
        GetLinearSymbolSequence(best_path_clat,&alignment,&words,&weight);

        //如果词符号表不为空
        if(word_sym != NULL){
            //如果有识别结果才进行打印
            if(!words.empty()) {
                //遍历words中全部的词id
                int32 unknum=0;
                for (size_t i = 0; i < words.size(); i++) {
                    std::string word = word_sym->Find(words[i]);
                    if (word == "")
                        KALDI_ERR << "词id" << words[i] << "不在符号表中";
                    //直接打印id对应的词
                    if(word == "<UNK>"){
                        unknum++;
                        continue;
                    }
                    std::cerr << word << ' ';
                }
                //一部分音频数据识别完了换行
                if(unknum==0)
                    std::cerr << std::endl;
            }
        }
    }
}

int main(int argc,char *argv[]){
    try{
        using namespace fst;
        using namespace kaldi;
        typedef kaldi::int32 int32;
        typedef kaldi::int64 int64;

        const int32 KTimeout = 500;

        const int32 KBuffersize = 327680;

        const int32 KSample_freq=8000;

        const int32 KReportInt = 4;

        const char *usage="使用不带ivector的dnn模型进行解码 音频通过麦克风采集 实时解码得到识别结果\n\n"
                          "示例:nnet3-local [options] nnet3-model fst words\n\n";

        ParseOptions po(usage);

        std::string word_symbol_table;

        OnlineNnet2FeaturePipelineConfig feature_opts;
        LatticeFasterDecoderConfig decoder_opts;
        nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
        OnlineEndpointConfig endpoint_opts;

        //采样率16000hz的情况下 每秒采样的音频数据点个数为4560
        //实际表示每次解码的数据块长度
        BaseFloat chunk_size_seconds=4.0;

        bool end_pointing=false;

        po.Register("word-symbol-table", &word_symbol_table,
                    "Symbol table for words [for debug output]");
        po.Register("do-endpointing", &end_pointing,
                    "If true, apply endpoint detection");
        po.Register("num-threads-startup", &g_num_threads,
                    "Number of threads used when initializing iVector extractor.");

        decoder_opts.Register(&po);
        decodable_opts.Register(&po);
        endpoint_opts.Register(&po);
        feature_opts.Register(&po);

        po.Read(argc,argv);

        if(po.NumArgs()!=2)
        {
            KALDI_ERR << "参数个数不匹配";
            return 1;
        }

        std::string nnet3_filename=po.GetArg(1),
                fst_filename=po.GetArg(2);

        //通过featurepipelineconfig初始化特征管道参数
        OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);

        //读取final.mdl文件中的声学模型
        TransitionModel trans_model;
        nnet3::AmNnetSimple am_nnet;
        {
            bool binary;
            Input ki(nnet3_filename,&binary);
            trans_model.Read(ki.Stream(),&binary);
            am_nnet.Read(ki.Stream(),&binary);
            nnet3::SetBatchnormTestMode(true,&(am_nnet.GetNnet()));
            nnet3::SetDropoutTestMode(true,&(am_nnet.GetNnet()));
            nnet3::CollapseModel(nnet3::CollapseModelConfig(),&(am_nnet.GetNnet()));

        }

        //可解码对象的参数配置
        nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts,&am_nnet);

        //读取解码图
        fst::Fst<fst::StdArc> *decode_fst=ReadFstKaldiGeneric(fst_filename);

        //读取词符号表
        fst::SymbolTable *word_sym=NULL;
        if(!word_symbol_table.empty()){
            if(!(word_sym=fst::SymbolTable::ReadText(word_symbol_table)))
                KALDI_ERR << "读取词符号表出错";
        }

        OnlineTimingStats Timer;
        //采集音频的对象
        OnlinePaSource au_arc(KTimeout,KSample_freq,KBuffersize,KReportInt);

        BaseFloat  nsample=KSample_freq*chunk_size_seconds;
        //存储每次解码音频数据的采样点数
        //auto nsamples= static_cast<int32 >(nsample);

        OnlineAudioMatrix audiodata(&au_arc,nsample);

        BaseFloat sample_rate=KSample_freq;

        std::cout<< "开始采集音频 并提取定长音频数据进行解码"<< std::endl;

//        OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
//
//        SingleUtteranceNnet3Decoder decoder(decoder_opts,trans_model,
//                                            decodable_info,*decode_fst,
//                                            &feature_pipeline);

        while(1){
            sleep(1);

            Matrix<BaseFloat > wavedata;
            //将采集的音频定长读取到audata矩阵中

            bool more_feats=audiodata.compute(wavedata);

            if(more_feats){
                SubVector<BaseFloat > data(wavedata,0);

                OnlineNnet2FeaturePipeline feature_pipeline(feature_info);
                //解码器 有报警告
                SingleUtteranceNnet3Decoder decoder(decoder_opts,trans_model,
                                                    decodable_info,*decode_fst,
                                                    &feature_pipeline);

                //接收音频数据
                feature_pipeline.AcceptWaveform(sample_rate,data);

                //特征输入完成，提取mfcc和pitch并拼接
                feature_pipeline.InputFinished();
                //加速解码
                decoder.AdvanceDecoding();

                //完成解码 清除剩余token
                decoder.FinalizeDecoding();

                CompactLattice lat;
                bool end_of_utterance=true;

                //获取词图
                decoder.GetLattice(end_of_utterance,&lat);
                //打印定长音频数据的识别结果
                GetResultAndPrintout(word_sym,lat);

            }

            if(!more_feats)
                break;
        }
        delete decode_fst;
        delete word_sym;
        return 1;
    }catch(const std::exception &e){
        std::cerr << e.what();
        return -1;
    }
}