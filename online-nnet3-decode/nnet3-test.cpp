//
// Created by smz on 18-12-29.
//

#include "feat/wave-reader.h"
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

#include <queue>

namespace kaldi {

        void GetDiagnosticsAndPrintOutput(const fst::SymbolTable *word_syms,
                                          const CompactLattice &clat) {
            //判断词图的状态个数 如果为0 则警告 退出
            if (clat.NumStates() == 0) {
                KALDI_WARN << "Empty lattice.";
                return;
            }
            //压缩词图
            CompactLattice best_path_clat;
            //求压缩词图的最短路径 这里得到最佳路径的词图
            CompactLatticeShortestPath(clat, &best_path_clat);

            Lattice best_path_lat;
            //把最佳路径的压缩词图转换成词图
            ConvertLattice(best_path_clat, &best_path_lat);


            LatticeWeight weight;
            //帧数
            //int32 num_frames;
            std::vector<int32> alignment;
            std::vector<int32> words;
            //使用最佳路径的词图 得到对齐状态 词id向量 以及词图权重
            GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);

            if (word_syms != NULL) {
                //std::cerr << utt << ' ';
                //打印识别结果
                for (size_t i = 0; i < words.size(); i++) {
                    std::string s = word_syms->Find(words[i]);
                    if (s == "")
                        KALDI_ERR << "Word-id " << words[i] << " not in symbol table.";
                    std::cerr << s << ' ';
                }
                std::cerr << std::endl;
            }
        }

}


int main(int argc, char *argv[]){
    try{
        using namespace kaldi;
        using namespace fst;

        typedef kaldi::int32 int32;
        typedef kaldi::int64 int64;

        const char *usage="接收来自客户端的音频数据 对特征进行处理后用神经网络模型进行解码 得到识别结果"
                          "nnet3-test model fst word \n\n";

        ParseOptions po(usage);

        std::string word_symbol_table;

        OnlineNnet2FeaturePipelineConfig feature_opts;
        nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
        LatticeFasterDecoderConfig decoder_opts;
        OnlineEndpointConfig endpoint_opts;
        //块长度 实际是数据块的采样时间（s）
        //采样率16k的情况下 每秒采样点个数为4560
        BaseFloat chunk_length_seconds=0.285;

        bool do_endpointing=false;

        po.Register("chunk-length", &chunk_length_seconds,
                    "Length of chunk size in seconds, that we process.  Set to <= 0 "
                    "to use all input in one chunk.");
        po.Register("word-symbol-table", &word_symbol_table,
                    "Symbol table for words [for debug output]");
        po.Register("do-endpointing", &do_endpointing,
                    "If true, apply endpoint detection");
        po.Register("num-threads-startup", &g_num_threads,
                    "Number of threads used when initializing iVector extractor.");

        feature_opts.Register(&po);
        decodable_opts.Register(&po);
        decoder_opts.Register(&po);
        endpoint_opts.Register(&po);

        po.Read(argc,argv);

        if(po.NumArgs() != 3)
        {
            KALDI_ERR<< "参数个数不匹配";
            return 1;
        }

        std::string model_filename=po.GetArg(1),
                    fst_rxfilename=po.GetArg(2),
                    udp_port_string=po.GetArg(3);

        int32 udp_port=atoi(udp_port_string.c_str());

        //在线特征管道的配置 通过pipelineconfig类初始化
        OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);

        TransitionModel trans_model;
        nnet3::AmNnetSimple am_nnet;
        {
            bool binary;
            Input ki(model_filename,&binary);
            trans_model.Read(ki.Stream(),&binary);
            am_nnet.Read(ki.Stream(),&binary);
            nnet3::SetBatchnormTestMode(true,&(am_nnet.GetNnet()));
            nnet3::SetDropoutTestMode(true,&(am_nnet.GetNnet()));
            nnet3::CollapseModel(nnet3::CollapseModelConfig,&(am_nnet.GetNnet()));
        }
        //可解码对象的参数
        nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts,&am_nnet);

        //读取解码图
        fst::Fst<fst::StdArc> *decode_fst=ReadFstKaldiGeneric(fst_rxfilename);

        //读取words.txt文件
        fst::SymbolTable *word_sym=NULL;
        if(!word_symbol_table.empty())
          if(!(word_sym = fst::SymbolTable::ReadText(word_symbol_table)))
              KALDI_ERR << "从文件中读取词符号表失败";

        //用于在线解码中的计时
        OnlineTimingStats timer;

        //设定采样率 得到实际解码的音频数据的采样点个数
        int32  nsample=16000*chunk_length_seconds;
        //udp传输 接收来自客户端传来的定长音频数据 长度由nsample决定
        OnlineVectorInput audioinput(udp_port,nsample);

        while (1){

            Matrix<BaseFloat > wavedata;
            //将矩阵拉长至行向量 行数为1  列数为所需要的音频采样点个数
            wavedata.Resize(1,nsample);
            //从客户端接收得到定长的音频数据用于解码
            audioinput.Compute(wavedata);

            SubVector<BaseFloat > audiodata(wavedata,0);
            //设定用于解码的特征管道
            OnlineNnet2FeaturePipeline feature_pipeline(feature_info);

            //单一语音解码器
            SingleUtteranceNnet3Decoder decoder(decoder_opts,
                                                 trans_model,decodable_info,
                                                 *decode_fst,&feature_pipeline);

            //采样率
            BaseFloat sample_freq=16000;
            //特征管道接收客户端传来的定长音频数据
            feature_pipeline.AcceptWaveform(sample_freq,audiodata);
            //音频数据输入完毕
            feature_pipeline.InputFinished();

            //加速解码
            decoder.AdvanceDecoding();
            if (do_endpointing && decoder.EndpointDetected(endpoint_opts)) {
                break;
            }
            //完成解码
            decoder.FinalizeDecoding();

            CompactLattice lat;
            bool end_of_utterance=true;
            //获取词图
            decoder.GetLattice(end_of_utterance,&lat);

            GetDiagnosticsAndPrintOutput(word_sym,lat);


        }
        delete decode_fst;
        delete word_sym;
        return 1;
    }catch(const std::exception& e){
        std::cerr<<e.what();
        return -1;
    }
}