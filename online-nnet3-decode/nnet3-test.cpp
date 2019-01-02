//
// Created by smz on 18-12-29.
//

#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"
#include "online2/onlinebin-util.h"
#include "online2/online-timing.h"
#include "online2/online-endpoint.h"
#include "fstext/fstext-lib.h"
#include "lat/lattice-functions.h"
#include "util/kaldi-thread.h"
#include "nnet3/nnet-utils.h"

namespace kaldi{

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
        //采样率8k的情况下 每秒采样点个数为4560
        BaseFloat chunk_length_seconds=0.57;

        bool do_endpointing=false;
        bool online=true;

        po.Register("chunk-length", &chunk_length_seconds,
                    "Length of chunk size in seconds, that we process.  Set to <= 0 "
                    "to use all input in one chunk.");
        po.Register("word-symbol-table", &word_symbol_table,
                    "Symbol table for words [for debug output]");
        po.Register("do-endpointing", &do_endpointing,
                    "If true, apply endpoint detection");
        po.Register("online", &online,
                    "You can set this to false to disable online iVector estimation "
                    "and have all the data for each utterance used, even at "
                    "utterance start.  This is useful where you just want the best "
                    "results and don't care about online operation.  Setting this to "
                    "false has the same effect as setting "
                    "--use-most-recent-ivector=true and --greedy-ivector-extractor=true "
                    "in the file given to --ivector-extraction-config, and "
                    "--chunk-length=-1.");
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

        OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);
//
//        if (!online) {
//            //如果为真 总是使用最新可用的ivector，
//            //而并非那些指定帧的ivector
//            feature_info.ivector_extractor_info.use_most_recent_ivector = true;
//            //如果为真，提取ivector时尽可能多的读取我们现有可用的帧。可能会改善ivector的质量
//            feature_info.ivector_extractor_info.greedy_ivector_extractor = true;
//            //读取所有输入作为一块
//            chunk_length_secs = -1.0;
//        }

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
        if(word_symbol_table != "")
          if(!(word_sym = fst::SymbolTable::ReadText(word_symbol_table)))
              KALDI_ERR << "从文件中读取词符号表失败";

        //用于在线解码中的计时
        OnlineTimingStats timer;



    }catch(const std::exception& e){
        std::cerr<<e.what();
        return -1;
    }
}