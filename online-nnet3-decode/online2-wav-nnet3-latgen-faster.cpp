//
// Created by smz on 18-12-26.
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

namespace kaldi {

    void GetDiagnosticsAndPrintOutput(const std::string &utt,
                                      const fst::SymbolTable *word_syms,
                                      const CompactLattice &clat,
                                      int64 *tot_num_frames,
                                      double *tot_like) {
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

        //似然度
        double likelihood;
        //词图权重
        LatticeWeight weight;
        //帧数
        int32 num_frames;
        std::vector<int32> alignment;
        std::vector<int32> words;
        //使用最佳路径的词图 得到对齐状态 词id向量 以及词图权重
        GetLinearSymbolSequence(best_path_lat, &alignment, &words, &weight);

        //这里实际得到了帧数
        num_frames = alignment.size();
        //似然度
        likelihood = -(weight.Value1() + weight.Value2());
        *tot_num_frames += num_frames;
        *tot_like += likelihood;
        KALDI_VLOG(2) << "Likelihood per frame for utterance " << utt << " is "
                      << (likelihood / num_frames) << " over " << num_frames
                      << " frames.";

        if (word_syms != NULL) {
            std::cerr << utt << ' ';
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

int main(int argc, char *argv[]) {
    try {
        using namespace kaldi;
        using namespace fst;

        typedef kaldi::int32 int32;
        typedef kaldi::int64 int64;

        const char *usage =
                "Reads in wav file(s) and simulates online decoding with neural nets\n"
                "(nnet3 setup), with optional iVector-based speaker adaptation and\n"
                "optional endpointing.  Note: some configuration values and inputs are\n"
                "set via config files whose filenames are passed as options\n"
                "\n"
                "Usage: online2-wav-nnet3-latgen-faster [options] <nnet3-in> <fst-in> "
                "<spk2utt-rspecifier> <wav-rspecifier>\n"
                "The spk2utt-rspecifier can just be <utterance-id> <utterance-id> if\n"
                "you want to decode utterance by utterance.\n";


        //**********************************************************************************************************//
        //**********************************************命令行参数配置***********************************************//
        //**********************************************************************************************************//


        ParseOptions po(usage);

        std::string word_syms_rxfilename;

        //特征参数配置  这里包含了自适应的ivector以及基本特征的配置
        OnlineNnet2FeaturePipelineConfig feature_opts;
        //可解码对象参数配置 声学比重 每一块包含的帧数 等等
        nnet3::NnetSimpleLoopedComputationOptions decodable_opts;
        //解码器的参数配置 词图光束大小 剪枝比例等等 解码器最大活跃状态(max-active)等等
        LatticeFasterDecoderConfig decoder_opts;
        //在线断点的配置
        OnlineEndpointConfig endpoint_opts;

        //每一块的秒数
        BaseFloat chunk_length_secs = 0.18;
        //不使用断点检测
        bool do_endpointing = false;
        //配置false可以阻止在线ivector的估计 使用一条语音的所有数据
        bool online = true;

        po.Register("chunk-length", &chunk_length_secs,
                    "Length of chunk size in seconds, that we process.  Set to <= 0 "
                    "to use all input in one chunk.");
        po.Register("word-symbol-table", &word_syms_rxfilename,
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

        //配置的结构体了解一下参数
        feature_opts.Register(&po);
        decodable_opts.Register(&po);
        decoder_opts.Register(&po);
        endpoint_opts.Register(&po);


        po.Read(argc, argv);

        if (po.NumArgs() != 4) {
            po.PrintUsage();
            return 1;
        }


        //**********************************************************************************************************//
        //**********************************************读取模型文件***********************************************//
        //**********************************************************************************************************//

        //读取nnet3神经网络模型 解码图 spk2utt wav文件读取路径文件
        std::string nnet3_rxfilename = po.GetArg(1),
                fst_rxfilename = po.GetArg(2),
                spk2utt_rspecifier = po.GetArg(3),
                wav_rspecifier = po.GetArg(4);
                //clat_wspecifier = po.GetArg(5);

         //从OnlineNnet2FeaturePipelineConfig类初始化的 该类从命令行读取选项
         //负责存储OnlineNnet2FeaturePipeline的配置变量、对象和选项
        OnlineNnet2FeaturePipelineInfo feature_info(feature_opts);

        //如果不使用online ivector 则设置ivector-extractor的参数
        if (!online) {
            //如果为真 总是使用最新可用的ivector，
            //而并非那些指定帧的ivector
            feature_info.ivector_extractor_info.use_most_recent_ivector = true;
            //如果为真，提取ivector时尽可能多的读取我们现有可用的帧。可能会改善ivector的质量
            feature_info.ivector_extractor_info.greedy_ivector_extractor = true;
            //读取所有输入作为一块
            chunk_length_secs = -1.0;
        }

        //读取神经网络声学模型
        TransitionModel trans_model;
        //神经网络声学模型 即dnn部分
        nnet3::AmNnetSimple am_nnet;
        //这部分以测试模式读取神经网络模型 即不会对模型参数造成影响 只是单纯使用模型
        {
            bool binary;
            Input ki(nnet3_rxfilename, &binary);
            trans_model.Read(ki.Stream(), binary);
            am_nnet.Read(ki.Stream(), binary);
            //将批度归一化组件设为测试模式
            SetBatchnormTestMode(true, &(am_nnet.GetNnet()));
            //将失活组件设置为测试模式
            SetDropoutTestMode(true, &(am_nnet.GetNnet()));
            //必须在设置了batchnorm和dropout组件为测试模式后才可以这么做
            //这个函数尝试着去除随即失活 批度归一化等等
            nnet3::CollapseModel(nnet3::CollapseModelConfig(), &(am_nnet.GetNnet()));
        }

        /*std::vector<int32 > remove_node;
        remove_node.push_back(0);
        am_nnet.GetNnet().RemoveSomeNodes(remove_node);
        */
        // 这个对象包含了由所有可解码对象使用的预计算部分。它采用了一个指向am_nnet的指针
        //因为如果它拥有ivector特征 它不得不去修改nnet在间歇期间接收ivector特征
        nnet3::DecodableNnetSimpleLoopedInfo decodable_info(decodable_opts,
                                                            &am_nnet);

        //读取解码图
        fst::Fst<fst::StdArc> *decode_fst = ReadFstKaldiGeneric(fst_rxfilename);

        //读取words.txt文件
        fst::SymbolTable *word_syms = NULL;
        if (word_syms_rxfilename != "")
            if (!(word_syms = fst::SymbolTable::ReadText(word_syms_rxfilename)))
                KALDI_ERR << "Could not read symbol table from file "
                          << word_syms_rxfilename;

        //完成的解码个数和报错次数
        int32 num_done = 0, num_err = 0;
        //总的似然度
        double tot_like = 0.0;
        //帧数
        int64 num_frames = 0;

        //读取说话人到语音的信息
        SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
        //得到wav文件读取的路径
        RandomAccessTableReader<WaveHolder> wav_reader(wav_rspecifier);
        //CompactLatticeWriter clat_writer(clat_wspecifier);

        //在线时间状态类 用于在线解码中的计时
        OnlineTimingStats timing_stats;


        //**********************************************************************************************************//
        //***************************************读取wav文件中音频数据并提取特征**************************************//
        //**********************************************************************************************************//

        //遍历所有说话人 其中spk2utt_reader的key为说话人 value为说话人对应的语音
        for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
            std::string spk = spk2utt_reader.Key();
            //把一个说话人的语音全部放在一个string类型的向量中
            const std::vector<std::string> &uttlist = spk2utt_reader.Value();
            //存储了在线ivector提取器的自适应状态
            //更加有益的初始化同一个说话人下一条语音的自适应状态(可以不受之前说话人的影响初始化自适应状态)
            OnlineIvectorExtractorAdaptationState adaptation_state(
                    feature_info.ivector_extractor_info);
            //对当前说话人的所有音频文件进行一个遍历 进行解码
            for (size_t i = 0; i < uttlist.size(); i++) {
                std::string utt = uttlist[i];
                if (!wav_reader.HasKey(utt)) {
                    KALDI_WARN << "Did not find audio for utterance " << utt;
                    num_err++;
                    continue;
                }
                //读取对应音频文件的音频数据到wave_data
                const WaveData &wave_data = wav_reader.Value(utt);
                // 从通道0获取音频数据 (如果信号不是单声道, 我们只使用第一个通道).
                //wave_data.Data()实际返回的是一个矩阵(单通道情况下行数为1)
                //把参数1得到矩阵的第一行初始化给subvector data对象
                //data中存储已经是一个wav文件的所有音频数据 数据类型为basefloat
                SubVector<BaseFloat> data(wave_data.Data(), 0);
                //添加了音高和ivector的特征(mfcc) 总维度为143(100+3+40)
                OnlineNnet2FeaturePipeline feature_pipeline(feature_info);

                //设定与上一条语音的同一个说话人自适应状态 包括cmvn的自适应状态 ivector维度等等
                feature_pipeline.SetAdaptationState(adaptation_state);

                //保留来自解码器的最佳路径回溯 解码时静音相关的设置
                OnlineSilenceWeighting silence_weighting(
                        trans_model,
                        feature_info.silence_weighting_config,
                        decodable_opts.frame_subsampling_factor);

                //使用nnet3神经网络模型解码一段语音时使用的解码器
                SingleUtteranceNnet3Decoder decoder(decoder_opts, trans_model,
                                                    decodable_info,
                                                    *decode_fst, &feature_pipeline);
                //用来存储解码这段语音所用的时间
                OnlineTimer decoding_timer(utt);
                //采样率
                BaseFloat samp_freq = wave_data.SampFreq();
                //存储请求的采样块长度(每次采样点个数 即单次解码的音频数据长度)
                //这里识别整条语音 故将块长度设为极大值
                int32 chunk_length;
                if (chunk_length_secs > 0) {
                    //得到采样块的长度
                    chunk_length = int32(samp_freq * chunk_length_secs);
                    if (chunk_length == 0) chunk_length = 1;
                } else {
                    //返回编译器允许的最大值
                    chunk_length = std::numeric_limits<int32>::max();
                }

                int32 samp_offset = 0;
                std::vector<std::pair<int32, BaseFloat> > delta_weights;

                //音频数据的偏置值
                while (samp_offset < data.Dim()) {
                    //当前剩下的音频数据个数
                    int32 samp_remaining = data.Dim() - samp_offset;
                    //从当前音频数据中请求数据 只要请求的音频数据长度小于当前剩余的音频数据长度
                    //则取请求的长度 否则取剩余的长度
                    int32 num_samp = chunk_length < samp_remaining ? chunk_length
                                                                   : samp_remaining;
                    //从data中取出从偏置值开始的一段音频数据 num_samp代表取的长度
                    SubVector<BaseFloat> wave_part(data, samp_offset, num_samp);
                    //从wave_part中接收音频数据 这步并不会真正的处理它们 仅仅是拷贝
                    //采样频率需和音频数据的采样率相同
                    feature_pipeline.AcceptWaveform(samp_freq, wave_part);

                    //偏置值按当前提取的数据递增
                    samp_offset += num_samp;
                    //模拟等待与当前语音长度相同的时间
                    decoding_timer.WaitUntil(samp_offset / samp_freq);
                    //如果偏置值等于音频数据总长度
                    if (samp_offset == data.Dim()) {
                        //没有更多的特征了 ,清洗最后几帧
                        //并且确定化音高特征
                        feature_pipeline.InputFinished();
                    }

                    //如果静音权重非1且静音字符串非空 并且ivector特征非空
                    if (silence_weighting.Active() &&
                        feature_pipeline.IvectorFeature() != NULL) {
                        //计算当前回溯路径 decoder.Decoder()返回在线词图解码器
                        silence_weighting.ComputeCurrentTraceback(decoder.Decoder());
                        //调用该函数获取权重的变化
                        silence_weighting.GetDeltaWeights(feature_pipeline.NumFramesReady(),
                                                          &delta_weights);
                        //将变化的权重提供给ivector特征
                        feature_pipeline.IvectorFeature()->UpdateFrameWeights(delta_weights);
                    }

                    //加速解码
                    decoder.AdvanceDecoding();
                    //是否使用断点检测
                    if (do_endpointing && decoder.EndpointDetected(endpoint_opts)) {
                        break;
                    }
                }
                //完成解码 清理剩余token 加快词图的获取
                decoder.FinalizeDecoding();

                //*****************************************************************************************************//
                //*******************************************生成词图并用词图解码************************************//
                //*****************************************************************************************************//

                //定义压缩词图
                CompactLattice clat;
                bool end_of_utterance = true;
                //由词图快速解码器得到压缩的词图
                decoder.GetLattice(end_of_utterance, &clat);

                //使用词图解码并返回识别结果
                GetDiagnosticsAndPrintOutput(utt, word_syms, clat,
                                             &num_frames, &tot_like);
                //得到解码一段音频所用的时间
                decoding_timer.OutputStats(&timing_stats);

                // In an application you might avoid updating the adaptation state if
                // you felt the utterance had low confidence.  See lat/confidence.h
                //这是为了在你认为语音自信度不高的情况下避免更新自适应状态
                feature_pipeline.GetAdaptationState(&adaptation_state);

                // we want to output the lattice with un-scaled acoustics.
                //打印词图
                //BaseFloat inv_acoustic_scale =
                //        1.0 / decodable_opts.acoustic_scale;
                //ScaleLattice(AcousticLatticeScale(inv_acoustic_scale), &clat);

                //clat_writer.Write(utt, clat);
                KALDI_LOG << "Decoded utterance " << utt;
                num_done++;
            }
        }
        timing_stats.Print(online);

        KALDI_LOG << "Decoded " << num_done << " utterances, "
                  << num_err << " with errors.";
        KALDI_LOG << "Overall likelihood per frame was " << (tot_like / num_frames)
                  << " per frame over " << num_frames << " frames.";
        delete decode_fst;
        delete word_syms; // will delete if non-NULL.
        return (num_done != 0 ? 0 : 1);
    } catch(const std::exception& e) {
        std::cerr << e.what();
        return -1;
    }
} // main()
