//
// Created by smz on 18-12-25.
//

#ifndef NNET_TEST_NNET3_DECODABLE_H
#define NNET_TEST_NNET3_DECODABLE_H

#endif //NNET_TEST_NNET3_DECODABLE_H

#include "base/kaldi-common.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"
#include "util/parse-options.h"
#include "online/online-feat-input.h"
#include "feat/feature-mfcc.h"
#include "online/online-faster-decoder.h"
#include "online/online-decodable.h"
#include "online2/online-nnet3-decoding.h"
#include "online2/online-nnet2-feature-pipeline.h"


namespace  kaldi{
    void  SendResult(const std::vector<int32>& words,
                     const fst::SymbolTable *word_syms,
                     const bool line_break){
        KALDI_ASSERT(word_syms!=NULL);
        std::stringstream ss;
        for(int i=0;i<words.size();i++){
            std::string s=word_syms->Find(words[i]);
            if(s =="")
                KALDI_ERR<<"word_id"<<words[i]<<"not in symbol table.";
            ss << s << ' ';
        }
        if(line_break)
            ss<< "\n";
        if(sizeof(ss.str()) > 0)
            std::cout << ss.str();
    }
}

using namespace kaldi;
//用于接受客户端数据的类(直接通过构造函数得到客户端传来的矩阵)

class onlinefeatureinput : public OnlineFeatInputItf {
public:
    //构造函数由客户端传来特征和特征维度初始化
    onlinefeatureinput(OnlineFeInput<Mfcc> *input,Matrix<BaseFloat> feat,int32 batch_size,MfccOptions *opts);
    //得到客户端传来特征
    virtual bool Compute(Matrix<BaseFloat> *output);
    //计算特征的维度
    virtual int32 Dim() const {return opts->num_ceps;}
private:
    //发送的特征批度
    int32 batch_size;
    //feature是客户端传来的特征
    //由于feature是私有变量 需要通过公有方法compute来访问
    OnlineFeInput<Mfcc> *input;
    MfccOptions *opts;
    Matrix<BaseFloat> feature;
};

onlinefeatureinput::onlinefeatureinput(OnlineFeInput<Mfcc> *input,Matrix<BaseFloat> feat,
                                       int32 batch_size,MfccOptions *opts):input(input),feature(feat),batch_size(batch_size),opts(opts){}

bool onlinefeatureinput::Compute(Matrix<BaseFloat> *output) {
    //如果接受到的特征维度不为空
    feature.Resize(batch_size,opts->num_ceps,kUndefined);
    //output->Resize(batch_size,opts->num_ceps,kUndefined);
    //得到对应大小的mfcc特征矩阵
    bool more_feats=input->Compute(&feature);
    if(feature.NumCols()>0){
        std::stringstream ss;
        feature.Write(ss,true);
        output->Read(ss,true);
        return true;
    }
    return false;
}

int main(int argc, char *argv[]) {
    try {

        using namespace fst;
        typedef kaldi::int32 int32;
        //输入的是mfcc特征 需要配置一些mfcc特征的参数
        typedef OnlineFeInput<Mfcc> FeInput;

        const int32 KDeltaOrder = 2;

        //portaudio源的超时时间间隔
        const int32 kTimeout = 500;
        // PortAudio的采样频率
        const int32 kSampleFreq = 16000;
        // PortAudio内部环状缓冲区大小
        const int32 kPaRingSize = 32768;
        // Report interval for PortAudio buffer overflows in number of feat. batches
        const int32 kPaReportInt = 4;
        //用法：从麦克风接收输入，提取特征并且通过网络连接将它们发送到语音识别服务器
        const char *usage =
                "Takes input using a microphone(PortAudio), extracts features and sends them\n"
                "to a speech recognition server then return the online decoding result directly\n\n"
                "Usage: online-test --acoustic-scale=0.0769 etc. model fst words silence lda\n\n";
        ParseOptions po(usage);
        //一次性发送和提取的特征向量数量
        BaseFloat acoustic_scale = 0.1;
        int32 cmn_window=600,min_cmn_window=100;
        int32 left_context=4,right_context=4;
        int32 batch_size = 27;

        kaldi::DeltaFeaturesOptions delta_opts;
        //给po对象配置delta特征的参数
        //(实际上该特征参数对象的构造函数初始化了一部分参数值 这里登记了全部参数值到po对象 下同)
        delta_opts.Register(&po);
        //给po配置在线快速解码的参数选项
        OnlineFasterDecoderOpts decoder_opts;
        //给po配置在线特征矩阵的参数选项(每一次传输的帧数以及尝试次数)
        OnlineFeatureMatrixOptions feature_reading_opts;
        //给po对象配置解码器参数
        decoder_opts.Register(&po, true);
        feature_reading_opts.Register(&po);

        //po.Register("batch-size", &batch_size,
        //           "The number of feature vectors to be extracted and sent in one go");
        //register有三个参数其中 参数1和3是字符串 参数2是任意类型的数据
        //登记输入的参数选项值
        po.Register("left-context", &left_context, "Number of frames of left context");
        po.Register("right-context", &right_context, "Number of frames of right context");
        //声学模型比例
        po.Register("acoustic-scale", &acoustic_scale,
                    "Scaling factor for acoustic likelihoods");
        po.Register("cmn-window", &cmn_window,
                    "Number of feat. vectors used in the running average CMN calculation");
        po.Register("min-cmn-window", &min_cmn_window,
                    "Minumum CMN window used at start of decoding (adds "
                    "latency only at start)");

        //std::queue<int32> a;
        //a.push(32);
        //a.size();

        po.Read(argc, argv);
        //如果参数不为4或者5 则输出函数用法并退出
        if (po.NumArgs() != 4 && po.NumArgs() !=5) {
            po.PrintUsage();
            return 1;
        }
        //读取模型相关文件地址信息
        std::string model_filename=po.GetArg(1),
                fst_filname=po.GetArg(2),
                words_filename=po.GetArg(3),
                silence_phones=po.GetArg(4),
                lda_mat=po.GetOptArg(5);

        Matrix<BaseFloat> lda_transform;
        if(lda_mat != ""){
            bool binary_in;
            Input ki(lda_mat,&binary_in);
            lda_transform.Read(ki.Stream(),&binary_in);
        }

        std::vector<int32> silence;
        if(!SplitStringToIntegers(silence_phones ,":" ,false ,&silence ))
            KALDI_ERR << " 无效的静音音素字符串 " << silence_phones;

        if(silence.empty())
            KALDI_ERR << "没有给出静音音素！";

        //读取声学模型
        TransitionModel trans_model;
        AmDiagGmm am_gmm;
        {
            bool binary_in;
            Input ki(model_filename,&binary_in);
            trans_model.Read(ki.Stream(),&binary_in);
            am_gmm.Read(ki.Stream(),&binary_in);
        }

        //获取词符号表
        fst::SymbolTable *word_syms=NULL;
        if(!(word_syms=fst::SymbolTable::ReadText(words_filename)))
            KALDI_ERR << " 无法从文件中读取词符号表 " << words_filename;

        //读取解码图
        fst::Fst<fst::StdArc> *decode_fst = ReadDecodeGraph(fst_filname);


        //设置mfcc特征的参数 包括是否采用能量 设置帧长 帧移等等
        MfccOptions mfcc_opts;

        mfcc_opts.use_energy = false;
        int32 frame_length = mfcc_opts.frame_opts.frame_length_ms = 25;
        int32 frame_shift = mfcc_opts.frame_opts.frame_shift_ms = 10;

        //au_src是一个在线音频源接口(onlineaudiosourceinterface)对象
        //设置音频采集时的一些参数 超时时间、采样频率、环形缓冲区的大小等
        OnlinePaSource au_src(kTimeout, kSampleFreq, kPaRingSize, kPaReportInt);
        Mfcc mfcc(mfcc_opts);
        //设置传输中的音频特征参数 包括帧长 帧移等
        FeInput fe_input(&au_src, &mfcc,
                         frame_length * (kSampleFreq / 1000),
                         frame_shift * (kSampleFreq / 1000));
        std::cerr << std::endl << "Sending features " << std::endl;

        //定义在线快速解码器类
        OnlineFasterDecoder decoder(*decode_fst,decoder_opts,
                                    silence,trans_model);

        //存放解码后的词图弧向量
        VectorFst<LatticeArc> out_fst;

        //int32 feature_dim = mfcc_opts.num_ceps; //默认为13维

        //**********************************************************************************************//
        //**********************************  传输提取的特征和接受识别的结果*******************************//
        //**********************************************************************************************//

        //矩阵用于存放每一次传输时的特征batch_size*num_ceps 默认为27*13
        Matrix<BaseFloat> feats;

        //初始化矩阵为客户端传输过来的特征相符合的大小
        //feats.Resize(batch_size, mfcc_opts.num_ceps, kUndefined);
        //从fe_input中得到mfcc特征矩阵 存入feats中
        //bool more_feats = fe_input.Compute(&feats);
        //如果有特征输入

        onlinefeatureinput feature_input(&fe_input,feats,batch_size,&mfcc_opts);
        OnlineCmnInput cmn_input(&feature_input,cmn_window,min_cmn_window);
        OnlineFeatInputItf *feat_transform = 0;

        if(lda_mat != ""){
            feat_transform = new OnlineLdaInput(
                    &cmn_input,lda_transform,
                    left_context,right_context);
        }else{
            DeltaFeaturesOptions opts;
            opts.order = KDeltaOrder;
            feat_transform = new OnlineDeltaInput(
                    opts,&cmn_input);
        }

        OnlineFeatureMatrix feature_matrix( feature_reading_opts , feat_transform );

        OnlineDecodableDiagGmmScaled decodable(am_gmm,trans_model,acoustic_scale,&feature_matrix);

        std::cerr << std::endl << " 开始对传输来的特征进行解码 " <<std::endl;

        bool result_res = false;
        while (1) {
            //获得当前的解码状态
            OnlineFasterDecoder::DecodeState dstate = decoder.Decode(&decodable);
            //存放识别的词id的向量
            std::vector<int32> word_ids;
            //如果话未说完
            if (dstate & (decoder.kEndFeats | decoder.kEndUtt)) {
                decoder.FinishTraceBack(&out_fst);

                fst::GetLinearSymbolSequence(out_fst,static_cast<vector<int32> *>(0),
                                             &word_ids,
                                             static_cast<LatticeArc::Weight *>(0));

                SendResult(word_ids,word_syms,result_res || word_ids.size() );

                result_res = false;
            }else{
                if(decoder.PartialTraceback(&out_fst)){

                    fst::GetLinearSymbolSequence(out_fst,static_cast<vector<int32> *>(0),
                                                 &word_ids,
                                                 static_cast<LatticeArc::Weight *>(0));

                    SendResult(word_ids,word_syms,false);
                    if(!result_res)
                        result_res = (word_ids.size() > 0);
                }
            }

        }
        //释放指针内存空间

        delete feat_transform;
        delete word_syms;
        delete decode_fst;
        return 0;
    } catch(const std::exception& e) {
        //无缓冲方式输出错误信息
        std::cerr << e.what();
        return -1;
    }
} // main()