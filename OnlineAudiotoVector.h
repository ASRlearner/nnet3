//
// Created by smz on 18-12-27.
//

#ifndef NNET_TEST_ONLINEAUDIOTOVECTOR_H
#define NNET_TEST_ONLINEAUDIOTOVECTOR_H

#endif //NNET_TEST_ONLINEAUDIOTOVECTOR_H

#include "itf/decodable-itf.h"
#include "util/parse-options.h"
#include "online/online-feat-input.h"
#include "feat/feature-mfcc.h"

using namespace kaldi;

class OnlineAudioVector : public OnlineFeatInputItf {
public:

    OnlineAudioVector(OnlineAudioSourceItf *input,MfccOptions *mfcc_opts,
                                                  const int32 batch_size);

    virtual bool compute(Vector<BaseFloat> &output);

    virtual int32 Dim() const { return opts_.num_ceps;}
private:
    OnlineAudioSourceItf *input;
    MfccOptions *opts_;
    const int32 frame_size;
    const int32 frame_shift;
    const int32 sample_rate;
    const int32 batch_size;
    //SubVector<BaseFloat> wave_;
};

OnlineAudioVector::OnlineAudioVector(OnlineAudioSourceItf *input, MfccOptions *mfcc_opts,
                                                                  const int32 batch_size):
                                     input(input),opts_(mfcc_opts),batch_size(batch_size),
                                     frame_size(opts_->frame_opts.frame_length_ms),
                                     frame_shift(opts_->frame_opts.frame_shift_ms),
                                     sample_rate(opts_->frame_opts.samp_freq){}

//每次输出一个定长的vector
bool OnlineAudioVector::compute(Vector<BaseFloat> &output) {
    int32 frame_window=frame_size*sample_rate/1000,
          shift_window=frame_shift*sample_rate/1000;
     int32 nrev=frame_window+(batch_size-1)*shift_window;
     Vector<BaseFloat> data(nrev);

     bool ans=input->Read(&data);
     output=data;

     return ans;
}



