cmake_minimum_required(VERSION 3.13)
project(nnet_test)

set(CMAKE_CXX_STANDARD 11)
set(KALDI_PATH /home/smz/asr/kaldi-5.4/src)
set(CMAKE_C_COMPILER gcc)

set(CMAKE_CXX_STANDARD g++)
set(CMAKE_CXX_STANDARD 11)

add_definitions(-msse
        -msse2
        -pthread
        -g
        -fPIC
        -D HAVE_CUDA=0
        -Wno-sign-compare
        -Wall
        -Wno-sign-compare
        -Wno-unused-local-typedefs
        -Wno-deprecated-declarations
        -Winit-self
        -D KALDI_DOUBLEPRECISION=0
        -D HAVE_EXECINFO_H=1
        -D HAVE_CXXABI_H
        -D HAVE_ATLAS
        -I /home/smz/asr/kaldi-5.4/src
        -L /home/smz/asr/kaldi-5.4/src
        -isystem ${KALDI_PATH}/../tools/openfst/include
        -o1)

include_directories(./
        usr/include
        ${KALDI_PATH}/online2
        ${KALDI_PATH}/../tools/ATLAS_headers/include)

include_directories(${KALDI_PATH}/../tools/openfst-1.6.7/include
                    ${KALDI_PATH}/../tools/portaudio/install/include)
include_directories(${KALDI_PATH}/../tools/portaudio/include)
include_directories(${KALDI_PATH}/../tools/portaudio/src/common)


#原kaldi中的demo
add_executable(nnet_test online-nnet3-decode/online2-wav-nnet3-latgen-faster.cpp)


# chain/nnet3强制截断识别
add_executable(nnet3-test online-nnet3-decode/nnet3-local.cpp
       )

# vad检测测试
add_executable(vad-test VAD-detecting/vad-test.cpp VAD-detecting/vad-application.cpp)

# gmmdemo测试
#add_executable(gmm-test main.cpp)

target_link_libraries(nnet_test
        ${KALDI_PATH}/online/kaldi-online.a
        ${KALDI_PATH}/online2/kaldi-online2.a
        ${KALDI_PATH}/ivector/kaldi-ivector.a
        ${KALDI_PATH}/nnet3/kaldi-nnet3.a
        ${KALDI_PATH}/chain/kaldi-chain.a
        ${KALDI_PATH}/nnet2/kaldi-nnet2.a
        ${KALDI_PATH}/cudamatrix/kaldi-cudamatrix.a
        ${KALDI_PATH}/decoder/kaldi-decoder.a
        ${KALDI_PATH}/lat/kaldi-lat.a
        ${KALDI_PATH}/fstext/kaldi-fstext.a
        ${KALDI_PATH}/hmm/kaldi-hmm.a
        ${KALDI_PATH}/feat/kaldi-feat.a
        ${KALDI_PATH}/transform/kaldi-transform.a
        ${KALDI_PATH}/gmm/kaldi-gmm.a
        ${KALDI_PATH}/tree/kaldi-tree.a
        ${KALDI_PATH}/util/kaldi-util.a
        ${KALDI_PATH}/matrix/kaldi-matrix.a
        ${KALDI_PATH}/base/kaldi-base.a
        ${KALDI_PATH}/../tools/openfst/lib/libfst.so
        ${KALDI_PATH}/../tools/portaudio/install/lib/libportaudio.a
        /usr/lib/x86_64-linux-gnu/liblapack.so
        /usr/lib/x86_64-linux-gnu/libcblas.so
        /usr/lib/x86_64-linux-gnu/libcblas.so
        /usr/lib/x86_64-linux-gnu/libf77blas.so
        asound
        rt
        pthread
        )

target_link_libraries(vad-test
        ${KALDI_PATH}/online/kaldi-online.a
        ${KALDI_PATH}/online2/kaldi-online2.a
        ${KALDI_PATH}/ivector/kaldi-ivector.a
        ${KALDI_PATH}/nnet3/kaldi-nnet3.a
        ${KALDI_PATH}/chain/kaldi-chain.a
        ${KALDI_PATH}/nnet2/kaldi-nnet2.a
        ${KALDI_PATH}/cudamatrix/kaldi-cudamatrix.a
        ${KALDI_PATH}/decoder/kaldi-decoder.a
        ${KALDI_PATH}/lat/kaldi-lat.a
        ${KALDI_PATH}/fstext/kaldi-fstext.a
        ${KALDI_PATH}/hmm/kaldi-hmm.a
        ${KALDI_PATH}/feat/kaldi-feat.a
        ${KALDI_PATH}/transform/kaldi-transform.a
        ${KALDI_PATH}/gmm/kaldi-gmm.a
        ${KALDI_PATH}/tree/kaldi-tree.a
        ${KALDI_PATH}/util/kaldi-util.a
        ${KALDI_PATH}/matrix/kaldi-matrix.a
        ${KALDI_PATH}/base/kaldi-base.a
        ${KALDI_PATH}/../tools/openfst/lib/libfst.so
        ${KALDI_PATH}/../tools/portaudio/install/lib/libportaudio.a
        /usr/lib/x86_64-linux-gnu/liblapack.so
        /usr/lib/x86_64-linux-gnu/libcblas.so
        /usr/lib/x86_64-linux-gnu/libcblas.so
        /usr/lib/x86_64-linux-gnu/libf77blas.so
        asound
        rt
        pthread
        )

target_link_libraries(nnet3-test
        ${KALDI_PATH}/online/kaldi-online.a
        ${KALDI_PATH}/online2/kaldi-online2.a
        ${KALDI_PATH}/ivector/kaldi-ivector.a
        ${KALDI_PATH}/nnet3/kaldi-nnet3.a
        ${KALDI_PATH}/chain/kaldi-chain.a
        ${KALDI_PATH}/nnet2/kaldi-nnet2.a
        ${KALDI_PATH}/cudamatrix/kaldi-cudamatrix.a
        ${KALDI_PATH}/decoder/kaldi-decoder.a
        ${KALDI_PATH}/lat/kaldi-lat.a
        ${KALDI_PATH}/fstext/kaldi-fstext.a
        ${KALDI_PATH}/hmm/kaldi-hmm.a
        ${KALDI_PATH}/feat/kaldi-feat.a
        ${KALDI_PATH}/transform/kaldi-transform.a
        ${KALDI_PATH}/gmm/kaldi-gmm.a
        ${KALDI_PATH}/tree/kaldi-tree.a
        ${KALDI_PATH}/util/kaldi-util.a
        ${KALDI_PATH}/matrix/kaldi-matrix.a
        ${KALDI_PATH}/base/kaldi-base.a
        ${KALDI_PATH}/../tools/openfst/lib/libfst.so
        ${KALDI_PATH}/../tools/portaudio/install/lib/libportaudio.a
        /usr/lib/x86_64-linux-gnu/liblapack.so
        /usr/lib/x86_64-linux-gnu/libcblas.so
        /usr/lib/x86_64-linux-gnu/libcblas.so
        /usr/lib/x86_64-linux-gnu/libf77blas.so
        asound
        rt
        pthread
        )