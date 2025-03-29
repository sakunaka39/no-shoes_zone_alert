# TensorRTのインクルードディレクトリを検索
find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS /usr/lib/aarch64-linux-gnu/include
)

# TensorRTライブラリを検索
find_library(TENSORRT_LIBRARY
  NAMES nvinfer
  HINTS /usr/lib/aarch64-linux-gnu/lib
)

# インクルードディレクトリとライブラリのパスを設定
include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIBRARY})

