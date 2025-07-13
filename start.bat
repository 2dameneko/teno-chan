rem It is better to use direct indication of the order of devices used, just as specified in your cmd when starting llama-cpp
rem set CUDA_VISIBLE_DEVICES=0,1
set CUDA_VISIBLE_DEVICES=0,1

call venv\Scripts\activate
python teno-chan.py ^
  --gguf-url %1 ^
  --context-length 16384 ^
  --context-quantization-size 8 ^
  --gpu-memory 0.9 ^
  --verbose
