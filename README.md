# teno-chan

<div align="center">
    <img src="https://count.getloli.com/get/@teno-chan?theme=asoul&padding=4" alt="Visitor count"><br>
</div>

**teno-chan** is an [llama.cpp](https://github.com/ggerganov/llama.cpp) optimization assistant for mix of experts (MOE) LLM models like [DeepSeek-R1](https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF), [Qwen3-235B-A22B-GGUF](https://huggingface.co/unsloth/Qwen3-235B-A22B-GGUF), [Qwen3-30B-A3B](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF) in GGUF format that generates optimal [experts tensor](https://unsloth.ai/blog/deepseek-r1-0528) (ffn_up_exps, ffn_down_exps, ffn_gate_exps) offloading configuration. It analyzes your GPU/CPU resources and model architecture to create custom command-line arguments that maximize inference speed while staying within VRAM limits.

## Features
* **Multi-GPU Support**: Generates balanced tensor splitting ratios across multiple GPUs
* **Resource-aware Optimization**: Considers GPU VRAM, system RAM, and context requirements
* **GGUF Model Analysis**: Parses single/multi-part local/huggingface GGUF files to extract tensor metadata
* **CUDA_VISIBLE_DEVICES**: Respect environment variable
* **KV Cache Calculation**: Precisely calculates key-value cache size for given context length
* **Expert Tensor Identification**: Automatically detects MoE expert tensors for CPU offloading
* **Memory Safety Checks**: Validates model fits in available memory before optimization
* **Regex Compression**: Creates compact regex patterns for tensor overrides

## Requirements
* Python 3.10+
* NVIDIA GPU with CUDA libs installed
* Dependencies: `gguf-py`, `gguf`, `psutil`, `requests`, `pycuda`

## Installation
```bash
git clone https://github.com/2dameneko/teno-chan
cd teno-chan
install
```

## Usage
```bash
start "Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf"
```
You can open and change the args in the `start.bat` yourself or use pure script:

Either:
```
set CUDA_VISIBLE_DEVICES=0,1
call venv\Scripts\activate
python teno-chan.py ^
  --gguf-url %1 ^
  --context-length 16384 ^
  --context-quantization-size 8 ^
  --gpu-memory 0.9 ^
  --verbose
```

Example output:
```
-ngl 999 -ts 0.34,0.66 -ot "blk\.(56|57|58|59|60)\.ffn_down_exps\.weight=CPU" -ot "blk\.(56|57|58|59|60)\.ffn_up_exps\.weight=CPU" -ot "blk\.(56|57|58|59|60)\.ffn_gate_exps\.weight=CPU"
```

### Recommended Workflow
1. Run teno-chan with your model and context length
2. Copy the generated command
3. Append to your llama.cpp command:

## Command Line Options
| Argument | Description | Default |
|----------|-------------|---------|
| `-g/--gguf-url` | GGUF file URL or path | **Required** |
| `-c/--context-length` | Context length for KV cache calculation | **Required** |
| `-q/--context-quantization-size` | Quantization size for KV cache (4,8,16) | 16 |
| `--gpu-memory` | Per-GPU memory limits (GB or fraction) | 0.9 of VRAM |
| `--no-check` | Skip resource availability check | False |
| `--verbose` | Enable detailed logging | False |
| `--quiet` | Only output final command | False |

## Key Configuration Notes
1. **CUDA_VISIBLE_DEVICES**: For best results, set before running:
   ```bash
   set CUDA_VISIBLE_DEVICES=0,1  # Use specific GPUs
   ```
2. **GPU Memory Specification**: Use either:
   - Fractions: `--gpu-memory 0.85,0.9`
   - Gigabytes: `--gpu-memory 31.3,22.6`

## Version History
* **0.2** (Enhanced Regex Compression):
* **0.1** (Initial Release):

## Important Notes
1. This is an **early version** - results may require manual tuning
2. Reduce `--gpu-memory` values if encountering OOM errors while loading llama.cpp
3. Models with unusual architectures may require manual overrides

## License
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)

## Credits
* Project inspiration: [gguf-tensor-overrider](https://github.com/k-koehler/gguf-tensor-overrider)
* GGUF parsing: [ggerganov/ggml](https://github.com/ggerganov/ggml)
* Quantization specs: [llama.cpp](https://github.com/ggerganov/llama.cpp)

Thank you for using teno-chan! Your GPU's new best friend 💖
