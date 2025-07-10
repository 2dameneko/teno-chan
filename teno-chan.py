#teno-chan 0.1

import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from gguf import GGUFReader, GGMLQuantizationType


class Log:
    """Unified logging interface with configurable verbosity."""
    verbose: bool = False
    no_log: bool = False

    @classmethod
    def log(cls, log_level: str, message: str) -> None:
        """Log messages with level-based filtering."""
        if cls.no_log:
            return
        if log_level == "info" and cls.verbose:
            print(f"[INFO] {message}")
        elif log_level == "default":
            print(message)

    @classmethod
    def warn(cls, message: str) -> None:
        """Log warning messages."""
        if cls.no_log:
            return
        print(f"[WARN] {message}")

    @classmethod
    def error(cls, message: str) -> None:
        """Log error messages to stderr."""
        if cls.no_log:
            return
        print(f"[ERROR] {message}", file=sys.stderr)


def get_ram_bytes() -> int:
    """
    Get total system RAM in bytes with fallback to 128GB estimate.
    
    Returns:
        Total system RAM in bytes
    """
    try:
        import psutil
        return psutil.virtual_memory().total
    except ImportError:
        Log.warn("psutil not installed, using default 128GB RAM estimate")
        return 128 * 1024**3  # 128 GB


def _fallback_nvidia_smi() -> List[Dict[str, Any]]:
    """
    Fallback to nvidia-smi with CUDA_VISIBLE_DEVICES support.
    
    Returns:
        List of GPU dictionaries from nvidia-smi output
    """
    try:
        cmd = "nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv,noheader"
        output = subprocess.check_output(
            cmd, shell=True, stderr=subprocess.DEVNULL
        ).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        Log.warn("nvidia-smi not available. No GPUs detected.")
        return []

    all_gpus = []
    for line in output.splitlines():
        if not line.strip():
            continue
        parts = line.split(',')
        if len(parts) < 4:
            continue
            
        index_str = parts[0].strip()
        uuid = parts[1].strip()
        mem_str = parts[-1].strip()
        name = ','.join(parts[2:-1]).strip()
        
        try:
            mem_mb = int(mem_str.replace(' MiB', ''))
            all_gpus.append({
                "index_str": index_str,
                "uuid": uuid,
                "name": name,
                "memory_bytes": mem_mb * 1024**2
            })
        except ValueError:
            Log.warn(f"Couldn't parse GPU memory: {mem_str}. Skipping GPU.")

    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        return [
            {
                "cuda_id": int(gpu["index_str"]),
                "name": gpu["name"],
                "memory_total_bytes": gpu["memory_bytes"]
            }
            for gpu in all_gpus
            if gpu["index_str"].isdigit()
        ]
    
    if visible.strip() == "":
        return []
        
    tokens = [t.strip() for t in visible.split(',')]
    visible_gpus = []
    
    for token in tokens:
        for gpu in all_gpus:
            if token == gpu["index_str"] or token == gpu["uuid"]:
                visible_gpus.append(gpu)
                break
        else:
            Log.warn(f"GPU specified in CUDA_VISIBLE_DEVICES not found: {token}")
    
    return [
        {
            "cuda_id": idx,
            "name": gpu["name"],
            "memory_total_bytes": gpu["memory_bytes"]
        }
        for idx, gpu in enumerate(visible_gpus)
    ]


def get_nvidia_gpus() -> List[Dict[str, Any]]:
    """
    Get NVIDIA GPU information respecting CUDA_VISIBLE_DEVICES.
    
    Returns:
        List of GPU dictionaries containing index, name, and memory
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
        return _fallback_nvidia_smi()
        
    try:
        import pycuda.driver as drv
        drv.init()
    except ImportError:
        Log.warn("pycuda not available. Falling back to nvidia-smi.")
        return _fallback_nvidia_smi()
    except Exception as e:
        Log.warn(f"pycuda initialization failed: {str(e)}. Falling back to nvidia-smi.")
        return _fallback_nvidia_smi()

    try:
        gpus = []
        for idx in range(drv.Device.count()):
            device = drv.Device(idx)
            gpus.append({
                "cuda_id": idx,
                "name": device.name(),
                "memory_total_bytes": device.total_memory()
            })
        return gpus
    except Exception as e:
        Log.warn(f"pycuda device query failed: {str(e)}. Falling back to nvidia-smi.")
        return _fallback_nvidia_smi()


class GGUFMultiPartReader:
    """Handles multi-part GGUF files with combined metadata and tensors"""
    def __init__(self, readers: List[GGUFReader]) -> None:
        if not readers:
            raise ValueError("At least one GGUFReader required")
        self.readers = readers
        self.fields = readers[0].fields
        self.tensors = []
        for reader in readers:
            self.tensors.extend(reader.tensors)


def download_single_gguf(url: str) -> GGUFReader:
    """
    Download and parse a single GGUF file with retries.
    
    Args:
        url: URL of GGUF file
    
    Returns:
        GGUFReader instance
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".gguf") as tmp:
        for attempt in range(3):
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                for chunk in response.iter_content(chunk_size=8192):
                    tmp.write(chunk)
                break
            except (requests.RequestException, ConnectionError) as e:
                if attempt == 2:
                    raise RuntimeError(f"Failed to download GGUF after 3 attempts: {str(e)}")
                Log.warn(f"Download failed (attempt {attempt+1}/3): {str(e)}")
        tmp_path = tmp.name
        
    try:
        reader = GGUFReader(tmp_path)
    finally:
        os.unlink(tmp_path)
        
    return reader


def download_gguf(url_or_path: str) -> Union[GGUFReader, GGUFMultiPartReader]:
    """
    Handles both single and multi-part GGUF files (local or HTTP).
    
    Args:
        url_or_path: URL or file path to GGUF file
    
    Returns:
        GGUFReader or GGUFMultiPartReader instance
    """
    match = re.match(r"(.+?)-(\d+)-of-(\d+)\.gguf", url_or_path)
    if not match:
        if url_or_path.startswith('http'):
            return download_single_gguf(url_or_path)
        return GGUFReader(url_or_path)
    
    base_path = match.group(1)
    part_num = int(match.group(2))
    total_parts = int(match.group(3))
    readers = []
    
    for i in range(1, total_parts + 1):
        part_name = f"{base_path}-{i:05d}-of-{total_parts:05d}.gguf"
        Log.log("info", f"Loading part {i} of {total_parts}: {part_name}")
        
        if url_or_path.startswith('http'):
            reader = download_single_gguf(part_name)
        else:
            reader = GGUFReader(part_name)
            
        readers.append(reader)
        
    return GGUFMultiPartReader(readers)


# Quantization sizes mapping (block_size, type_size)
GGML_QUANT_SIZES = {
    GGMLQuantizationType.F32: (1, 4),
    GGMLQuantizationType.F16: (1, 2),
    GGMLQuantizationType.BF16: (1, 2),
    GGMLQuantizationType.F64: (1, 8),

    GGMLQuantizationType.I8: (1, 1),
    GGMLQuantizationType.I16: (1, 2),
    GGMLQuantizationType.I32: (1, 4),
    GGMLQuantizationType.I64: (1, 8),

    GGMLQuantizationType.Q4_0: (32, 18),
    GGMLQuantizationType.Q4_1: (32, 20),
    GGMLQuantizationType.Q5_0: (32, 22),
    GGMLQuantizationType.Q5_1: (32, 24),
    GGMLQuantizationType.Q8_0: (32, 34),
    GGMLQuantizationType.Q8_1: (32, 36),

    GGMLQuantizationType.Q2_K: (256, 92),
    GGMLQuantizationType.Q3_K: (256, 112),
    GGMLQuantizationType.Q4_K: (256, 144),
    GGMLQuantizationType.Q5_K: (256, 176),
    GGMLQuantizationType.Q6_K: (256, 208),
    GGMLQuantizationType.Q8_K: (256, 256),

    GGMLQuantizationType.IQ1_M: (256, 50),
    GGMLQuantizationType.IQ1_S: (256, 57),
    GGMLQuantizationType.IQ2_XXS: (256, 70),
    GGMLQuantizationType.IQ2_XS: (256, 78),
    GGMLQuantizationType.IQ2_S: (256, 80),
    GGMLQuantizationType.IQ3_XXS: (256, 103),
    GGMLQuantizationType.IQ3_S: (256, 113),
    GGMLQuantizationType.IQ4_NL: (256, 128),
    GGMLQuantizationType.IQ4_XS: (256, 136),
    
    GGMLQuantizationType.TQ1_0: (256, 54),
    GGMLQuantizationType.TQ2_0: (256, 108),
}


def bytes_to_mib(bytes_val: float) -> float:
    """Convert bytes to mebibytes."""
    return bytes_val / (1024**2)


def _get_field_value(gguf: Union[GGUFReader, GGUFMultiPartReader], 
                    field_name: str, 
                    default: Optional[float] = None) -> float:
    """Extract field value from GGUF metadata with fallback."""
    if field_name in gguf.fields:
        field = gguf.fields[field_name]
        content = field.contents()
        if isinstance(content, list) and content:
            return float(content[0])
        return float(content)
            
    alt_names = {
        "block_count": ["n_layer", "num_hidden_layers"],
        "embedding_length": ["n_embd", "hidden_size"],
        "attention.head_count": ["n_head", "num_attention_heads"],
        "attention.head_count_kv": ["n_head_kv", "num_key_value_heads"],
    }
    
    field_key = field_name.split('.')[-1]
    for alt in alt_names.get(field_key, []):
        if alt in gguf.fields:
            field = gguf.fields[alt]
            content = field.contents()
            if isinstance(content, list) and content:
                return float(content[0])
            return float(content)
                
    if default is not None:
        return default
        
    raise ValueError(f"Missing required field: {field_name}")


def extract_metadata(gguf: Union[GGUFReader, GGUFMultiPartReader]) -> Dict[str, float]:
    """
    Robust metadata extraction that works with GGUFReader structure.
    
    Args:
        gguf: GGUFReader or GGUFMultiPartReader instance
    
    Returns:
        Dictionary containing model metadata
    """
    arch_field = gguf.fields.get("general.architecture")
    arch = "llama"  # default
    
    if arch_field is not None:
        arch_content = arch_field.contents()
        if isinstance(arch_content, bytes):
            arch = arch_content.decode('utf-8')
        else:
            arch = str(arch_content)
            
    arch = arch.lower().replace('-', '').replace('_', '')
    Log.log("info", f"Detected model architecture: {arch}")
    
    ARCH_FIELD_MAP = {
        'qwen3moe': {
            'hidden_size': 'qwen3moe.embedding_length',
            'num_attention_heads': 'qwen3moe.attention.head_count',
            'num_layers': 'qwen3moe.block_count',
            'num_key_value_heads': 'qwen3moe.attention.head_count_kv',
        },
        'qwen3': {
            'hidden_size': 'qwen3.embedding_length',
            'num_attention_heads': 'qwen3.attention.head_count',
            'num_layers': 'qwen3.block_count',
            'num_key_value_heads': 'qwen3.attention.head_count_kv',
        },
        'llama': {
            'hidden_size': 'llama.embedding_length',
            'num_attention_heads': 'llama.attention.head_count',
            'num_layers': 'llama.block_count',
            'num_key_value_heads': 'llama.attention.head_count_kv',
        },
        'llama4': {
            'hidden_size': 'llama.embedding_length',
            'num_attention_heads': 'llama.attention.head_count',
            'num_layers': 'llama.block_count',
            'num_key_value_heads': 'llama.attention.head_count_kv',
        },
        'deepseek2': {
            'hidden_size': 'deepseek2.embedding_length',
            'num_attention_heads': 'deepseek2.attention.head_count',
            'num_layers': 'deepseek2.block_count',
            'num_key_value_heads': 'deepseek2.attention.head_count_kv',
        },
        'dots1': {
            'hidden_size': 'dots1.embedding_length',
            'num_attention_heads': 'dots1.attention.head_count',
            'num_layers': 'dots1.block_count',
            'num_key_value_heads': 'dots1.attention.head_count_kv',
        }
    }
    
    field_map = ARCH_FIELD_MAP.get(arch, ARCH_FIELD_MAP['llama'])
    
    try:
        hidden_size = _get_field_value(gguf, field_map['hidden_size'])
        num_attention_heads = _get_field_value(gguf, field_map['num_attention_heads'])
        num_layers = _get_field_value(gguf, field_map['num_layers'])
        
        try:
            num_key_value_heads = _get_field_value(gguf, field_map['num_key_value_heads'])
        except ValueError:
            num_key_value_heads = num_attention_heads
            
    except ValueError as e:
        Log.error(f"Metadata extraction failed: {str(e)}")
        
        # Fallback to direct attributes
        hidden_size = getattr(gguf, 'n_embd', 4096.0)
        num_attention_heads = getattr(gguf, 'n_head', 32.0)
        num_layers = getattr(gguf, 'n_layer', 32.0)
        num_key_value_heads = getattr(gguf, 'n_head_kv', num_attention_heads)
            
        Log.warn(f"Using fallback meta hidden_size={int(hidden_size)}, "
                 f"num_attention_heads={int(num_attention_heads)}, num_layers={int(num_layers)}, "
                 f"num_key_value_heads={int(num_key_value_heads)}")
    
    head_size = hidden_size / num_attention_heads
    
    Log.log("info", f"Extracted meta hidden_size={int(hidden_size)}, "
                    f"num_attention_heads={int(num_attention_heads)}, num_layers={int(num_layers)}, "
                    f"num_key_value_heads={int(num_key_value_heads)}, head_size={int(head_size)}")
    
    return {
        "hidden_size": hidden_size,
        "num_attention_heads": num_attention_heads,
        "num_layers": num_layers,
        "num_key_value_heads": num_key_value_heads,
        "head_size": head_size,
    }


def calculate_kv_cache_size_bytes(
    gguf: Union[GGUFReader, GGUFMultiPartReader], 
    context_length: int, 
    context_quantization_size: int
) -> float:
    """
    Calculate KV cache size with validation.
    
    Args:
        gguf: GGUF model instance
        context_length: Maximum context length
        context_quantization_size: Quantization size in bits
    
    Returns:
        Required KV cache size in bytes
    """
    metadata = extract_metadata(gguf)
    context_quantization_byte_size = context_quantization_size / 8
    
    return (
        2 *  # key + value
        context_quantization_byte_size *
        metadata["num_layers"] *
        context_length *
        metadata["num_key_value_heads"] *
        metadata["head_size"]
    )


def calculate_tensor_size_bytes(tensor: Any) -> int:
    """
    Calculate tensor size using correct quantization block math.
    
    Args:
        tensor: GGUF tensor object
    
    Returns:
        Size in bytes
    """
    quant_type = tensor.tensor_type
    if quant_type in GGML_QUANT_SIZES:
        block_size, type_size = GGML_QUANT_SIZES[quant_type]
        n_elements = 1
        for dim in tensor.shape:
            n_elements *= dim
        return (n_elements * type_size) // block_size
    
    # Fallback for unknown types
    Log.warn(f"Unknown quantization type: {quant_type} for tensor {tensor.name}")
    n_elements = 1
    for dim in tensor.shape:
        n_elements *= dim
    return n_elements * 4  # Assume 32-bit float as fallback


def calculate_tensors_size_bytes(gguf: Union[GGUFReader, GGUFMultiPartReader]) -> float:
    """
    Calculate total size of all tensors with progress.
    
    Args:
        gguf: GGUF model instance
    
    Returns:
        Total size in bytes
    """
    total = 0
    for i, tensor in enumerate(gguf.tensors):
        if i % 100 == 0 and Log.verbose:
            Log.log("info", f"Calculating tensor sizes: {i}/{len(gguf.tensors)}")
        total += calculate_tensor_size_bytes(tensor)
    return total


def verify_allocation(allocator: Any, gguf: Any) -> float:
    """
    Verify allocation matches tensor sizes.
    
    Args:
        allocator: DeviceAllocator instance
        gguf: GGUF model instance
    
    Returns:
        Total tensor size in bytes
    """
    total_allocated = sum(device.bytes_allocated for device in allocator.devices)
    total_tensor_size = calculate_tensors_size_bytes(gguf)
    
    if abs(total_allocated - total_tensor_size) > 1024:  # 1KB tolerance
        Log.warn(f"Allocation mismatch: {bytes_to_mib(total_allocated):.2f} MiB allocated vs "
                 f"{bytes_to_mib(total_tensor_size):.2f} MiB tensor size")
                 
        if Log.verbose:
            for tensor in gguf.tensors:
                size = calculate_tensor_size_bytes(tensor)
                device = allocator.tensor_map.get(tensor.name, "UNALLOCATED")
                Log.log("info", f"{tensor.name}: {bytes_to_mib(size):.2f} MiB -> {device}")
                
    return total_tensor_size


class Device:
    """Enhanced device with memory balancing"""
    def __init__(self, name: str, memory_bytes: int, priority: int, gpu_percentage: float) -> None:
        self.name = name
        self.memory_total_bytes = memory_bytes
        self.priority = priority
        self.utilization_percentage = gpu_percentage
        self.bytes_allocated = 0
        self.unsafe = False

    @property
    def safe_memory(self) -> float:
        """Calculate safe memory limit based on utilization percentage."""
        return self.memory_total_bytes * self.utilization_percentage

    def can_allocate(self, required: float) -> bool:
        """Check if device can allocate requested memory."""
        if self.unsafe:
            return True
        return (self.bytes_allocated + required) <= self.safe_memory

    def set_unsafe(self) -> None:
        """Set device to ignore memory limits."""
        self.unsafe = True

    def allocate(self, required: float) -> None:
        """Allocate memory on device."""
        if not self.can_allocate(required):
            raise MemoryError(f"Insufficient memory on {self.name} "
                             f"({bytes_to_mib(self.bytes_allocated):.2f}/"
                             f"{bytes_to_mib(self.safe_memory):.2f} MiB used, "
                             f"need {bytes_to_mib(required):.2f} MiB more)")
        self.bytes_allocated += required


class DeviceAllocator:
    """Advanced allocator with memory balancing and fallback"""
    def __init__(self, devices: List[Device]) -> None:
        self.devices = sorted(devices, key=lambda d: d.name, reverse=True)
        self.tensor_map = {}

    def allocate(self, required: float, tensor_name: Optional[str] = None) -> Device:
        """Allocate memory to first suitable device."""
        for device in self.devices:
            if device.can_allocate(required):
                device.allocate(required)
                if tensor_name:
                    self.tensor_map[tensor_name] = device.name
                return device
        raise MemoryError(f"Cannot allocate {bytes_to_mib(required):.2f} MiB on any device")

    def allocate_on_device(
        self, 
        device_name: str, 
        required: float, 
        tensor_name: Optional[str] = None
    ) -> Device:
        """Allocate memory to specific device."""
        device = next((d for d in self.devices if d.name == device_name), None)
        if not device:
            raise ValueError(f"Device {device_name} not found")
        if not device.can_allocate(required):
            raise MemoryError(f"Insufficient memory on {device_name}")
        device.allocate(required)
        if tensor_name:
            self.tensor_map[tensor_name] = device.name
        return device


def model_fits_in_memory(
    gguf: Union[GGUFReader, GGUFMultiPartReader],
    gpus: List[Dict[str, Any]],
    ram_bytes: int,
    context_length: int,
    context_quantization_size: int,
    gpu_memory_spec: Optional[List[Union[float, int]]] = None
) -> bool:
    """
    Check if model fits in combined GPU and RAM memory with new spec.
    
    Args:
        gguf: GGUF model instance
        gpus: List of GPU dictionaries
        ram_bytes: Available system RAM in bytes
        context_length: Maximum context length
        context_quantization_size: Quantization size in bits
        gpu_memory_spec: Optional GPU memory specifications
    
    Returns:
        Boolean indicating if model fits in available memory
    """
    total_gpu_memory = 0
    
    if gpu_memory_spec is None:
        for gpu in gpus:
            total_gpu_memory += gpu["memory_total_bytes"] * 0.9
    else:
        for i, spec in enumerate(gpu_memory_spec):
            if i >= len(gpus):
                break
            gpu = gpus[i]
            if isinstance(spec, float) and spec <= 1:
                total_gpu_memory += gpu["memory_total_bytes"] * spec
            elif isinstance(spec, float) and spec > 1:
                total_gpu_memory += spec * 1024**3
    
    tensor_size = calculate_tensors_size_bytes(gguf)
    kv_cache_size = calculate_kv_cache_size_bytes(
        gguf, context_length, context_quantization_size
    )
    total_required = tensor_size + kv_cache_size
    total_available = total_gpu_memory + ram_bytes
    
    Log.log("info", f"Total required memory: {bytes_to_mib(total_required):.2f} MiB")
    Log.log("info", f"Available memory: GPU={bytes_to_mib(total_gpu_memory):.2f} MiB, "
                    f"RAM={bytes_to_mib(ram_bytes):.2f} MiB, Total={bytes_to_mib(total_available):.2f} MiB")
    
    return total_required <= total_available


def is_token_embedding(tensor_name: str) -> bool:
    """Check if tensor is a token embedding."""
    name = tensor_name.lower()
    if "position" in name or "pos" in name:
        return False
        
    patterns = [
        "token_embd", "token_embed", "tok_embeddings", "embeddings.token",
        "token.embedding", "word_embeddings", "wte", "token_emb", "tokenemb",
        "tok_emb", "tok_embeddings"
    ]
    
    return any(pattern in name for pattern in patterns) or \
        ("token" in name and ("emb" in name or "embed" in name))


def group_tensors_by_layer(gguf: Union[GGUFReader, GGUFMultiPartReader]) -> Dict[int, Dict[str, List]]:
    """
    Group tensors by layer and category for optimized allocation.
    
    Args:
        gguf: GGUF model instance
    
    Returns:
        Dictionary mapping layer IDs to tensor groups
    """
    layer_tensors = defaultdict(lambda: defaultdict(list))
    layer_patterns = [
        r"blk\.(\d+)\.",
        r"layers\.(\d+)\.",
        r"model\.layers\.(\d+)\.",
        r"h\.(\d+)\."
    ]
    
    for tensor in gguf.tensors:
        layer_id = None
        name = tensor.name.lower()
        
        for pattern in layer_patterns:
            match = re.search(pattern, name)
            if match:
                try:
                    layer_id = int(match.group(1))
                    break
                except (ValueError, IndexError):
                    continue
                    
        if layer_id is None:
            continue
            
        if "attention" in name or "attn" in name:
            layer_tensors[layer_id]['attention'].append(tensor)
        elif "ffn" in name or "feed_forward" in name:
            if "expert" in name or "moe" in name:
                layer_tensors[layer_id]['experts'].append(tensor)
            else:
                layer_tensors[layer_id]['ffn'].append(tensor)
        elif "gate" in name:
            layer_tensors[layer_id]['gate'].append(tensor)
        elif "norm" in name or "ln" in name:
            layer_tensors[layer_id]['norm'].append(tensor)
        else:
            layer_tensors[layer_id]['other'].append(tensor)
            
    return layer_tensors


def _allocate_tensors(
    allocator: DeviceAllocator,
    tensors: List[Any],
    seen_tensors: set,
    category_bytes: float,
    category_name: str
) -> float:
    """Helper to allocate a group of tensors."""
    total_bytes = category_bytes
    for tensor in tensors:
        if tensor.name in seen_tensors:
            continue
        try:
            size = calculate_tensor_size_bytes(tensor)
            allocator.allocate(size, tensor.name)
            seen_tensors.add(tensor.name)
            total_bytes += size
        except MemoryError:
            try:
                allocator.allocate_on_device("CPU", size, tensor.name)
                seen_tensors.add(tensor.name)
                total_bytes += size
                Log.warn(f"{category_name} tensor offloaded to CPU: {tensor.name}")
            except MemoryError:
                Log.error(f"Failed to allocate {category_name} tensor: {tensor.name}")
    return total_bytes


def optimize(
    gguf: Union[GGUFReader, GGUFMultiPartReader],
    gpus: List[Dict[str, Any]],
    ram_bytes: int,
    context_length: int,
    context_quantization_size: int,
    check: bool = True,
    granular_gpu_percentage: Optional[List[Union[float, int]]] = None,
) -> str:
    """Main optimization function to generate allocation strategy."""
    metadata = extract_metadata(gguf)
    num_layers = int(metadata["num_layers"])
    
    if not gpus:
        Log.error(f"No CUDA devices found. Unable to offload.")
        sys.exit(1)
    
    cpu = Device("CPU", ram_bytes, 0, 1.0)
    if not check:
        cpu.set_unsafe()
        
    gpu_devices = []
    for i, gpu in sorted(enumerate(gpus), key=lambda d: d):
        if granular_gpu_percentage:
            spec = granular_gpu_percentage[i]
            if isinstance(spec, float) and spec < 1:
                percentage = spec
            elif isinstance(spec, float) and spec >= 1:
                if spec == 0:
                    continue
                percentage = spec * 1024**3 / gpu["memory_total_bytes"]
            else:
                percentage = 0.9
        else:
            percentage = 0.9
            
        device = Device(
            f"CUDA{gpu['cuda_id']}",
            gpu["memory_total_bytes"],
            gpu["memory_total_bytes"],
            percentage
        )
        gpu_devices.append(device)
        
    allocator = DeviceAllocator([cpu] + gpu_devices)
    seen_tensors = set()
    
    if check and not model_fits_in_memory(
        gguf, 
        gpus, 
        ram_bytes, 
        context_length, 
        context_quantization_size, 
        granular_gpu_percentage
    ):
        raise MemoryError(
            "Model does not fit in combined GPU and RAM memory. "
            "Try reducing context length or quantization size."
        )
        
    kv_cache_per_layer = calculate_kv_cache_size_bytes(
        gguf, context_length, context_quantization_size
    ) / num_layers
    Log.log("info", f"KV cache per layer: {bytes_to_mib(kv_cache_per_layer):.2f} MiB")
    
    # Allocate embeddings on CPU
    for tensor in [t for t in gguf.tensors if is_token_embedding(t.name)]:
        try:
            size = calculate_tensor_size_bytes(tensor)
            allocator.allocate_on_device("CPU", size, tensor.name)
            seen_tensors.add(tensor.name)
            Log.log("info", f"Embedding tensor {tensor.name} allocated on CPU: {bytes_to_mib(size):.2f} MiB")
        except MemoryError as e:
            Log.warn(f"Failed to allocate embedding tensor: {str(e)}")
            
    is_moe = any("expert" in t.name.lower() for t in gguf.tensors)
    Log.log("info", f"Detected MOE model: {is_moe}")
    layer_tensors = group_tensors_by_layer(gguf)
    
    # Track memory usage
    total_attention_bytes = 0
    total_norm_bytes = 0
    total_ffn_bytes = 0
    total_gate_bytes = 0
    total_expert_bytes = 0
    
    # Process layers
    for layer_id in range(num_layers):
        layer_data = layer_tensors.get(layer_id, {})
        
        # Attention weights
        attention_tensors = layer_data.get('attention', [])
        total_attention_bytes = _allocate_tensors(
            allocator, attention_tensors, seen_tensors, total_attention_bytes, "Attention"
        )
        
        # KV cache
        try:
            allocator.allocate(kv_cache_per_layer)
            total_attention_bytes += kv_cache_per_layer
        except MemoryError as e:
            Log.error(f"KV cache allocation failed for layer {layer_id}: {str(e)}")
            
        # Normalization weights
        norm_tensors = layer_data.get('norm', [])
        total_norm_bytes = _allocate_tensors(
            allocator, norm_tensors, seen_tensors, total_norm_bytes, "Norm"
        )
        
        # FFN non-expert weights
        ffn_tensors = layer_data.get('ffn', [])
        total_ffn_bytes = _allocate_tensors(
            allocator, ffn_tensors, seen_tensors, total_ffn_bytes, "FFN"
        )
        
        # Gate weights
        gate_tensors = layer_data.get('gate', [])
        total_gate_bytes = _allocate_tensors(
            allocator, gate_tensors, seen_tensors, total_gate_bytes, "Gate"
        )
        
        # Expert weights (MoE)
        expert_tensors = layer_data.get('experts', [])
        if is_moe and expert_tensors:
            try:
                for tensor in expert_tensors:
                    if tensor.name in seen_tensors:
                        continue
                    size = calculate_tensor_size_bytes(tensor)
                    allocator.allocate(size, tensor.name)
                    seen_tensors.add(tensor.name)
                    total_expert_bytes += size
            except MemoryError:
                for tensor in expert_tensors:
                    if tensor.name in seen_tensors:
                        continue
                    size = calculate_tensor_size_bytes(tensor)
                    try:
                        allocator.allocate_on_device("CPU", size, tensor.name)
                        seen_tensors.add(tensor.name)
                        total_expert_bytes += size
                        Log.log("info", f"Offloaded expert {tensor.name} to CPU")
                    except MemoryError:
                        Log.error(f"Failed to offload expert: {tensor.name}")
                        
    # Log allocation stats
    Log.log("info", f"Total attention bytes: {bytes_to_mib(total_attention_bytes):.2f} MiB")
    Log.log("info", f"Total norm bytes: {bytes_to_mib(total_norm_bytes):.2f} MiB")
    Log.log("info", f"Total FFN bytes: {bytes_to_mib(total_ffn_bytes):.2f} MiB")
    Log.log("info", f"Total gate bytes: {bytes_to_mib(total_gate_bytes):.2f} MiB")
    Log.log("info", f"Total expert bytes: {bytes_to_mib(total_expert_bytes):.2f} MiB")
    
    # Remaining tensors
    total_other_bytes = 0
    for tensor in gguf.tensors:
        if tensor.name in seen_tensors:
            continue
        try:
            size = calculate_tensor_size_bytes(tensor)
            allocator.allocate(size, tensor.name)
            seen_tensors.add(tensor.name)
            total_other_bytes += size
        except MemoryError as e:
            Log.warn(f"Failed to allocate tensor {tensor.name}: {str(e)}")
    Log.log("info", f"Total other bytes: {bytes_to_mib(total_other_bytes):.2f} MiB")
    
    # Calculate GPU utilization ratios
    cpu_static_size = 0
    gpu_static_sizes = []
    gpu_devices = [d for d in sorted(allocator.devices, key=lambda d: d.name) if d.name.startswith("CUDA")]
    
    for tensor in gguf.tensors:
        if tensor.name in seen_tensors and allocator.tensor_map.get(tensor.name) == "CPU":
            cpu_static_size += calculate_tensor_size_bytes(tensor)
    
    for device in gpu_devices:
        static_size = 0
        for tensor in gguf.tensors:
            if tensor.name in seen_tensors and allocator.tensor_map.get(tensor.name) == device.name:
                static_size += calculate_tensor_size_bytes(tensor)
        gpu_static_sizes.append(static_size)
        
    if gpu_static_sizes:
        gpu_static_sizes[-1] += cpu_static_size
        
    total_model_size = calculate_tensors_size_bytes(gguf)
    gpu_ratios = [static_size / total_model_size for static_size in gpu_static_sizes]
    
    for i, device in enumerate(gpu_devices):
        Log.log("info", f"Ratio: {device.name} - {gpu_ratios[i]:.4f}")
        
    ratio_str = ",".join(f"{r:.4f}" for r in gpu_ratios)
    Log.log("info", f"GPU model ratios: {ratio_str} (sum={sum(gpu_ratios):.4f})")
    
    cpu_tensor_map = {
        name: device 
        for name, device in allocator.tensor_map.items() 
        if device.startswith("CPU") and "exps" in name
    }
    
    if not cpu_tensor_map:
        Log.log("info", "The model fits completely into the VRAM. There is no need to override experts' tensors to CPU. Only GPU splitting fractions arg is sufficient or use automatic splitting by GPU (without -ts arg).")
        
    # Generate command
    compressed_rules = compress_tensor_overrides(cpu_tensor_map)
    command = "-ngl 999 "
    if len(gpu_devices) > 1:
        command += f"-ts {ratio_str} "
    for pattern, device_name in compressed_rules:
        command += f'-ot "{pattern}={device_name}" '
        
    return command.strip()


def compress_tensor_overrides(tensor_map: Dict[str, str]) -> List[Tuple[str, str]]:
    """Compress tensor overrides into regex patterns."""
    base_groups = defaultdict(lambda: defaultdict(list))
    no_number_tensors = []
    
    for tensor, device in tensor_map.items():
        parts = re.split(r'(\d+)', tensor, 1)
        if len(parts) < 3:
            no_number_tensors.append((tensor, device))
            continue
        prefix, num_str, rest = parts
        try:
            num = int(num_str)
            base_groups[(prefix, rest)][device].append(num)
        except ValueError:
            no_number_tensors.append((tensor, device))
            
    compressed_rules = []
    
    # Tensors without numbers
    for tensor, device in no_number_tensors:
        escaped = re.escape(tensor)
        compressed_rules.append((escaped, device))
        
    # Grouped tensors with numeric ranges
    for (prefix, rest), device_numbers in base_groups.items():
        all_ranges = []
        for device, numbers in device_numbers.items():
            numbers.sort()
            ranges = []
            if not numbers:
                continue
            start = end = numbers[0]
            for num in numbers[1:]:
                if num == end + 1:
                    end = num
                else:
                    ranges.append((start, end))
                    start = end = num
            ranges.append((start, end))
            all_ranges.extend((s, e, device) for s, e in ranges)
            
        all_ranges.sort(key=lambda x: x[0])
        
        for start, end, device in all_ranges:
            if start == end:
                num_pattern = str(start)
            else:
                num_pattern = "|".join(str(i) for i in range(start, end + 1))
            pattern = re.escape(prefix) + "(" + num_pattern + ")" + re.escape(rest)
            compressed_rules.append((pattern, device))
            
    compressed_rules.sort(key=lambda x: (x[0], x[1]))
    return compressed_rules


def _parse_gpu_memory_spec(spec_str: str, gpu_count: int) -> List[Union[float, int]]:
    """Parse GPU memory specification string."""
    values = [v.strip() for v in spec_str.split(",")]
    parsed_values = []
    
    for v in values:
        try:
            num = float(v)
        except ValueError:
            raise ValueError(f"'{v}' is not a valid number")
            
        if num < 0:
            raise ValueError("Values must be non-negative")
        elif num < 1:
            parsed_values.append(num)
        else:
            parsed_values.append(num)
            
    if len(parsed_values) == 1:
        return parsed_values * gpu_count
    elif len(parsed_values) != gpu_count:
        raise ValueError("GPU memory values count must match GPU count")
    return parsed_values


def main() -> None:
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Teno-chan will help to offload expert tensors for faster inference.")
    
    parser.add_argument("-g", "--gguf-url", required=True, help="URL or path of GGUF file")
    parser.add_argument("-c", "--context-length", type=int, required=True, help="Context length for optimization")
    parser.add_argument("-q", "--context-quantization-size", type=int, default=16,
                        choices=[4, 8, 16], help="Context quantization size")
    parser.add_argument("--no-check", action="store_true", help="Skip system resource checks")
    parser.add_argument("--gpu-memory", type=str, help="Comma-separated GPU memory limits")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--quiet", action="store_true", help="Only output command string")
    
    args = parser.parse_args()
    
    Log.verbose = args.verbose
    Log.no_log = args.quiet
    
    gpus = get_nvidia_gpus()
    ram_bytes = get_ram_bytes()
    
    granular_percentages = None
    if args.gpu_memory:
        try:
            granular_percentages = _parse_gpu_memory_spec(args.gpu_memory, len(gpus))
        except ValueError as e:
            Log.error(f"Invalid --gpu-memory format: {str(e)}")
            sys.exit(1)
            
    try:
        Log.log("info", f"Loading GGUF model from: {args.gguf_url}")
        gguf = download_gguf(args.gguf_url)
        Log.log("info", f"Model loaded with {len(gguf.tensors)} tensors")
    except Exception as e:
        Log.error(f"Failed to load GGUF: {str(e)}")
        sys.exit(1)
        
    try:
        command = optimize(
            gguf=gguf,
            gpus=gpus,
            ram_bytes=ram_bytes,
            context_length=args.context_length,
            context_quantization_size=args.context_quantization_size,
            check=not args.no_check,
            granular_gpu_percentage=granular_percentages,
        )
        
        if not args.quiet:
            print("\nOptimization complete. Use this command with llama.cpp:\n")
            print(command)
            print("\nIf the model does not load with the calculated parameters, try reducing the VRAM usage factor (--gpu-memory arg) - e.g. from 0.9 to 0.85.\n")          
        else:
            print(command)
            
    except Exception as e:
        Log.error(f"Optimization failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()