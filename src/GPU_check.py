import torch

if torch.cuda.is_available():
    print("✅ GPU có sẵn")
    print(f"Tên GPU: {torch.cuda.get_device_name(0)}")
    print(f"Số lượng GPU: {torch.cuda.device_count()}")
    
    # Kiểm tra bộ nhớ đã sử dụng
    gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**2
    gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**2
    print(f"Bộ nhớ đang dùng: {gpu_memory_allocated:.2f} MB")
    print(f"Bộ nhớ đã cấp phát: {gpu_memory_reserved:.2f} MB")
else:
    print("❌ Không tìm thấy GPU")
