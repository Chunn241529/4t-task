import subprocess
import sys
import os
import platform
import argparse

# --- Cấu hình ---
VENV_DIR = ".venv"
PYTHON_MIN_VERSION = (3, 8)

def run_command(command, check=True):
    """Thực thi một lệnh shell và xử lý lỗi nếu có."""
    try:
        print(f"Đang chạy lệnh: {' '.join(command)}")
        subprocess.run(command, check=check, shell=False)
    except subprocess.CalledProcessError as e:
        print(f"LỖI: Lệnh {' '.join(command)} thất bại với mã lỗi {e.returncode}")
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy lệnh '{command[0]}'. Hãy đảm bảo nó đã được cài đặt và có trong PATH.")
        sys.exit(1)

def get_python_executable(venv_path):
    """Lấy đường dẫn đến file thực thi python trong venv cho HĐH hiện tại."""
    if platform.system() == "Windows":
        return os.path.join(venv_path, "Scripts", "python.exe")
    else: # Linux, macOS, etc.
        return os.path.join(venv_path, "bin", "python")

def main():
    """Hàm chính để thiết lập môi trường và cài đặt dependencies."""
    parser = argparse.ArgumentParser(
        description="Script cài đặt môi trường cho dự án.",
        formatter_class=argparse.RawTextHelpFormatter # Để hiển thị help message đẹp hơn
    )
    parser.add_argument(
        '--pytorch', 
        default='cuda121',  # <-- THAY ĐỔI QUAN TRỌNG: Mặc định là CUDA 12.1
        choices=['auto', 'cpu', 'cuda118', 'cuda121'],
        help="Chọn phiên bản PyTorch để cài đặt:\n"
             "  - auto:    Tự động tìm phiên bản tốt nhất (có thể không đáng tin cậy).\n"
             "  - cpu:     Chỉ cài đặt phiên bản cho CPU.\n"
             "  - cuda118: Cài đặt cho NVIDIA GPU với CUDA 11.8.\n"
             "  - cuda121: Cài đặt cho NVIDIA GPU với CUDA 12.1 (khuyên dùng cho driver mới)."
    )
    args = parser.parse_args()

    # 1. Kiểm tra phiên bản Python
    if sys.version_info < PYTHON_MIN_VERSION:
        print(f"Yêu cầu Python {PYTHON_MIN_VERSION[0]}.{PYTHON_MIN_VERSION[1]} trở lên.")
        sys.exit(1)
    
    print("Bắt đầu quá trình cài đặt môi trường...")

    # 2. Tạo/Kiểm tra virtual environment
    if not os.path.exists(VENV_DIR):
        print(f"Đang tạo virtual environment tại '{VENV_DIR}'...")
        run_command([sys.executable, "-m", "venv", VENV_DIR])
    
    python_in_venv = get_python_executable(VENV_DIR)
    
    if not os.path.exists(python_in_venv):
        print(f"LỖI: Không tìm thấy file thực thi Python tại '{python_in_venv}'.")
        sys.exit(1)

    print(f"Sử dụng Python interpreter từ: {python_in_venv}")

    # 3. Cập nhật pip
    print("\nĐang cập nhật pip...")
    run_command([python_in_venv, "-m", "pip", "install", "--upgrade", "pip"])

    # 4. Cài đặt các thư viện từ requirements.txt
    requirements_file = "requirements.txt"
    if os.path.exists(requirements_file):
        print(f"\nĐang cài đặt các thư viện từ {requirements_file}...")
        run_command([python_in_venv, "-m", "pip", "install", "-r", requirements_file])

    # 5. Cài đặt PyTorch theo lựa chọn
    print(f"\nĐang cài đặt PyTorch (phiên bản đã chọn: {args.pytorch})...")
    
    base_command = [python_in_venv, "-m", "pip", "install", "torch", "torchvision", "torchaudio"]
    
    if args.pytorch == 'cuda121':
        install_command = base_command + ["--index-url", "https://download.pytorch.org/whl/cu121"]
    elif args.pytorch == 'cuda118':
        install_command = base_command + ["--index-url", "https://download.pytorch.org/whl/cu118"]
    elif args.pytorch == 'cpu':
        # Một số hệ thống có thể cần chỉ định rõ nguồn CPU
        install_command = base_command + ["--index-url", "https://download.pytorch.org/whl/cpu"]
    else: # auto
        install_command = base_command

    run_command(install_command)
    
    print("\n✅ Cài đặt hoàn tất!")
    print(f"Để kích hoạt môi trường ảo, hãy chạy lệnh sau:")
    if platform.system() == "Windows":
        print(f"   .\\{VENV_DIR}\\Scripts\\activate")
    else:
        print(f"   source ./{VENV_DIR}/bin/activate")
    print("Sau đó, bạn có thể chạy file chính của mình.")

if __name__ == "__main__":
    main()