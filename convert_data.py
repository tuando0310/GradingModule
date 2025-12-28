import sys
from pathlib import Path

# Setup đường dẫn module
sys.path.append(str(Path(__file__).parent / 'src'))

from services.io_service import IOService
from services.preprocess_service import PreprocessService

# CẤU HÌNH
BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / 'data' / 'keypoints3d' # Folder chứa 313 file con
OUTPUT_FILE = BASE_DIR / 'data' / 'merged_camera_data.json' # File kết quả
TOTAL_FILES = 313 # Từ 000000 đến 000312

def main():
    print("--- BẮT ĐẦU QUÁ TRÌNH TIỀN XỬ LÝ DỮ LIỆU ---")
    
    # 1. Khởi tạo Service
    preprocessor = PreprocessService()
    io = IOService() # Tái sử dụng IO Service cũ để lưu file

    # 2. Thực hiện gộp
    final_data = preprocessor.merge_keypoints_files(INPUT_DIR, TOTAL_FILES)

    # 3. Lưu kết quả
    if final_data:
        io.save_json(final_data, OUTPUT_FILE)
        print(f"--- HOÀN TẤT: Dữ liệu đã lưu tại {OUTPUT_FILE} ---")
    else:
        print("--- THẤT BẠI: Không có dữ liệu nào được xử lý ---")

if __name__ == "__main__":
    main()