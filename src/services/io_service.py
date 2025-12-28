import json
import os
from pathlib import Path

class IOService:
    @staticmethod
    def load_json(file_path: Path):
        """Đọc file JSON từ đường dẫn cho trước."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Không tìm thấy file tại: {file_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Lỗi định dạng JSON trong file: {file_path}")

    @staticmethod
    def save_json(data, file_path: Path):
        """Lưu dữ liệu dictionary/list xuống file JSON."""
        # Đảm bảo thư mục cha tồn tại
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"[IO] Đã lưu file thành công: {file_path}")