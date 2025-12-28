import json
import os
from pathlib import Path

class PreprocessService:
    def __init__(self):
        # Mảng ánh xạ index từ bộ 25 khớp (Input) sang bộ 13 khớp (Output)
        # Thứ tự này tuân theo file index13JointsMap.json
        self.MAPPING_INDICES = [8, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14]

    def merge_keypoints_files(self, input_dir: Path, total_files: int):
        """
        Đọc hàng loạt file JSON, lọc khớp xương và gộp thành 1 list.
        """
        merged_data = []
        
        print(f"[PROCESS] Bắt đầu xử lý {total_files} file từ: {input_dir}")

        for i in range(total_files):
            # Format tên file: 0 -> 000000.json, 10 -> 000010.json
            file_name = f"{i:06d}.json"
            file_path = input_dir / file_name

            if not file_path.exists():
                print(f"[WARNING] File không tồn tại: {file_name}. Bỏ qua.")
                continue

            try:
                # 1. Đọc file
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_content = json.load(f)
                    
                # Cấu trúc file raw là list chứa 1 dict: [{"keypoints3d": [...]}]
                if not raw_content or 'keypoints3d' not in raw_content[0]:
                    print(f"[WARNING] Dữ liệu sai định dạng: {file_name}")
                    continue
                
                source_points = raw_content[0]['keypoints3d']

                # 2. Transform (Lọc và sắp xếp lại điểm)
                refined_points = []
                for idx in self.MAPPING_INDICES:
                    if idx < len(source_points):
                        refined_points.append(source_points[idx])
                    else:
                        # Fallback nếu dữ liệu thiếu điểm (chèn điểm 0,0,0)
                        refined_points.append([0.0, 0.0, 0.0]) 

                # 3. Tạo cấu trúc đích
                frame_obj = {
                    "frame": i,  # Sử dụng biến đếm vòng lặp làm frame ID
                    "data": refined_points
                }
                
                merged_data.append(frame_obj)

            except Exception as e:
                print(f"[ERROR] Lỗi khi xử lý file {file_name}: {e}")

        print(f"[SUCCESS] Đã gộp thành công {len(merged_data)} frame.")
        return merged_data