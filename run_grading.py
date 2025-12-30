import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / 'src'))

from services.io_service import IOService
from services.grading_service import GradingService

# --- CẤU HÌNH ---
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
OUTPUT_DIR = BASE_DIR / 'output'

USER_FILE = DATA_DIR / 'aligned_grading_data.json'
MODEL_FILE = DATA_DIR / 'model_data.json'
MAPPING_FILE = OUTPUT_DIR / 'dtw_mapping_result.json'

# --- CẤU HÌNH NGƯỠNG (QUAN TRỌNG) ---
GRADING_CONFIG = {
    # Ngưỡng tổng lỗi 
    "total_error_threshold": 1.5, 

    # Ngưỡng mặc định cho 1 khớp bất kỳ
    "default_joint_threshold": 0.15, 

    # Ngưỡng riêng cho từng khớp (nếu cần khắt khe hoặc lỏng lẻo hơn)
    "joint_thresholds": {
    }
}

def main():
    
    io = IOService()
    grader = GradingService()

    try:
        # 1. Load Data
        user_data = io.load_json(USER_FILE)
        model_data = io.load_json(MODEL_FILE)
        mapping_obj = io.load_json(MAPPING_FILE)
        
        user_seq = [item['data'] for item in user_data]
        model_seq = [item['data'] for item in model_data]
        
        if isinstance(mapping_obj, dict): mapping_arr = mapping_obj['mapping']
        else: mapping_arr = mapping_obj

        # 2. Tính toán chi tiết
        print(" Đang phân tích lỗi từng khớp...")
        detailed_results = grader.evaluate_performance_detailed(
            user_seq, model_seq, mapping_arr, GRADING_CONFIG
        )

        # 3. Tổng hợp báo cáo
        pass_threshold = GRADING_CONFIG['total_error_threshold']
        passing_frames = sum(1 for r in detailed_results if r['total_error'] < pass_threshold)
        total_frames = len(detailed_results)

        print(f"\n[KẾT QUẢ] Đạt: {passing_frames}/{total_frames} ({passing_frames/total_frames*100:.2f}%)")

        # 4. Lưu báo cáo
        final_report = {
            "config": GRADING_CONFIG,
            "summary": {
                "total_frames": total_frames,
                "passed": passing_frames,
                "score": round(passing_frames/total_frames * 100, 2)
            },
            "frames_detail": []
        }

        # Format lại kết quả cho gọn gàng khi lưu file
        for item in detailed_results:
            frame_info = {
                "frame": item['frame_index'],
                "map_to": item['mapped_model_frame'],
                "error": round(item['total_error'], 4),
                "status": "PASS" if item['total_error'] < pass_threshold else "FAIL"
            }
            
            # CHỈ LƯU TRƯỜNG "bad_joints" NẾU CÓ LỖI (status FAIL hoặc có khớp lệch)
            if item['bad_joints']:
                frame_info['bad_joints'] = item['bad_joints']
            
            final_report["frames_detail"].append(frame_info)

        out_path = OUTPUT_DIR / 'detailed_grading_report.json'
        io.save_json(final_report, out_path)

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()