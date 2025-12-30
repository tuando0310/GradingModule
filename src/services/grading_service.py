import numpy as np

class GradingService:
    def __init__(self):
        # Mapping index sang tên khớp (Dựa trên cấu trúc 13 khớp đã thống nhất)
        self.INDEX_MAP = {
            0: "hip", 1: "right_shoulder", 2: "right_elbow", 3: "right_wrist",
            4: "left_shoulder", 5: "left_elbow", 6: "left_wrist", 7: "right_hip",
            8: "right_knee", 9: "right_ankle", 10: "left_hip", 11: "left_knee", 12: "left_ankle"
        }

    def _calculate_frame_detail(self, user_frame, model_frame, joint_thresholds, default_threshold):
        """
        Tính toán chi tiết sai số cho từng khớp trong 1 frame.
        """
        p_user = np.array(user_frame)
        p_model = np.array(model_frame)
        
        # 1. Tính khoảng cách Euclidean cho từng cặp khớp
        diff = p_user - p_model
        dists = np.sqrt(np.sum(diff**2, axis=1)) # Mảng 13 phần tử
        
        total_error = np.sum(dists)
        bad_joints = []

        # 2. Kiểm tra từng khớp xem có vượt ngưỡng không
        for i, dist in enumerate(dists):
            # Lấy ngưỡng riêng cho khớp i, nếu không có thì dùng ngưỡng mặc định
            threshold = joint_thresholds.get(i, default_threshold)
            
            if dist > threshold:
                joint_name = self.INDEX_MAP.get(i, f"unknown_{i}")
                bad_joints.append(joint_name)
                
        return total_error, bad_joints

    def evaluate_performance_detailed(self, user_sequence, model_sequence, mapping, config):
        """
        Phiên bản nâng cấp: Trả về cả tổng lỗi và danh sách khớp sai.
        """
        results = []
        
        joint_thresholds = config.get('joint_thresholds', {})
        default_threshold = config.get('default_joint_threshold', 0.1)
        
        n_frames = len(user_sequence)
        max_model_idx = len(model_sequence) - 1 # Chỉ số lớn nhất cho phép
        
        for i in range(n_frames):
            user_frame = user_sequence[i]
            
            # --- BẮT ĐẦU SỬA LỖI TẠI ĐÂY ---
            # Lấy index từ mapping
            raw_model_idx = mapping[i]
            
            # Đảm bảo index không vượt quá giới hạn của mảng model
            if raw_model_idx > max_model_idx:
                model_idx = max_model_idx
            elif raw_model_idx < 0:
                model_idx = 0
            else:
                model_idx = raw_model_idx
            # --- KẾT THÚC SỬA LỖI ---

            model_frame = model_sequence[model_idx]
            
            # Gọi hàm tính chi tiết
            total_err, bad_joints = self._calculate_frame_detail(
                user_frame, model_frame, joint_thresholds, default_threshold
            )
            
            results.append({
                "frame_index": i,
                "mapped_model_frame": model_idx,
                "total_error": total_err,
                "bad_joints": bad_joints
            })
            
        return results