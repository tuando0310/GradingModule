import numpy as np

class DtwService:
    def _calculate_frame_distance(self, frame_a, frame_b):
        """
        Tính tổng khoảng cách Euclidean của các khớp giữa 2 frame.
        Input: List hoặc Numpy array (Nx3)
        """
        fa = np.array(frame_a)
        fb = np.array(frame_b)
        
        # Đảm bảo 2 frame có cùng số lượng khớp
        min_len = min(len(fa), len(fb))
        diff = fa[:min_len] - fb[:min_len]
        
        # Tính khoảng cách Euclidean cho từng khớp rồi tổng hợp lại
        dist_sq = np.sum(diff**2, axis=1)
        dists = np.sqrt(dist_sq)
        
        return np.sum(dists)

    def run_dtw_alignment(self, user_sequence, model_sequence):
        N = len(user_sequence)
        M = len(model_sequence)
        
        print(f"[DTW] Bắt đầu tính toán ma trận chi phí ({N}x{M})...")

        # 1. Tính Ma trận Khoảng cách (Distance Matrix)
        dist_mat = np.zeros((N, M))
        for i in range(N):
            for j in range(M):
                dist_mat[i, j] = self._calculate_frame_distance(user_sequence[i], model_sequence[j])
            
        # 2. Tính Ma trận Chi phí Tích lũy (Accumulated Cost Matrix)
        acc_cost = np.full((N, M), np.inf)
        acc_cost[0, 0] = dist_mat[0, 0]
        
        # Khởi tạo dòng đầu và cột đầu
        for i in range(1, N):
            acc_cost[i, 0] = acc_cost[i-1, 0] + dist_mat[i, 0]
        for j in range(1, M):
            acc_cost[0, j] = acc_cost[0, j-1] + dist_mat[0, j]
            
        # Quy hoạch động (Dynamic Programming)
        for i in range(1, N):
            for j in range(1, M):
                # Chọn đường đi có chi phí nhỏ nhất từ 3 ô lân cận
                acc_cost[i, j] = dist_mat[i, j] + min(
                    acc_cost[i-1, j],   # Insertion
                    acc_cost[i, j-1],   # Deletion
                    acc_cost[i-1, j-1]  # Match
                )
        
        # 3. Truy vết (Backtracking) để tìm đường đi tối ưu
        print("[DTW] Đang truy vết đường đi tối ưu...")
        path = []
        i, j = N-1, M-1
        path.append((i, j))
        
        while i > 0 or j > 0:
            if i == 0:
                j -= 1
            elif j == 0:
                i -= 1
            else:
                candidates = [
                    acc_cost[i-1, j],
                    acc_cost[i, j-1],
                    acc_cost[i-1, j-1]
                ]
                idx = np.argmin(candidates)
                if idx == 0:      i -= 1
                elif idx == 1:    j -= 1
                else:             i -= 1; j -= 1
            path.append((i, j))
        
        path.reverse() # Đảo ngược để có thứ tự từ 0 -> N
        
        # 4. Tạo mảng Mapping (Result Array)
        # Với mỗi frame i của User, tìm frame j của Model tốt nhất trên đường đi
        mapping = np.zeros(N, dtype=int)
        
        # Gom nhóm: Một i có thể khớp với nhiều j, ta chọn j có khoảng cách nhỏ nhất
        i_to_js = {}
        for (r, c) in path:
            if r not in i_to_js: i_to_js[r] = []
            i_to_js[r].append(c)
            
        for i in range(N):
            if i in i_to_js:
                candidates = i_to_js[i]
                # Tìm j trong các ứng viên sao cho khoảng cách là nhỏ nhất (Best Fit)
                best_j = min(candidates, key=lambda idx: dist_mat[i, idx])
                mapping[i] = best_j
            else:
                mapping[i] = mapping[i-1] if i > 0 else 0

        return mapping.tolist()