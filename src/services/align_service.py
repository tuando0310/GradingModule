import numpy as np

class AlignmentService:
    def calculate_umeyama_params(self, source_points, target_points):
        """
        Chỉ tính toán và trả về tham số biến đổi (Scale, R, t) dựa trên 2 tập điểm.
        KHÔNG trả về điểm đã biến đổi.
        """
        P = np.array(source_points)
        Q = np.array(target_points)
        n, m = P.shape

        # 1. Centroid
        centroid_P = np.mean(P, axis=0)
        centroid_Q = np.mean(Q, axis=0)

        # 2. Centering
        P_centered = P - centroid_P
        Q_centered = Q - centroid_Q

        # 3. Covariance & SVD
        H = P_centered.T @ Q_centered / n
        U, S, Vt = np.linalg.svd(H)

        # 4. Rotation Matrix
        D = np.eye(m)
        if np.linalg.det(U) * np.linalg.det(Vt.T) < 0:
            D[m-1, m-1] = -1
        R = U @ D @ Vt

        # 5. Scale
        var_P = np.var(P_centered, axis=0).sum()
        c = 1/var_P * np.trace(np.diag(S) @ D)

        # 6. Translation
        t = centroid_Q - c * (centroid_P @ R)

        return c, R, t

    def apply_transform(self, points, scale, R, t):
        """
        Áp dụng tham số biến đổi (đã tính trước) vào một tập điểm bất kỳ.
        Công thức: P_new = (scale * (P @ R)) + t
        """
        P = np.array(points)
        P_transformed = (scale * (P @ R)) + t
        return P_transformed.tolist()