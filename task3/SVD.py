from struct import pack, calcsize, unpack

import PIL
import numpy as np


class SVD:

    def __init__(self, compression_ratio):
        self.u = None
        self.s = None
        self.v = None
        self.ratio = compression_ratio

    def compress(self, img: PIL.Image):
        u, s, v = np.linalg.svd(img, full_matrices=False)
        num_singular_values = int(len(s) * self.ratio)

        self.u = u[:, :num_singular_values]
        self.s = s[:num_singular_values]
        self.v = v[:num_singular_values, :]

        return self

    def decompress(self):
        return np.dot(np.dot(self.u, np.diag(self.s)), self.v).astype(np.uint8)

    def to_bytes(self):
        if self.u is None or self.s is None or self.v is None:
            raise ValueError('Cannot convert to bytes an empty values')

        u_shape_flat = self.u.shape[0] * self.u.shape[1]
        s_len = len(self.s)
        v_shape_0, v_shape_1 = self.v.shape

        shapes_bytes = pack('<LIII', u_shape_flat, s_len, v_shape_0, v_shape_1)

        u_bytes = self.u.tobytes()
        s_bytes = self.s.tobytes()
        v_bytes = self.v.tobytes()

        return shapes_bytes + u_bytes + s_bytes + v_bytes

    def read(self, data: bytes):
        offset = calcsize('<LIII')
        u_shape_flat, s_len, v_shape_0, v_shape_1 = unpack('<LIII', data[:offset])

        u_shape_0 = u_shape_flat // v_shape_1
        u_shape_1 = v_shape_1

        u_size = u_shape_0 * u_shape_1 * 4
        s_size = s_len * 4

        u_end = offset + u_size
        s_end = u_end + s_size

        self.u = np.frombuffer(data[offset:u_end], dtype=np.float32).reshape((u_shape_0, u_shape_1))
        self.s = np.frombuffer(data[u_end:s_end], dtype=np.float32)
        self.v = np.frombuffer(data[s_end:], dtype=np.float32).reshape((v_shape_0, v_shape_1))

        return self


class PrimitiveSVD:
    def __init__(self, compression_ratio, num_iterations=10):
        self.u = None
        self.s = None
        self.v = None
        self.ratio = compression_ratio
        self.num_iterations = num_iterations

    def compress(self, img: np.ndarray):
        m, n = img.shape
        k = int(min(m, n) * self.ratio)
        self.u, self.s, self.v = self._svd(img, k)
        return self

    def decompress(self):
        return np.clip(np.dot(np.dot(self.u, np.diag(self.s)), self.v), 0, 255).astype(np.uint8)

    def _svd(self, matrix, k):
        m, n = matrix.shape
        u = np.zeros((m, k))
        s = np.zeros(k)
        v = np.zeros((k, n))

        for i in range(k):
            v_i = self._power_iteration(matrix.T @ matrix)
            u_i = matrix @ v_i
            s_i = np.linalg.norm(u_i)
            u_i /= s_i

            u[:, i] = u_i
            s[i] = s_i
            v[i] = v_i

            matrix = matrix - s_i * np.outer(u_i, v_i)

        return u, s, v

    def _power_iteration(self, matrix):
        n = matrix.shape[1]
        v = np.random.rand(n)
        v /= np.linalg.norm(v)

        for _ in range(self.num_iterations):
            v = matrix @ v
            v /= np.linalg.norm(v)

        return v

    def to_bytes(self):
        if self.u is None or self.s is None or self.v is None:
            raise ValueError('Cannot convert to bytes an empty values')

        u_shape_flat = self.u.shape[0] * self.u.shape[1]
        s_len = len(self.s)
        v_shape_0, v_shape_1 = self.v.shape

        shapes_bytes = pack('<LIII', u_shape_flat, s_len, v_shape_0, v_shape_1)

        u_bytes = self.u.tobytes()
        s_bytes = self.s.tobytes()
        v_bytes = self.v.tobytes()

        return shapes_bytes + u_bytes + s_bytes + v_bytes

    def read(self, data: bytes):
        offset = calcsize('<LIII')
        u_shape_flat, s_len, v_shape_0, v_shape_1 = unpack('<LIII', data[:offset])

        u_shape_0 = u_shape_flat // v_shape_1
        u_shape_1 = v_shape_1

        u_size = u_shape_0 * u_shape_1 * 4
        s_size = s_len * 4

        u_end = offset + u_size
        s_end = u_end + s_size

        self.u = np.frombuffer(data[offset:u_end], dtype=np.float32).reshape((u_shape_0, u_shape_1))
        self.s = np.frombuffer(data[u_end:s_end], dtype=np.float32)
        self.v = np.frombuffer(data[s_end:], dtype=np.float32).reshape((v_shape_0, v_shape_1))

        return self


class BlockPowerSVD:
    def __init__(self, compression_ratio, num_iterations=10, block_size=30):
        self.u = None
        self.s = None
        self.v = None
        self.ratio = compression_ratio
        self.num_iterations = num_iterations
        self.block_size = block_size

    def compress(self, img: np.ndarray):
        m, n = img.shape
        k = int(min(m, n) * self.ratio)
        self.u, self.s, self.v = self._svd(img, k)
        return self

    def decompress(self):
        return np.clip(np.dot(np.dot(self.u, np.diag(self.s)), self.v), 0, 255).astype(np.uint8)

    def _svd(self, matrix, k):
        m, n = matrix.shape
        block_size = min(self.block_size, k)
        U = np.random.rand(m, block_size)
        V = np.random.rand(n, block_size)

        for _ in range(self.num_iterations):
            # Bidiagonalization step
            V = np.linalg.qr(matrix.T @ U)[0][:, :block_size]
            U = np.linalg.qr(matrix @ V)[0][:, :block_size]

        # Compute singular values and vectors from the bidiagonalized matrices
        U, s, Vt = np.linalg.svd(matrix @ V, full_matrices=False)
        U = U[:, :k]
        s = s[:k]
        V = V @ Vt.T
        V = V[:, :k]

        return U, s, V.T

    def to_bytes(self):
        if self.u is None or self.s is None or self.v is None:
            raise ValueError('Cannot convert to bytes an empty values')

        u_shape_flat = self.u.shape[0] * self.u.shape[1]
        s_len = len(self.s)
        v_shape_0, v_shape_1 = self.v.shape

        shapes_bytes = pack('<LIII', u_shape_flat, s_len, v_shape_0, v_shape_1)

        u_bytes = self.u.tobytes()
        s_bytes = self.s.tobytes()
        v_bytes = self.v.tobytes()

        return shapes_bytes + u_bytes + s_bytes + v_bytes

    def read(self, data: bytes):
        offset = calcsize('<LIII')
        u_shape_flat, s_len, v_shape_0, v_shape_1 = unpack('<LIII', data[:offset])

        u_shape_0 = u_shape_flat // v_shape_1
        u_shape_1 = v_shape_1

        u_size = u_shape_0 * u_shape_1 * 4
        s_size = s_len * 4

        u_end = offset + u_size
        s_end = u_end + s_size

        self.u = np.frombuffer(data[offset:u_end], dtype=np.float32).reshape((u_shape_0, u_shape_1))
        self.s = np.frombuffer(data[u_end:s_end], dtype=np.float32)
        self.v = np.frombuffer(data[s_end:], dtype=np.float32).reshape((v_shape_0, v_shape_1))

        return self
