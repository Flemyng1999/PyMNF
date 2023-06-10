from typing import Tuple

import numpy as np
from scipy.linalg import svd
import matplotlib.pyplot as plt


def whiten_noise(data_: np.ndarray, N_: np.ndarray) -> np.ndarray:
    """
    对输入的数据进行白化处理，即减少其内部的相关性。

    Args:
        data_: 二维数组，表示要处理的数据。
        N_: 二维数组, 表示噪声估计矩阵, 其大小应与data相同。

    Returns:
        经过白化处理后的数据，为二维数组。

    Raises:
        AssertionError: 输入的数据与噪声估计矩阵的大小不一致。

    """
    assert data_.shape == N_.shape, "输入的数据与噪声估计矩阵的大小不一致。"

    # 计算噪声估计矩阵的SVD分解
    Un, Sn, V = np.linalg.svd(N_, full_matrices=False)

    # 将噪声估计矩阵的奇异值转化为对角矩阵
    Sn = np.diag(Sn)

    # 计算噪声白化矩阵
    F = np.dot(Un, np.linalg.inv(Sn))

    # 对输入数据进行白化处理
    data_whitened = np.dot(F.T, data_)

    return data_whitened


def unravel_spatial_coords(data_: np.ndarray) -> np.ndarray:
    """
    对输入的三维数据进行展开，得到一个二维数组。其中第一维是输入数据前两维的乘积，第二维是输入数据的第三维。

    Args:
        data_: 三维数组，表示要处理的数据。

    Returns:
        展开后得到的结果，为二维数组。

    Raises:
        AssertionError: 输入数据不是三维数组。

    """
    dims = data_.shape

    assert len(dims) == 3, "输入数据不是三维数组。"

    out = np.zeros((dims[0] * dims[1], dims[2]))

    # 将二维切片展开为一维
    data_slice = np.zeros((dims[0], dims[1]))

    for i_ in range(dims[2]):
        data_slice[:, :] = data_[:, :, i_]  # 保持二维形状
        out[:, i_] = data_slice.ravel()

    return out


def noise_estimate(data_: np.ndarray, m1_: int, m2_: int) -> np.ndarray:
    """
    根据输入的一维数据，估计其对应的噪声矩阵。

    Args:
        data_: 一维数组，表示要进行噪声估计的数据。
        m1_: 整数，表示数据的第一维大小。
        m2_: 整数，表示数据的第二维大小。

    Returns:
        根据输入数据进行估计得到的噪声矩阵，为二维数组。

    """
    # 将一维数据重塑为三维数据
    data_3D = np.reshape(data_, (m1_, m2_, -1))

    N_ = np.zeros(data_3D.shape)

    # 对于除最后一列外的所有列，与相邻左列做差
    N_[:, :-1, :] = data_3D[:, :-1, :] - data_3D[:, 1:, :]

    # 对于最后一列，与相邻右列做差
    N_[:, -1, :] = N_[:, -2, :]

    # 将所有二维切片展开为一维，组成一个二维矩阵作为噪声矩阵
    N_ = unravel_spatial_coords(N_)

    return N_


def mnf(data_: np.ndarray, m1_: int, m2_: int, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    对输入的数据进行MNF变换，得到重构数据和前k个主成分。

    Args:
        data_: 一维数组，表示要进行MNF变换的数据。
        m1_: 整数，表示数据的第一维大小。
        m2_: 整数，表示数据的第二维大小。
        k: 整数，表示要保留的主成分个数。

    Returns:
        一个元组，包含两个元素：
        - reconst: 一维数组，表示经过MNF变换后得到的重构数据。
        - A: 二维数组，表示从输入数据中得到的前k个主成分。

    """
    # 估计数据的噪声矩阵
    N_ = noise_estimate(data_, m1_, m2_)

    # 中心化数据
    means_by_wavenumber = np.sum(data_, axis=0) / (m1_ * m2_)
    means_matrix = np.ones((m1_ * m2_, 1)) * means_by_wavenumber
    data_ = data_ - means_matrix

    # 对数据进行白化处理
    data_whitened = whiten_noise(data_, N_)

    # 提取前k个主成分
    _, _, A_ = svd(data_whitened, full_matrices=False)
    A_ = A_[:k, :]

    # 将数据投影到主成分空间中，得到重构数据
    reconst_ = np.dot(data_, np.dot(A_.T, A_)) + means_matrix

    return reconst_, A_


def test():
    # Test the noise_estimate function
    m1 = 3
    m2 = 4
    n = 5

    np.random.seed(42)
    A = np.random.rand(m1 * m2, n)
    N = noise_estimate(A, m1, m2)

    # Test the reshaping
    ints = np.arange(1, 6)
    B = np.ones((m1 * m2, 1)) * ints
    should_be_0 = noise_estimate(B, m1, m2)
    new_B = np.reshape(B, (m1, m2, -1))
    same_as_B = unravel_spatial_coords(new_B)

    # Test the mnf function
    Y, T = mnf(A, m1, m2, n)

    # Test the noise whitening procedure
    m1 = 180
    m2 = 50
    n = 100
    vec = np.arange(1, m1 + 1) * 2 / m1
    pic = np.outer(vec, np.ones(m2)).flatten()
    data = np.zeros((m1 * m2, n))
    noise_fractions = 0.1 * np.random.rand(n)  # amount of noise in each band
    for i in range(n):
        data[:, i] = (pic + noise_fractions[i] * np.random.randn(m1 * m2)) * i / n

    reconstruct, A1 = mnf(data, m1, m2, 2)

    # Plot the results
    plt.rc('font', family='Times New Roman', )
    fig, axs = plt.subplots(2, figsize=(8, 8), constrained_layout=1, dpi=300)

    axs[0].plot(data[:, 2], label='picture with noise',
                alpha=0.7, solid_capstyle='round')
    axs[0].plot(reconstruct[:, 2], label='picture denoised with mnf',
                alpha=0.7, solid_capstyle='round')
    axs[0].legend()

    axs[1].plot(data[2, :], label='spectrum with noise',
                alpha=0.7, solid_capstyle='round')
    axs[1].plot(reconstruct[2, :], label='spectrum denoised with mnf',
                alpha=0.7, solid_capstyle='round')
    axs[1].legend()

    plt.show()


if __name__ == '__main__':
    test()
