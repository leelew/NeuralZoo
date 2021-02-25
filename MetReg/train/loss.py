import tensorflow as tf
import numpy as np


class MaskMSE(tf.keras.losses.Loss):

    def __init__(self, na_mask, name='MaskMSE'):
        super().__init__(name=name)
        self.na_mask = na_mask
        H, W = na_mask.shape
        self.x = []
        self.y = []
        for i in range(H):
            for j in range(W):
                if np.isnan(na_mask[i, j]):
                    self.x.append(i)
                    self.y.append(j)
                    self.na_mask[i, j] = 0
                else:
                    self.na_mask[i, j] = 1
        self.na_mask = tf.convert_to_tensor(self.na_mask)
        self.na_mask = tf.cast(self.na_mask, dtype=tf.float32)

    def set_value(self, matrix, x, y, val):
        # 得到张量的宽和高，即第一维和第二维的Size
        w = int(matrix.get_shape()[1])
        h = int(matrix.get_shape()[2])
        print(w)
        print(h)
        # 构造一个只有目标位置有值的稀疏矩阵，其值为目标值于原始值的差
        val_diff = val - matrix[x][y]
        diff_matrix = tf.sparse.to_dense(tf.sparse.SparseTensor(
            indices=[x, y], values=[val_diff], dense_shape=[w, h]))
        # 用 Variable.assign_add 将两个矩阵相加
        matrix.assign_add(diff_matrix)
        return matrix

    def call(self, y_true, y_pred):
        mse2 = tf.square(y_true - y_pred)
        #mse2 = self.set_value(mse2, self.x, self.y, 0)
        mse2 = tf.math.multiply(mse2, self.na_mask)

        return tf.math.reduce_mean(mse2)


class SSIM(tf.keras.losses.Loss):

    def __init__(self, name='SSIM'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        return 1 - tf.math.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))


def G_loss(y_truth, G_pred, D_pred):
    """generator loss construct of square-error loss
    """
    G_mse_loss = MeanSquaredError()(y_truth, G_pred)
    G_bce_loss = BinaryCrossentropy()(tf.zeros_like(D_pred), D_pred)
    # print(tf.print(G_mse_loss))
    # print(tf.print(G_bce_loss))
    return G_mse_loss - 0.0003*G_bce_loss


def D_loss(D_truth, D_pred):
    """discriminator loss

    Args:
        D_truth ([type]): [description]
        D_pred ([type]): [description]
    """
    D_truth_loss = BinaryCrossentropy()(tf.ones_like(D_truth), D_truth)
    D_pred_loss = BinaryCrossentropy()(tf.zeros_like(D_pred), D_pred)
    return D_truth_loss + D_pred_loss
