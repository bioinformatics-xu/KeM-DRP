
import itertools
import logging
import numpy as np
import pandas as pd

from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.python.ops.numpy_ops import int32

from data.pathways.reactome import ReactomeNetwork

from tensorflow.keras import layers

class DynamicTanh(layers.Layer):
    def __init__(self, normalized_shape, channels_last, alpha_init_value=0.5, **kwargs):
        super().__init__(**kwargs)
        self.normalized_shape = normalized_shape
        self.alpha_init_value = alpha_init_value
        self.channels_last = channels_last

        # 初始化可学习参数
        self.alpha = self.add_weight(
            name="alpha",
            shape=(1,),
            initializer=tf.keras.initializers.Constant(value=alpha_init_value),
            trainable=True,
        )
        self.weight = self.add_weight(
            name="weight",
            shape=normalized_shape,
            initializer=tf.keras.initializers.Ones(),
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=normalized_shape,
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

    def call(self, inputs):
        # 应用 tanh 激活函数
        x = tf.tanh(self.alpha * inputs)

        # 根据 channels_last 的值调整 weight 和 bias 的广播方式
        if self.channels_last:
            x = x * self.weight + self.bias
        else:
            # 如果 channels_last 为 False，需要调整 weight 和 bias 的形状以匹配输入
            weight = tf.reshape(self.weight, [-1, 1, 1] + list(self.normalized_shape))
            bias = tf.reshape(self.bias, [-1, 1, 1] + list(self.normalized_shape))
            x = x * weight + bias

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "normalized_shape": self.normalized_shape,
            "alpha_init_value": self.alpha_init_value,
            "channels_last": self.channels_last,
        })
        return config
import tensorflow as tf
from tensorflow.keras import layers

class MultiAttentionBlock(layers.Layer):
    def __init__(self, num_heads, key_dim,input_shape_, kernel_regularizer=None, **kwargs):
        super(MultiAttentionBlock, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.kernel_regularizer = kernel_regularizer
        self.input_shape_ = input_shape_


        self.mlp_Q = layers.Dense(key_dim , use_bias=True, kernel_regularizer=kernel_regularizer)
        self.mlp_K = layers.Dense(key_dim , use_bias=True, kernel_regularizer=kernel_regularizer)
        self.mlp_V= layers.Dense(key_dim , use_bias=True, kernel_regularizer=kernel_regularizer)
        # 初始化多头注意力层
        self.attention_layer = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        # self.activation_layer = tf.keras.layers.Activation('tanh')
        # 初始化后续的 Dense 层
        self.l1 = layers.Dense(key_dim // 2, use_bias=True, kernel_regularizer=kernel_regularizer)
        self.l2 = layers.Dense(1, use_bias=True, kernel_regularizer=kernel_regularizer)

        # 初始化 DynamicTanh 层
        self.dyt = DynamicTanh(normalized_shape=[input_shape_], channels_last=True, alpha_init_value=0.5)
        # 创建 LayerNormalization 层
        self.layer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)

        self.d1 = layers.Dense(key_dim*2, use_bias=True, kernel_regularizer=kernel_regularizer)
        self.d2 = layers.Dense(key_dim, use_bias=True, kernel_regularizer=kernel_regularizer)

    def call(self, input1):
        x=tf.expand_dims(input1, axis=-1)
        query = self.mlp_Q(x)
        key = self.mlp_K(x)
        value = self.mlp_V(x)

        output = self.attention_layer(query, key, value)


        output = self.l1(output)
        output = self.l2(output)

        output = tf.squeeze(output, axis=-1)

        output += input1


        output = self.layer_norm(output)

        return output

    def get_config(self):
        config = super().get_config()  # 获取父类的配置
        config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
        })
        return config

def CGRM(h_outcome, m_outcome, pet_outcome,pnetshape1):

    project_input = tf.concat([h_outcome, m_outcome, pet_outcome], axis=-1)


    mlp = tf.keras.Sequential([
        tf.keras.layers.Dense(128,input_dim=project_input.shape[1]),
        tf.keras.layers.Dense(256,activation='tanh'),
        tf.keras.layers.Dense(pnetshape1)
    ])
    project_output = mlp(project_input)

    return project_output

def Hybridcontrastivebinary_loss(binary_weight=0.5, cosine_weight=0.5):
    cosine_similarity = tf.keras.losses.CosineSimilarity(axis=-1)
    def loss(y_true, y_pred):


        n = (int32)((y_pred.shape[1] - 1)/2)
        main_pred, features_pre, features_true = tf.split(y_pred, [1, n, n], axis=-1)

        binary_loss = BinaryCrossentropy(from_logits=False)(y_true, main_pred)

        cs = cosine_similarity(features_true, features_pre)

        cosine_loss = 1+cs


        tf.print("binary_loss:", binary_loss, "cosine_loss:", cosine_loss)

        total_loss = binary_weight * binary_loss + cosine_weight * cosine_loss


        return total_loss

    return loss

def get_map_from_layer(layer_dict):
    # pathways = layer_dict.keys()
    pathways = list(layer_dict.keys())  # 将 pathways 转换为列表
    print ('pathways', len(pathways))
    genes = list(itertools.chain.from_iterable(layer_dict.values()))
    genes = list(np.unique(genes))
    print ('genes', len(genes))

    n_pathways = len(pathways)
    n_genes = len(genes)

    mat = np.zeros((n_pathways, n_genes))
    for p, gs in layer_dict.items():
        g_inds = [genes.index(g) for g in gs]
        p_ind = pathways.index(p)
        mat[p_ind, g_inds] = 1

    df = pd.DataFrame(mat, index=pathways, columns=genes)
    # for k, v in layer_dict.items():
    #     print k, v
    #     df.loc[k,v] = 1
    # df= df.fillna(0)
    return df.T


def get_layer_maps(genes, n_levels, direction, add_unk_genes):
    reactome_layers = ReactomeNetwork().get_layers(n_levels, direction)
    filtering_index = genes
    maps = []
    for i, layer in enumerate(reactome_layers[::-1]):
        print ('layer #', i)
        mapp = get_map_from_layer(layer)
        filter_df = pd.DataFrame(index=filtering_index)
        print ('filtered_map', filter_df.shape)
        filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='left')
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
        print ('filtered_map', filter_df.shape)
        # filtered_map = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')

        # UNK, add a node for genes without known reactome annotation
        if add_unk_genes:
            print('UNK ')
            filtered_map['UNK'] = 0
            ind = filtered_map.sum(axis=1) == 0
            filtered_map.loc[ind, 'UNK'] = 1
        ####

        filtered_map = filtered_map.fillna(0)
        print ('filtered_map', filter_df.shape)
        # filtering_index = list(filtered_map.columns)
        filtering_index = filtered_map.columns
        logging.info('layer {} , # of edges  {}'.format(i, filtered_map.sum().sum()))
        maps.append(filtered_map)
    return maps


def shuffle_genes_map(mapp):
    # print mapp[0:10, 0:10]
    # print sum(mapp)
    # logging.info('shuffling the map')
    # mapp = mapp.T
    # np.random.shuffle(mapp)
    # mapp= mapp.T
    # print mapp[0:10, 0:10]
    # print sum(mapp)
    logging.info('shuffling')
    ones_ratio = np.sum(mapp) / np.prod(mapp.shape)
    logging.info('ones_ratio {}'.format(ones_ratio))
    mapp = np.random.choice([0, 1], size=mapp.shape, p=[1 - ones_ratio, ones_ratio])
    logging.info('random map ones_ratio {}'.format(ones_ratio))
    return mapp





