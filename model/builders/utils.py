
import logging
import numpy as np
import pandas as pd
from tensorflow.keras.layers import Dense
from transformers import TFBertModel
from tensorflow.keras.regularizers import l2

from model.builders.builders_utils import get_layer_maps,shuffle_genes_map,DynamicTanh
from model.layers_custom import Diagonal, SparseTF
import  tensorflow as tf
from tensorflow.keras import layers



def separate_features(features,cols,file1, file2, file3):
    # 将 features 转换为 DataFrame，并使用 cols 作为列名
    features_df = pd.DataFrame(features, columns=list(cols))
    # 读取三个文件的列名，去掉第一列
    cols1 = pd.read_csv(file1, nrows=0).columns.tolist()[1:]
    cols2 = pd.read_csv(file2, nrows=0).columns.tolist()[1:]
    cols3 = pd.read_csv(file3, nrows=0).columns.tolist()[1:]

    # 在features中提取每个文件的列数据
    feature1 = features_df [cols1]
    feature2 = features_df [cols2]
    feature3 = features_df [cols3]
    feature4 = features_df[['gene_exert']]

    # 在features中删除这些列
    features_df = features_df .drop(cols1 + cols2 + cols3+['gene_exert'], axis=1)

    return features_df , feature1, feature2, feature3,feature4
def embedding_infor(input2,input3, input4_ids,attention_mask,feature_categories,kernel_regularize,max_len=512):

    dyt = DynamicTanh(normalized_shape=[input2.shape[1]], channels_last=True, alpha_init_value=0.5)
    i2 = dyt(input2)
    # layer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)
    # i2 = layer_norm(input2)
    x2 = i2


    dyt2 = DynamicTanh(normalized_shape=[input3.shape[1]], channels_last=True, alpha_init_value=0.5)
    i3 = dyt2(input3)
    # layer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)
    # i3 = layer_norm(input3)
    x3=i3


    bert_model = TFBertModel.from_pretrained(r"D:\P-net\p-net1\pnet_prostate_paper\bert_tf",  from_pt=False )
    bert_model.trainable = False
    bert_output = bert_model(input4_ids, attention_mask=attention_mask)
    output = layers.GlobalAveragePooling1D()(bert_output.last_hidden_state)
    dyt3 = DynamicTanh(normalized_shape=[output .shape[1]], channels_last=True, alpha_init_value=0.5)
    x4 = dyt3(output)
    # layer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)
    # x4 = layer_norm(output)
    return x2, x3,x4
def get_pnet(inputs, features, genes, n_hidden_layers, direction, activation, activation_decision, w_reg,
             w_reg_outcomes, dropout, sparse, add_unk_genes, batch_normal, kernel_initializer, use_bias=False,
             shuffle_genes=False, attention=False, dropout_testing=False, non_neg=False, sparse_first_layer=True):
    feature_names = {}
    n_features = len(features)
    n_genes = len(genes)

    if not type(w_reg) == list:
        w_reg = [w_reg] * 10

    if not type(w_reg_outcomes) == list:
        w_reg_outcomes = [w_reg_outcomes] * 10

    if not type(dropout) == list:
        dropout = [w_reg_outcomes] * 10

    w_reg0 = w_reg[0]
    reg_l = l2
    constraints = {}

    if sparse:
        if shuffle_genes == 'all':
            ones_ratio = float(n_features) / np.prod([n_genes, n_features])
            logging.info('ones_ratio random {}'.format(ones_ratio))
            mapp = np.random.choice([0, 1], size=[n_features, n_genes], p=[1 - ones_ratio, ones_ratio])
            layer1 = SparseTF(n_genes, mapp, activation=activation, W_regularizer=reg_l(w_reg0),
                              name='h{}'.format(0), kernel_initializer=kernel_initializer, use_bias=use_bias,
                              **constraints)
        else:
            layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg0),
                              use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer, **constraints)
    else:
        if sparse_first_layer:
            #
            layer1 = Diagonal(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg0),
                              use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer, **constraints)
        else:
            layer1 = Dense(n_genes, input_shape=(n_features,), activation=activation, W_regularizer=l2(w_reg0),
                           use_bias=use_bias, name='h0', kernel_initializer=kernel_initializer)
    outcome = layer1(inputs)




    if n_hidden_layers > 0:
        maps = get_layer_maps(genes, n_hidden_layers, direction, add_unk_genes)
        w_regs = w_reg[1:]
        for i, mapp in enumerate(maps[0:-1]):
            w_reg = w_regs[i]
            names = mapp.index
            mapp = mapp.values
            if shuffle_genes in ['all', 'pathways']:
                mapp = shuffle_genes_map(mapp)
            n_genes, n_pathways = mapp.shape
            logging.info('n_genes, n_pathways {} {} '.format(n_genes, n_pathways))
            print ('layer {}, dropout  {} w_reg {}'.format(i, dropout, w_reg))
            layer_name = 'h{}'.format(i + 1)
            if sparse:
                hidden_layer = SparseTF(n_pathways, mapp, activation=activation, W_regularizer=reg_l(w_reg),
                                        name=layer_name, kernel_initializer=kernel_initializer,
                                        use_bias=use_bias, **constraints)
            else:
                hidden_layer = Dense(n_pathways, activation=activation, W_regularizer=reg_l(w_reg),
                                     name=layer_name, kernel_initializer=kernel_initializer, **constraints)

            outcome = hidden_layer(outcome)
            feature_names['h{}'.format(i)] = names

        i = len(maps)
        feature_names['h{}'.format(i - 1)] = maps[-1].index
        # 将 output 通过 DynamicTanh 层

        l1=layers.Dense(outcome.shape[1], activation='linear', use_bias=True)
        inputs1 = l1(inputs)
        outcome += inputs1
        dyt = DynamicTanh(normalized_shape=[outcome.shape[1]], channels_last=True, alpha_init_value=0.5)
        outcome = dyt(outcome)
        #none , 26
    return outcome, feature_names

class ClinicalInformationDecoder(layers.Layer):
    def __init__(self,sp, d_in=8, d_out_kq=8, d_out_v=8):
        super(ClinicalInformationDecoder, self).__init__()
        self.d_in = d_in
        self.d_out_kq = d_out_kq
        self.d_out_v = d_out_v
        self.shape = sp
        self.built = False

    def build(self, input_shape):
        if not self.built:

            self.dense33 = layers.Dense(26, activation='linear', use_bias=True, kernel_initializer='random_uniform')
            self.dense20 = layers.Dense(26, activation='linear', use_bias=True,kernel_initializer='random_uniform')
            self.dense21 = layers.Dense(32, activation='tanh', use_bias=True,kernel_initializer='random_uniform')
            self.dense22 = layers.Dense(26, activation='linear', use_bias=True,kernel_initializer='random_uniform')

            self.dense1 = layers.Dense(64, activation='linear', use_bias=True,kernel_initializer='random_uniform')
            self.dense11 = layers.Dense(32, activation='linear', use_bias=True,kernel_initializer='random_uniform')
            self.dense12 = layers.Dense(16, activation='linear', use_bias=True,kernel_initializer='random_uniform')
            self.dense13 = layers.Dense(8, activation='linear', use_bias=False,kernel_initializer='random_uniform')
            self.dense2 = layers.Dense(1, activation='sigmoid', use_bias=False,kernel_initializer='random_uniform')

            self.mlp_Q = layers.Dense(self.d_out_kq, use_bias=True,kernel_initializer='random_uniform')
            self.mlp_K = layers.Dense(self.d_out_kq, use_bias=True,kernel_initializer='random_uniform')
            self.mlp_V = layers.Dense(self.d_out_kq, use_bias=True,kernel_initializer='random_uniform')
            self.l1 = layers.Dense(self.d_out_kq, use_bias=True, kernel_initializer='random_uniform')
            self.l2 = layers.Dense(1, use_bias=True, kernel_initializer='random_uniform')
            self.pool = layers.GlobalAveragePooling1D()
            # 创建层归一化层
            self.layer_norm = layers.LayerNormalization(axis=-1, epsilon=1e-6)
            self.dense5 = layers.Dense(self.shape, activation='linear', use_bias=False, kernel_initializer='random_uniform')
            self.attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=self.d_out_v,kernel_initializer='random_uniform')
            self.dyt = DynamicTanh(normalized_shape=[1], channels_last=True, alpha_init_value=0.5)

            # 创建 LayerNormalization 层
            self.layer_norm = tf.keras.layers.LayerNormalization()

            self.built = True

    def call(self, pnet_outcome, h_outcome, m_outcome, pet_outcome):


        concatenated = tf.concat([pet_outcome, h_outcome, m_outcome], axis=-1)
        in1 = tf.expand_dims(concatenated, axis=1)
        in2 = tf.expand_dims(pnet_outcome, axis=1)
        att_output = self.attention_layer(in2,in1, in1)
        output = tf.squeeze(att_output, axis=1)


        output += pnet_outcome
        output1 = self.dyt(output)
        # output1 =self.layer_norm(output)
        output1 = self.dense20(output1)
        output1 = self.dense21(output1)
        output1 = self.dense22(output1)
        output1 += output

        output1 = self.dense1(output1)
        output1 = self.dense11(output1)
        output1 = self.dense12(output1)
        output1 = self.dense13(output1)
        output1 = self.dense2(output1)

        return output1

    def get_config(self):
            config = super().get_config()
            config.update({
                "d_in": self.d_in,
                "d_out_kq": self.d_out_kq,
                "d_out_v": self.d_out_v,
            })
            return config

