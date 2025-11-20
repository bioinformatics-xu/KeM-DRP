import logging
import numpy as np
from tensorflow.keras.models import Model
from data.data_access import Data
from model.builders.builders_utils import CGRM,Hybridcontrastivebinary_loss, MultiAttentionBlock
from model.builders.utils import get_pnet, separate_features, embedding_infor,ClinicalInformationDecoder


# 其他代码
from model.layers_custom import f1
from model.model_utils import print_model, get_layers
from tensorflow.keras.layers import Input,Lambda
import  tensorflow as tf
from pipeline.one_split import  preprocess_sentences_for_bert
tf.keras.backend.set_floatx('float32')
from tensorflow.keras.regularizers import l2

def ClinMultiDLBCL(optimizer, w_reg, w_reg_outcomes,file1,file2,file3, add_unk_genes=True, sparse=True, loss_weights=1.0, dropout=0.5,
                use_bias=False, activation='tanh', loss='binary_crossentropy', data_params=None, n_hidden_layers=1,
                direction='root_to_leaf', batch_normal=False, kernel_initializer='glorot_uniform', shuffle_genes=False,
                attention=False, dropout_testing=False, non_neg=False, repeated_outcomes=True, sparse_first_layer=True,):

    print(data_params)
    print('n_hidden_layers', n_hidden_layers)
    data = Data(**data_params)
    x, y, info, cols = data.get_data()
    cols_n = np.array(list(cols))
    feature1_df, feature2_df,feature3_df, feature4_df, feature5_df= separate_features(x, cols_n,file1,file2,file3)

    # 获取每个特征集的列数
    n_features1 = feature1_df.shape[1]
    n_features2 = feature2_df.shape[1]
    n_features3 = feature3_df.shape[1]
    n_features5 = feature5_df.shape[1]
    sentences = feature4_df['PET/CT'].tolist()
    bert_input,bert_mask = preprocess_sentences_for_bert(sentences)
    n_features4 = bert_input.shape[1]
    cols= feature1_df.columns
    print(x.shape)
    print(y.shape)
    print(info.shape)
    features = cols

    if loss == 'binary_crossentropy':
        activation_decision = 'sigmoid'
    else:
        activation_decision = 'linear'
    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols_n.shape))

    logging.info('x shape {} , y shape {} info {} genes {}'.format(x.shape, y.shape, info.shape, cols_n.shape))


    if hasattr(cols, 'levels'):
        genes = cols.levels[0]
    else:
        genes = cols
        # 调用函数



    # 定义 Keras 输入层
    input1 = Input(shape=(n_features1,), dtype='float32', name='input1')
    input2 = Input(shape=(n_features2,), dtype='float32', name='input2')
    input3 = Input(shape=(n_features3,), dtype='float32', name='input3')
    # input4 = Input(shape=(n_features4,), dtype='string', name='input4')
    # 定义模型
    input4_ids = tf.keras.layers.Input(shape=(n_features4,), dtype=tf.int32, name='input_ids')
    attention_mask = tf.keras.layers.Input(shape=(n_features4,), dtype=tf.int32, name='attention_mask')
    if_gense = tf.keras.layers.Input(shape=(n_features5,), dtype=tf.int32, name='if_gense')
    feature_categories = [2]


    h_outcome, m_outcome, pet_outcome = embedding_infor(input2, input3, input4_ids,attention_mask,feature_categories, l2(0.01))# 添加 L2 正则化)


    MA=  MultiAttentionBlock(8,16,input1.shape[1],l2(0.01))
    input_1  = MA(input1)
    pnet_outcome, feature_n = get_pnet(              input_1,
                                                     features=features,
                                                     genes=genes,
                                                     n_hidden_layers=n_hidden_layers,
                                                     direction=direction,
                                                     activation=activation,
                                                     activation_decision=activation_decision,
                                                     w_reg=w_reg,
                                                     w_reg_outcomes=w_reg_outcomes,
                                                     dropout=dropout,
                                                     sparse=sparse,
                                                     add_unk_genes=add_unk_genes,
                                                     batch_normal=batch_normal,
                                                     sparse_first_layer=sparse_first_layer,
                                                     use_bias=use_bias,
                                                     kernel_initializer=kernel_initializer,
                                                     shuffle_genes=shuffle_genes,
                                                     attention=attention,
                                                     dropout_testing=dropout_testing,
                                                     non_neg=non_neg
                                                     )



    project_out = CGRM(h_outcome, m_outcome, pet_outcome,pnet_outcome.shape[1])

    project_input_flattened2 = tf.keras.layers.Lambda(
        lambda inputs: tf.where(
            tf.equal(inputs[0], 1),  # 条件：if_gense == 1
            inputs[1],
            inputs[2])  # if_gense == 0 时的逻辑
        )([if_gense,pnet_outcome, project_out])
    output_combiner = ClinicalInformationDecoder(pnet_outcome.shape[1]+m_outcome.shape[1]+h_outcome.shape[1]+pet_outcome.shape[1],d_in=8, d_out_kq=8, d_out_v=8)
    outcome1 = output_combiner(pnet_outcome, h_outcome, m_outcome, pet_outcome)
    outcome2 = output_combiner(project_out, h_outcome, m_outcome, pet_outcome)
    # 将所有输入传递给 Lambda 层
    # 使用 Lambda 层实现条件逻辑
    outcome = tf.keras.layers.Lambda(
        lambda inputs: tf.where(
            tf.equal(inputs[0], 1),  # 条件：if_gense == 1
            inputs[1],  # if_gense == 1 时的逻辑
            inputs[2])  # if_gense == 0 时的逻辑
        )([if_gense,outcome1,outcome2])

    feature_names = feature_n
    feature_names['inputs'] = cols

    print('Compiling...')


    output = Lambda(lambda x: tf.concat(x, axis=-1))([outcome,project_out,project_input_flattened2])

    model = Model(inputs=[input1,input2,input3,input4_ids,attention_mask,if_gense], outputs=output)




    print('loss_weights', loss_weights)
    model.compile(optimizer=optimizer,
                  loss=Hybridcontrastivebinary_loss(binary_weight=0.4, cosine_weight=0.6), metrics=[f1])


    logging.info('done compiling')

    print_model(model)
    print(get_layers(model))
    print(model.output)
    logging.info(model.summary())
    logging.info('# of trainable params of the model is %s' % model.count_params())
    return model, feature_names
