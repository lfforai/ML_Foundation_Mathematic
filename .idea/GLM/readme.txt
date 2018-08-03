该项目属于公司商业项目，由于数据保密性，未上传车基和出险数据
目前使用迭代法计算GLM模型的算法有2中梯度下降和迭代法：
迭代法：GLM-iteration算法进行参数拟合（与sas和matlab一致）
梯度下降法：（input-ps是基于异步计算的模型）和GLM_STochastic_gradien（是单GPU同步模型）
结论：随机梯度的收敛性不好，原因待查，迭代法效率已经接近sas速度效果很好

数据处理：1、mapd的python接口利用sql处理出险和车基数据.
         2、利用tensorflow和sas拟合GLM模型、其中tensorflow迭代法和梯度下降法为自己编写，sas直接使用了genmod过程
            两类模型效果基本一致.
         3、在模型数据处理中，数据拟合前现将数据汇总，对赔付项进行求平均。
         4、在拟合GLM模型时候，随机生成各指标的区间切割，生成若干模型结果演算，由人工对大量模型结果进行识别


其中使用mapd 是基于gpu的sql数据库速度非常快，和python的panads无缝链接

使用工具包括：
tensorflow
mapd
python pandas

预计增减机器对模型结果识别模块，机器筛选最佳指标组合和分组