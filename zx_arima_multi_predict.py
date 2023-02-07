from utils import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from itertools import product
from tqdm import tqdm_notebook, tqdm
import warnings
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA

# 忽视在模型拟合中遇到的错误
warnings.filterwarnings("ignore")
shape = 1
target_dim = 0  # 10: 功率，0: CPU使用
# 导入数据
resource_pool = "xar10"
server_id = "computer-b0503-01"
train_data = pd.read_csv(f'data/ZX/train/{server_id}.csv', sep=",", header=None, skiprows=1).dropna(axis=0)
test_data = pd.read_csv(f'data/ZX/test/{server_id}.csv', sep=",", header=None, skiprows=1).dropna(axis=0)
train_data.drop(0, axis=1, inplace=True)
test_data.drop(0, axis=1, inplace=True)
# data = pd.concat([train_data, test_data]).iloc[:, 10]
data = train_data.iloc[:, target_dim]
plt.figure(figsize=(8 * shape, 6 * shape))
plt.plot(range(len(data.values)), data.values)
plt.plot(range(len(data.values)), data.diff(1).values)
plt.savefig(f"output/ZX/analysis/origin_time_series_{target_dim}.pdf")

# 分析平稳性 周期为24*12 = 288。实际上1阶查分后就基本平稳了，没有周期性
# 平稳性检测
print(u'一阶差分序列的平稳性检验结果为：\n', adfuller(data.diff(1).dropna()))  # 返回统计量和p值
# p值小于1% 5% 10%的值 拒绝非平稳的原假设，序列平稳
# 白噪声检测
print(u'一阶差分序列的白噪声检验结果为：\n', acorr_ljungbox(data.diff(1).dropna(), lags=77))  # 返回统计量和p值
# p值均远小于0.05 拒绝白噪声检验的原假设，序列为非白噪声

# ARIMA模型分析：
# 先把ACF图和PACF图画出来看看：lags=sort(n), n为样本数量
# 判断：ACF图在1之后截尾，而PACF图在4之后截尾。模型可为ARIMA(1,1,4).
plt.cla()
fig = plt.figure(figsize=(8 * shape, 6 * shape))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(data.diff(1).iloc[1:].dropna(), lags=77, ax=ax1)  # 注意：要去掉第1个空值
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(data.diff(1).iloc[1:].dropna(), lags=77, ax=ax2)  # 注意：要去掉第1个空值
fig.savefig(f"output/ZX/analysis/ACF_PACF_{target_dim}.pdf")


# 找最优的参数 SARIMAX
def find_best_params(data: np.array, params_list):
    result = []
    best_bic = 100000
    for param in tqdm_notebook(params_list):
        # 模型拟合
        # model = SARIMAX(data,order=(param[0], param[1], param[2]),seasonal_order=(param[3], param[4], param[5], 12)).fit(disp=-1)
        # model = SARIMAX(data, order=(param[0], param[1], param[2])).fit(disp=-1)
        model = ARIMA(data, order=(param[0], param[1], param[2])).fit()
        bicc = model.bic  # 拟合出模型的BIC值
        # print(bic)
        # 寻找最优的参数
        if bicc < best_bic:
            best_mode = model
            best_bic = bicc
            best_param = param
        param_1 = (param[0], param[1], param[2])
        # param_2 = (param[3], param[4], param[5], 12)
        # param = 'SARIMA{0}x{1}'.format(param_1, param_2)
        param = f'ARIMA{param_1}'
        print(param)
        result.append([param, model.bic])

    result_table = pd.DataFrame(result)
    result_table.columns = ['parameters', 'bic']
    result_table = result_table.sort_values(by='bic', ascending=True).reset_index(drop=True)
    return result_table


# 使用准则自动判断ARIMA的参数
def get_beat_params(data):
    # ARIMA的参数
    ps = range(0, 2)
    d = range(0, 2)
    qs = range(0, 5)
    # 季节项相关的参数
    # Ps = range(0, 1)
    # D = range(0, 2)
    # Qs = range(0, 2)
    # 将参数打包，传入下面的数据，是哦那个BIC准则进行参数选择
    # params_list = list(product(ps, d, qs, Ps, D, Qs))
    params_list = list(product(ps, d, qs))
    print(params_list)

    result_table = find_best_params(data, params_list)
    print(result_table)
    # BIC 结果 ARIMA(1,1,2)最优
get_beat_params(data)

# 进行残差检验 todo debug
# # ma1 = SARIMAX(df, order=(0, 1, 1), seasonal_order=(0, 1, 1, 12)).fit(disp=-1)
# # ma1 = SARIMAX(data, order=(1, 1, 2), enforce_stationarity=False, enforce_invertibility=False).fit(disp=-1)
# ma1 = SARIMAX(data, order=(1, 1, 2)).fit(disp=-1)
# # ma1 = ARIMA(data, order=(1, 1, 2)).fit()
# resid = ma1.resid
# # fig = ma1.plot_diagnostics(figsize=(15, 12))
# # fig = ma1.plot_diagnostics()
# # fig.savefig(r'output/ZX/analysis/residual_test.pdf')
# print(ma1.summary())

# 建立ARIMA模型
model = ARIMA(data, order=(1, 1, 2)).fit()  # 使用最小二乘，‘mle’是极大似然估计
# 画图比较一下预测值和真实观测值之间的关系
plt.cla()
fig = plt.figure(figsize=(8 * shape, 6 * shape))
ax = fig.add_subplot(111)
ax.plot(range(len(data.values)), data.values, color='blue', label='Power')
predict = model.fittedvalues
ax.plot(range(len(predict.values)), predict.values, color='red', label='Predicted Power')
plt.legend(loc='lower right')
plt.savefig(f"output/ZX/analysis/train_{target_dim}.pdf")


# 最后，把预测值还原为原始数据的形式，预测值是差分数值，需要转化
def forecast(step, var, modelname):
    diff = list(modelname.predict(var.index[len(var) - 1], var.index[len(var) - 1] + step, dynamic=True))
    prediction = []
    prediction.append(var[var.index[len(var) - 1]])
    seq = []
    seq.append(var[var.index[len(var) - 1]])
    seq.extend(diff)
    for i in range(step):
        v = prediction[i]  # + seq[i + 1]
        prediction.append(v)
    return prediction[1:]  # 第一个值是原序列最后一个值，故第二个值是预测值。


test = test_data.iloc[:, target_dim]
prediction_step = 12
# short—term prediction ---- works
predicts = []
for i in tqdm(range(100, len(test), prediction_step)):
    test_ = test[i - 100:i]
    predict = forecast(prediction_step, test_, model)
    predicts.append(predict)
test__ = test[100:]
tru_date = test__.values.reshape(-1, 1)
pre_data = np.array(predicts).reshape(-1, 1)
pre_data = pre_data[:len(tru_date)]  # 对齐数据

plt.cla()
plt.figure(figsize=(8 * shape, 6 * shape))
plt.plot(range(len(tru_date)), tru_date, color='blue', label='Power')
plt.plot(range(len(pre_data)), pre_data, color='red', label='Predicted Power', alpha=0.5)
plt.legend(loc='lower right')
plt.savefig(f"output/ZX/analysis/test_{target_dim}_consistent_multi_prediction.pdf")

# 标准化后计算损失
tru, scaler = normalize_data(tru_date, scaler=None)
pre, _ = normalize_data(pre_data, scaler=scaler)
predicts_loss = np.sqrt((np.array(pre) - tru) ** 2).mean()
print(f"mes:{predicts_loss}")

# todo 预测未来一小时内所有时刻的序列值
# todo 预测CPU和功率使用情况
# todo 可视化功率限制前后，CPU与功率的使用情况
