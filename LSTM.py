import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from sqlite3 import Error
import sqlite3
import struct
from datetime import datetime , timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score




torch.manual_seed(0)

########

def sql_connection(file):
    try:
        con = sqlite3.connect(file)
        print("Connection is established with: ", file)
 
    except Error:
        print(Error)
 
    finally:
        pass
        #con.close()
    return con







def msm_data_dic(msm_output_file = 'outputParameter.db',check_BIM = True):
        con = sql_connection(msm_output_file)
        cursorObj_1 = con.cursor()
        cursorObj_1.execute('SELECT * FROM table_curveData')
        rows_1 = cursorObj_1.fetchall()

        cursorObj_2 = con.cursor()
        cursorObj_2.execute('SELECT * FROM table_outputParameter')
        rows_2 = cursorObj_2.fetchall()

        cursorObj_3 = con.cursor()
        cursorObj_3.execute('SELECT * FROM table_operations')
        rows_3 = cursorObj_3.fetchall()
        msm_dic = {}

        ##travel curve , coil current , time stamp
        for row in rows_1:
            if check_BIM:
                temp = {'optype':None,'Time_stamp':None,
                                 'Travel_curve':None,'NO':None,'NC':None}

            else:
                temp = {'optype':None,'Time_stamp':None,
                                 'Travel_curve':None}
            opid =row[1]
            curve_data = row[6]*np.array([struct.unpack(">h",row[3][i*2:i*2+2]) for i in range(len(row[3])//2)]) #signed int 16
            if opid not in msm_dic.keys():
                msm_dic[opid] =temp 
                # {'optype':None,'Time_stamp':{'Travel':None,'Coil':None},
                #                  'Travel_curve':None,'Coil_current':None,'Vel':None,'Timing':None,'Total_travel':None,'52a':None,'52b':None}
            if row[0]=='CircuitBreaker1_general_travelCurve':
                 msm_dic[opid]['Travel_curve'] = curve_data
                 msm_dic[opid]['Time_stamp'] = row[4]
        ##vel , timing , total travel
        for element in rows_2:
            if check_BIM:
                temp = {'optype':None,'Time_stamp':None,
                                 'Travel_curve':None,'NO':None,'NC':None}

            else:
                temp = {'optype':None,'Time_stamp':None,
                                 'Travel_curve':None}
            opid_2 = element[1]
            if opid_2 not in msm_dic.keys():
                msm_dic[opid_2] = temp
                # {'optype':None,'Time_stamp':{'Travel':None,'Coil':None},
                #                  'Travel_curve':None,'Coil_current':None,'Vel':None,'Timing':None,'Total_travel':None,'52a':None,'52b':None}
            
            if(element[0]== 'CircuitBreaker1_general_NC'):
                msm_dic[opid_2]['NC'] = element[6]
            elif(element[0]== 'CircuitBreaker1_general_NO'):
                msm_dic[opid_2]['NO'] = element[6]


        ##optype
        for data in rows_3:
            if check_BIM:
                temp = {'optype':None,'Time_stamp':None,
                                 'Travel_curve':None,'NO':None,'NC':None}

            else:
                temp = {'optype':None,'Time_stamp':None,
                                 'Travel_curve':None}
            opid_3 = data[0]
            if opid_3 not in msm_dic.keys():
                msm_dic[opid_3] = temp
                # {'optype':None,'Time_stamp':{'Travel':None,'Coil':None},
                #                  'Travel_curve':None,'Coil_current':None,'Vel':None,'Timing':None,'Total_travel':None,'52a':None,'52b':None}
            msm_dic[opid_3]['optype'] = data[2]

        return msm_dic

def get_time_diff(timestamp1 ,timestamp2 ):
        '''
        Get time difference (condiser carries)
        output:time diff in s
        timestamp1:time stamp of travel curve(read from MSM)
        timestamp2:tiem stamp of coil current(read from MSM)
        '''
        dt1 = datetime.strptime(timestamp1, "%Y-%m-%dT%H:%M:%S.%fZ")
        dt2 = datetime.strptime(timestamp2, "%Y-%m-%dT%H:%M:%S.%fZ")

        delta = dt2 - dt1

        # unit:s
        return delta.total_seconds() 



def msm_dic_clean(msm_dic):
        '''
        clean data misssing and small time interval operation
        return cleaned data & removed 2 types of data in dic

        '''
        deleted_dic_missing = {}  #store missing data 
        deleted_dic_interval = {} # store small interval data
        #delete data missing operation
        for k, v in list(msm_dic.items()):
            if any(val is None for val in v.values()):
                deleted_dic_missing[k] = msm_dic.pop(k)

        #delete small interval data operation
        keys = list(msm_dic.keys())
        i = 0
        while i < len(keys) - 1:
            time_diff = get_time_diff(msm_dic[keys[i + 1]]['Time_stamp'], msm_dic[keys[i]]['Time_stamp'])
            time_diff = abs(time_diff)
            if time_diff < 4:    #4s
                deleted_dic_interval[keys[i]] = msm_dic.pop(keys[i])
                deleted_dic_interval[keys[i + 1]] = msm_dic.pop(keys[i + 1])
                i += 2  # jump
            else:
                i += 1 #follow loop
        
        return msm_dic

def classify_by_optype(msm_dic):
        optype_0 = {}
        optype_1 = {}

        i = 0
        for key, value in msm_dic.items():
            if value['optype'] == 0:
                optype_0[key] = value
            elif value['optype'] == 1:
                optype_1[key] = value
            i+=1
        return optype_0, optype_1
msm_dic = msm_data_dic()
msm_dic_cleaned = msm_dic_clean(msm_dic)
msm_0,msm_1=classify_by_optype(msm_dic_cleaned)
print('test')
#######





# 数据生成
# def generate_data(length=1000, num_samples=800):
#     # 生成模拟数据
#     X = np.zeros((num_samples, length, 1))
#     y = np.zeros((num_samples, length, 1))

#     for i in range(num_samples):
#         # 随机生成一个行程曲线
#         curve = np.sin(np.linspace(0, 4*np.pi, length)) + np.random.normal(0, 0.1, length)
#         # 选择两个时间点
#         numbers_1 = np.arange(100, 131)
#         t1= np.random.choice(numbers_1)
#         numbers_2 = np.arange(800, 831)
#         t2= np.random.choice(numbers_2)
#         # t1, t2 = np.random.choice(length, size=2, replace=False)
#         # 将这两个时间点的值放入X
#         X[i, t1, 0] = curve[t1]
#         X[i, t2, 0] = curve[t2]
#         # y包含完整的行程曲线
#         y[i, :, 0] = curve

#     return X, y








###生成数据集


# def generate_data_from_actual_journeys(actual_journeys_dic, length=1200):
#     # 生成模拟数据
#     num_samples = len(actual_journeys_dic.keys())
#     X = np.zeros((num_samples, length, 1))
#     y = np.zeros((num_samples, length, 1))
#     i =0
#     for key,value in actual_journeys_dic.items():
#         # 提取时间戳
#         timestamp_first_point = value['Time_stamp']
#         timestamp_t1 = value['NO']
#         timestamp_t2 = value['NC']

#         # 假设我们有一个函数可以生成一个行程曲线
#         curve = value['Travel_curve'][:length]

#         con = get_time_diff(timestamp_t1 , timestamp_first_point)/0.0001
#         # 将这两个时间点的值放入X
#         X[i, int(get_time_diff(timestamp_t1 , timestamp_first_point)/0.0001), 0] = 55   #NC distance
#         X[i, int(get_time_diff(timestamp_t2 , timestamp_first_point)/0.0001), 0] = 196  #NO distance
        
#         # y包含完整的行程曲线
#         y[i, :, 0] = curve[:,0]
#         i+=1
#     return X, y;



### with normalization 

def generate_data_from_actual_journeys(actual_journeys_dic, length=1200):
    # 生成模拟数据
    num_samples = len(actual_journeys_dic.keys())
    X = np.zeros((num_samples, length, 1))
    y = np.zeros((num_samples, length, 1))
    i = 0
    max_distance = 206 # 假设这是辅助接点最大可能的距离
    
    # 首先遍历一遍数据，找到 y 的最大值和最小值
    # min_y = float('inf')
    # max_y = float('-inf')
    
    # for key, value in actual_journeys_dic.items():
    #     curve = value['Travel_curve'][:length]
    #     min_y = min(min_y, np.min(curve[:, 0]))
    #     max_y = max(max_y, np.max(curve[:, 0]))
    min_y= 0
    max_y = 206
        
    for key, value in actual_journeys_dic.items():
        # 提取时间戳
        timestamp_first_point = value['Time_stamp']
        timestamp_t1 = value['NO']
        timestamp_t2 = value['NC']

        # 假设我们有一个函数可以生成一个行程曲线
        curve = value['Travel_curve'][:length]

        con = int(get_time_diff(timestamp_t1, timestamp_first_point) / 0.0001)
        
        # 将这两个时间点的值放入X
        X[i, con, 0] = 196 / max_distance  # NC distance
        X[i, int(get_time_diff(timestamp_t2, timestamp_first_point) / 0.0001), 0] = 53 / max_distance  # NO distance
        
        # y包含完整的行程曲线
        y[i, :, 0] = curve[:, 0]
        i += 1
    
    # 归一化 y 数据
    y_normalized = (y - min_y) / (max_y - min_y)
    
    return X, y_normalized

# 使用示例
# X, y = generate_data_from_actual_journeys(actual_journeys_dic)




# 转换为 PyTorch 张量
X, y = generate_data_from_actual_journeys(msm_1)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.1, random_state=42)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)  # 测试时通常不需要shuffle

# 定义模型
class LSTMSeq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMSeq2Seq, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out)
        return out

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 实例化模型
input_size = 1
hidden_size = 64
num_layers = 2
output_size = 1
model = LSTMSeq2Seq(input_size, hidden_size, num_layers, output_size).to(device)

# 定义损失函数和优化器
# criterion = nn.MSELoss()
# criterion = nn.L1Loss()
def log_cosh_loss(input, target):
    x = input - target
    return torch.mean(torch.log(torch.cosh(x)))
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 100
writer = SummaryWriter(log_dir="runs/my_experiment_real_11_change-loss")




def evaluate_model(dataloader):
    model.eval()
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())
            ground_truths.append(targets.cpu().numpy())

    predictions = np.concatenate(predictions)
    ground_truths = np.concatenate(ground_truths)
    predictions = predictions.reshape(-1, predictions.shape[-1])
    ground_truths = ground_truths.reshape(-1, ground_truths.shape[-1])
    
    mse = mean_squared_error(ground_truths, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(ground_truths, predictions)
    r2 = r2_score(ground_truths, predictions)
    
    return mse, rmse, mae, r2



for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, targets in train_dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # 前向传播
        outputs = model(inputs)
        
        # 计算损失
        # loss = criterion(outputs, targets)
        loss = log_cosh_loss(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataloader.dataset)
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    mse, rmse, mae, r2 = evaluate_model(test_dataloader)
    writer.add_scalar("MSE/val", mse, epoch)
    writer.add_scalar("RMSE/val", rmse, epoch)
    writer.add_scalar("MAE/val", mae, epoch)
    writer.add_scalar("R2/val", r2, epoch)

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, "
          f"MSE: {mse:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
writer.close()



torch.save({
    'epoch': num_epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss.item(),
}, 'LSTM_11.pth')  # 替换为你的文件路径

# # 使用两个点预测整个行程
# known_points = [0.3, 0.7]  # 这些是时间点的位置
# known_times = [np.sin(2 * np.pi * p) for p in known_points]

# # 构建输入张量
# test_input = np.zeros((1, 1000, 1))
# for i, point in enumerate(known_points):
#     index = int(point * 1000)
#     test_input[0, index, 0] = known_times[i]

# # 转换为 PyTorch 张量并移动到设备上
# test_input_tensor = torch.tensor(test_input, dtype=torch.float32).to(device)

# # 使用模型预测
# with torch.no_grad():
#     predicted_curve = model(test_input_tensor)
#     predicted_curve = predicted_curve.cpu().numpy().squeeze()

# 打印预测结果
# print("Predicted Curve:", predicted_curve)


model = LSTMSeq2Seq(input_size, hidden_size, num_layers, output_size).to(device)
# 加载模型权重
checkpoint = torch.load('LSTM_11.pth')  # 替换为你的模型路径
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # 设置为评估模式


# 预测
predictions = []
ground_truth = []
with torch.no_grad():
    for inputs, traget in test_dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())

# 合并所有批次的预测结果
predictions = np.concatenate(predictions, axis=0)
test_targets = np.concatenate([y.cpu().numpy() for _, y in test_dataloader], axis=0)

plt.figure(figsize=(10, 5))
for i in range(10):  # 只显示前10个样本
    plt.plot(test_targets[i], label='Actual')
    plt.plot(predictions[i], label='Predicted', linestyle='--')
plt.legend()
plt.title("Actual vs Predicted Travel Curves")
plt.show()
print('test')