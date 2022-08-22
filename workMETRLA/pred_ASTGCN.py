import shutil
import sys
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import Metrics
import Utils
from ASTGCN import *
from Param import *
from Param_ASTGCN import *

# ASTGCN just has one timestep parameter Tp for both in and out, which means TIMESTEP_IN must equal TIMESTEP_OUT.
# Thus, in this script we only keep TIMESTEP_OUT as Tp.

assert TIMESTEP_IN == TIMESTEP_OUT, \
    "ASTGCN just has one timestep parameter Tp, which means TIMESTEP_IN must equal TIMESTEP_OUT."


def getXSYS(data, mode):
    start_index = max(TIMESTEP_PER_HOUR * 24 * 7 * WEEK, TIMESTEP_PER_HOUR * 24 * DAY, TIMESTEP_PER_HOUR * HOUR)
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':
        for i in range(TRAIN_NUM - TIMESTEP_OUT - start_index + 1):
            x = None
            if WEEK != 0:
                week_index = [j for j in range(start_index - TIMESTEP_PER_HOUR * 24 * 7 * WEEK + i, start_index + i,
                                               TIMESTEP_PER_HOUR * 24 * 7)]
                week_sample = np.concatenate([data[k:k + TIMESTEP_OUT] for k in week_index], axis=0)
                x = week_sample
            if DAY != 0:
                day_index = [j for j in range(start_index - TIMESTEP_PER_HOUR * 24 * 1 * DAY + i, start_index + i,
                                              TIMESTEP_PER_HOUR * 24 * 1)]
                day_sample = np.concatenate([data[k:k + TIMESTEP_OUT] for k in day_index], axis=0)
                if str(type(x)) == "<class 'NoneType'>":
                    x = day_sample
                else:
                    x = np.concatenate([x, day_sample], axis=0)
            if HOUR != 0:
                hour_index = [j for j in range(start_index - TIMESTEP_PER_HOUR * 1 * HOUR + i, start_index + i,
                                               TIMESTEP_PER_HOUR * 1)]
                hour_sample = np.concatenate([data[k:k + TIMESTEP_OUT] for k in hour_index], axis=0)
                if str(type(x)) == "<class 'NoneType'>":
                    x = hour_sample
                else:
                    x = np.concatenate([x, hour_sample], axis=0)
            y = data[i + start_index:i + start_index + TIMESTEP_OUT]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - start_index, data.shape[0] - TIMESTEP_OUT - start_index + 1):
            x = None
            if WEEK != 0:
                week_index = [j for j in range(start_index - TIMESTEP_PER_HOUR * 24 * 7 * WEEK + i, start_index + i,
                                               TIMESTEP_PER_HOUR * 24 * 7)]
                week_sample = np.concatenate([data[k:k + TIMESTEP_OUT] for k in week_index], axis=0)
                x = week_sample
            if DAY != 0:
                day_index = [j for j in range(start_index - TIMESTEP_PER_HOUR * 24 * 1 * DAY + i, start_index + i,
                                              TIMESTEP_PER_HOUR * 24 * 1)]
                day_sample = np.concatenate([data[k:k + TIMESTEP_OUT] for k in day_index], axis=0)
                if str(type(x)) == "<class 'NoneType'>":
                    x = day_sample
                else:
                    x = np.concatenate([x, day_sample], axis=0)
            if HOUR != 0:
                hour_index = [j for j in range(start_index - TIMESTEP_PER_HOUR * 1 * HOUR + i, start_index + i,
                                               TIMESTEP_PER_HOUR * 1)]
                hour_sample = np.concatenate([data[k:k + TIMESTEP_OUT] for k in hour_index], axis=0)
                if str(type(x)) == "<class 'NoneType'>":
                    x = hour_sample
                else:
                    x = np.concatenate([x, hour_sample], axis=0)
            y = data[i + start_index:i + start_index + TIMESTEP_OUT]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    # if name == 'ASTGCN':
    XS = XS[:, :, np.newaxis, :]
    XS = XS.transpose(0, 3, 2, 1)
    YS = YS.transpose(0, 2, 1)
    return XS, YS


def getModel(name):
    distance = adj_tans(sensor_ids_file=SENSOR_IDS, distance_file=DISTANCES)
    adj_mx, _ = get_adjacency_matrix(distance, N_NODE)
    L_tilde = scaled_Laplacian(adj_mx)
    cheb_polynomials = [torch.from_numpy(i).type(torch.FloatTensor).to(device) for i in cheb_polynomial(L_tilde, K=3)]
    model = ASTGCN(device, cheb_polynomials=cheb_polynomials, in_channels=CHANNEL,
                   num_for_predict=TIMESTEP_OUT, len_input=TIMESTEP_OUT * (WEEK + DAY + HOUR),
                   num_of_vertices=N_NODE).to(device)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)
    return model


def evaluateModel(model, criterion, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x)
            l = criterion(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n


def predictModel(model, data_iter):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            YS_pred_batch = model(x)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred


def trainModel(name, mode, XS, YS):
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_OUT, TIMESTEP_OUT)  # ASTGCN
    model = getModel(name)
    summary(model, (N_NODE, CHANNEL, TIMESTEP_OUT * (WEEK + DAY + HOUR)), device=device)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1 - TRAINVALSPLIT))
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, BATCHSIZE, shuffle=False)

    min_val_loss = np.inf
    wait = 0

    print('LOSS is :', LOSS)
    if LOSS == "MaskMAE":
        criterion = Utils.masked_mae
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    if OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARN)
    if OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)
    for epoch in range(EPOCH):
        starttime = datetime.now()
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        train_loss = loss_sum / n
        val_loss = evaluateModel(model, criterion, val_iter)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), PATH + '/' + name + '.pt')
        else:
            wait += 1
            if wait == PATIENCE:
                print('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time, " seconds ", "train loss:", train_loss, ", validation loss:",
              val_loss)
        with open(PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % (
                "epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:",
                val_loss))

    torch_score = evaluateModel(model, criterion, train_iter)

    YS_pred = predictModel(model, torch.utils.data.DataLoader(trainval_data, BATCHSIZE, shuffle=False))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS = YS.transpose(0, 2, 1)
    YS_pred = YS_pred.transpose(0, 2, 1)
    YS, YS_pred = scaler.inverse_transform(np.squeeze(YS).reshape(-1, N_NODE)), scaler.inverse_transform(
        np.squeeze(YS_pred).reshape(-1, N_NODE))
    YS, YS_pred = YS.reshape(-1, TIMESTEP_OUT, N_NODE), YS_pred.reshape(-1, TIMESTEP_OUT, N_NODE)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
        f.write("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print("%s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (name, mode, MSE, RMSE, MAE, MAPE))
    print('Model Training Ended ...', time.ctime())


def testModel(name, mode, XS, YS):
    print('Model Testing Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_OUT, TIMESTEP_OUT)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
    model = getModel(name)
    model.load_state_dict(torch.load(PATH + '/' + name + '.pt'))
    print('LOSS is :', LOSS)
    if LOSS == "MaskMAE":
        criterion = masked_mae
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    torch_score = evaluateModel(model, criterion, test_iter)

    YS_pred = predictModel(model, test_iter)
    YS = YS.transpose(0, 2, 1)
    YS_pred = YS_pred.transpose(0, 2, 1)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    YS, YS_pred = scaler.inverse_transform(np.squeeze(YS).reshape(-1, N_NODE)), scaler.inverse_transform(
        np.squeeze(YS_pred).reshape(-1, N_NODE))
    YS, YS_pred = YS.reshape(-1, TIMESTEP_OUT, N_NODE), YS_pred.reshape(-1, TIMESTEP_OUT, N_NODE)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    print('*' * 40)
    print("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    f = open(PATH + '/' + name + '_prediction_scores.txt', 'a')
    f.write("%s, %s, Torch MSE, %.10e, %.10f\n" % (name, mode, torch_score, torch_score))
    print("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (
        name, mode, MSE, RMSE, MAE, MAPE))
    f.write("all pred steps, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (
        name, mode, MSE, RMSE, MAE, MAPE))
    for i in range(TIMESTEP_OUT):
        MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS[:, i, :], YS_pred[:, i, :])
        print("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f" % (
            i + 1, name, mode, MSE, RMSE, MAE, MAPE))
        f.write("%d step, %s, %s, MSE, RMSE, MAE, MAPE, %.10f, %.10f, %.10f, %.10f\n" % (
            i + 1, name, mode, MSE, RMSE, MAE, MAPE))
    f.close()
    print('Model Testing Ended ...', time.ctime())


################# Parameter Setting #######################
MODELNAME = 'ASTGCN'
KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = '../save/' + KEYWORD
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)
# torch.backends.cudnn.deterministic = True
import os

cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

###########################################################
GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
###########################################################
data = pd.read_hdf(FLOWPATH).values
scaler = StandardScaler()
data = scaler.fit_transform(data)
print('data.shape', data.shape)


###########################################################
def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('Param_ASTGCN.py', PATH)
    shutil.copy2('ASTGCN.py', PATH)

    trainXS, trainYS = getXSYS(data, 'TRAIN')
    testXS, testYS = getXSYS(data, 'TEST')

    print(KEYWORD, 'training started', time.ctime())
    print('TRAIN XS.shape YS,shape', trainXS.shape, trainYS.shape)

    trainModel(MODELNAME, 'TRAIN', trainXS, trainYS)

    print(KEYWORD, 'testing started', time.ctime())
    print('TEST XS.shape, YS.shape', testXS.shape, testYS.shape)
    testModel(MODELNAME, 'TEST', testXS, testYS)


if __name__ == '__main__':
    main()
