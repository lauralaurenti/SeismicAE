import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from tqdm.auto import tqdm
from transformers import EncodecModel
from sklearn.model_selection import train_test_split,KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import os
import utils as u

from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

seeds=[1,12,123,1234,12345,123456,1234567,12345678,123456789,1234567890]
# seed =123
embedded_input = True # if True it embedds the input, if false it uses the original model
model_gcn=False

network = 1
test_set_size = 0.2
trace_len =1000
print(torch.cuda.is_available())
if torch.cuda.is_available():  
    dev = "cuda" 
    map_location=None
else:  
    dev = "cpu"  
    map_location='cpu'
device = torch.device(dev)


for seed in seeds:
    print(seed)
    u.seed_everything(seed)

    if network == 1:
        inputs = np.load('Data/inputs_ci.npy', allow_pickle=True)
        targets = np.load('Data/targets.npy', allow_pickle=True)
        graph_input = u.graph_creator('network_1', 0.3) # np.load('data/minmax_normalized_laplacian.npy', allow_pickle=True)
        graph_input = np.array([graph_input] * inputs.shape[0])
        graph_features = np.load('Data/station_coords.npy', allow_pickle=True)
        graph_features = np.array([graph_features] * inputs.shape[0])
    if network == 2:
        inputs = np.load('othernetwork/inputs_cw.npy', allow_pickle=True)
        targets = np.load('othernetwork/targets.npy', allow_pickle=True)
        graph_input = u.graph_creator('network_1', 0.3)  #np.load('data/othernetwork/minmax_normalized_laplacian.npy', allow_pickle=True)
        graph_input = np.array([graph_input] * inputs.shape[0])
        graph_features = np.load('othernetwork/station_coords.npy', allow_pickle=True)
        graph_features = np.array([graph_features] * inputs.shape[0])

    if embedded_input:
        inputs_norm = u.normalize_for_emb(inputs[:, :, :trace_len, :])
        dset = TensorDataset(torch.tensor(inputs_norm, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32),torch.tensor(graph_features, dtype=torch.float32),torch.tensor(graph_input, dtype=torch.float32))
        d_loader = DataLoader(dset, batch_size=1, shuffle=False)

        net_H =  EncodecModel.from_pretrained("facebook/encodec_24khz")
        net_E =  EncodecModel.from_pretrained("facebook/encodec_24khz")
        net_Z =  EncodecModel.from_pretrained("facebook/encodec_24khz")

        # net_E.load_state_dict(torch.load("C:\\Users\\Laura\\Desktop\\Laura\\Istruzione\\PhD\\Los Alamos\\Project\\Encoder\\ours\\best_models_enc\\epoch47_finetuningSTEAD_INSTANCE_lr00001_BEST_EPOCH_ON_signal_ratio_lossSpectra_chE.pth", map_location='cuda:0'))
        # net_H.load_state_dict(torch.load("C:\\Users\\Laura\\Desktop\\Laura\\Istruzione\\PhD\\Los Alamos\\Project\\Encoder\\ours\\best_models_enc\\epoch48_finetuningSTEAD_INSTANCE_lr00001_BEST_EPOCH_ON_signal_ratio_lossSpectra_shouldBeN.pth", map_location='cuda:0'))
        # net_Z.load_state_dict(torch.load("C:\\Users\\Laura\\Desktop\\Laura\\Istruzione\\PhD\\Los Alamos\\Project\\Encoder\\ours\\best_models_enc\\epoch47_finetuningSTEAD_INSTANCE_lr00001_BEST_EPOCH_ON_signal_ratio_lossSpectra_chZ.pth", map_location='cuda:0'))
        net_E.load_state_dict(torch.load("C:\\Users\\Laura\\Desktop\\Laura\\Istruzione\\PhD\\Los Alamos\\Project\\Encoder\\ours\\best_models_enc\\epoch61_finetuningSTEAD_lr00001_BEST_EPOCH_ON_loss_ch0.pth", map_location='cuda:0'))
        net_H.load_state_dict(torch.load("C:\\Users\\Laura\\Desktop\\Laura\\Istruzione\\PhD\\Los Alamos\\Project\\Encoder\\ours\\best_models_enc\\epoch8_finetuningSTEAD_lr00001_BEST_EPOCH_ON_loss_ch1.pth", map_location='cuda:0'))
        net_Z.load_state_dict(torch.load("C:\\Users\\Laura\\Desktop\\Laura\\Istruzione\\PhD\\Los Alamos\\Project\\Encoder\\ours\\best_models_enc\\epoch6_finetuningSTEAD_lr00001_BEST_EPOCH_ON_loss_ch2.pth", map_location='cuda:0'))
        net_H.to(device)
        net_E.to(device)
        net_Z.to(device)

        net_H.eval()
        net_E.eval()
        net_Z.eval()

        print('Encoding:')
        def hook(module, input, output):
            outputs["conv"] = output

        denc_X = []
        denc_target = []
        denc_features = []
        denc_graph = []
        # outputs = {}

        hook_handle_E = net_E.encoder.layers[15].register_forward_hook(hook)
        hook_handle_H = net_H.encoder.layers[15].register_forward_hook(hook)
        hook_handle_Z = net_Z.encoder.layers[15].register_forward_hook(hook)

        with torch.no_grad():
            for idx, batch in tqdm(enumerate(d_loader), total=len(d_loader)):
                # print(batch[0].shape)
                reshaped_batch = batch[0].reshape(-1,trace_len,3).permute(0,2,1)
                # print("reshaped_batch", reshaped_batch.shape)
                E_btc = reshaped_batch[:,0,:].unsqueeze(1).float().to(device)
                H_btc = reshaped_batch[:,1,:].unsqueeze(1).float().to(device)
                Z_btc = reshaped_batch[:,2,:].unsqueeze(1).float().to(device)
                # print("Z_btc", Z_btc.shape)
                outputs = {}
                output = net_E(E_btc)
                out_E = outputs["conv"]

                outputs = {}
                output = net_H(H_btc)
                out_H = outputs["conv"]

                outputs = {}
                output = net_Z(Z_btc)
                out_Z = outputs["conv"]
                # print("out_Z",  out_Z.shape)

                out_chs= torch.cat((out_E,out_H,out_Z), axis=2)
                # print("out_chs", out_chs.shape)

                denc_X.append(out_chs)
                denc_target.append(batch[1].reshape(-1,5))
                denc_features.append(batch[2].reshape(-1,2))
                denc_graph.append(batch[3].reshape(-1,39))
                # print("enc_X, target,feat", len(enc_X),len(enc_target),len(enc_features))

        denc_X = torch.stack(denc_X)
        denc_target = torch.stack(denc_target)
        denc_features = torch.stack(denc_features)
        denc_graph = torch.stack(denc_graph)

        print("denc_X, target, feat, graph", denc_X.shape,denc_target.shape,denc_features.shape,denc_graph.shape)


    idx_dset=np.arange(len(inputs))
    np.random.shuffle(idx_dset)
    idx_dset=idx_dset.tolist()
    idx_train_set=idx_dset[:len(idx_dset)-int(len(idx_dset)*test_set_size)]
    idx_test_set=idx_dset[len(idx_dset)-int(len(idx_dset)*test_set_size):]
    print("len(idx_dset)", len(idx_dset),"len(idx_train_set)", len(idx_train_set),"len(idx_test_set)", len(idx_test_set))

    if embedded_input:
        enc_X = denc_X[idx_train_set]
        enc_target = denc_target[idx_train_set]
        enc_features = denc_features[idx_train_set]
        enc_graph = denc_graph[idx_train_set]
        print("enc_X, target, feat, graph", enc_X.shape,enc_target.shape,enc_features.shape,enc_graph.shape)

        enc_test_X = denc_X[idx_test_set]
        enc_test_target = denc_target[idx_test_set]
        enc_test_features = denc_features[idx_test_set]
        enc_test_graph = denc_graph[idx_test_set]
        print("enc_test_X, target, feat, graph", enc_test_X.shape,enc_test_target.shape,enc_test_features.shape,enc_test_graph.shape)
    train_features = graph_features[idx_train_set]
    test_features = graph_features[idx_test_set]
    train_graph_input = graph_input[idx_train_set]
    test_graph_input = graph_input[idx_test_set]
    train_inputs = inputs[idx_train_set]
    test_inputs = inputs[idx_test_set]
    train_targets = targets[idx_train_set]
    test_targets = targets[idx_test_set]

    if embedded_input:
        train_inputs = u.normalize_for_emb(train_inputs[:, :, :trace_len, :])#(train_inputs[:, :, :, :])
        test_inputs = u.normalize_for_emb(test_inputs[:, :, :trace_len, :])#(train_inputs[:, :, :, :])
    else:    
        train_inputs = u.normalize(train_inputs[:, :, :trace_len, :])#(train_inputs[:, :, :, :])
        test_inputs = u.normalize(test_inputs[:, :, :trace_len, :])#(train_inputs[:, :, :, :])

    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(torch.tensor(train_inputs, dtype=torch.float32), torch.tensor(train_targets, dtype=torch.float32),torch.tensor(train_features, dtype=torch.float32),torch.tensor(train_graph_input, dtype=torch.float32))
    test_dataset = TensorDataset(torch.tensor(test_inputs, dtype=torch.float32), torch.tensor(test_targets, dtype=torch.float32),torch.tensor(test_features, dtype=torch.float32), torch.tensor(test_graph_input, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=True)

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    batch_size = 20
    num_epochs = 100
    patience = 50  # Numero di epoche da attendere dopo l'ultimo miglioramento prima di interrompere l'addestramento

    best_losses_per_fold = [] 
    if embedded_input:
        X_data=enc_X
        Y_data=enc_target
        feat_data=enc_features
        graph_input_data=enc_graph
    else:
        X_data=torch.tensor(train_inputs, dtype=torch.float32)
        Y_data=torch.tensor(train_targets, dtype=torch.float32)
        feat_data=torch.tensor(train_features, dtype=torch.float32)
        graph_input_data=torch.tensor(train_graph_input, dtype=torch.float32)

    for fold, (train_index, val_index) in enumerate(kf.split(X_data)):
        if embedded_input:
            if model_gcn:
                model= u.Model_gcn_for_embedding().to(device)
            else:
                model = u.Model_cnn_for_embedding().to(device)
        else:
            if model_gcn:
                model= u.OriginalModel_gcn().to(device)
            else:
                model= u.OriginalModel_cnn().to(device)
        best_model=type(model)().to(device)
        # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Aggiungi regolarizzazione L2 con weight_decay
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.0001, weight_decay=1e-4)  # Aggiungi regolarizzazione L2 con weight_decay
        loss_function = nn.MSELoss()
        
        print(f"Starting fold {fold+1}/{n_splits}")
        
        best_val_loss = float('inf')
        best_val_losses = []
        epochs_without_improvement = 0  # Resetta il contatore per l'early stopping
        
        train_inputs, val_inputs = X_data[train_index], X_data[val_index]
        train_targets, val_targets  = Y_data[train_index], Y_data[val_index]
        train_features, val_features = feat_data[train_index], feat_data[val_index]
        train_graph_input, val_graph_input = graph_input_data[train_index], graph_input_data[val_index]

        train_dataset = TensorDataset(train_inputs, train_targets,train_features, train_graph_input)
        val_dataset = TensorDataset(val_inputs, val_targets, val_features, val_graph_input)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(num_epochs):
            model.train()
            train_total_loss = 0 
            # for inputs, targets, features in tqdm(train_loader, total=len(train_loader)):
            for inputs, targets, features, graph_input in train_loader:
                optimizer.zero_grad()
                if embedded_input:
                    inputs = inputs.permute(0,3,2,1) 
                pga_output, pgv_output, sa03_output, sa10_output, sa30_output = model(inputs.to(device), features.to(device), graph_input.to(device))
                loss = nn.MSELoss()(pga_output, targets[:, :, 0].to(device)) + nn.MSELoss()(pgv_output, targets[:, :, 1].to(device)) + nn.MSELoss()(sa03_output, targets[:, :, 2].to(device)) + nn.MSELoss()(sa10_output, targets[:, :, 3].to(device)) + nn.MSELoss()(sa30_output, targets[:, :, 4].to(device))
                loss.backward()
                optimizer.step()
                train_total_loss += loss.item()
            
            if epoch % 1 == 0:
                print(f"Train Epoch {epoch+1}, Loss: {train_total_loss/ len(train_loader)}")

            model.eval()
            val_total_loss = 0 
            with torch.no_grad():
                #for inputs, targets, features in tqdm(val_loader, total=len(val_loader)):
                for inputs, targets, features, graph_input in val_loader:
                    if embedded_input:
                        inputs = inputs.permute(0,3,2,1) 
                    pga_output, pgv_output, sa03_output, sa10_output, sa30_output = model(inputs.to(device), features.to(device), graph_input.to(device))
                    loss = nn.MSELoss()(pga_output, targets[:, :, 0].to(device)) + nn.MSELoss()(pgv_output, targets[:, :, 1].to(device)) + nn.MSELoss()(sa03_output, targets[:, :, 2].to(device)) + nn.MSELoss()(sa10_output, targets[:, :, 3].to(device)) + nn.MSELoss()(sa30_output, targets[:, :, 4].to(device))
                    val_total_loss += loss.item()
                
                avg_val_loss = val_total_loss/ len(val_loader)
                if epoch % 1 == 0:
                    print(f"Val Epoch {epoch+1}, Loss: {avg_val_loss}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_val_losses = [best_val_loss]
                    best_model_path = "./Checkpoint/model_embedding_"+str(embedded_input)+"_gcn_"+str(model_gcn)+"_fold_"+str(fold+1)+"_seed_"+str(seed)+".pth"
                    torch.save(model.state_dict(), best_model_path)
                    del best_model
                    best_model = type(model)()
                    best_model.load_state_dict(model.state_dict())
                    print(f"New best model saved for fold {fold+1} with val loss: {best_val_loss}")
                    epochs_without_improvement = 0  # Resetta il contatore poiché abbiamo trovato un miglioramento
                else:
                    epochs_without_improvement += 1
                best_losses_per_fold.append(best_val_losses)
            


            model.eval()
            if embedded_input:
                test_dataset = TensorDataset(enc_test_X, enc_test_target,enc_test_features, enc_test_graph)
            else:
                test_dataset = TensorDataset(torch.tensor(test_inputs, dtype=torch.float32), torch.tensor(test_targets, dtype=torch.float32), torch.tensor(test_features, dtype=torch.float32),  torch.tensor(test_graph_input, dtype=torch.float32))
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_total_loss=0

            with torch.no_grad():
                #for inputs, targets, features in tqdm(test_loader, total=len(test_loader)):
                for inputs, targets, features, graph_input in test_loader:
                    if embedded_input:
                        inputs = inputs.permute(0,3,2,1) 
                    pga_output, pgv_output, sa03_output, sa10_output, sa30_output = model(inputs.to(device), features.to(device), graph_input.to(device))
                    loss = nn.MSELoss()(pga_output, targets[:, :, 0].to(device)) + nn.MSELoss()(pgv_output, targets[:, :, 1].to(device)) + nn.MSELoss()(sa03_output, targets[:, :, 2].to(device)) + nn.MSELoss()(sa10_output, targets[:, :, 3].to(device)) + nn.MSELoss()(sa30_output, targets[:, :, 4].to(device))
                    test_total_loss += loss.item()
                
                avg_test_loss = test_total_loss/ len(test_loader)
                if epoch % 1 == 0 or epochs_without_improvement == 0:
                    print(f"Test Epoch {epoch+1}, Loss: {avg_test_loss}")
                    
            if epochs_without_improvement >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement.")
                    break
      
te_losses=[]
mse_lists=[]
rmse_lists=[]
mae_lists=[]
for fold in range(n_splits):
    for seed in seeds:
        del best_model
        best_model = type(model)()
        print("Model: Checkpoint/model_embedding_"+str(embedded_input)+"_gcn_"+str(model_gcn)+"_fold_"+str(fold+1)+"_seed_"+str(seed)+".pth")
        best_model.load_state_dict(torch.load("Checkpoint/model_embedding_"+str(embedded_input)+"_gcn_"+str(model_gcn)+"_fold_"+str(fold+1)+"_seed_"+str(seed)+".pth", map_location=torch.device(device)))
        best_model.to(device)               
        best_model.eval()

        if embedded_input:
            test_dataset = TensorDataset(enc_test_X, enc_test_target,enc_test_features, enc_test_graph)
        else:
            test_dataset = TensorDataset(torch.tensor(test_inputs, dtype=torch.float32), torch.tensor(test_targets, dtype=torch.float32), torch.tensor(test_features, dtype=torch.float32), torch.tensor(test_graph_input, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_total_loss=0

        with torch.no_grad():
            predictions = []
            mse_list = []
            rmse_list = []
            mae_list = []

            for inputs, targets, features, graph_input,  in tqdm(test_loader, total=len(test_loader)):
                if embedded_input:
                    inputs = inputs.permute(0,3,2,1) 
                pga_output, pgv_output, sa03_output, sa10_output, sa30_output = best_model(inputs.to(device), features.to(device), graph_input.to(device))
                predictions.append(torch.stack([pga_output, pgv_output, sa03_output, sa10_output, sa30_output]))
                loss = nn.MSELoss()(pga_output, targets[:, :, 0].to(device)) + nn.MSELoss()(pgv_output, targets[:, :, 1].to(device)) + nn.MSELoss()(sa03_output, targets[:, :, 2].to(device)) + nn.MSELoss()(sa10_output, targets[:, :, 3].to(device)) + nn.MSELoss()(sa30_output, targets[:, :, 4].to(device))
                test_total_loss += loss.item()
            
            avg_te_loss = test_total_loss/ len(test_loader)
            print(f"Loss: {avg_te_loss}")
            predictions_dropLast = torch.stack(predictions[:-1])
            predictions_dropLast=predictions_dropLast.permute(0,2,3,1)
            predictions_dropLast=predictions_dropLast.reshape(-1,predictions_dropLast.shape[2],predictions_dropLast.shape[3]).cpu()

            for i in range(5):
                test_targets_dropLast=test_targets[:predictions_dropLast.shape[0]]
                
                mse_list.append(mean_squared_error(test_targets_dropLast[:, :, i], predictions_dropLast[:, :, i]))
                rmse_list.append(mean_squared_error(test_targets_dropLast[:, :, i], predictions_dropLast[:, :, i], squared=False))
                mae_list.append(mean_absolute_error(test_targets_dropLast[:, :, i], predictions_dropLast[:, :, i]))

            # Output results
            print('All averages:')
            print('MSE score:', np.mean(mse_list))
            print('RMSE score:', np.mean(rmse_list))
            print('MAE score:', np.mean(mae_list))
            mse_lists.append(mse_list)
            rmse_lists.append(rmse_list)
            mae_lists.append(mae_list)
            te_losses.append(avg_te_loss)
print("loss mean:", np.array(te_losses).mean())
# if not embedded_input:
print("mse mean: ", np.array(mse_lists).mean(),
      "PGA", f"{np.array([i[0] for i in mse_lists]).mean():.3f}", u"\u00B1", f"{np.array([i[0] for i in mse_lists]).std():.3f}",
      "PGV", f"{np.array([i[1] for i in mse_lists]).mean():.3f}", u"\u00B1", f"{np.array([i[1] for i in mse_lists]).std():.3f}",
      "PSA03", f"{np.array([i[2] for i in mse_lists]).mean():.3f}", u"\u00B1", f"{np.array([i[2] for i in mse_lists]).std():.3f}",
      "PSA1", f"{np.array([i[3] for i in mse_lists]).mean():.3f}", u"\u00B1", f"{np.array([i[3] for i in mse_lists]).std():.3f}",
      "PSA3", f"{np.array([i[4] for i in mse_lists]).mean():.3f}", u"\u00B1", f"{np.array([i[4] for i in mse_lists]).std():.3f}")

print("rmse mean: ", np.array(rmse_lists).mean(),
      "PGA", f"{np.array([i[0] for i in rmse_lists]).mean():.3f}", u"\u00B1", f"{np.array([i[0] for i in rmse_lists]).std():.3f}",
      "PGV", f"{np.array([i[1] for i in rmse_lists]).mean():.3f}", u"\u00B1", f"{np.array([i[1] for i in rmse_lists]).std():.3f}",
      "PSA03", f"{np.array([i[2] for i in rmse_lists]).mean():.3f}", u"\u00B1", f"{np.array([i[2] for i in rmse_lists]).std():.3f}",
      "PSA1", f"{np.array([i[3] for i in rmse_lists]).mean():.3f}", u"\u00B1", f"{np.array([i[3] for i in rmse_lists]).std():.3f}",
      "PSA3", f"{np.array([i[4] for i in rmse_lists]).mean():.3f}", u"\u00B1", f"{np.array([i[4] for i in rmse_lists]).std():.3f}")

print("mae mean: ", np.array(mae_lists).mean(),
      "PGA", f"{np.array([i[0] for i in mae_lists]).mean():.3f}", u"\u00B1", f"{np.array([i[0] for i in mae_lists]).std():.3f}",
      "PGV", f"{np.array([i[1] for i in mae_lists]).mean():.3f}",u"\u00B1", f"{np.array([i[1] for i in mae_lists]).std():.3f}",
      "PSA03", f"{np.array([i[2] for i in mae_lists]).mean():.3f}",u"\u00B1", f"{np.array([i[2] for i in mae_lists]).std():.3f}",
      "PSA1", f"{np.array([i[3] for i in mae_lists]).mean():.3f}",u"\u00B1", f"{np.array([i[3] for i in mae_lists]).std():.3f}",
      "PSA3", f"{np.array([i[4] for i in mae_lists]).mean():.3f}",u"\u00B1", f"{np.array([i[4] for i in mae_lists]).std():.3f}")