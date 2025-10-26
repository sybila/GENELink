from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torch.optim import Adam
from scGNN import GENELink
from torch.optim.lr_scheduler import StepLR
from utils import scRNADataset, load_data, adj2saprse_tensor
import pandas as pd
import numpy as np
import random
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
parser.add_argument('--epochs', type=int, default= 20, help='Number of epoch.')
parser.add_argument('--num_head', type=list, default=[3,3], help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--hidden_dim', type=int, default=[128,64,32], help='The dimension of hidden layer')
parser.add_argument('--output_dim', type=int, default=16, help='The dimension of latent layer')
parser.add_argument('--batch_size', type=int, default=256, help='The size of each batch')
parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
parser.add_argument('--seed', type=int, default=8, help='Random seed')
parser.add_argument('--Type',type=str,default='dot', help='score metric')
parser.add_argument('--flag', type=str, default="", help='the identifier whether to conduct causal inference')
parser.add_argument('--reduction',type=str,default='concate', help='how to integrate multihead attention')

# ADDED 
parser.add_argument('--expr-data', type=str, help="Path to expression data (should be in genes x cells format)")
parser.add_argument('--tf-index', type=str, help="Path to tf file containing names of the tfs and indices (columns `TF` and `index`)")
parser.add_argument('--target-index', type=str, help="Path to tf file containing names of the tfs and indices")

parser.add_argument('--train', type=str, help="path to train data")

parser.add_argument('--validation', type=str, help="path to validation data (is it needed?)")

parser.add_argument('--model-dir', type=str, help="folder to store model")
parser.add_argument('--model-path', type=str, help = "name of model file")

parser.add_argument('--result-path', type=str, help="path to where to store the result of validation data eval")
parser.add_argument('--test-data-deindexed', type=str, help="path to validation data with genes instead of indices, included for easier processing")


if __name__ == "__main__":
    args = parser.parse_args()

    seed = args.seed

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    causal_flag = True if args.flag == "TRUE" else False

    exp_file = args.expr_data
    tf_file = args.tf_index
    target_file = args.target_index
    train_file = args.train

    test_file = args.validation
    model_path = args.model_path
    model_dir = args.model_dir


    data_input = pd.read_csv(exp_file,index_col=0)
    loader = load_data(data_input)
    feature = loader.exp_data()
    tf = pd.read_csv(tf_file,index_col=0)['index'].values.astype(np.int64)
    target = pd.read_csv(target_file,index_col=0)['index'].values.astype(np.int64)
    feature = torch.from_numpy(feature)
    tf = torch.from_numpy(tf)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_feature = feature.to(device)
    tf = tf.to(device)


    train_data = pd.read_csv(train_file).values
    test_data = pd.read_csv(test_file).values

    train_load = scRNADataset(train_data, feature.shape[0], flag=causal_flag)
    adj = train_load.Adj_Generate(tf,loop=args.loop)

    adj = adj2saprse_tensor(adj)

    train_data = torch.from_numpy(train_data)
    val_data = torch.from_numpy(test_data)

    model = GENELink(
        input_dim = feature.size()[1],
        hidden1_dim = args.hidden_dim[0],
        hidden2_dim = args.hidden_dim[1],
        hidden3_dim = args.hidden_dim[2],
        output_dim = args.output_dim,
        num_head1 = args.num_head[0],
        num_head2 = args.num_head[1],
        alpha = args.alpha,
        device = device,
        type = args.Type,
        reduction = args.reduction
    )

    adj = adj.to(device)
    model = model.to(device)
    train_data = train_data.to(device)
    test_data = val_data.to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)


    for epoch in range(args.epochs):
        running_loss = 0.0

        for train_x, train_y in DataLoader(train_load, batch_size=args.batch_size, shuffle=True):
            model.train()
            optimizer.zero_grad()

            if causal_flag:
                train_y = train_y.to(device)
            else:
                train_y = train_y.to(device).view(-1, 1)


            # train_y = train_y.to(device).view(-1, 1)
            pred = model(data_feature, adj, train_x)

            #pred = torch.sigmoid(pred)
            if causal_flag:
                pred = torch.softmax(pred, dim=1)
            else:
                pred = torch.sigmoid(pred)

            loss_BCE = F.binary_cross_entropy(pred, train_y)

            loss_BCE.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss_BCE.item()


        model.eval()
        score = model(data_feature, adj, test_data)
        if causal_flag:
            score = torch.softmax(score, dim=1)
        else:
            score = torch.sigmoid(score)


        print('Epoch:{}'.format(epoch + 1),
                'train loss:{}'.format(running_loss))


    print(f"{model_dir}/{model_path}")
    torch.save(model.state_dict(), f"{model_dir}/{model_path}")


    model.eval()
    print("evaluating")
    res_score = model(data_feature, adj, test_data)
    sig_score = torch.sigmoid(res_score)
    print(f"saving to {args.result_path}")

    testdata = pd.read_csv(args.test_data_deindexed)
    testdata.loc[:, "Score_original"] = res_score.detach()
    testdata.loc[:, "Score"] = sig_score.detach()

    testdata.to_csv(args.result_path)
