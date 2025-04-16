from Margs1 import args_parser
from Mget_data1 import nn_seq, setup_seed
from Mmodels import SAEG_Net
from Muilts1 import practice,practice_test
from Mget_data import adj2coo
setup_seed(42)
def main():
    args = args_parser()
    train_loader,val_loader,test_loader, scaler, edge_index = nn_seq(args)
    practice(args, train_loader, edge_index)
    practice_test(args, test_loader,scaler, edge_index)
if __name__ == '__main__':
    main()
