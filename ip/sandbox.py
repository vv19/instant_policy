import torch
import ip
import torch_geometric
import lightning

if __name__ == '__main__':
    print(torch.__version__)
    print(torch.version.cuda)
    print(torch_geometric.__version__)
    print(lightning.__version__)
