import torch
import hydra
import os,sys

sys.path.append('..')


from bridge.runners.ipf import IPFSequential


# SETTING PARAMETERS

@hydra.main(config_path="./conf", config_name="config")
def main(args):

    print('Directory: ' + os.getcwd())
    ipf = IPFSequential(args)
    ipf.train()
    

if __name__ == '__main__':
    main()  
