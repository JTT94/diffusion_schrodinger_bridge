import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import time


class CacheLoader(Dataset):
    def __init__(self, fb,
                 sample_net,
                 dataloader_b,
                 num_batches,
                 langevin,
                 n,
                 mean, std,
                 batch_size, device='cpu',
                 dataloader_f=None,
                 transfer=False):

        super().__init__()
        start = time.time()
        shape = langevin.d
        num_steps = langevin.num_steps
        self.data = torch.zeros(
            (num_batches, batch_size*num_steps, 2, *shape)).to(device)  # .cpu()
        # self.steps_data = torch.zeros(
        #     (num_batches, batch_size*num_steps, 1), dtype=torch.long).to(device)  # .cpu() # steps
        self.steps_data = torch.zeros(
            (num_batches, batch_size*num_steps, 1)).to(device)  # .cpu() # steps
        with torch.no_grad():
            for b in range(num_batches):
                if fb == 'b':
                    batch = next(dataloader_b)[0]
                    batch = batch.to(device)
                elif fb == 'f' and transfer:
                    batch = next(dataloader_f)[0]
                    batch = batch.to(device)
                else:
                    batch = mean + std * \
                        torch.randn((batch_size, *shape), device=device)

                if (n == 1) & (fb == 'b'):
                    x, out, steps_expanded = langevin.record_init_langevin(
                        batch)
                else:
                    x, out, steps_expanded = langevin.record_langevin_seq(
                        sample_net, batch, ipf_it=n)

                # store x, out
                x = x.unsqueeze(2)
                out = out.unsqueeze(2)
                batch_data = torch.cat((x, out), dim=2)
                flat_data = batch_data.flatten(start_dim=0, end_dim=1)
                self.data[b] = flat_data

                # store steps
                flat_steps = steps_expanded.flatten(start_dim=0, end_dim=1)
                self.steps_data[b] = flat_steps

        self.data = self.data.flatten(start_dim=0, end_dim=1)
        self.steps_data = self.steps_data.flatten(start_dim=0, end_dim=1)

        stop = time.time()
        print('Cache size: {0}'.format(self.data.shape))
        print("Load time: {0}".format(stop-start))

    def __getitem__(self, index):
        item = self.data[index]
        x = item[0]
        out = item[1]
        steps = self.steps_data[index]

        return x, out, steps

    def __len__(self):
        return self.data.shape[0]
