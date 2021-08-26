
from pytorch_lightning.loggers import NeptuneLogger as _NeptuneLogger

from pytorch_lightning.loggers import CSVLogger as _CSVLogger


class Logger:
    def log_metrics(self, metric_dict, step, save=False):
        pass

    def log_hparams(self, hparams_dict):
        pass


class CSVLogger(Logger):

    def __init__(self, directory='./', name='logs', save_stride=1):
        self.logger = _CSVLogger(directory, name=name)
        self.count = 0
        self.stride = save_stride

    def log_metrics(self, metrics, step=None,save=False):
        self.count += 1
        self.logger.log_metrics(metrics, step=step)
        if self.count % self.stride == 0:
            self.logger.save()
            self.logger.metrics = []

        if self.count > self.stride * 10:
            self.count = 0
            
        if save:
            self.logger.save()
    
    def log_hparams(self, hparams_dict):
        self.logger.log_hyperparams(hparams_dict)
        self.logger.save()


class NeptuneLogger(Logger):
    def __init__(self, project_name, api_key, save_folder='./'):
        self.directory = save_folder
        self.logger = _NeptuneLogger(api_key=api_key, project_name=project_name)

    def log_metrics(self, metrics, step=None):
        self.logger.log_metrics(metrics,step=step)

    def log_hparams(self, hparams_dict):
        self.logger.log_hyperparams(hparams_dict)

