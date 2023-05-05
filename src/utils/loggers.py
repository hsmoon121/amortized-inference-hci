import os, datetime
from torch.utils.tensorboard import SummaryWriter

class Logger(object):
    def __init__(self, name, last_step=0, board=True, board_path="./data/board"):
        self.name = name
        self.train_step = last_step
        self.board = board
        if self.board:
            self._set_tb_writer(board_path)

    def _set_tb_writer(self, board_path):
        os.makedirs(f"{board_path}/{self.name}", exist_ok=True)
        datetime_str = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.writer = SummaryWriter(f"{board_path}/{self.name}/{datetime_str}")

    def step(self):
        self.train_step += 1

    def write_scalar(self, verbose=False, **kwargs):
        if self.board:
            for key, value in kwargs.items():
                self.writer.add_scalar(key, value, self.train_step)
