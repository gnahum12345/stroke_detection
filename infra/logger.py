import torch
import os 
from torch.utils.tensorboard import SummaryWriter

class Logger(object): 
    
    def __init__(self, log_dir, comment=''): 
        self.writer = SummaryWriter(log_dir=log_dir, comment=comment)

    def scalar_summary(self, tag, value, step): 
        self.writer.add_scalar(tag, value, global_step=step)
        self.writer.flush()
    
    def combined_scalars_summary(self, main_tag, tag_scalar_dict, step): 
        self.writer.add_scalars(main_tag, tag_scalar_dict, step)
        self.writer.flush()

    def log(self, tag, text_string, step=0): 
        self.writer.add_text(tag, text_string, step)
        self.writer.flush()
        
    def log_model(self, model, inputs): 
        self.writer.add_graph(model, inputs)
        self.writer.flush()

    def log_model_state(self, model, step): 
        path = os.path.join(self.writer.get_logdir(), type(model).__name__ + '_%d.pt' % step)
        torch.save(model.state_dict(), path)
        
        
    def close(self): 
        self.writer.close()

