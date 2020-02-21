import torch
import os 
from torch.utils.tensorboard import SummaryWriter

class Logger(object): 
    
    def __init__(self, log_dir, comment=''): 
        self.writer = SummaryWriter(log_dir=log_dir, comment=comment)
        self.imgs_dict = {}
        
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
    
    def get_dir(self): 
        return self.writer.get_logdir()
    
    def log_model_state(self, model, step): 
        path = os.path.join(self.writer.get_logdir(), type(model).__name__ + '_%d.pt' % step)
        torch.save(model.state_dict(), path)
    
    def log_video(self, tag, global_step=None, img_tns=None,finished_video=False, video_tns=None, debug=False):
        '''
            Logs video to tensorboard. 
            Video_tns will be empty. If given image tensors, then when finished_video = True, the video of the past tensors will be made into one video. 
            If vide_tns is not empty, then that will be marked the video and the other arguments will be ignored. 
        ''' 
        if debug:
            import pdb; pdb.set_trace()
        if img_tns is None and video_tns is None:
            if not finished_video or tag not in self.imgs_dict.keys(): 
                return None 
            lst_img_tns = self.imgs_dict[tag] 
            self.writer.add_video(tag, torch.tensor(lst_img_tns), global_step=global_step, fps=4)
            self.writer.flush()
            self.imgs_dict[tag] = []
            return None 
        elif video_tns is not None: 
            self.writer.add_video(tag, video_tns, global_step=global_step, fps=4)
            self.writer.flush()
            return None 
        
        if tag in self.imgs_dict.keys(): 
            lst_img_tns = self.imgs_dict[tag] 
        else: 
            lst_img_tns = []
            self.imgs_dict[tag] = lst_img_tns 
    
        lst_img_tns.append(img_tns)
        
        if finished_video: 
            self.writer.add_video(tag, torch.tensor(lst_img_tns), global_step=global_step, fps=4)
            self.writer.flush()
            self.imgs_dict[tag].clear()
        
    
    def close(self): 
        self.writer.close()

