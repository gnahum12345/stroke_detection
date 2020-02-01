import matplotlib.pyplot as plt 
from textwrap import wrap 

def show_images(images, save=None, size=None): 
    """
    Display a list of images with matplotlib and save them afterwards. 
    
    Params: 
        images: a tensor that contains a list of lists of numpy arrays.
            i.e. [[np.arr], [np.arr], ....]
        save: file path to where the images will be saved, if it is None, then it won't save. 
        size: Size of the figure, if None then a default figure will be used. 
    """
    assert len(images) > 0, "images should contain at least 1 element"
    assert len(images[0].shape) == 3, "each image should contain 3 fiels (w,h, c)"
    
    fig, ax = plt.subplots(nrows=images[0].shape(-1), ncols=len(images))
    
    for i in range(len(images)): 
        for j in range(len(images[0].shape[-1])): 
            ax[i,j].imshow(images[i][:,:,j], cmap='gray')
    
    plt.show()
    

    
class ImagePlot():
    
    def __init__(self, volumes, titles=None, nrows=1):
        ''' 
            Volumes is a list of 3D images and titles is the correpsonding titles. 
        '''
        self.v = volumes
        self.titles = titles
        self.nrows = nrows

    def render(self): 
        assert len(self.v) > 0, 'Volumes should be a list of 3D images and should not be empty.'
        return self.multi_slice_viewer()
    
    def remove_keymap_conflicts(self, new_keys_set):
        for prop in plt.rcParams:
            if prop.startswith('keymap.'):
                keys = plt.rcParams[prop]
                remove_list = set(keys) & new_keys_set
                for key in remove_list:
                    keys.remove(key)

    def multi_slice_viewer(self):
        self.remove_keymap_conflicts({'j', 'k', 'l', 'h', 's', 't', 'left', 'right', 'up', 'down'})
        plt.rcParams["axes.titlesize"] = 8
        volumes = self.v
        self.ncols = len(volumes)//self.nrows 
        if self.ncols != len(volumes)/ self.nrows: 
            self.ncols += 1 
          
        fig, axes = plt.subplots(ncols=self.ncols, nrows=self.nrows)
        
        for i in range(len(volumes)): 
            if len(volumes) != 1: 
                if self.nrows != 1 and self.ncols != 1:
                    ax = axes[i%self.nrows, i//self.nrows]
                else: 
                    ax = axes[i] 
            else: 
                ax = axes 
            ax.volume = volumes[i]
            ax.index = volumes[i].shape[0] // 2
            if self.titles is not None: 
                ax.set_title('\n'.join(wrap(self.titles[i] + ": %d" % ax.index)))
            ax.imshow(volumes[i][ax.index], cmap='gray')
        
        if self.nrows != 1: 
            for i in range(len(volumes), self.nrows*self.ncols): 
                axes[i%self.nrows, i//self.nrows].set_visible(False)
        
        fig.canvas.mpl_connect('key_press_event', self.process_key)
        fig.tight_layout(h_pad=0.4)

    def process_key(self, event):
        fig = event.canvas.figure
        axes = fig.axes
        if event.key in ('left', 'j') :
            self.previous_slice(axes)
        elif event.key in ('right', 'k'):
            self.next_slice(axes)
#         elif event.key in ('up', 'l', 'down', 'h'): 
        elif event.key == 't': 
            self.transpose_images(fig)
        elif event.key == 's': 
            self.save()
        fig.canvas.draw()

    def transpose_images(self, fig): 
        axes = fig.axes
        for i in range(len(axes)): 
            axes[i].volume = axes[i].volume.T
           
        fig.axes = axes
        self.next_slice(axes)
        self.previous_slice(axes)
        fig.canvas.draw()
    
    def save(self): 
        path = './'
        for t in self.titles: 
            tmp = '_'.join(t.split(' '))
            path = path + tmp + '-'
        path = path[:-1] + '.png'
        plt.savefig(path)
        
    def previous_slice(self, axes):
        for ax in axes: 
            volume = ax.volume
            t = ax.get_title().split(':')[0]
            ax.set_title(t + ': %d' % ax.index)
            ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
            ax.images[0].set_array(volume[ax.index])

    def next_slice(self, axes):
        for ax in axes: 
            volume = ax.volume
            ax.index = (ax.index + 1) % volume.shape[0]
            t = ax.get_title().split(':')[0]
            ax.set_title(t + ': %d' % ax.index)
            ax.images[0].set_array(volume[ax.index])
        return axes

    