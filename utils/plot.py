import matplotlib.pyplot as plt 


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
    

    
    