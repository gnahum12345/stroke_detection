from sd.infra.dataset import * 

ad = AtlasDataset('/datavol/brain_data/atlas')
print('ad: {}'.format(len(ad)))

last_res = ad[0]
print(last_res.shape)
for i in range(1,len(ad)): 
    print(i)
    current_res = ad[i]
    assert last_res.shape ==  current_res.shape, 'There is a descripency between scans'

    last_res = current_res

