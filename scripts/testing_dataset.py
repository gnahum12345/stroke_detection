from sd.infra.dataset import AtlasDataset

ad = AtlasDataset('/datavol/brain_data/atlas')

sample = ad[0]
print(f'Expected type of sample: dict\nActual type of sample: {type(sample)}')
print(f'Expected keys of smaple: ["scan", "mask"]\nActual keys of sample: {sample.keys()}')
print(f'Expected shape of scan: (197, 1, 233, 189)\nActual shape of scan: {sample["scan"].shape}')
print(f'Expected shape of masks: (197, 1, 233, 189)\n Actual shape of masks: {sample["mask"].shape}')
print(f'Expected type of scan: torch.float64 tensor \nActual type of scan: {type(sample["scan"])}')
print(f'Expected type of mask: torch.float64 tensor \nActual type of mask: {type(sample["mask"])}')

