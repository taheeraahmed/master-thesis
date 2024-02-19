from datasets import ChestXRay14BBoxDatast
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import set_up

if __name__ == "__main__":
    root_folder = '/cluster/home/taheeraa/datasets/chestxray-14/'
    set_up()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = ChestXRay14BBoxDatast(root_folder, transform=transform)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1)

    for i, sample in enumerate(dataloader):
        print(i, sample['img'].size(), sample['bbox'])
        if i == 3:
            break
