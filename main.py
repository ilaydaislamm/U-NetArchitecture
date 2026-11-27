import torch.optim
from engine import train_model
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from Unet import UNet
from preprocess import create_weight_maps,train_validation_split
from dataset import create_data_loaders
if __name__ == "__main__":

    train_root = r"C:\Users\omen\PycharmProjects\UNetArchitecture\stage1_train"
    out_dir = "stage1_weightmaps"
    test_path = r"C:\Users\omen\PycharmProjects\UNetArchitecture\stage1_test"

    weight_dir = create_weight_maps(train_root,out_dir)


    split_file = train_validation_split(train_root,0.2)

    train_loader , val_loader,test_loader = create_data_loaders(train_root,weight_dir,test_path,split_file)


    model = UNet(in_channels=3,out_classes=2)

    loss = CrossEntropyLoss(reduction="none")
    optimizer = Adam(params = model.parameters(),lr = 1e-3)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    EPOCH = 10
    for epoch in range(EPOCH):
        result = train_model(train_loader,model,loss,optimizer,device)
        print(result)

    # Model, Optimizer ve Eğitim Ayarlarını Kaydetme (Save Checkpoint)
    checkpoint = {
        'epoch': EPOCH,  # Kaldığınız epoch sayısı
        'model_state_dict': model.state_dict(),  # Öğrenilmiş ağırlıklar
        'optimizer_state_dict': optimizer.state_dict(),  # Optimizer parametreleri
        'model_config': {  # Modeli başka yerde sıfırdan kurmak için gereken parametreler
            'in_channels': 3,
            'out_classes': 2
        }
    }

    # Dosyayı diske kaydet
    torch.save(checkpoint, 'unet_checkpoint.pth')
    print("Model ve eğitim durumu başarıyla 'unet_checkpoint.pth' dosyasına kaydedildi!")
