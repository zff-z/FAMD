from siamese import *
from meta_train import *

inputsize = 57
hiddensize = 114
outputsize = 16
model = SiameseNet(inputsize, hiddensize, outputsize)
criterion = ContrastiveLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

dataset = SiameseDataset('./dataset/Test/CIC2019_per.csv')
train_size = int(len(dataset) * 0.8)
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
dataset_loader = DataLoader(dataset, batch_size=32, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

for epoch in range(10):
    for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
        optimizer.zero_grad()
        output1 = model(anchor)
        output2 = model(positive)
        output3 = model(negative)
        loss = criterion(output1, output2, output3)
        loss.backward()
        optimizer.step()

    print('Fine_tune Epoch: {} Loss: {:.6f}'.format(epoch + 1, loss.item()))

#就是要让训练的时候不知道最终的支持集和query是啥，微调只能说是拿到一些存在的类。
# torch.save(model.state_dict(), './model/Siamese_per_API_opcode.pth')
num_tasks = 5
num_query = 2
num_support = 5
num_classes = 5
epochs = 10
criterion_test = nn.CrossEntropyLoss()
for epoch in range(epochs):
    # train_loss = train(model, optimizer, criterion, dataset, num_classes, num_support, num_query, num_tasks)
    val_loss, val_acc = evaluate(model, criterion_test, test_dataset, num_classes, num_support, num_query, num_tasks)
    print(f'Epoch {epoch + 1}:  Val Loss={val_loss:.4f} Val Acc={val_acc:.4f}')