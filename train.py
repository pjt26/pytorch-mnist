import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.init as nnInit
from model import LeNet5

kwargs = {'num_workers': 4, 'pin_memory': True} if torch.cuda.is_available() else {}

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081, ))
        ])),
    batch_size=64, shuffle=True,**kwargs)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/', train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081, ))
        ])),
    batch_size=64, shuffle=False,**kwargs)

# create model instance
model = LeNet5()
# initialize weights and bias
for m in model.modules():
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nnInit.xavier_normal(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()

if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.SGD(model.parameters(),
    lr=0.1,
    momentum=0.9,
    weight_decay=1e-4,
    nesterov=True)

if torch.cuda.is_available():
    criterion = nn.CrossEntropyLoss().cuda()
else:
    criterion = nn.CrossEntropyLoss()




def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader) :
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print'[training] loss', loss.data[0]

def test(epoch):
    model.eval ()
    correct = 0
    for batch_idx, (data, target) in enumerate(test_loader) :
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        output = model(data)
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()
    print'[testing] accuracy', 100*correct/len(test_loader.dataset)
    return model,correct

if __name__ == "__main__":
    maxCorrect = 0 
    for epoch in range(1, 6):
        train(epoch)
        model,correct = test(epoch)
        if correct >= maxCorrect :
            bestModel = model
            maxCorrect = correct
            print'found better model ! accuracy: ',correct/100.0
    torch.save(bestModel,'model.pkl')