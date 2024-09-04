
#the original algorithm in tensorflow https://github.com/kaiqzhan/admm-pruning
import argparse

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms 
from torch.utils.data import random_split  

from model import LeNet

prune_percent = 92

def projection(weight_arr,percent = 10):
    pcen = np.percentile(abs(weight_arr),percent)
    print ("percentile " + str(pcen))
    weight_arr[abs(weight_arr)< pcen] = 0
    return weight_arr

def prune_weight(w,percent):
  # to work with admm, we calculate percentile based on all elements instead of nonzero elements.
  pcen = np.percentile(abs(w.detach().numpy()),percent)
  print ("percentile " + str(pcen))
  above_th = abs(w)>= pcen
  w=w*above_th
  return above_th,w

def apply_prune(W,prune_percent):
    masks=[]
    for w in W:
        print ("before pruning #non zero parameters " + str(np.sum(w=0)))
        before = np.sum(w!=0)
        mask,w_pruned = prune_weight(w,prune_percent)
        after = np.sum(w_pruned!=0)
        print ("pruned "+ str(before-after))
        print ("after prunning #non zero parameters " + str(np.sum(w_pruned!=0)))
        w.data=w_pruned
        masks.append(mask)
    return masks

def loss_with_l1(outputs,targets,model,lossf,eps=0.01):
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.sum(torch.abs(param))
    return lossf(outputs,targets)+eps*l2_reg

def loss_with_l2(outputs,targets,model,lossf,eps=0.00005):
    l2_reg = 0
    for param in model.parameters():
        l2_reg += torch.sum(torch.square(param))
    return lossf(outputs,targets)+eps*l2_reg


def loss_with_l2_A(outputs,targets,model,U,Z,lossf,eps=0.0001):
    l2_reg = 0
    for i,param in enumerate(model.parameters()):
        l2_reg += torch.sum(torch.square(param-Z[i]+U[i]))
    return lossf(outputs,targets)+eps*l2_reg

def print_accuracy(model,dataloader):
    correct_prediction = []
    for inputs, targets in dataloader:
        correct_prediction.append( (targets==model(inputs).argmax(dim=1)).sum().float() )
    p=np.array(correct_prediction).mean()
    print("test accuracy %g"%p)
    return p

def main(args):
    num_epochs=args.num_epochs
    threash=args.threshold
    batchsize=args.batchsize
    lr=args.learningrate
    data_dir=args.data_dir
    max_iteration=args.max_iteration    
    
    tr=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    if(args.dataset=="CIFAR10"):
        dataset_train = datasets.CIFAR10(data_dir, train=True, download=True,transform=tr)
        dataset_test = datasets.CIFAR10(data_dir, train=False, transform=tr)
        label_num=10
    elif(args.dataset=="CIFAR100"):
        dataset_train = datasets.CIFAR100(data_dir, train=True, download=True,transform=tr)
        dataset_train2 = datasets.CIFAR100(data_dir, train=True, download=True,transform=tr)
        dataset_test = datasets.CIFAR100(data_dir, train=False,transform=tr)
        #subset_size = len(dataset) // 3  
        #subset1, subset2, subset3 = random_split(dataset, [subset_size, subset_size, dataset_size - 2 * subset_size])  
        label_num=100
    else:
        tr=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        dataset_train = datasets.MNIST(data_dir, train=True, download=True,transform=tr)
        dataset_test = datasets.MNIST(data_dir, train=False,transform=tr)
        dataset_train2 =dataset_train
        label_num=10

    dataloader_train = DataLoader(dataset_train, batchsize, shuffle=True)
    dataloader_train2 = DataLoader(dataset_train2, batchsize, shuffle=True)
    dataloader_test = DataLoader(dataset_test, batchsize, shuffle=True)    

    model = LeNet(label_num)

    W=[w.data for w in model.parameters()]
    U=[torch.zeros_like(w) for w in W]
    Z=[torch.zeros_like(w) for w in W]
    
    #grads=[w.data.grad() for w in params]

    criterion = nn.CrossEntropyLoss()

    if(args.opt=="Adam"):
        optimizer = optim.Adam(list(model.parameters()),  lr = 1e-4)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.01)

    #initial training
    for epoch in range(num_epochs):
        for i, d in enumerate(tqdm(dataloader_train)):
            inputs, targets = d
            optimizer.zero_grad()

            outputs = model(inputs)

            #https://python-code.dev/articles/242716633
#            outputs_softmax = nn.functional.log_softmax(outputs, dim=1)  
#            print("outputs",outputs,"shape",outputs.shape)
#            print("outputs_softmax",outputs_softmax,"shape",outputs_softmax.shape)
#            print("label",targets)
            
            loss = loss_with_l2(outputs,targets,model,criterion)
            loss.backward()
            optimizer.step()
            if(i>max_iteration):
                break

    print_accuracy(model,dataloader_test)

    torch.save(model,"lenet_5_simple_model.pt")

    Z=[projection(w,threash) for w in W]

    #ADMM training
    for epoch in range(num_epochs):
        for j, d in enumerate(tqdm(dataloader_train2)):
            inputs, targets = d
        
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            outputs = model(inputs)
            
            loss = loss_with_l2_A(outputs,targets,model,U,Z,criterion)
            loss.backward()
            optimizer.step()

        for i ,w in enumerate(W):
            Z[i]=projection(w+U[i],threash)
            U[i]=U[i]+w-Z[i]

        print_accuracy(model,dataloader_test)

    masks=apply_prune(W,threash)
    # Specify the parameters to update  
    #optimizer01 = optim.SGD([  {'params': nonpruned_w} ], lr=0.01)  
    torch.save(model,"lenet_5_simple_model.pt")    

    print ("start retraining after pruning")

    for i, d in enumerate(tqdm(dataloader_train2)):
      inputs, targets = d
    #for inputs, targets in dataloader_train2:
      optimizer.zero_grad()
      outputs = model(inputs)

      loss = loss_with_l2(outputs,targets,model,criterion)
      loss.backward()
      #https://ohke.hateblo.jp/entry/2019/12/07/230000
      #param_b = (param_b - param_b.grad * learning_rate).detach().requires_grad_()
      #   with torch.no_grad():  
      for p,m in zip(model.parameters(),masks):
         p=(p-p.grad*m*lr).detach().requires_grad_()
      
      #optimizer01.step()
      #apply_gradient_op.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
      
      if i % 1000 == 0:
        print_accuracy(model,dataloader_train)


    print_accuracy(model,dataloader_test)
    for w in W:
        print(np.sum(w.data!=0))

    torch.save(model,"lenet_5_pruned_model.pt")
    #torch.save(model.state_dict(),"lenet_5_pruned_model.pth")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,default='data', help='Directory for storing input data')
  parser.add_argument('-n', '--num_epochs',default=10,type=int)
  parser.add_argument('-bs', '--batchsize',default=4,type=int)
  parser.add_argument('-ds', '--dataset',default="mnist")    
  parser.add_argument('-l', '--learningrate',default=1e-4,type=float)
  parser.add_argument('-th', '--threshold',default=0.001,type=float)
  parser.add_argument('-opt', '--opt',default="SGD")    
  parser.add_argument('-m', '--max_iteration',default=1000000,type=int)

  FLAGS, unparsed = parser.parse_known_args()
  args = parser.parse_args()
  main(args)
  
