import os
import torch
import time
import matplotlib.pyplot as plt
from dataloader import get_dataloader
from dagmm import DAGMM
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR
from utils import EarlyStop


def plot_loss_moment(losses,args):
    _, ax = plt.subplots(figsize=(10,5), dpi=80)
    ax.plot(losses, 'blue', label='train', linewidth=1)
    ax.set_title('Loss change in training')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    ax.legend(loc='upper right')
    plt.savefig('./result/loss_dagmm_{}.png'.format(args.n_gmm))
    
    
def train(args, train_loader, valid_loader):
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DAGMM(args)
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(),args.lr, amsgrad=True)
    scheduler = MultiStepLR(optim, [5, 8], 0.1)
    
    # best_loss = 1e+5

    loss_total = 0
    recon_error_total = 0
    e_total = 0
    p_total = 0

    valid_loss_total = 0
    valid_recon_error_total = 0
    valid_e_total = 0
    valid_p_total = 0

    loss_plot = []
    early_stop = EarlyStop(patience=10)

    for epoch in range(args.epochs):
        for step, (input_data, _) in enumerate(train_loader):
            input_data = input_data.to(device)
            model.train()
            optim.zero_grad()
            input_data = input_data.squeeze(1)
            enc,dec,z,gamma = model(input_data)
            input_data,dec,z,gamma = input_data.cpu(),dec.cpu(),z.cpu(),gamma.cpu()
            loss, recon_error, e, p = model.loss_func(input_data, dec, gamma, z)
    
            loss_total += loss.item()
            recon_error_total += recon_error.item()
            e_total += e.item()
            p_total += p.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optim.step()
            
            if (step+1) % args.print_iter == 0:
                log = "Epoch [{}/{}], Iter [{}/{}], lr {} ".format(epoch+1, args.epochs, step+1, len(train_loader), optim.param_groups[0]['lr'])
                
                log+= 'loss {:.2f}, recon_error {:.2f}, energy {:.2f}, p_diag {:.2f}'.format(loss_total/args.print_iter,
                                                                    recon_error/args.print_iter,e_total/args.print_iter, p_total/args.print_iter)
                loss_plot.append(loss_total/args.print_iter)
                print(log)
                loss_total = 0
                recon_error_total = 0 
                e_total = 0
                p_total = 0

        with torch.no_grad():
            model.eval()
            for input_data,_ in valid_loader:
                input_data = input_data.to(device)
                input_data = input_data.squeeze(1)
                enc,dec,z,gamma = model(input_data)
                input_data,dec,z,gamma = input_data.cpu(),dec.cpu(),z.cpu(),gamma.cpu()
                loss, recon_error, e, p = model.loss_func(input_data, dec, gamma, z)

                valid_loss_total += loss.item()
                valid_recon_error_total += recon_error.item()
                valid_e_total += e.item()
                valid_p_total += p.item()

            print('[Dev] loss {:.2f}, recon_error {:.2f}, energy {:.2f} p_diag {:.2f}'.format(valid_loss_total/len(valid_loader), 
                                                                            valid_recon_error_total/len(valid_loader), 
                                                                            valid_e_total/len(valid_loader), valid_p_total/len(valid_loader)))            
            if (early_stop(valid_loss_total/len(valid_loader), model, optim)):
                plot_loss_moment(loss_plot,args)
                exit(0)
                break
            valid_loss_total = 0
            valid_recon_error_total = 0
            valid_e_total = 0
            valid_p_total = 0
        scheduler.step()
        


                    # if valid_loss_total/len(valid_loader) < best_loss:
                    #         torch.save(model.state_dict(), os.path.join(args.save_path, 'checkpoint.pth'))
                    #         best_loss = valid_loss_total/len(valid_loader)
                    

                    
    plot_loss_moment(loss_plot,args)
