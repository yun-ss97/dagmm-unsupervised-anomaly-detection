import os
import torch
import time
import matplotlib.pyplot as plt
from dataloader import get_dataloader
from dagmm import DAGMM
from tqdm import tqdm
from torch.optim.lr_scheduler import MultiStepLR


def plot_loss_moment(losses,args):
    _, ax = plt.subplots(figsize=(10,5), dpi=80)
    ax.plot(losses, 'blue', label='train', linewidth=1)
    ax.set_title('Loss change in training')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Iteration')
    ax.legend(loc='upper right')
    plt.savefig(os.path.join(args.img_dir, './result/loss_dagmm_{}.png'.format(args.data_name[:-2])))
    
    
def train(args, train_loader, valid_loader):
 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DAGMM(args)
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(),args.lr, amsgrad=True)
    scheduler = MultiStepLR(optim, [5, 8], 0.1)
    # iter_wrapper = lambda x: tqdm(x, total=len(train_loader))
    
    best_loss = 1e+5

    loss_total = 0
    recon_error_total = 0
    e_total = 0

    valid_loss_total = 0
    valid_recon_error_total = 0
    valid_e_total = 0

    loss_plot = []
    start_time = time.time()
    
    for epoch in range(args.epochs):
        for i, input_data in enumerate(train_loader):
            input_data = input_data.to(device)
            model.train()
            optim.zero_grad()

            enc,dec,z,gamma = model(input_data)
            input_data,dec,z,gamma = input_data.cpu(),dec.cpu(),z.cpu(),gamma.cpu()
            loss, recon_error, e, p = model.loss_func(input_data, dec, gamma, z)
            # print('loss',loss,'recon_error',recon_error,'e',e,'p',p)
     
            loss_total += loss.item()
            recon_error_total += recon_error.item()
            e_total += e.item()

            # model.zero_grad()

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optim.step()
            

            if (i+1) % args.print_iter == 0:
                elapsed = time.time() - start_time
                
                log = "Time {:.2f}, Epoch [{}/{}], Iter [{}/{}], lr {} ".format(elapsed, epoch+1, args.epochs, i+1, len(train_loader), optim.param_groups[0]['lr'])
                
                log+= 'loss {:.4f}, recon_error {:.4f}, energy {:.4f} '.format(loss_total/args.print_iter,
                                                                   recon_error/args.print_iter,e_total/args.print_iter)
                loss_plot.append(loss_total/args.print_iter)
                loss_total = 0
                recon_error_total = 0 
                e_total = 0
                print(log)
        

        with torch.no_grad():
            model.eval()
            for i, input_data in enumerate(valid_loader):
                input_data = input_data.to(device)

                enc,dec,z,gamma = model(input_data)
                input_data,dec,z,gamma = input_data.cpu(),dec.cpu(),z.cpu(),gamma.cpu()
                loss, recon_error, e, p = model.loss_func(input_data, dec, gamma, z)

                # print('loss',loss,'recon_error',recon_error,'e',e,'p',p)

                valid_loss_total += loss.item()
                valid_recon_error_total += recon_error.item()
                valid_e_total += e.item()

            print('[Dev] loss {:.4f}, recon_error {:.4f}, energy {:.4f} '.format(valid_loss_total/len(valid_loader), 
                                                                            valid_recon_error_total/len(valid_loader), 
                                                                            valid_e_total/len(valid_loader)))            

            # if (epoch+1) % args.savestep_epoch == 0:
            if valid_loss_total/len(valid_loader) < best_loss:
                    print('save the best model at epoch {}!'.format(epoch+1))
                    torch.save(model.state_dict(),
                        os.path.join(args.save_path, 'ngmm{}_{}.pth'.format(args.n_gmm, args.data_name[:-2])))
                    best_loss = valid_loss_total/len(valid_loader)
            
            valid_loss_total = 0
            valid_recon_error_total = 0
            valid_e_total = 0

        scheduler.step()
                
    plot_loss_moment(loss_plot,args)