from email.policy import default
from re import I
import torch
import configargparse
import imageio
from model import *
from load_llff import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
np.random.seed(0)

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument('--ckpt', type = str, 
                        help='ckpt path')
    parser.add_argument('--image', type = str, 
                        help='image path')
    parser.add_argument('--expname', type = str, 
                        help='expname path')

    parser.add_argument('--render_only', action = "store_true", 
                        help='only render with ckpt')
    parser.add_argument('--lrate', type = float, default = 5e-4,
                        help='learning rate')
    parser.add_argument('--lrate_decay', type = int, default = 250,
                        help='learning rate decay to 0.1 per (lrate_decay*1000) steps')
    parser.add_argument('--N_samples',  type = int, default = 64,
                        help='number of coarse samples per ray')
    parser.add_argument('--N_importance',  type = int, default = 0,
                        help='number of additional fine samples per ray')
    return parser.parse_args()

def create_nerf(args):
    model = NeRF(input_x = 3, input_d = 3).to(device)
    para = list(model.parameters())
    model_fine = None
    if(args.N_importance > 0):
        model_fine = NeRF(input_x = 3, input_d = 3).to(device)
        para += list(model_fine)

    optimizer = torch.optim.Adam(para, lr = args.lrate,betas = (0.9, 0.999))
    
    if args.ckpt != None: # load ckpt
        ckpt = torch.load(args.ckpt)
        print('load ckpt from: '+args.ckpt)
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if(args.N_importance > 0):
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    return model, model_fine, optimizer

def poses2rays(render_poses, args):
    hwf = render_poses[0,:,-1]
    H, W, f = int(hwf[0]), int(hwf[1]), hwf[2]
    print(hwf)

    render_poses = torch.from_numpy(render_poses).to(device)

    render_poses = render_poses[:, :, :4] # N,3,4 去除hwf
    bottom = torch.tile(torch.tensor([[0,0,0,1]]), (render_poses.shape[0],1,1))
    p44 = torch.cat((render_poses, bottom), axis = -2) # N,4,4

    hi, wi = torch.meshgrid(torch.linspace(0,H-1,H), torch.linspace(0,W-1,W), indexing = 'ij')
    d = torch.stack([(wi-W//2)/f,(hi-H//2)/f, -torch.ones(wi.shape)], axis = -1)
    d = torch.cat((d,torch.tensor([1]).expand(d.shape[0],d.shape[1],1) ), axis = -1).transpose(0,-1).reshape(-1,4,1)
    print(d.shape)
    p44 = p44.reshape(-1, 1, 4, 4)
    print(p44.shape)
    

    for i in range(p44.shape[0]):
        if i == 0:
            dir = p44[i] @ d 
        else:
            dir = torch.ca

    print(dir.shape)
    return None

def poses2rays_np(render_poses, args):
    hwf = render_poses[0,:,-1]
    H, W, f = int(hwf[0]), int(hwf[1]), hwf[2]
    print(hwf)

    render_poses = render_poses[:, :, :4] # N,3,4 去除hwf
    bottom = np.tile([[0,0,0,1]], (render_poses.shape[0],1,1))
    p44 = np.concatenate((render_poses, bottom), axis = -2) # N,4,4

    hi, wi = np.meshgrid(np.linspace(0,H-1,H), np.linspace(0,W-1,W), indexing = 'ij')
    d = np.stack([(wi-W//2)/f,(hi-H//2)/f, -np.ones(wi.shape)], axis = -1)
    d = np.concatenate((d,np.broadcast_to(np.array([1]), (d.shape[0],d.shape[1],1) )), axis = -1).reshape((-1,4,1))

    p44 = p44.reshape(-1, 1, 4, 4)
    dir = p44 @ d
    
    print(dir.shape)
    print(dir[0,0,:,0])
    return None
    


def render_poses_fn(render_poses, model, args):
    rays = poses2rays_np(render_poses, args)

    imgs = None
    return imgs

if __name__ =='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    args = config_parser()
    print(args)
    model, model_fine, optimizer = create_nerf(args)
    poses, render_poses, images, bds = load_llff(args)
    
    images = [f/255 for f in images]
    
    render_imgs = render_poses_fn(render_poses, poses, args)

    # print(images[0])
