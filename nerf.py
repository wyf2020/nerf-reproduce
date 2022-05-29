from ast import arg
from email.policy import default
from operator import truediv
import torch
import configargparse
import imageio
import time
from model import *
from load_llff import *

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
np.random.seed(0)

start = 0

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
    parser.add_argument('--spherify', action = "store_true", default = True,
                        help='pictures are shoted inwarding 360 degrees')
    parser.add_argument('--lindisp', action = "store_true", default = True,
                        help='pictures are shoted inwarding 360 degrees')
    parser.add_argument('--shared_near_far', action = "store_true", default = True,
                        help='all poses share the same near and far while sampling points')

    parser.add_argument('--render_only', action = "store_true",
                        help='only render with ckpt')
    parser.add_argument('--render_train', action = "store_true", default = False,
                        help='render_poses == train_poses with ckpt')
    parser.add_argument('--render_test', action = "store_true", default = True,
                        help='render_poses == test_poses with ckpt')
    parser.add_argument('--lrate', type = float, default = 5e-4,
                        help='learning rate')
    parser.add_argument('--lrate_decay', type = int, default = 250,
                        help='learning rate decay to 0.1 per (lrate_decay*1000) steps')
    parser.add_argument('--N_samples',  type = int, default = 64,
                        help='number of coarse samples per ray')
    parser.add_argument('--N_importance',  type = int, default = 0,
                        help='number of additional fine samples per ray')
    parser.add_argument('--batch_train',  type = int, default = 1024,
                        help='batch size of rays while training')
    parser.add_argument('--batch_rays',  type = int, default = 1024*8,
                        help='batch size of rays')
    parser.add_argument('--batch_points',  type = int, default = 1024*16,
                        help='batch size of points') # batch_rays*N_samples > batch_points
    parser.add_argument('--encode_x',  type = int, default = 10,
                        help='encoded vector dimension for position')
    parser.add_argument('--encode_d',  type = int, default = 4,
                        help='encoded vector dimension for direction')
    return parser.parse_args()

def encode(x, L):
    fn = [lambda x, i : torch.sin(torch.tensor(2.0**i) * x),
    lambda x, i : torch.cos(torch.tensor(2.0**i)  * x)] # 原版实现和论文有出入,这里原版实现没有*pi
    res = []
    res.append(x) # 同时原版实现多了3维自身
    for i in range(L):
        for f in fn:
            res.append(f(x,i))
    return torch.cat([t for t in res], axis = -1)

def create_nerf(args):
    model = NeRF(input_x = 3*args.encode_x*2 + 3, input_d = 3*args.encode_d*2 + 3).to(device)
    para = list(model.parameters())
    model_fine = None
    if(args.N_importance > 0):
        model_fine = NeRF(input_x = 3*args.encode_x*2 + 3, input_d = 3*args.encode_d*2 + 3).to(device)
        para += list(model_fine.parameters())

    optimizer = torch.optim.Adam(para, lr = args.lrate, betas = (0.9, 0.999), eps = 1e-7)
    
    start = 0
    if args.ckpt != None: # load ckpt
        ckpt = torch.load(args.ckpt)
        print('load ckpt from: '+args.ckpt)
        start = ckpt['global_step']
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        model.load_state_dict(ckpt['network_fn_state_dict'])
        if(args.N_importance > 0):
            model_fine.load_state_dict(ckpt['network_fine_state_dict'])

    return model, model_fine, optimizer, start

def poses2rays(render_poses, args):
    
    hwf = render_poses[0,:,-1]
    H, W, f = int(hwf[0]), int(hwf[1]), hwf[2]

    render_poses = torch.from_numpy(render_poses).to(device).to(torch.float32)

    hi, wi = torch.meshgrid(torch.linspace(0,H-1,H), torch.linspace(0,W-1,W), indexing = 'ij')
    d = torch.stack([(wi-W//2)/f, -(hi-H//2)/f, -torch.ones(wi.shape)], axis = -1)
    '''z轴需要加负号,这是因为相机面朝-z; 但是y轴加负号只是改变了光线的顺序, 这里与原版保持一致'''
    # version 1 比cpu的numpy版本快4倍
    # new_d = render_poses[:,None,:,:3] @ d.reshape(-1,3,1) # N,H*W,3,1
    # version 2 在gpu上300-600倍快于version1
    new_d = (render_poses[:,:,None,:3] * d.reshape(-1,3) ).sum(-1).movedim(1,-1) # N,H*W,3

    new_o = render_poses[:,None,:,3].expand(new_d.shape)
    rays = torch.stack((new_o, new_d), axis = -1).reshape((-1,3,2)) # N*H*W,3,2
    return rays

def poses2rays_np(render_poses, args):
    hwf = render_poses[0,:,-1]
    H, W, f = int(hwf[0]), int(hwf[1]), hwf[2]

    hi, wi = np.meshgrid(np.linspace(0,H-1,H), np.linspace(0,W-1,W), indexing = 'ij')
    d = np.stack([(wi-W//2)/f,-(hi-H//2)/f, -np.ones(wi.shape)], axis = -1) # H,W,3

    # version 1
    # new_d = render_poses[:, None, :, :3] @ d.reshape(-1, 3, 1) # N,H*W,3,1
    # version 2 但在cpu事实上version 1 更快一点
    new_d = (render_poses[:, :, None, None, :3] * d).sum(-1).transpose((0,2,3,1)) # N,H,W,3

    new_o = np.broadcast_to(render_poses[:,None, None,:,3], new_d.shape)
    rays = np.stack((new_o,new_d), axis = -1).reshape((-1,3,2)) # N*H*W,3,2

    return rays

def rays2points(rays, near, far, bds, args):
    if args.shared_near_far == True:
        rays_o, rays_d = rays[...,0], rays[...,1]
        
        p = torch.linspace(0, 1, args.N_samples)
        if args.lindisp:
            p = 1. / (1 / near + p * (1. / far - 1. / near) )
        else:
            p = near + p * (far - near)
        points = rays_o[:,None,:] + p[:,None] * rays_d[:,None, :] # near,far为各poses的z轴范围, 这里虽然norm(rays_d)不为1,但是rays_d的z值为1, 故直接相乘  (N_rays, N_points, 3)
        # embedder add here
        points = torch.cat((points, rays_d[:,None,:].expand(points.shape)), axis = -1) # [N*H*W, N_samples, 6]
    else:
        pass
    
    return points

def raws2pixels(raws, points, args):
    raws = raws.reshape(-1, args.N_samples, 4)
    points = points.reshape(-1, args.N_samples, 3+3)
    x,d = points[:,:,:3], points[:,0,3:6] # x[N*H*W,N_samples,3], d[N*H*W,3]
    rgb, sigma = raws[:,:,:3], raws[:,:,3] # rgb[N*H*W,N_samples,3], sigma[N*H*W,N_samples]

    last_dist = 1e5
    dist = torch.cat([torch.linalg.norm(x[:,1:,:] - x[:,:-1,:], dim = -1),
        torch.tensor(last_dist).expand(x.shape[0],1)], dim = -1)

    alpha = torch.tensor(1) - torch.exp(-sigma * dist) # alpha[N_rays, N_samples]
    eps = 1e-10
    T = torch.cumprod(torch.cat([torch.tensor(1).expand(alpha.shape[0],1), 1.-alpha[:,:-1]+eps],dim = -1), dim = 1)
    pixels = torch.sum(T[...,None] * alpha[...,None] * rgb, dim = -2)
    return pixels

def points2pixels(points, model, model_fine, args):
    points = points.reshape((-1,3+3)) # points[N*H*W*N_samples, (x_dim + dir_dim)]
    x, d = points.split([3,3], dim = -1)
    view_d = d /torch.norm(d, dim = -1, keepdim = True) 
    # 这里论文要求输入模型的view_dir为unit vector. norm之前不是unit vector 而是z值为1的向量
    x_encoded , d_encoded = encode(x, args.encode_x), encode(view_d, args.encode_d)
    points_encoded = torch.cat([x_encoded, d_encoded], axis = -1)
    # exit()
    pixels_list = []
    for i in range(0, points_encoded.shape[0], args.batch_points):
        if i + args.batch_points > points.shape[0]:
            end = points_encoded.shape[0]
        else:
            end = i + args.batch_points
        raws = model(points_encoded[i:end,:])
        assert(args.batch_points % args.N_samples == 0)
        pixels = raws2pixels(raws, points[i:end], args)
        pixels_list.append(pixels)
        # print(i, points_encoded.shape[0])
    pixels = torch.cat(pixels_list, dim = 0)
    return pixels

def batch_rays(rays, bds, model, model_fine, args):
    if args.shared_near_far == True:
        near = bds.min()*0.9
        far = bds.max()*1.0

        pixels_list = []
        for i in range(0, rays.shape[0], args.batch_rays):
            print(i)
            if i + args.batch_rays > rays.shape[0]:
                end = rays.shape[0]
            else:
                end = i + args.batch_rays
            batch_points = rays2points(rays[i:end,...], near, far, bds, args) # batch_points[N*H*W,N_sample,6] (x+dir)
            pixels = points2pixels(batch_points, model, model_fine, args)
            pixels_list.append(pixels)
            print("rays", i/196608., rays.shape[0])

        pixels = torch.cat(pixels_list, dim = 0)
        print(len(pixels_list))
        print(pixels.shape)
    else:
        pass
    return pixels

def render_poses_fn(render_poses, bds, model, model_fine, args): # render_poses[N,3,5], bds[N,2]
    hwf = render_poses[0,:,-1]
    H, W, f = int(hwf[0]), int(hwf[1]), hwf[2]
    rays = poses2rays(render_poses, args) # rays[N*H*W,3,2] (rays_o+rays_d)
    pixels = batch_rays(rays, bds, model, model_fine, args) # 
    print(pixels.shape)
    imgs = pixels.reshape((render_poses.shape[0],H,W,3))
    return imgs

def train(train_poses, train_images, bds, model, model_fine, optimizer, args):
    rays = poses2rays(train_poses, args) # rays[N*H*W,3,2] (rays_o+rays_d)

    train_images = np.array([f/255 for f in train_images]).reshape((-1,3,1))
    print(train_images.shape)
    train_images = torch.from_numpy(train_images).to(device).to(torch.float32)
    print(train_images.shape)
    rays_rgb = torch.cat([rays, train_images], axis = -1)
    batch_i = 0
    for step in range(start, 200000, 1):
        print(step)
        if(batch_i + args.batch_train <= rays_rgb.shape[0]):
            chunk = rays_rgb[batch_i : batch_i + args.batch_train]
            batch_i += args.batch_train
        else:
            rays_rgb = rays_rgb[torch.randperm(rays_rgb.shape[0])]
            batch_i = arg.batch_train
            chunk = rays_rgb[:batch_i]
        ray, rgb = chunk[...,:2], chunk[...,2]
        pixels = batch_rays(ray, bds, model, model_fine, args)
        loss = torch.mean((pixels-torch.tensor(rgb).to(device))**2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        new_lrate = args.lrate * (0.1 ** (step / (args.lrate_decay*1000)))
        for p in optimizer.param_groups:
            p['lr'] = new_lrate
        
        if step % 100 == 0 and step != 0:
            path = "logs/" + args.expname + '/ckpt/' + str(step) + '.tar'
            if model_fine != None:
                model_fine_state = model_fine.state_dict()
            else:
                model_fine_state = None
            torch.save({
                'global_step' : step,
                'optimizer_state_dict' : optimizer.state_dict(),
                'network_fn_state_dict' : model.state_dict(),
                'network_fine_state_dict' : model_fine_state
            }, path
            )
            print('save ckpt to: '+path)
        

if __name__ =='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    args = config_parser()
    print(args)
    model, model_fine, optimizer, start = create_nerf(args)
    poses, train_poses, render_poses, train_images, render_images, bds, i_test = load_llff(args)
    if args.render_only == True:
        with torch.no_grad():
            render_images = [f/255 for f in render_images]
            # for i in range(render_poses.shape[0]):
            render_poses = render_poses[::20]
            render_images = render_images[::20]
            render_imgs = render_poses_fn(render_poses, bds, model, model_fine, args)
            rgb8 = (255 * (render_imgs).cpu().numpy()).astype(np.uint8)
            for i in range(render_imgs.shape[0]):
                imageio.imwrite("./img_" +  str(i) + ".png", rgb8[i])
                imageio.imwrite("./img_GT_" + str(i) + ".png", render_images[i])
                mse = torch.mean((render_imgs[i]-torch.tensor(render_images[i]).to(device))**2)
                psnr = -10. * torch.log(mse) / torch.log(torch.tensor([10.]))
                print(i, psnr)
    else:
        train(train_poses, train_images, bds, model, model_fine, optimizer, args)
