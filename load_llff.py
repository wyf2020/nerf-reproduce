from math import radians
import numpy as np
import os
import imageio

def norm(x):
    return x / np.linalg.norm(x)
def recenter(poses):
    hwf = poses[:,:, 4:5]

    xyz = np.mean(poses[:,:,3], axis = 0)
    z_axis = norm(np.sum(poses[:,:,2], axis = 0))
    y_axis = norm(np.sum(poses[:,:,1], axis = 0))
    x_axis = norm(np.cross(y_axis, z_axis))
    y_axis = norm(np.cross(z_axis, x_axis))

    p34 = np.stack([x_axis,y_axis,z_axis,xyz], axis = -1)
    p44 = np.concatenate([p34,[[0,0,0,1]] ], axis = -2)
    print(p44)
    bottom = np.tile([[0,0,0,1]], (poses.shape[0],1,1))
    poses = np.concatenate((poses[:,:,:4], bottom), axis = -2) # N,4,4

    new_poses = np.linalg.inv(p44) @ poses
    new_poses = np.concatenate([new_poses[:,:3,:], hwf], axis = -1)
    return new_poses


def min_line_dist(rays_o, rays_d): # official version
    A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
    b_i = -A_i @ rays_o
    pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
    return pt_mindist

# def min_line_dist2(rays_o, rays_d): # not implemented yet
#     pt_mindist = np.ones(3)
#     pt_mindist[0] = np.sum(rays_o[:, 0] * (1-rays_d[:, 0]), axis = 0) / np.sum((1-rays_d[:, 0]), axis = 0)
#     pt_mindist[1] = np.sum(rays_o[:, 1] * (1-rays_d[:, 1]), axis = 0) / np.sum((1-rays_d[:, 1]), axis = 0)
#     pt_mindist[2] = np.sum(rays_o[:, 2] * (1-rays_d[:, 2]), axis = 0) / np.sum((1-rays_d[:, 2]), axis = 0)
#     return pt_mindist

def load_llff(args):
    '''load pose bds imgs'''
    poses_bounds = np.load(os.path.join('./data', args.expname, 'poses_bounds.npy')) # N*17
    print(poses_bounds.shape)
    poses = poses_bounds[:,:15].reshape([-1,3,5]) # N 3 5
    bds = poses_bounds[:, -2:] # N 2

    imgpath_root = os.path.join('./data', args.expname, args.image)
    imgpath = [os.path.join('./data', args.expname, args.image, f) for f in sorted(os.listdir(imgpath_root))]

    print('img==pose: %r' % (len(imgpath) == poses.shape[0]))

    images = []
    for p in imgpath:
        if p.endswith('png'):
            images += [imageio.imread(p, ignoregamma = True)]
        else:
            images += [imageio.imread(p)]

    '''modify poses'''
    # 相机坐标系变换到openGL格式下,由[x down,y right,z backward]变换为[x right, y up, z backward], xyz和hwf不变
    poses = np.concatenate([poses[:, :, 1:2], -poses[:, :, 0:1], poses[:, :, 2:], ], axis = -1)
    print(poses[0,:,:])

    # 这里需要hwf/2 因为目前没有实现改变图像分辨率,而现有的check_point是在分辨率减半的图像上训练出来,为了可以使用之前的check_point
    poses[:,:,4] = poses[:,:,4]/2
    
    bd_factor = 0.75 * bds.min()
    poses[:,:,3] = poses[...,3] / bd_factor
    bds = bds / bd_factor

    poses = recenter(poses)
    print(poses[0])

    # pt_mindist2 = min_line_dist2(poses[:, :, 2:3], poses[:, :, 3:4])
    # print(pt_mindist)
    # print(pt_mindist2)
    if args.spherify == True:
        
        hwf = poses[:,:, 4:5]
    
        pt_mindist = min_line_dist(poses[:, :, 3:4], poses[:, :, 2:3])
        print(pt_mindist)
        # 其实三个轴选择很随意 这里与原版数值保持一致
        z_axis = norm((poses[:,:,3] - pt_mindist).mean(0))
        x_axis = norm(np.cross([0.1,0.2,0.3], z_axis))
        y_axis = norm(np.cross(z_axis, x_axis))

        p34 = np.stack([x_axis,y_axis,z_axis,pt_mindist], axis = -1)
        p44 = np.concatenate([p34,[[0,0,0,1]] ], axis = -2)
        print(p44)
        
        bottom = np.tile([[0,0,0,1]], (poses.shape[0],1,1))
        poses = np.concatenate((poses[:,:,:4], bottom), axis = -2) # N,4,4

        poses = np.linalg.inv(p44) @ poses
        poses = np.concatenate([poses[:,:3,:], hwf], axis = -1)

        radius = np.sqrt(np.mean(np.sum(np.square(poses[:,:3,3]), -1)))
        # 原版实现为上面[先平均在开根], 但计算平均半径实际应该是下面两行等价的代码[先开根再平均]
        # radius = np.mean(np.linalg.norm(poses[:,:,3], axis = -1), axis = 0)
        # radius = np.mean(np.sqrt(np.sum(np.square(poses[:,:3,3]), -1)))
        poses[:,:,3] *= 1/radius


    render_poses = poses

    return poses, render_poses, images, bds