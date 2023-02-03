#%%
import functools
import numpy as np
import torch
from tqdm import tqdm
from NURBSDiff.surf_eval import SurfEval
import matplotlib as mpl
from torch.autograd import Variable
import sys
import copy
import json
import matplotlib.pyplot as plt
from geomdl import NURBS, multi
from itertools import chain
from scipy.ndimage import rotate
from scipy.spatial.distance import cdist
import geomdl
import warnings
from pytorch3d.loss import chamfer_distance
from scipy.optimize import linear_sum_assignment
from geomdl.helpers import find_span_binsearch, basis_function
from geomdl.construct import extract_curves
from geomdl.operations import length_curve
from geomdl.visualization import VisMPL
from geomdl.fitting import approximate_curve
from geomdl.convert import bspline_to_nurbs

sys.path.insert(1, 'nurbspy')
mpl.use('Qt5Agg')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def extract_json(jsonInFileName, dimension=3):
    '''
    extract the NURBS info from json file. It will be replaced by OCC later
    :param jsonInFileName:
    :param dimension:
    :return: dictionary
    '''
    with open(jsonInFileName, 'r') as f:
        surface = json.load(f)
    degree = [surface['shape']['data']['degree_u'], surface['shape']['data']['degree_v']]
    CtrlPtsCountUV = [surface['shape']['data']['size_u'], surface['shape']['data']['size_v']]
    CtrlPtsTotal = CtrlPtsCountUV[0] * CtrlPtsCountUV[1]

    knotU = surface['shape']['data']['knotvector_u']
    knotV = surface['shape']['data']['knotvector_v']
    knotU -= np.min(knotU)
    knotV -= np.min(knotV)
    knotU = knotU / np.max(knotU)
    knotV = knotV / np.max(knotV)

    CtrlPtsNoW = np.array(surface['shape']['data']['control_points']['points'])
    Weights = np.array(surface['shape']['data']['control_points']['weights'])
    CtrlPts = [CtrlPtsNoW, Weights]
    return {'uv_degree': [np.array(degree)], "uv_ctp_count": [np.array(CtrlPtsCountUV)], "ctp_count": [np.array(CtrlPtsTotal)], "knot_u": [knotU], "knot_v": [knotV], "ctp": [CtrlPtsNoW], "windweight": [Weights]}


def extract_pc(dataFileName, sample_rate=1):
    '''
    Extract the point cloud information.
    :param dataFileName:
    :param sample_rate: sample rate of the point cloud
    :return:
    '''
    target = np.genfromtxt(dataFileName, delimiter='\t', dtype=np.float32) # returns n*3 or n*6 np.array
    if target.shape[1] > 3: # remove colors if exists
        target = target[:,:3]
    target = target * 0.196/1.1544 # adjust the interval of point cloud
    samp_ind=np.arange(0,target.shape[0],int(1/sample_rate))
    target = target[samp_ind]
    return target


def match_surface_id(CAD, ptcloud):
    '''
    Simple Match the point cloud to the CAD surface. Minimize the sum of chamfer distance.
    :param CAD: list of 3D points
    :param ptcloud: list of 3D points
    :return: the index sequence for point cloud
    '''
    dist = np.zeros((len(CAD), len(ptcloud)))
    for i in range(len(CAD)):
        for j in range(len(ptcloud)):
            dist[i,j] = chamfer_distance(torch.tensor(CAD[i].reshape(-1,3), dtype=torch.float).unsqueeze(0),
                                         torch.tensor(ptcloud[j].reshape(-1,3), dtype=torch.float).unsqueeze(0))[0].item()
    row_ind, col_ind = linear_sum_assignment(dist)
    return col_ind


def reindex_ctp_from_all(surf_i):
    '''
    Return the index of the surface parameters for assigned surface id.
    :param surf_i:
    :return: A slice of index.
    '''
    return slice(sum([x.size_u*x.size_v for x in solid_faces[:surf_i]]), sum([x.size_u*x.size_v for x in solid_faces[:surf_i+1]]))


def area_from_points(ptcloud):
    '''
    Calculate the rectangle area to approximate the area of surface by discrete sample points.
    :param ptcloud:
    :return:
    '''
    if ptcloud.ndim != 3:
        raise TypeError("The input matrix should be N x M x 3")
        return
    if not isinstance(ptcloud, torch.Tensor):
        base_length_u = np.sqrt(((ptcloud[:-1, :-1, :] - ptcloud[1:, :-1, :]) ** 2).sum(-1).squeeze())
        base_length_v = np.sqrt(((ptcloud[:-1, :-1, :] - ptcloud[:-1, 1:, :]) ** 2).sum(-1).squeeze())
        surf_areas_base = np.sum(np.multiply(base_length_u, base_length_v))
    else:
        base_length_u = torch.sqrt(((ptcloud[:-1, :-1, :] - ptcloud[1:, :-1, :]) ** 2).sum(-1).squeeze())
        base_length_v = torch.sqrt(((ptcloud[:-1, :-1, :] - ptcloud[:-1, 1:, :]) ** 2).sum(-1).squeeze())
        surf_areas_base = torch.sum(torch.multiply(base_length_u, base_length_v))
    return surf_areas_base


def chamfer_distance_one_side(pred, gt, face, side=1):
    """
    Computes average chamfer distance prediction and groundtruth
    but is one sided
    :param pred: Prediction: B x N x 3
    :param gt: ground truth: B x M x 3
    :return:
    """
    # side = 1:min distance from each ground truth to surface
    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).to(device)

    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).to(device)

    pred = torch.unsqueeze(pred, 1)
    gt = torch.unsqueeze(gt, 2)

    diff = pred - gt # BxMxNx3, diff[b, m, n, :] means the pred[n] - gt[m] for batch b
    diff = torch.sum(diff ** 2, 3)
    if side == 0:
        cd = torch.mean(torch.min(diff, 1)[0], 1)
    elif side == 1:
        cd = torch.mean(torch.min(diff, 2)[0], 1)
    cd = torch.mean(cd) #mean of the batch
    return cd


def Hausdorff_distance_one_side(pred, gt, side=1):
    """
    Computes average chamfer distance prediction and groundtruth
    but is one sided
    :param pred: Prediction: B x N x 3
    :param gt: ground truth: B x M x 3
    :return:
    """

    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).to(device)

    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).to(device)

    pred = torch.unsqueeze(pred, 1)
    gt = torch.unsqueeze(gt, 2)

    diff = pred - gt
    diff = torch.sum(diff ** 2, 3) # row 0 means dist from gt[0] to all pred
    if side == 0:
        hd = torch.max(torch.min(diff, 1)[0], 1)[0]
    elif side == 1:
        hd = torch.max(torch.min(diff, 2)[0], 1)[0]
    hd = torch.mean(hd) #mean of the batch
    return hd


def Hausdorff_distance_index(pred, gt, side=1):
    """
    Computes average chamfer distance prediction and groundtruth
    but is one sided
    :param pred: Prediction: B x N x 3
    :param gt: ground truth: B x M x 3
    :return:
    """

    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).to(device)

    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).to(device)

    pred = torch.unsqueeze(pred, 1)
    gt = torch.unsqueeze(gt, 2)

    diff = pred - gt
    diff = torch.sum(diff ** 2, 3).squeeze() # row 0 means dist from gt[0] to all pred
    hd = torch.min(diff, 1)[0]
    hd_ind = torch.argmax(hd)

    # hd = torch.mean(torch.topk(hd,10)[0])
    hd2 = torch.min(diff, 0)[0]
    hd2_ind = torch.argmax(hd2)
    # hd2 = torch.mean(torch.topk(hd2,10)[0])

    return (hd_ind,hd2_ind)
    # return (hd+hd2)/2


def Hausdorff_distance_cust(pred, gt, side=1):
    """
    Computes average chamfer distance prediction and groundtruth, using largest few values
    but is one sided
    :param pred: Prediction: B x N x 3
    :param gt: ground truth: B x M x 3
    :return:
    """
    # customized hausdorff distance, when side = 1, find the ten farthest gt to the surface
    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).to(device)

    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).to(device)

    pred = torch.unsqueeze(pred, 1)
    gt = torch.unsqueeze(gt, 2)

    diff = pred - gt
    diff = torch.sum(diff ** 2, 3) # row 0 means dist from gt[0] to all pred
    if side == 0:
        hd = torch.min(diff, 1)[0]
        hd = torch.topk(hd, 10, 1)[0].mean(1)
        hd = torch.mean(hd) #mean of the batch
    elif side == 1:
        hd = torch.min(diff, 2)[0]
        hd = torch.topk(hd, 10, 1)[0].mean(1)
        hd = torch.mean(hd) #mean of the batch
    return hd



def Hausdorff_distance_cust2(pred, gt, side=1):
    """
    Computes average chamfer distance prediction and groundtruth, using largest few values
    but is one sided
    :param pred: Prediction: B x N x 3
    :param gt: ground truth: B x M x 3
    :return:
    """
    # customized hausdorff distance, when side = 1, find the  farthest gt to the surface, but the distance is the average of ten closest points
    if isinstance(pred, np.ndarray):
        pred = Variable(torch.from_numpy(pred.astype(np.float32))).to(device)

    if isinstance(gt, np.ndarray):
        gt = Variable(torch.from_numpy(gt.astype(np.float32))).to(device)

    pred = torch.unsqueeze(pred, 1)
    gt = torch.unsqueeze(gt, 2)

    diff = pred - gt
    diff = torch.sum(diff ** 2, 3) # row 0 means dist from gt[0] to all pred
    if side == 0:
        hd = -torch.topk(-diff, 10, 1)[0].mean(1)
        hd = hd.max(1)[0]
        hd = torch.mean(hd) #mean of the batch
    elif side == 1:
        hd = -torch.topk(-diff, 10, 2)[0].mean(2)
        hd = hd.max(1)[0]
        hd = torch.mean(hd) #mean of the batch
    return hd


def grid_avg_grad(input_grad, max_range):
    '''
    Average the gradient of the nearby control points.
    :param input_grad:
    :param max_range:
    :return: gradient
    '''
    if not input_grad.is_cuda:
        input_grad = input_grad.to(device)
    input_type = input_grad.dtype
    dist_weight = torch.pow(2, -torch.tensor(range(max_range + 1), dtype=input_type))
    dist_weight = dist_weight / dist_weight.sum()
    filter = torch.ones([1, 1], dtype=torch.float64, device=device) * dist_weight[0]
    for current_size in range(max_range):
        current_size += 1
        filter = torch.nn.functional.pad(filter, (1, 1, 1, 1), value=dist_weight[current_size] / (
                    (current_size + 2) ** 2 - current_size ** 2))
    avg_grad = torch.cat([torch.nn.functional.conv2d(input_grad[..., now_dim].unsqueeze(0).unsqueeze(0),
                                                     filter.unsqueeze(0).unsqueeze(0), padding=max_range) for now_dim in
                          range(3)]).squeeze()
    avg_grad = avg_grad.movedim(0, 2)
    avg_grad = avg_grad.view(-1, 3)
    return avg_grad


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def create_cube(width, length, height, degree:[tuple]=[(3,3)]*6, size:[tuple]=[(6,6)]*6, weight:[np.ndarray]=None):\
    # todo return Solid instead of tuple
    '''
    Create a simple cube with given data, the control points are equally assigned in the given surface area. The centroid of the cube is the origin.
    :param width: width of cube
    :param length: length of cube
    :param height: height of cube
    :param degree: the degree of each NURBS surface
    :param size: the number of control points for each surface
    :param weight: weight for each NURBS surface
    :return: list of geomdl.NURBS.Surface
    '''
    def ctp_gen(a,b,na,nb,c,dim_index,weight):
        '''
        return the control points of a square surface, including the weight
        :param a: the u direction length
        :param b: the v direction length
        :param na: # of ctp on u direction
        :param nb: # of ctp on v direction
        :param c: the location of the surface in 3d space
        :param dim_index: index of the c dimension, e.g. 0 or 1 or 2
        :param weight: weight list
        :return:
        '''
        tmp = np.stack(np.meshgrid(np.linspace(-a/2,a/2,na), np.linspace(-b/2,b/2,nb))+[np.ones((na,nb))*c]+[weight],2)
        tmp[:,:,[dim_index, 2]] = tmp[:,:,[2,dim_index]]
        return tmp

    if weight==None:
        weight = [np.ones(x) for x in size]


    ctps = [
        ctp_gen(width, length, size[0][0], size[0][1], height / 2, 2, weight[0]),
        ctp_gen(width, length, size[1][0], size[1][1], -height / 2, 2, weight[1]),
        ctp_gen(width, height, size[2][0], size[2][1], length / 2, 1, weight[2]),
        ctp_gen(width, height, size[3][0], size[3][1], -length / 2, 1, weight[3]),
        ctp_gen(height, length, size[4][0], size[4][1], width / 2, 0, weight[4]),
        ctp_gen(height, length, size[5][0], size[5][1], -width / 2, 0, weight[5])
    ]
    surfs = []
    for i in range(6):
        surf = NURBS.Surface()
        surf.degree_u = degree[i][0]
        surf.degree_v = degree[i][1]
        surf.set_ctrlpts(ctps[i].reshape(-1,4).tolist(), size[i][0], size[i][1])
        surf.knotvector_u = [0,0,0]+np.linspace(0,1,size[i][0]-2).tolist()+[1,1,1]
        surf.knotvector_v = [0,0,0]+np.linspace(0,1,size[i][1]-2).tolist()+[1,1,1]
        surfs.append(surf)
    curves = [[None]*6 for _ in range(6)]
    # todo can auto create the edge by conter clockwise create the edges for each surf, remember the edge is not on same direction for connected surfaces
    curves[0][2] = ((1,1), (1,0))
    curves[0][3] = ((0,0), (0,1))
    curves[0][4] = ((0,1), (1,1))
    curves[0][5] = ((1,0), (0,0))
    curves[1][2] = ((1,0), (1,1))
    curves[1][3] = ((0,1), (0,0))
    curves[1][4] = ((1,1), (0,1))
    curves[1][5] = ((0,0), (1,0))
    curves[2][0] = ((1,1), (1,0))
    curves[2][1] = ((0,0), (0,1))
    curves[2][4] = ((0,1), (1,1))
    curves[2][5] = ((1,0), (0,0))
    curves[3][0] = ((1,0), (1,1))
    curves[3][1] = ((0,1), (0,0))
    curves[3][4] = ((1,1), (0,1))
    curves[3][5] = ((0,0), (1,0))
    curves[4][0] = ((0,1), (1,1))
    curves[4][1] = ((1,0), (0,0))
    curves[4][3] = ((0,1), (0,0))
    curves[4][2] = ((1,0), (1,1))
    curves[5][0] = ((1,1), (0,1))
    curves[5][1] = ((0,0), (1,0))
    curves[5][3] = ((0,0), (0,1))
    curves[5][2] = ((1,1), (1,0))
    ####check the correctness of trim direction####
    # for i in range(6):
    #     for j in np.arange(i, 6):
    #         if curves[i][j]:
    #             a = surfs[i].evaluate_list(curves[i][j])
    #             b = surfs[j].evaluate_list(curves[j][i])
    #             if a != b:
    #                 print(i, j)


    return (surfs, curves)


def plot_list_geomdl_surf(surfs: [geomdl.NURBS.Surface], new_fig=True, ax=None, fig=None):
    if new_fig:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')
    else:
        if fig==None or ax == None:
            ValueError("Plase provide figure handle if you set new_fig to be False")
    colorpanel = (np.arange(len(surfs)))/(len(surfs)-1)
    cmap = mpl.cm.get_cmap('PRGn')
    colorpanel = cmap(colorpanel)


    for i in range(len(surfs)):
        na, nb = surfs[i].sample_size
        evalpts = np.array(surfs[i].evalpts)
        # ax.scatter(evalpts[:,0],evalpts[:,1],evalpts[:,2])
        evalpts = evalpts.reshape(na,nb,3)
        ax.plot_surface(evalpts[:, :, 0], evalpts[:, :, 1], evalpts[:, :, 2], color=colorpanel[i], alpha=0.8)

    ax.azim = 130  # -89 85
    ax.dist = 6
    ax.elev = 50  # 16 50
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_axis_off()
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.tight_layout()
    plt.xlabel('x')
    plt.ylabel('y')
    set_axes_equal(ax)
    return fig, ax


def plot_solid(solid, ctpall, new_fig=True, ax=None, fig=None, ignore_surf=[]):
    if new_fig:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')
    else:
        if fig==None or ax == None:
            ValueError("Plase provide figure handle if you set new_fig to be False")
    n_surfs = len(solid.faces)
    colorpanel = (np.arange(n_surfs))/(n_surfs-1)
    cmap = mpl.cm.get_cmap('PRGn')
    colorpanel = cmap(colorpanel)
    # colorpanel = ['b','c','g','k','m','r']
    i = 0
    for face_id, face in solid.faces.items():  # evaluate the points on each surface
        if face_id in ignore_surf:
            continue
        geomdl_face = face.create_3dNURBS(ctpall[face.ctp_index[0]:face.ctp_index[1], :])
        geomdl_face.delta = 0.1
        na, nb = geomdl_face.sample_size
        evalpts = np.array(geomdl_face.evalpts).reshape(na, nb, 3)
        ax.plot_surface(evalpts[:, :, 0], evalpts[:, :, 1], evalpts[:, :, 2], color=colorpanel[i], alpha=0.5)
        i+=1

    ax.azim = 130  # -89 85
    ax.dist = 6
    ax.elev = 50  # 16 50
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_axis_off()
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.tight_layout()
    plt.xlabel('x')
    plt.ylabel('y')
    set_axes_equal(ax)
    return fig, ax


def plot_list_points(points, new_fig=True, ax=None, fig=None):
    if new_fig:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')
    else:
        if fig==None or ax == None:
            ValueError("Plase provide figure handle if you set new_fig to be False")
    colorpanel = (np.arange(len(points)))/(len(points)-1)
    cmap = mpl.cm.get_cmap('PRGn')
    colorpanel = cmap(colorpanel)


    for i in range(len(points)):
        ax.scatter(points[i][:,0],points[i][:,1],points[i][:,2],color=colorpanel[i], s=2)

    ax.azim = 130  # -89 85
    ax.dist = 6
    ax.elev = 50  # 16 50
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.set_axis_off()
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.tight_layout()
    plt.xlabel('x')
    plt.ylabel('y')
    set_axes_equal(ax)
    return fig, ax


def binary_grid_search(face0_geomdl, loc_3d):
    '''
    Search the 2d location according to 3d location for surface
    :param face0_geomdl: geomdl.surface
    :param loc_3d: 3d location of a point
    :return: 2d location of the point on surface
    '''
    u = [0,1]
    v = [0,1]
    d = 1
    n_loop = 0
    while d >= 1e-8 and n_loop <= 200:
        grid_size = 10
        ul = np.linspace(u[0], u[1], grid_size)
        vl = np.linspace(v[0], v[1], grid_size)
        grid = np.stack(np.meshgrid(ul, vl),2)
        eval_grid = np.array(face0_geomdl.evaluate_list(grid.reshape(-1,2)))
        eval_grid = eval_grid-loc_3d
        eval_grid = np.array(eval_grid).reshape(10,10,3)
        eval_grid = (eval_grid**2).sum(2)
        d = np.min(eval_grid)
        x,y = np.where(eval_grid==d)
        u = [ul[max(y[0]-1, 0)], ul[min(y[0]+1, grid_size-1)]]
        v = [vl[max(x[0]-1, 0)], vl[min(x[0]+1, grid_size-1)]]
        d = sum((np.array(face0_geomdl.evaluate_single([np.mean(u), np.mean(v)]))-loc_3d)**2)
        n_loop += 1
    if d>=1e-8:
        warnings.warn('The binary search is not convergent. The final square error is: {:E}'.format(d))
    return [np.mean(u), np.mean(v)]


def deform_2d_surf0(solid, scale_u=1, scale_v=1, angle=0): # todo renameing as scretching_rotate_in_2d_space(...):
    '''
    deform the surface0. One can scale or rotate it.
    :param solid: Solid object
    :param scale_u: scale on u direction
    :param scale_v: scale on v direction
    :param angle: clockwise rotate angle
    :return: Solid object
    '''
    face0_name = list(solid.faces.keys())[0]
    face0 = solid.faces[face0_name]
    face0_geomdl_old = face0.create_3dNURBS(ctp=ctpall[face0.ctp_index[0]:face0.ctp_index[1], :])
    angle = angle/180*np.pi # rotation angle
    rot_mat = np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    ctpall[face0.ctp_index[0]:face0.ctp_index[1],:2] = ctpall[face0.ctp_index[0]:face0.ctp_index[1],:2]@rot_mat
    ctpall[face0.ctp_index[0]:face0.ctp_index[1],0] = ctpall[face0.ctp_index[0]:face0.ctp_index[1],0]*scale_u
    ctpall[face0.ctp_index[0]:face0.ctp_index[1],1] = ctpall[face0.ctp_index[0]:face0.ctp_index[1],1]*scale_v
    face0_geomdl = face0.create_3dNURBS(ctp=ctpall[face0.ctp_index[0]:face0.ctp_index[1], :])
    if angle == 0:
        # only need to change the two end point of line segment if no rotation
        us = []
        vs = []
        for surf_id, curve in face0.edges.items():
            # print(surf_id)
            # curve_old = curve.trims[face0.surf_id]
            # ctp_val_3d = face0_geomdl_old.evaluate_list(curve_old.ctrlpts)
            # new_ctp_val = [binary_grid_search(face0_geomdl, np.array(loc_3d)) for loc_3d in ctp_val_3d] # assume the 2d parameter space do not change much, binary search in the original neighbor region
            # curve_new1 = copy.deepcopy(curve_old)
            # curve_new1.ctrlpts = new_ctp_val
            # curve.trims[face0.surf_id].ctrlpts = new_ctp_val
            # us+=[new_ctp_val[0][0], new_ctp_val[1][0]]
            # vs+=[new_ctp_val[0][1], new_ctp_val[1][1]]
        # solid.faces[face0_name].DiffSurf = SurfEval(face0.size_u, face0.size_v, knot_u=face0.knotvector_u, knot_v=face0.knotvector_v,
        #                     dimension=3, p=face0.degree_u, q=face0.degree_v, out_dim_u=face0.uEvalPtSize,
        #                     out_dim_v=face0.vEvalPtSize, dvc=device)
            print(surf_id)
            curve_old = curve.trims[face0.surf_id]
            curve_old.sample_size = 30
            pt_2d_old = curve_old.evalpts
            pt_val_3d = face0_geomdl_old.evaluate_list(pt_2d_old)
            new_pt_val = [binary_grid_search(face0_geomdl, np.array(loc_3d)) for loc_3d in pt_val_3d]
            curve_fit = approximate_curve(new_pt_val, degree=3, ctrlpts_size=6, centripetal=True)
            curve_fit = bspline_to_nurbs(curve_fit)
            curve_new = NURBS.Curve()
            curve_new.degree = curve_fit.degree
            curve_new.set_ctrlpts(curve_fit.ctrlptsw)
            curve_new.knotvector = curve_fit.knotvector
            # curve_new.set_ctrlpts(np.concatenate([np.array([curve_fit.evalpts[0],(np.array(curve_fit.evalpts[0])*0.4+np.array(curve_fit.evalpts[-1])*0.6),curve_fit.evalpts[-1]]), np.array([[1], [1], [1]])], 1).tolist())
            # curve_new.knotvector = np.array([0, 0, 0.6, 1, 1])
            pt = new_pt_val
            for i in range(len(pt)):
                plt.scatter(pt[i][0], pt[i][1], color='red')
            pt = curve_new.evalpts
            for i in range(len(pt)):
                plt.scatter(pt[i][0], pt[i][1], color='blue')
            curve.trims[face0.surf_id] = curve_new
            # pt = pt_2d_old
            # pt = curve_new.evalpts
            # for i in range(len(pt) - 1):
            #     start = pt[i]
            #     delta = [pt[i + 1][0] - start[0], pt[i + 1][1] - start[1]]
            #     plt.arrow(start[0], start[1], delta[0], delta[1], width=0.1, head_length=0.1)
    else:
        for surf_id, curve in face0.edges.items():
            # we sample 25 (default) points on the edges and find the deformed 2d location. Finally fit the new NURBS curve for the edge
            print(surf_id)
            curve_old = curve.trims[face0.surf_id]
            curve_old.sample_size = 100
            pt_2d_old = curve_old.evalpts
            pt_val_3d = face0_geomdl_old.evaluate_list(pt_2d_old)
            new_pt_val = [binary_grid_search(face0_geomdl, np.array(loc_3d)) for loc_3d in pt_val_3d]
            curve_new = approximate_curve(new_pt_val, degree=3, ctrlpts_size=np.max(face0_geomdl_old.cpsize))
            curve.trims[face0.surf_id] = curve_new

            # pt = pt_2d_old
            # pt = curve_new.evalpts
            # for i in range(len(pt) - 1):
            #     start = pt[i]
            #     delta = [pt[i + 1][0] - start[0], pt[i + 1][1] - start[1]]
            #     plt.arrow(start[0], start[1], delta[0], delta[1], width=0.1, head_length=0.1)


            # pt = new_pt_val
            # for i in range(len(pt)):
            #     plt.scatter(pt[i][0], pt[i][1], color='red')
            # pt = curve_new.evalpts
            # for i in range(len(pt)):
            #     plt.scatter(pt[i][0], pt[i][1], color='blue')
        # plt.xlim((0,1))
        # plt.ylim((0,1))
    return solid


def plot_solid_from_geomdl(solid, ctp):
    '''
    plot the solid using geomdl, changed Axes3D to matplotlib official style!
    :param solid: Solid
    :param ctp: control point list for all surfaces
    :return: None
    '''
    geomdl_face = []
    for face_id, face in solid.faces.items(): # evaluate the points on each surface
        geomdl_tmp = face.create_3dNURBS(ctp[face.ctp_index[0]:face.ctp_index[1],:])
        for edge in face.edges.values():
            geomdl_tmp.add_trim(edge.trims[face_id])
        geomdl_face.append(geomdl_tmp)

    vis_config = VisMPL.VisConfig(legend=False, axes=False, figure_dpi=120,evalpts=False,trims=True)
    s=multi.SurfaceContainer(geomdl_face)
    # Create a visualization method instance using the configuration above
    s.vis = VisMPL.VisSurface(vis_config)
    # Set the visualization method of the curve object
    # Plot the curve
    s.render()


def plot_surface_trim_2d(solid, surf_id):
    '''
    plot 2d trim curve on surf_id
    :param solid: Solid
    :param surf_id: string
    :return: None
    '''
    geomdl_curves = []
    for edge in solid.faces[surf_id].edges.values():
        geomdl_curves.append(edge.trims[surf_id])
    vis_config = VisMPL.VisConfig(legend=False, axes=False, figure_dpi=120, evalpts=False)
    s = multi.CurveContainer(geomdl_curves)
    # Create a visualization method instance using the configuration above
    s.vis = VisMPL.VisCurve3D(vis_config)
    # Set the visualization method of the curve object
    # Plot the curve
    s.render()


def plot_2dSurf_trims(solid, surf_id, ctpall):
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, adjustable='box')
    target_surf = solid.faces[surf_id]
    geomdl_surf = target_surf.create_3dNURBS(ctpall[target_surf.ctp_index[0]:target_surf.ctp_index[1]])
    curves_evalpts = []
    for curve_id, curve in target_surf.edges.items():
        ctps = np.array(curve.trims[surf_id].ctrlpts)
        evalpts = np.array(curve.trims[surf_id].evalpts)
        curves_evalpts.append(evalpts)
        ax.scatter(ctps[:,0], ctps[:,1])
        ax.plot(evalpts[:,0], evalpts[:,1])

    curves_evalpts = np.array(curves_evalpts)
    x_range = np.linspace(np.min(curves_evalpts[...,0]), np.max(curves_evalpts[...,0]), curves_evalpts.shape[0]*curves_evalpts.shape[1])
    up_low = []
    for x in x_range:
        min_ind = np.argmin(abs(curves_evalpts[...,0]-x),1)
        min_ind2 = np.argsort(np.min(abs(curves_evalpts[...,0]-x),1))[:2]
        up_low.append(np.sort(np.array([curves_evalpts[min_ind2[0], min_ind[min_ind2[0]], 1], curves_evalpts[min_ind2[1], min_ind[min_ind2[1]], 1]])))
    up_low = np.array(up_low)
    ax.fill_between([0,1],[0,0],[1,1],color='red',alpha=0.3)
    ax.fill_between(x_range, up_low[:,0], up_low[:,1] ,color='blue',alpha=0.3)
    ax.grid()


class Solid:
    faces = None # a dict of Surface objects
    edges = None # a dict of Curve objects
    device = None
    '''
    The class is immutatble
    '''
    def __init__(self, faces: list=None, edges: list=None, device='cpu'):
        self.device = device
        self.faces = {}
        self.edges = {}
        return
        ctp_index = 0
        for idx, face in enumerate(faces):
            tmp_surf = Surface(solid=self, control_point=face.ctrlpts, knotvector_u=face.knotvector_u, knotvector_v=face.knotvector_v,
                        surface_id='Surface'+str(idx), ctp_start_index=ctp_index, degree_u=face.degree_u, degree_v=face.degree_v,
                        size_u=face.cpsize[0], size_v=face.cpsize[0], weight=face.weights, device=device)
            self.faces['Surface'+str(idx)] = tmp_surf
            ctp_index = tmp_surf.ctp_index[1]
        trim_id = 0
        for i in range(len(faces)):
            for j in np.arange(i, len(faces)):
                if edges[i][j]:
                    self.edges['Curve'+str(trim_id)] = Curve.get_line_segment(points_1=edges[i][j], points_2=edges[j][i], curve_id='Curve'+str(trim_id), surf_id1='Surface'+str(i), surf_id2='Surface'+str(j), solid=self, device=device)
                    trim_id += 1
        pass


    @classmethod
    def create_cube(cls, length, width, height, n_ctp, device='cpu'):
        def ctp_gen(x_range, y_range, z_range, uv_seq, z_index, size):
            xyz = [x_range, y_range, z_range]
            u, v = xyz[uv_seq[0]], xyz[uv_seq[1]]
            tmp = []
            for u_value in np.linspace(u[0], u[1], size[0]):
                for v_value in np.linspace(v[0], v[1], size[1]):
                    loc = [0, 0, 0, 1]
                    loc[z_index] = xyz[z_index]
                    loc[uv_seq[0]] = u_value
                    loc[uv_seq[1]] = v_value
                    tmp.append(loc)
            return np.array(tmp).reshape(size[0], size[1], 4)


        ctp = [ctp_gen([-length / 2, length / 2], [-width / 2, width / 2], height / 2, [0, 1], 2, n_ctp[0]),
               ctp_gen([-length / 2, length / 2], [-width / 2, width / 2], -height / 2, [0, 1], 2, n_ctp[1]),
               ctp_gen([-length / 2, length / 2], -width / 2, [-height / 2, height / 2], [0, 2], 1, n_ctp[2]),
               ctp_gen([-length / 2, length / 2], width / 2, [-height / 2, height / 2], [0, 2], 1, n_ctp[3]),
               ctp_gen(length / 2, [-width / 2, width / 2], [-height / 2, height / 2], [1, 2], 0, n_ctp[4]),
               ctp_gen(-length / 2, [-width / 2, width / 2], [-height / 2, height / 2], [1, 2], 0, n_ctp[5])]
        knotvectors_u = [[0,0,0]+np.linspace(0,1,n_ctp[i][0]-2).tolist()+[1,1,1] for i in range(6)]
        knotvectors_v = [[0,0,0]+np.linspace(0,1,n_ctp[i][1]-2).tolist()+[1,1,1] for i in range(6)]

        solid = Solid(device=device)
        ctp_index= 0
        vertices = []
        line24 = []
        line24_pair = []
        conterclock_verts = [[0, 0], [1, 0], [1, 1], [0, 1]]
        for face_id in range(6):
            tmp_surf = Surface(solid=solid, control_point=ctp[face_id][...,:-1], knotvector_u=knotvectors_u[face_id], knotvector_v=knotvectors_v[face_id],
                        surface_id='Surface'+str(face_id), ctp_start_index=ctp_index, degree_u=3, degree_v=3,
                        size_u=n_ctp[face_id][0], size_v=n_ctp[face_id][1], weight=ctp[face_id][...,-1], device=solid.device)
            ctp_index += n_ctp[face_id][0]*n_ctp[face_id][1]
            solid.faces['Surface'+str(face_id)] = tmp_surf
            for vert in conterclock_verts:
                vertices.append(tmp_surf.evaluate_single(ctp[face_id][...,:-1].reshape(-1,3), vert))
            line24+=np.moveaxis(np.array([conterclock_verts, np.roll(conterclock_verts, -1, 0)]), 0, 1).tolist()
            conterclock_verts[1], conterclock_verts[3] = conterclock_verts[3], conterclock_verts[1]
        vert_dist = cdist(vertices, vertices) + np.eye(24)
        np.where(vert_dist[0,:]==0)[0]/4
        for count in range(24):
            surfi_start_index = count
            surfi_end_index = count+1-(not (count+1)%4)*4
            surfj_end_index = np.where(vert_dist[surfi_start_index,:]==0)[0]
            surfj_start_index = np.where(vert_dist[surfi_end_index,:]==0)[0]
            surfj_index, counts = np.unique([surfj_start_index//4, surfj_end_index//4], return_counts=True)
            surfj_index = surfj_index[counts==2]
            surfj_start_index = surfj_start_index[surfj_start_index // 4 == surfj_index]
            surfj_end_index = surfj_end_index[surfj_end_index // 4 == surfj_index]
            line24_pair.append([count, surfj_start_index[0]])
        line24_pair = set([tuple(sorted(_)) for _ in line24_pair])
        line24_pair = np.array(list(line24_pair))
        for curve_id in range(12):
            curve = Curve(solid=solid, curve_id=curve_id, device=solid.device)
            curve.trims = {}
            solid.edges['Curve' + str(curve_id)] = curve
        for line_id in range(24):
            surf_id = line_id//4
            curve_id = np.where(line24_pair==line_id)[0][0]
            curve = solid.edges['Curve'+str(curve_id)]
            curve.set_line_trim(line24[line_id], 'Surface'+str(surf_id))
            solid.faces['Surface'+str(surf_id)].set_edge(curve)
        return solid, [ctp[i][...,:-1] for i in range(6)]


class Surface:
    init_ctp = None
    weight = None
    degree_u = None
    degree_v = None
    knotvector_u = None
    knotvector_v = None
    size_u = None
    size_v = None
    surf_id = None
    uEvalPtSize = None
    vEvalPtSize = None
    device = None
    DiffSurf = None
    uvEvalPtSize_max = 512
    edges = None
    solid = None
    ctp_index = None
    '''
    The class is immutable.
    '''
    def __init__(self, solid, control_point, knotvector_u, knotvector_v, surface_id, ctp_start_index, degree_u=None, degree_v=None, size_u=None, size_v=None, weight=None, device='cpu'):
        # document the mutability of attributes (in what situation)
        # maybe move the control points outside the class, making the class immutable
        # wrapper class to include the control point (changing) and this immutable class.
        self.solid = solid
        self.init_ctp = np.array(control_point) # initial control points
        if self.init_ctp.ndim < 3:
            if size_u == None and size_v == None:
                raise ValueError('The dimension of the control point is wrong. You should either provide a 3 dimension matrix or 2 dimension matrix with assigned control point size.')
            else:
                self.init_ctp = self.init_ctp.reshape(size_u, size_v, 3)
        self.knotvector_u = np.array(knotvector_u) # knotvector in u direction
        self.knotvector_v = np.array(knotvector_v) # knotvector in v direction
        if weight is not None:
            self.weight = np.array(weight).reshape(self.init_ctp.shape[:-1]) # weight matrix
        else:
            self.weight = np.ones(control_point.shape[:1]).tolist()
        if size_u is not None:
            self.size_u = size_u  # number of control points in u direction
        else:
            self.size_u = self.init_ctp.shape[0]
        if size_v is not None:
            self.size_v = size_v# number of control points in v direction
        else:
            self.size_v = self.init_ctp.shape[1]
        if degree_u is not None:
            self.degree_u = degree_u # the degree in u
        else:
            self.degree_u = len(knotvector_u)-self.size_u-1
        if degree_v is not None:
            self.degree_v = degree_v # the degree in v
        else:
            self.degree_v = len(knotvector_v) - self.size_v - 1
        self.ctp_index = (ctp_start_index, self.size_u*self.size_v+ctp_start_index)
        self.device = device # compute device cpu or cuda
        self.surf_id = surface_id # surface id
        self.set_uv_EvalPts()  # set the number of evaluated points
        self.create_diff_NURBS() # create a differentiable NURBS object. It does not change during optimization because it does not include the control points
        pass

    def __str__(self):
        return 'Surface'

    def __repr__(self):
        return self.surf_id

    def create_3dNURBS(self, ctp):
        '''
        It creates the 3D representation of the NURBS surface via geomdl.
        :return:
        '''
        surf3d = NURBS.Surface()
        surf3d.degree_u = self.degree_u
        surf3d.degree_v = self.degree_v
        ctp = ctp.reshape(self.size_u, self.size_v, 3)
        ctp = np.concatenate((ctp, self.weight.reshape(self.size_u, self.size_v, 1)), -1)
        surf3d.set_ctrlpts(ctp.reshape(-1, 4).tolist(), self.size_u, self.size_v)
        surf3d.knotvector_u = self.knotvector_u
        surf3d.knotvector_v = self.knotvector_v
        surf3d.delta = 0.01
        return surf3d


    def _calculate_uv_EvalPts(self):
        '''
        It calculates the number of points to be evaluated on the surface in u or v direction. The total number of
        evaluated points should not exceed the max value. The number of evaluated points on each direction is proportional
        to the average length of curves on u and v direction.
        :return: a set of integer, size=2
        '''
        NURBS3d = self.create_3dNURBS(self.init_ctp)
        curves_dict = extract_curves(NURBS3d)
        u_length = np.array([length_curve(curve_u) for curve_u in curves_dict['u']]).mean()
        v_length = np.array([length_curve(curve_v) for curve_v in curves_dict['v']]).mean()
        unit_u = np.sqrt(self.uvEvalPtSize_max/(v_length/u_length))
        uEvalPtSize = int(np.floor(unit_u))
        vEvalPtSize = int(np.floor(unit_u*(v_length/u_length)))
        return (uEvalPtSize, vEvalPtSize)


    def set_uv_EvalPts(self, uvsize=None):
        '''
        Set the number of evaluated points on u and v direction.
        :param uvsize: a set of intergers, size=2
        :return:
        '''
        if not uvsize:
            uvsize = self._calculate_uv_EvalPts()
        self.uEvalPtSize = uvsize[0]
        self.vEvalPtSize = uvsize[1]
        pass


    def create_diff_NURBS(self):
        '''
        It creates a differentiable NURBS surface via NURBSDiff.
        :return:
        '''
        try_i = 0
        tmp = [np.nan]
        while try_i <= 5 and sum([np.sum(np.isnan(x)) for x in tmp]):
            surf = SurfEval(self.size_u, self.size_v, knot_u=self.knotvector_u, knot_v=self.knotvector_v, dimension=3,
                            p=self.degree_u, q=self.degree_v, out_dim_u=self.uEvalPtSize, out_dim_v=self.vEvalPtSize,
                            dvc=self.device)
            self.DiffSurf = surf
            # check whether there is null value error in the evaluated points or not
            tmp = self.evalpts(self.init_ctp)
        if sum([np.sum(np.isnan(x)) for x in tmp]):
            self.DiffSurf = None
            raise Exception('Fail to create differentiable surface. try to decrease the evaluate size.')
        pass


    def evalpts(self, ctp):
        '''
        It evaluate points using differentiable NURBS surface. The number of evaluated points is predetermiend.
        :param ctp:
        :return: an array of points in 3D space
        '''
        return self.DiffSurf(torch.cat((torch.tensor(ctp, dtype=torch.float, device=self.device).unsqueeze(0),
                                         torch.tensor(self.weight, dtype=torch.float, device=self.device).unsqueeze(
                                             0).unsqueeze(-1)), axis=-1)).cpu().detach().numpy()


    def evaluate_single(self, ctp, param):
        '''
        It evaluates the single point at 3D space given 2D parameters.
        :param param: a point in parameter space. size=2
        :return: an array in 3D space. size=3
        '''
        NURBS3d = self.create_3dNURBS(ctp)
        return NURBS3d.evaluate_single(param)


    def evaluate_list(self, ctp, param_list):
        '''
        It evaluates a list of points at 3D space given 2D parameters.
        :param param_list: a list of points in parameter space. size=n*2
        :return: a list of points in 3D space. size=n*3
        '''
        NURBS3d = self.create_3dNURBS(ctp)
        return NURBS3d.evaluate_list(param_list)


    def set_device(self, device):
        '''
        set the device to be a specific cuda or cpu
        :param device: string
        :return:
        '''
        self.device = device
        pass


    def set_edge(self, edge):
        '''
        Bound the trim curve to the surface.
        :param trims: a list of Curve
        :return: None
        '''
        if self.edges == None:
            self.edges = {}
        self.edges[edge.curve_id] = edge
        pass


class Curve:
    device = None
    trims = None # dictionary of 2d trim curve representation in each surface, type: geomdl.NURBS.Curve.
    curve_id = None
    solid = None
    '''
    The class is immutable
    '''
    def __init__(self, solid, curve_id, device='cpu'):
        self.solid = solid
        self.curve_id = 'Curve' + str(curve_id)
        self.device = device
        pass


    @classmethod
    def get_line_segment(cls, points_1, points_2, curve_id, surf_id1, surf_id2, solid, device='cpu'):
        '''
        Create the Curve object via line segment
        :param points_1: 2d representation of two end points on surface 1
        :param points_2: 2d representation of two end points on surface 2
        :param curve_id: curve id, string
        :param surf_id1: surface 1 id, string
        :param surf_id2: surface 2 id, string
        :param device: data device, cpu or cuda
        :return: Curve object
        '''
        curve = Curve(solid=solid, curve_id=curve_id, device=device)
        curve.trims = {}
        curve.set_line_trim(points_1, surf_id1)
        curve.set_line_trim(points_2, surf_id2)
        curve.solid.faces[surf_id1].set_edge(curve)
        curve.solid.faces[surf_id2].set_edge(curve)
        return curve


    def set_line_trim(self, points, surf_id):
        '''
        Set the 2d representation of the trim on surface
        :param points: 2d representation of two end points on surface
        :param surf_id: surface id, string
        :return: None
        '''
        curve = NURBS.Curve()
        curve.degree = 1
        curve.set_ctrlpts(np.concatenate([np.array(points), np.array([[1],[1]])],1).tolist())
        curve.knotvector = np.array([0,0,1,1])
        self.trims[surf_id] = curve


    def evaluate_single(self, ctp, point, surf_id: str=None, dim=None):
        '''
        evaluate the 3d location according to 1d location
        :param point: 1d point location
        :param surf_id: surface id
        :return: point 3d location
        '''
        if surf_id==None:
            surf_id = list(self.trims.keys())[0]
        location_2d = self.trims[surf_id].evaluate_single(point)
        if dim == '2d':
            return location_2d
        surf = self.solid.faces[surf_id]
        location_3d = surf.evaluate_single(ctp[surf.ctp_index[0]:surf.ctp_index[1],:], location_2d)
        return location_3d


    def evaluate_list(self, ctp, points: list, surf_id: str=None, dim=None):
        '''
        evaluate a list of 3d location according to 1d location
        :param points: list of 1d point location
        :param surf_id: surface id
        :return: a list of point 3d location
        '''
        if surf_id==None:
            surf_id = list(self.trims.keys())[0]
        location_2d = self.trims[surf_id].evaluate_list(points)
        if dim=='2d':
            return location_2d
        location_3d = self.solid.faces[surf_id].evaluate_list(ctp, location_2d)
        return location_3d


    def __str__(self):
        return 'Curve'

    def __repr__(self):
        return self.curve_id


def angle_estimate(center,start,end):
    edge1 = start-center
    edge2 = end-center
    angle = np.arcsin(np.cross(edge1,edge2)/(np.linalg.norm(edge1)*np.linalg.norm(edge2)))
    return angle


def winding_number_curve(point, curve):
    pass





#%%
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
outputname = "fig/20220925/case2_5/twist"
uvEvalPtSize_max = 512
device = 'cpu'
device = 'cuda'
dataFileNames = ['data/NDE/pointCloud_twist - Cloud'+str(i+1)+'.txt' for i in np.arange(0,6)]
# jsonOutFileName = "AMSurface.out.json"
DELTA = 1e-8
dimension = 3
################# read in json and point cloud#######################
'''
The following part needs to be replaced by the OCC later. 
'''
##############Rotate and enlarge the parametric plane of solid surface 0  #####################
n_ctp = [[12,12],[5,6],[5,6],[5,6],[5,6],[5,6]]
solid, ctpall = Solid.create_cube(127, 19.05, 12.7, n_ctp, device=device)
ctpall = [x.reshape(-1,3) for x in ctpall]
ctpall = np.concatenate(ctpall).reshape(-1, 3)
solid = deform_2d_surf0(solid, 2, 5, 10) # scale/rotate the surface 0
# plot_solid_from_geomdl(solid, ctpall)
# plot_surface_trim_2d(solid, 'Surface0')
def in_trim(point, surf_id):
    edge_list = solid.faces[surf_id].edges
    edge_names = list(edge_list.keys())
    degs = []
    for edge_name in edge_names:
        curve = edge_list[edge_name]
        curve_pts = curve.evaluate_list(ctpall[solid.faces[surf_id].ctp_index[0]:solid.faces[surf_id].ctp_index[1]], np.arange(0,1,0.01), surf_id=surf_id, dim='2d')
        curve_pts = np.array(curve_pts)
        centroid = np.array(point)
        if np.min(cdist(np.array([0,0.012]).reshape(1,-1), curve_pts)) <=0.01: # if the point is on the curve
            return 360
        for i in range(len(curve_pts)-1):
            degs.append(angle_estimate(centroid, curve_pts[i], curve_pts[i+1]))
    return sum(degs)/np.pi*180

# degs = []
# s0 = solid.faces['Surface0']
# for u in np.linspace(0,1,s0.uEvalPtSize):
#     print(u)
#     for v in np.linspace(0,1,s0.vEvalPtSize):
#         degs.append(in_trim((u,v), 'Surface0'))
# degs = np.array(degs)
# a = degs.reshape(s0.uEvalPtSize, s0.vEvalPtSize)
# inner = abs(a)>=300
# outer = abs(a)<300
# inner = inner.reshape(-1)
# outer = outer.reshape(-1)
# u,v = np.meshgrid(np.linspace(0,1,s0.uEvalPtSize), np.linspace(0,1,s0.vEvalPtSize))
# loc3d = s0.evaluate_list(ctpall[s0.ctp_index[0]:s0.ctp_index[1]], np.stack([u.transpose(), v.transpose()],-1).reshape(-1,2))
# loc3d = np.array(loc3d)
# fig, ax = plot_solid(solid, ctpall)
# ax.scatter(loc3d[inner,0],loc3d[inner,1],loc3d[inner,2],color='red')


for face_id in solid.faces:
    print(face_id)
    degs = []
    face = solid.faces[face_id]
    for u in np.linspace(0, 1, face.uEvalPtSize):
        for v in np.linspace(0, 1, face.vEvalPtSize):
            degs.append(in_trim((u, v), face_id))
    degs = np.array(degs)
    degs = degs.reshape(face.uEvalPtSize, face.vEvalPtSize)
    inner = abs(degs) >= 300
    face.inside_evalpts = inner

fig, ax = plot_solid(solid, ctpall, ignore_surf=["Surface1", "Surface2","Surface3","Surface4","Surface5"])
for face_id in solid.faces:
    face = solid.faces[face_id]
    u, v = np.meshgrid(np.linspace(0, 1, face.uEvalPtSize), np.linspace(0, 1, face.vEvalPtSize))
    loc3d = face.evaluate_list(ctpall[face.ctp_index[0]:face.ctp_index[1]],
                             np.stack([u.transpose(), v.transpose()], -1).reshape(-1, 2))
    loc3d = np.array(loc3d)
    inner = face.inside_evalpts.reshape(-1)
face_id = 'Surface0'
face = solid.faces[face_id]
u, v = np.meshgrid(np.linspace(0, 1, face.uEvalPtSize), np.linspace(0, 1, face.vEvalPtSize))
loc3d = face.evaluate_list(ctpall[face.ctp_index[0]:face.ctp_index[1]],
                           np.stack([u.transpose(), v.transpose()], -1).reshape(-1, 2))
loc3d = np.array(loc3d)
inner = face.inside_evalpts.reshape(-1)
ax.scatter(loc3d[~inner, 0], loc3d[~inner, 1], loc3d[~inner, 2], color='black')
ax.scatter(loc3d[inner, 0], loc3d[inner, 1], loc3d[inner, 2], color='red')
#%%
##############Read in the point cloud data #####################
targets = []
for i in dataFileNames:
    targets += [extract_pc(i, sample_rate=1/15)]
target_centroid = np.mean([x.mean(0) for x in targets],0)
targets = [x-target_centroid for x in targets] # move the centroid of point cloud to origin
#################registration and mathcing##############
evaluated_ptcloud = []
for face_id, face in solid.faces.items(): # evaluate the points on each surface
    # geomdl_face = face.create_3dNURBS(ctpall[face.ctp_index[0]:face.ctp_index[1],:])
    # evaluated_ptcloud.append(np.array(geomdl_face.evalpts))
    evaluated_ptcloud.append(face.evalpts(ctpall[face.ctp_index[0]:face.ctp_index[1]].reshape(face.size_u,face.size_v, 3)).squeeze(0)[face.inside_evalpts,:])
col_ind = match_surface_id(evaluated_ptcloud, targets) # find the correct surface id between CAD and NDT ptcloud
targets_rearrange = [targets[i] for i in col_ind]
targets = targets_rearrange
# fig, ax = plot_solid(solid, ctpall)
# plot_list_points(targets, new_fig=False, ax=ax, fig=fig)
#################prepare feed in data and optimizer##################
inpCtrlPts_all = torch.nn.Parameter(torch.from_numpy(ctpall).to(device))
inpWeight_all = torch.nn.Parameter(torch.from_numpy(np.concatenate([copy.deepcopy(face.weight).reshape(-1) for face in solid.faces.values()]))).to(device)
opt = torch.optim.SGD(iter([inpCtrlPts_all]), lr=1e-2)# ....
scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=np.arange(2000, 40000, 2000), gamma=0.5)
#############find the basis of A*pA=B*pB, pA,pB are control points, A,B are linear vectors composed of NURBS basis function Ni,p and Nj,q#############
'''
The all basis matrix is (n_edges*sample_points)*sum_of_parameters. Each row is a constraint. sparse matrix. The index
of parameters on each surface is determined arec calculated by the function reindex_ctp_from_all()
'''
sample_size = 15
all_basis = np.zeros((sample_size*len(solid.edges), ctpall.shape[0]))
coef_sign = 1
for edge_idx, current_trim in enumerate(solid.edges.values()):
    coef_sign = 1
    for surf_name, edge in current_trim.trims.items():
        if surf_name == 'Surface0':
            print('yeee')
            pass
        # d3 = []
        # for surf_name, edge in current_trim.trims.items():
        #     edge_cp = copy.deepcopy(edge)
        #     if surf_name == 'Surface0':
        #         edge_cp.reverse()
        #     tmp = edge_cp.evalpts
        #     d3.append(solid.faces[surf_name].evaluate_list(ctpall[solid.faces[surf_name].ctp_index[0]:solid.faces[surf_name].ctp_index[1]], tmp))
        # d3 = np.array(d3)
        # print(d3[0,...]-d3[1,...])
        on_surf = solid.faces[surf_name]
        edge_cp = copy.deepcopy(edge)
        edge_cp.sample_size = sample_size
        if coef_sign == 1:
            uv_list = edge_cp.evalpts
            uv_list_3d = on_surf.evaluate_list(ctpall[on_surf.ctp_index[0]:on_surf.ctp_index[1]], uv_list)
        else: #todo uv_list在两个曲面上的trim有不同的knotvector，所以不能直接用evalpts，要通过反向搜索找到对应的2d坐标，但还是不对？
            #todo 0123 最后的点binary search错误？但没报错，为什么？
            uv_list = [binary_grid_search(on_surf.create_3dNURBS(ctpall[on_surf.ctp_index[0]:on_surf.ctp_index[1]]), np.array(loc_3d)) for loc_3d in uv_list_3d]
            uv_list_3d2 = on_surf.evaluate_list(ctpall[on_surf.ctp_index[0]:on_surf.ctp_index[1]], uv_list)
            if ((np.array(uv_list_3d2) - np.array(uv_list_3d)) ** 2).sum() >= 1.1:
                print('loss')
        for idx, uv in enumerate(uv_list):
            u_span = find_span_binsearch(3, on_surf.knotvector_u, on_surf.size_u, uv[0])
            v_span = find_span_binsearch(3, on_surf.knotvector_v, on_surf.size_v, uv[1])
            Nu = np.zeros((on_surf.size_u, 1))
            Nv = np.zeros((on_surf.size_v, 1))
            Nu_cp = np.array(basis_function(3, on_surf.knotvector_u, u_span, uv[0]))
            Nv_cp = np.array(basis_function(3, on_surf.knotvector_v, v_span, uv[1]))
            Nu[u_span - 3:u_span + 1] = Nu_cp[:, np.newaxis]
            Nv[v_span - 3:v_span + 1] = Nv_cp[:, np.newaxis]
            N = Nu @ Nv.transpose()
            all_basis[idx+edge_idx*sample_size, on_surf.ctp_index[0]:on_surf.ctp_index[1]] = N.reshape(-1)*coef_sign
        coef_sign*=-1
edge_idx = 7
current_trim=list(solid.edges.values())[7]
surf_name, edge = list(current_trim.trims.items())[0]
all_basis = np.array(all_basis)
from scipy.linalg import orth
norm_basis=orth(all_basis.T, sys.float_info.epsilon*all_basis.max()*1e13).T
print(norm_basis.shape)
#%%
pbar = tqdm(range(10000))
All_loss = []
All_chamfer = []
All_area = []
All_curv = []
ALl_hausdorff = []
saved_ctrl_pts = torch.zeros([1, 6, 6, 3])
fig_i = 0
now_loss = np.inf
controller = 0
loss_str=1
saved_inpCtrlPts_all = []
for i in pbar:
    saved_inpCtrlPts_all.append(inpCtrlPts_all.detach().cpu())
    def closure(controller):
        opt.zero_grad()
        lossVal = 0
        loss_iter = 0
        chamfer_iter = 0
        area_iter = 0
        curv_iter = 0
        hausdorff_iter = 0


        for surf_i, face in enumerate(solid.faces.values()):
            weight = torch.tensor(face.weight, device=device).unsqueeze(-1)
            inpCtrlPts = inpCtrlPts_all[face.ctp_index[0]:face.ctp_index[1], ...]
            inpCtrlPts = inpCtrlPts.view(face.size_u, face.size_v, 3)
            layer = face.DiffSurf
            target = torch.tensor(targets[surf_i])
            numPoints = target.shape


            out = layer(torch.cat((inpCtrlPts.unsqueeze(0), weight.unsqueeze(0)), axis=-1))
            out_inside = out[0,face.inside_evalpts,:]
            if torch.sum(torch.isnan(out)):
                0/0

            surf_area = area_from_points(out.squeeze())
            der1 = (- out[:, 0:-2, 1:-1, :] + out[:, 2:, 1:-1, :]).squeeze()
            der2 = (- out[:, 1:-1, 0:-2, :] + out[:, 1:-1, 2:, :]).squeeze()
            norm = torch.cross(der1,der2)
            der11 = (- 2 * out[:, 1:-1, 1:-1, :] + out[:, 0:-2, 1:-1, :] + out[:, 2:, 1:-1, :]).squeeze()
            der22 = (- 2 * out[:, 1:-1, 1:-1, :] + out[:, 1:-1, 0:-2, :] + out[:, 1:-1, 2:, :]).squeeze()
            der12 = (- 2 * out[:, 1:-1, 1:-1, :] + out[:, 0:-2, 1:-1, :] + out[:, 1:-1, 2:, :]).squeeze()
            der21 = (- 2 * out[:, 1:-1, 1:-1, :] + out[:, 1:-1, 0:-2, :] + out[:, 2:, 1:-1, :]).squeeze()

            E = (der1*der1).sum(-1)
            F = (der1*der2).sum(-1)
            G = (der2*der2).sum(-1)

            L = (der11*norm).sum(-1)
            M = (der12*norm).sum(-1)
            N = (der22*norm).sum(-1)

            K_nom = (L*N-M*M)
            K_den = (E*G-F*F)

            out_ = torch.ones(out.size())
            out_[:, ~face.inside_evalpts, :] = torch.nan
            der1_ = (- out_[:, 0:-2, 1:-1, :] + out_[:, 2:, 1:-1, :]).squeeze()
            der2_ = (- out_[:, 1:-1, 0:-2, :] + out_[:, 1:-1, 2:, :]).squeeze()
            norm_ = torch.cross(der1_,der2_)
            der11_ = (- 2 * out_[:, 1:-1, 1:-1, :] + out_[:, 0:-2, 1:-1, :] + out_[:, 2:, 1:-1, :]).squeeze()
            der22_ = (- 2 * out_[:, 1:-1, 1:-1, :] + out_[:, 1:-1, 0:-2, :] + out_[:, 1:-1, 2:, :]).squeeze()
            der12_ = (- 2 * out_[:, 1:-1, 1:-1, :] + out_[:, 0:-2, 1:-1, :] + out_[:, 1:-1, 2:, :]).squeeze()
            der21_ = (- 2 * out_[:, 1:-1, 1:-1, :] + out_[:, 1:-1, 0:-2, :] + out_[:, 2:, 1:-1, :]).squeeze()

            E_ = (der1_*der1_).sum(-1)
            F_ = (der1_*der2_).sum(-1)
            G_ = (der2_*der2_).sum(-1)

            L_ = (der11_*norm_).sum(-1)
            M_ = (der12_*norm_).sum(-1)
            N_ = (der22_*norm_).sum(-1)

            K_nom_ = (L_*N_-M_*M_)
            K_den_ = (E_*G_-F_*F_)


            # K_nom = K_nom[~torch.isnan(K_nom_)]
            # K_den = K_den[~torch.isnan(K_den_)]

            K = K_nom/K_den
            if torch.sum(torch.isnan(K)):
                0/0
            max_curv = torch.max(torch.abs(K))


            # Use the cos value among control points
            diff11 = (inpCtrlPts[:-1, :, :] - inpCtrlPts[1:, :, :]).square().sum(-1)
            diff12 = (inpCtrlPts[:-2, :, :] - inpCtrlPts[2:, :, :]).square().sum(-1)
            cos1 = (diff11[:-1, ...] + diff11[1:, ...] - diff12) / ((4 * diff11[:-1, ...] * diff11[1:, ...]).sqrt())

            diff21 = (inpCtrlPts[:, :-1, :] - inpCtrlPts[:, 1:, :]).square().sum(-1)
            diff22 = (inpCtrlPts[:, :-2, :] - inpCtrlPts[:, 2:, :]).square().sum(-1)
            cos2 = (diff21[:, :-1] + diff21[:, 1:] - diff22) / ((4 * diff21[:, :-1] * diff21[:, 1:]).sqrt())
            max_cos = torch.max(torch.max(cos1), torch.max(cos2))+1


            # lossVal = 0
            if True:
                chamfer = chamfer_distance_one_side(out_inside.view(1, -1, 3),
                                                    target.view(1, numPoints[0], 3).to(device), face=face, side=1)
                if loss_str==1:
                    hausdorff = Hausdorff_distance_cust(out_inside.view(1, -1, 3),
                                                    target.view(1, numPoints[0], 3).to(device), side=0)
                else:
                    hausdorff = Hausdorff_distance_one_side(out_inside.view(1, -1, 3),
                                                       target.view(1, numPoints[0], 3).to(device))
            else:
                lossVal = chamfer_distance_one_side(out.view(1, face.uEvalPtSize * face.vEvalPtSize, 3),
                                                    target.view(1, numPoints[0], 3),face=face)
                chamfer = lossVal.item()
            if controller and max_cos > 1.6:
                controller = 0
            elif not controller and max_cos <= 1.4:
                controller = 1
            controller = 0
            # Minimize maximum curvature
            lambda_0 = 1
            lambda_1 = 3
            lambda_2 = 500
            lambda_3 = 20
            # lossVal += lambda_0*chamfer+lambda_1*hausdorff
            lossVal += lambda_0*chamfer+lambda_1*hausdorff+500*max_curv+20*max_cos
            loss_iter += lossVal.item()
            chamfer_iter += chamfer.item()
            area_iter += surf_area.item()
            curv_iter += max_curv.item()
            hausdorff_iter += hausdorff.item()

        # Back propagate
        lossVal.backward(retain_graph=True)
        current_grad = copy.deepcopy(inpCtrlPts_all.grad.detach())
        for index_i, face in enumerate(solid.faces.values()):
            input_grad = current_grad[face.ctp_index[0]:face.ctp_index[1],:]
            input_grad = input_grad.reshape(face.size_u, face.size_v, 3)
            new_grad = grid_avg_grad(input_grad, 2)
            current_grad[face.ctp_index[0]:face.ctp_index[1],:] = new_grad
        inpCtrlPts_all.grad = current_grad

        return loss_iter, chamfer_iter, hausdorff_iter, area_iter, curv_iter



    iter_loss_sum = 0
    # Optimize step

    lossVal, chamfer, hausdorff, surf_area, max_curv = opt.step(functools.partial(closure, controller=controller))
    for xi in range(3):
        orthogonal = 0
        tmp = torch.multiply(inpCtrlPts_all[..., xi].cpu().clone().detach(), torch.tensor(norm_basis)).sum(-1).numpy()
        for constr_i in range(len(tmp)):
            orthogonal += tmp[constr_i] * norm_basis[constr_i,...]
        orthogonal = torch.tensor(orthogonal)
        a = inpCtrlPts_all.data[...,xi].cpu().clone().detach()-orthogonal
        inpCtrlPts_all.data[...,xi] = a
    iter_loss_sum += lossVal
    scheduler.step()
    pbar.set_description("Total loss is %s: %s, the new area is %s" % (i + 1, lossVal, surf_area))
    pass


#%%
ctp_now = copy.deepcopy(inpCtrlPts_all.cpu().detach())
fig, ax = plot_solid(solid, ctp_now)
plot_list_points(targets, new_fig=False, ax=ax, fig=fig)
#%%
face = list(solid.faces.values())[0]
a=face.create_3dNURBS(ctpall[face.ctp_index[0]:face.ctp_index[1], :])
edge1 = list(face.edges.values())[0]
edge1_geomdl = edge1.trims[face.surf_id]
edge1_geomdl.ctrlpts = [[0.5,0],[1,1]]
a.add_trim(edge1_geomdl)
#%%
plot_solid_from_geomdl(solid, inpCtrlPts_all.cpu().detach())
#%%
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')
colorpanel = np.arange(6) / (6 - 1)
cmap = mpl.cm.get_cmap('PRGn')
colorpanel = cmap(colorpanel)
color_count = 0
ctp_now = copy.deepcopy(inpCtrlPts_all.cpu().detach())
for face_id in solid.faces:
    face = solid.faces[face_id]
    pts = face.evalpts(ctp_now[face.ctp_index[0]:face.ctp_index[1]].reshape(face.size_u,face.size_v,3))
    pts = pts.squeeze(0)
    ax.scatter(pts[face.inside_evalpts,0],pts[face.inside_evalpts,1],pts[face.inside_evalpts,2],color=colorpanel[color_count])
    color_count += 1

pt = solid.faces['Surface0'].edges['Curve4'].trims['Surface0'].evalpts
for i in range(len(pt)):
    plt.scatter(pt[i][0], pt[i][1], color='red')
pt = solid.faces['Surface0'].edges['Curve1'].trims['Surface0'].evalpts
for i in range(len(pt)):
    plt.scatter(pt[i][0], pt[i][1], color='blue')
pt = solid.faces['Surface0'].edges['Curve7'].trims['Surface0'].evalpts
for i in range(len(pt)):
    plt.scatter(pt[i][0], pt[i][1], color='red')
pt = solid.faces['Surface0'].edges['Curve10'].trims['Surface0'].evalpts
for i in range(len(pt)):
    plt.scatter(pt[i][0], pt[i][1], color='blue')
#%%
#Calculate gapl
surf_old = [None]*6
for i in range(6):
    surf_old[i] = NURBS.Surface()
    surf_old[i].degree_u = info['uv_degree'][i][0]
    surf_old[i].degree_v = info['uv_degree'][i][1]
    ctp = info['ctp'][i]
    ctp = np.concatenate((ctp, np.ones((info['uv_ctp_count'][i][0], info['uv_ctp_count'][i][0], 1))), -1)
    surf_old[i].set_ctrlpts(ctp.reshape(-1,4).tolist(),info['ctp'][i].shape[0],info['ctp'][i].shape[1])
    surf_old[i].knotvector_u = info['knot_u'][i]
    surf_old[i].knotvector_v = info['knot_v'][i]
u_array = np.array([0, 0, 1, 1])
v_array = np.array([0, 1, 0, 1])
diff_area = []
count = 0
for i in range(6-1):
    for j in np.arange(i+1,6):
        u_array_i = u_array / (factor if i == factor_face else 1)
        v_array_i = v_array / (factor if i == factor_face else 1)
        u_array_j = u_array / (factor if j == factor_face else 1)
        v_array_j = v_array / (factor if j == factor_face else 1)
        points_i = np.array(surf[target_surf_list[i]].evaluate_list([*zip(u_array_i,v_array_i)]))
        points_j = np.array(surf[target_surf_list[j]].evaluate_list([*zip(u_array_j,v_array_j)]))
        points_i=points_i[:,np.newaxis,:]
        points_j=points_j[np.newaxis,:,:]
        dist = np.square(points_i-points_j).sum(-1)
        locs = np.where(dist < DELTA)
        if len(locs[0]):
            u_i = np.linspace(u_array_i[locs[0][0]], u_array_i[locs[0][1]], num=100)
            v_i = np.linspace(v_array_i[locs[0][0]], v_array_i[locs[0][1]], num=100)
            u_j = np.linspace(u_array_j[locs[1][0]], u_array_j[locs[1][1]], num=100)
            v_j = np.linspace(v_array_j[locs[1][0]], v_array_j[locs[1][1]], num=100)
            points_i_new = np.array(surf_new[i].evaluate_list(zip(u_i,v_i)))
            points_j_new = np.array(surf_new[j].evaluate_list(zip(u_j,v_j)))
            length_i = np.sqrt(((points_i_new[:-1,:]-points_i_new[1:,:])**2).sum(-1))
            diff_ij = np.sqrt(((points_i_new - points_j_new)**2).sum(-1))
            diff_ij = (diff_ij[:-1]+diff_ij[1:])/2
            # print(max(diff_ij))
            diff_area.append(np.sum(np.abs(diff_ij)) / np.sum(np.abs(length_i)))
print(sum(diff_area))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')
chamfer_final = []
hausdorff_final = []
for face_id, face in solid.faces.items():
    print(face_id)
    if face_id == 'Surface0':
        geomdl_surf = face.create_3dNURBS(inpCtrlPts_all.cpu().detach()[face.ctp_index[0]:face.ctp_index[1], :].reshape(face.size_u, face.size_v, 3))
        geomdl_surf.delta = (0.01,0.01)
        uv_list = [[u,v]  for u in np.linspace(0,1,100) for v in np.linspace(0,1,100)]
        degs = [in_trim((u,v), face_id) for u,v in uv_list]
        inner = abs(np.array(degs)) >= 300
        a = np.array(geomdl_surf.evalpts)
        a = a[inner].reshape(1,-1,3)
    else:
        geomdl_surf = face.create_3dNURBS(inpCtrlPts_all.cpu().detach()[face.ctp_index[0]:face.ctp_index[1], :].reshape(face.size_u, face.size_v, 3))
        geomdl_surf.delta = (0.01,0.01)
        a = np.array(geomdl_surf.evalpts)
        a = a.reshape(1,-1,3)
    b = targets[int(face_id[-1])][np.newaxis, ...]
    chamfer_final.append(chamfer_distance_one_side(a, b, face=face).cpu().detach().numpy().tolist())
    hausdorff_final.append(Hausdorff_distance_one_side(a, b).cpu().detach().numpy().tolist())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')
    ax.scatter(a[0, :, 0], a[0, :, 1], a[0, :, 2], color='red')
    ax.scatter(b[0, :, 0], b[0, :, 1], b[0, :, 2], color='blue')

print(chamfer_final)
print(hausdorff_final)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d', adjustable='box', proj_type='ortho')
ax.scatter(a[0,:,0],a[0,:,1],a[0,:,2],color='red')
ax.scatter(b[0,:,0],b[0,:,1],b[0,:,2],color='blue')

print(face_id)
degs = []
face = solid.faces[face_id]
for u in np.linspace(0, 1, face.uEvalPtSize):
    for v in np.linspace(0, 1, face.vEvalPtSize):
        degs.append(in_trim((u, v), face_id))
degs = np.array(degs)
degs = degs.reshape(face.uEvalPtSize, face.vEvalPtSize)
inner = abs(degs) >= 300
face.inside_evalpts = inner