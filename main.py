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
from geomdl import NURBS
from pytorch3d.loss import chamfer_distance
from scipy.optimize import linear_sum_assignment
from geomdl.helpers import find_span_binsearch, basis_function
from geomdl.construct import extract_curves
from geomdl.operations import length_curve
sys.path.insert(1, 'nurbspy')
mpl.use('Qt5Agg')

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
    return {'uv_degree': [np.array(degree)], "uv_ctp_count": [np.array(CtrlPtsCountUV)], "ctp_count": [np.array(CtrlPtsTotal)], "knot_u": [knotU], "knot_v": [knotV], "ctp": [CtrlPtsNoW], "weight": [Weights]}


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


def chamfer_distance_one_side(pred, gt, side=1):
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


class Surface:
    _init_ctp = None
    _ctp = None
    weight = None
    degree_u = None
    degree_v = None
    knotvector_u = None
    knotvector_v = None
    size_u = None
    size_v = None
    surf_id = None
    NURBS3d = None
    uEvalPtSize = None
    vEvalPtSize = None
    device = None
    DiffSurf = None
    uvEvalPtSize_max = 512
    trim3d = []
    trim2d = []
    def __init__(self, surface, surf_id, device='cpu'):
        self._init_ctp = surface['ctp'][0] # the inital control points: 3d list type, u*v*3
        self._ctp = surface['ctp'][0] # the current control points: 3d list type u*v*3
        self.weight = surface['weight'][0] # the weight of control points: 2d list type u*v
        self.degree_u = surface['uv_degree'][0][0] # int
        self.degree_v = surface['uv_degree'][0][1] # int
        self.knotvector_u = surface['knot_u'][0] # list
        self.knotvector_v = surface['knot_v'][0] # list
        self.size_u = surface['uv_ctp_count'][0][0] # int
        self.size_v = surface['uv_ctp_count'][0][1] # int
        self.device = device
        self.surf_id = surf_id
        self._create_3dNURBS()
        self.set_uv_EvalPts()
        self.create_diff_NURBS()
        pass

    def _create_3dNURBS(self):
        '''
        It creates the 3D representation of the NURBS surface via geomdl.
        :return:
        '''
        surf3d = NURBS.Surface()
        surf3d.degree_u = self.degree_u
        surf3d.degree_v = self.degree_v
        ctp = self._ctp
        ctp = np.concatenate((ctp, self.weight.reshape(self.size_u,self.size_v, 1)), -1)
        surf3d.set_ctrlpts(ctp.reshape(-1, 4).tolist(), self.size_u, self.size_v)
        surf3d.knotvector_u = self.knotvector_u
        surf3d.knotvector_v = self.knotvector_v
        surf3d.delta = 0.01
        self.NURBS3d = surf3d
        pass


    def _calculate_uv_EvalPts(self):
        '''
        It calculates the number of points to be evaluated on the surface in u or v direction. The total number of
        evaluated points should not exceeds the max value. The number of evaluated points on each direction is proportional
        to the average length of curves on u and v direction.
        :return: a set of integer, size=2
        '''
        curves_dict = extract_curves(self.NURBS3d)
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
                            dvc=device)
            self.DiffSurf = surf
            # check whether there is null value error in the evaluated points or not
            tmp = self.evalpts()
        if sum([np.sum(np.isnan(x)) for x in tmp]):
            self.DiffSurf = None
            raise Exception('Fail to create differntiable surface. try to decrease the evaluate size.')
        pass


    def evalpts(self, ctp=None):
        '''
        It evaluate points using differentiable NURBS surface. The number of evaluated points is predetermiend.
        :param ctp:
        :return: an array of points in 3D space
        '''
        if not ctp:
            ctp = self._ctp
        return self.DiffSurf(torch.cat((torch.tensor(self._ctp, dtype=torch.float, device=self.device).unsqueeze(0),
                                         torch.tensor(self.weight, dtype=torch.float, device=self.device).unsqueeze(
                                             0).unsqueeze(-1)), axis=-1)).cpu().detach().numpy()

    def evaluate_single(self, param):
        '''
        It evaluates the single point at 3D space given 2D parameters.
        :param param: a point in parameter space. size=2
        :return: an array in 3D space. size=3
        '''
        return self.NURBS3d.evaluate_single(param)


    def evaluate_list(self, param_list):
        '''
        It evaluates a list of points at 3D space given 2D parameters.
        :param param_list: a list of points in parameter space. size=n*2
        :return: a list of points in 3D space. size=n*3
        '''
        return self.NURBS3d.evaluate_list(param_list)


    def set_device(self, device):
        '''
        set the device to be a specific cuda or cpu
        :param device: string
        :return:
        '''
        self.device = device
        pass


    def set_trim(self, trim_curve, curve_info):
        '''
        Bound the trim curve to the surface. It also creates the 2d representation of the trim curve on the surface.
        :param trim_curve:
        :param curve_info:
        :return:
        '''
        self.trim3d.append(trim_curve)
        ctp = curve_info['ctp']
        weight = curve_info['weight']
        degree = curve_info['degree']
        knotvector = curve_info['knot']
        size = curve_info['ctp_count']
        curve2d = NURBS.Curve()
        curve2d.degree = degree
        ctp = np.concatenate((ctp, weight.reshape(size, 1)), -1)
        curve2d.set_ctrlpts(ctp.reshape(-1, 3).tolist(), size)
        curve2d.knotvector = knotvector.tolist()
        curve2d.delta = 0.01
        self.trim2d.append(curve2d)
        return len(self.trim2d)-1


class Curve:
    _init_ctp = None
    _ctp = None
    weight = None
    degree = None
    knotvector = None
    size = None
    NURBS3d = None
    surf_i = None
    surf_j = None
    surf_i_index = None
    surf_j_index= None
    def __init__(self, curve, curve_id):
        self._init_ctp = curve['ctp'] # 3d list type, u*v*3
        self._ctp = curve['ctp'] # 3d list type u*v*3
        self.weight = curve['weight'] # 2d list type u*v
        self.degree = curve['degree'] # int
        self.knotvector = curve['knot'] # list
        self.size = curve['ctp_count'] # int
        self._create_3dNURBS()
        self.surf_i = curve['face_i']
        self.surf_j = curve['face_j']
        self.curve_id = curve_id

    def _create_3dNURBS(self):
        '''
        Create the 3D NURBS curve via geomdl
        :return:
        '''
        curve3d = NURBS.Curve()
        curve3d.degree = self.degree
        ctp = self._ctp
        ctp = np.concatenate((ctp, self.weight.reshape(self.size, 1)), -1)
        curve3d.set_ctrlpts(ctp.reshape(-1, 4).tolist(), self.size)
        curve3d.knotvector = self.knotvector.tolist()
        curve3d.delta = 0.01
        self.NURBS3d = curve3d
        pass

#%%
outputname = "fig/20220925/case2_5/twist"
uvEvalPtSize_max = 512
device = 'cpu'
dataFileNames = ['data/NDE/pointCloud_twist - Cloud'+str(i+1)+'.txt' for i in np.arange(0,6)] #
jsonInFileNames = ['data/case2_5/surf_'+str(i)+'.json' for i in range(6)]
jsonOutFileName = "AMSurface.out.json"
DELTA = 1e-8
dimension = 3
trim_list = [[0,2,0,1,0,0,1,1,0,0],
             [0,3,0,0,0,1,0,0,0,0],
             [0,4,0,1,1,1,0,0,1,0],
             [0,5,1,1,0,1,1,1,0,0],
             [1,2,0,1,0,0,0,0,0,0],
             [1,3,0,0,0,1,1,1,0,0],
             [1,4,0,1,1,1,1,1,1,0],
             [1,5,1,1,0,1,0,0,0,0],
             [2,3,0,1,0,0,0,1,0,0],
             [2,5,0,1,1,1,0,1,1,0],
             [3,4,0,1,1,1,0,1,1,0],
             [4,5,0,1,1,1,0,1,1,1]] # surf_i,surf_j,ui1,ui2,vi1,vi2,uj1,uj2,vj1,vj2
################# read in json and point cloud#######################
'''
The following part needs to be replaced by the OCC later. 
'''
info = {'uv_degree': [], "uv_ctp_count": [], "ctp_count": [], "knot_u": [], "knot_v": [], "ctp": [], "weight": []}
solid_faces = []
for idx, file_i in enumerate(jsonInFileNames):
    solid_faces.append(Surface(extract_json(file_i), idx, device=device))
solid_edges = []
for curve_id in range(len(trim_list)):
    face_i = solid_faces[trim_list[curve_id][0]]
    face_j = solid_faces[trim_list[curve_id][1]]
    ui_range = np.linspace(trim_list[curve_id][2],trim_list[curve_id][3],4)
    vi_range = np.linspace(trim_list[curve_id][4],trim_list[curve_id][5],4)
    uj_range = np.linspace(trim_list[curve_id][6],trim_list[curve_id][7],4)
    vj_range = np.linspace(trim_list[curve_id][8],trim_list[curve_id][9],4)
    ctp = np.array(face_i.evaluate_list(list(zip(ui_range, vi_range))))
    knot = np.array([0,0,0,0,1,1,1,1])
    degree = 3
    weight = np.ones(4)
    ctp_count = 4
    curve_info = {'ctp':ctp, 'knot':knot,'degree':degree,'weight':weight,'ctp_count':ctp_count,'face_i':face_i,'face_j':face_j}
    curve_info_i = {'ctp':np.array([ui_range,vi_range]).T, 'knot':knot,'degree':degree,'weight':weight,'ctp_count':ctp_count}
    curve_info_j = {'ctp':np.array([uj_range,vj_range]).T, 'knot':knot,'degree':degree,'weight':weight,'ctp_count':ctp_count}
    edge = Curve(curve_info, curve_id)
    solid_edges.append(edge)
    index_i = face_i.set_trim(edge, curve_info_i)
    index_j = face_j.set_trim(edge, curve_info_j)
    edge.surf_i_index = index_i
    edge.surf_j_index = index_j


##############Read in the point cloud data #####################
targets = []
for i in dataFileNames:
    targets += [extract_pc(i, sample_rate=1/15)]

#################registration and mathcing##############
evaluated_ptcloud = []
for face in solid_faces:
    evaluated_ptcloud.append(face.evalpts()) # shift the scanned point clouds to align with CAD according to area weighted centroid
col_ind = match_surface_id(evaluated_ptcloud, targets) # find the correct surface id between CAD and NDT ptcloud
targets_rearrange = [targets[i] for i in col_ind]
targets = targets_rearrange
#################prepare feed in data and optimizer##################
inpCtrlPts_all = torch.nn.Parameter(torch.from_numpy(np.concatenate([copy.deepcopy(face._ctp).reshape(-1, 3) for face in solid_faces])).to(device))
inpWeight_all = torch.nn.Parameter(torch.from_numpy(np.concatenate([copy.deepcopy(face.weight).reshape(-1) for face in solid_faces]))).to(device)
opt = torch.optim.SGD(iter([inpCtrlPts_all]), lr=1e-2)# ....
scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=np.arange(2000, 40000, 2000), gamma=0.5)
#############find the basis of A*pA=B*pB, pA,pB are control points, A,B are linear vectors composed of NURBS basis function Ni,p and Nj,q#############
'''
The all basis matrix is (n_edges*sample_points)*sum_of_parameters. Each row is a constraint. sparse matrix. The index
of parameters on each surface is determined arec calculated by the function reindex_ctp_from_all()
'''
all_basis = []
for current_trim in solid_edges:
    uv_list_i = current_trim.surf_i.trim2d[0].evalpts
    uv_list_j = current_trim.surf_j.trim2d[0].evalpts
    for uvi, uvj in zip(uv_list_i, uv_list_j):
        ui_span = find_span_binsearch(3, current_trim.surf_i.knotvector_u, current_trim.surf_i.size_u, uvi[0])
        vi_span = find_span_binsearch(3, current_trim.surf_i.knotvector_v, current_trim.surf_i.size_v, uvi[1])
        Nui = np.zeros((current_trim.surf_i.size_u, 1))
        Nvi = np.zeros((current_trim.surf_i.size_v, 1))
        Nui_cp = np.array(basis_function(3, current_trim.surf_i.knotvector_u, ui_span, uvi[0]))
        Nvi_cp = np.array(basis_function(3, current_trim.surf_i.knotvector_v, vi_span, uvi[1]))
        Nui[ui_span - 3:ui_span + 1] = Nui_cp[:, np.newaxis]
        Nvi[vi_span - 3:vi_span + 1] = Nvi_cp[:, np.newaxis]
        Ni = Nui @ Nvi.transpose()

        uj_span = find_span_binsearch(3, current_trim.surf_j.knotvector_u, current_trim.surf_j.size_u, uvj[0])
        vj_span = find_span_binsearch(3, current_trim.surf_j.knotvector_v, current_trim.surf_j.size_v, uvj[1])
        Nuj = np.zeros((current_trim.surf_j.size_u, 1))
        Nvj = np.zeros((current_trim.surf_j.size_v, 1))
        Nuj_cp = np.array(basis_function(3, current_trim.surf_j.knotvector_u, uj_span, uvj[0]))
        Nvj_cp = np.array(basis_function(3, current_trim.surf_j.knotvector_v, vj_span, uvj[1]))
        Nuj[uj_span - 3:uj_span + 1] = Nuj_cp[:, np.newaxis]
        Nvj[vj_span - 3:vj_span + 1] = Nvj_cp[:, np.newaxis]
        Nj = Nuj @ Nvj.transpose()

        basis = np.zeros((sum([x.size_u*x.size_v for x in solid_faces])))  # n_surf*n_ctp_u*n_ctp_v
        basis[reindex_ctp_from_all(current_trim.surf_i.surf_id)] = Ni.reshape(-1)
        basis[reindex_ctp_from_all(current_trim.surf_j.surf_id)] = -Nj.reshape(-1)
        all_basis.append(basis)
############Re
all_basis = np.array(all_basis)
from scipy.linalg import orth
norm_basis=orth(all_basis.T, sys.float_info.epsilon*all_basis.max()*1e10).T
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
        for surf_i in range(len(solid_faces)):
            weight = torch.tensor(solid_faces[surf_i].weight, device=device).unsqueeze(-1)
            inpCtrlPts = inpCtrlPts_all[reindex_ctp_from_all(surf_i), ...]
            inpCtrlPts = inpCtrlPts.view(solid_faces[surf_i].size_u,solid_faces[surf_i].size_v,3)
            layer = solid_faces[surf_i].DiffSurf
            target = torch.tensor(targets[surf_i])
            numPoints = target.shape


            out = layer(torch.cat((inpCtrlPts.unsqueeze(0), weight.unsqueeze(0)), axis=-1))

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
                chamfer = (chamfer_distance_one_side(out.view(1, solid_faces[surf_i].uEvalPtSize * solid_faces[surf_i].vEvalPtSize, 3),
                                                    target.view(1, numPoints[0], 3).to(device), side=1)+
                           chamfer_distance_one_side(out.view(1, solid_faces[surf_i].uEvalPtSize * solid_faces[surf_i].vEvalPtSize, 3),
                                                    target.view(1, numPoints[0], 3).to(device), side=1))/2
                if loss_str==1:
                    hausdorff = Hausdorff_distance_cust(out.view(1, solid_faces[surf_i].uEvalPtSize * solid_faces[surf_i].vEvalPtSize, 3),
                                                    target.view(1, numPoints[0], 3).to(device), side=0)
                else:
                    hausdorff = Hausdorff_distance_one_side(out.view(1, solid_faces[surf_i].uEvalPtSize * solid_faces[surf_i].vEvalPtSize, 3),
                                                       target.view(1, numPoints[0], 3).to(device))
            else:
                lossVal = chamfer_distance_one_side(out.view(1, solid_faces[surf_i].uEvalPtSize * solid_faces[surf_i].vEvalPtSize, 3),
                                                    target.view(1, numPoints[0], 3))
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
            lossVal += lambda_0*chamfer+lambda_1*hausdorff+500*max_curv+20*max_cos
            loss_iter += lossVal.item()
            chamfer_iter += chamfer.item()
            area_iter += surf_area.item()
            curv_iter += max_curv.item()
            hausdorff_iter += hausdorff.item()

        # Back propagate
        lossVal.backward(retain_graph=True)
        current_grad = copy.deepcopy(inpCtrlPts_all.grad.detach())
        for index_i in range(6):
            input_grad = current_grad[reindex_ctp_from_all(index_i),:]
            input_grad = input_grad.reshape(solid_faces[index_i].size_u,solid_faces[index_i].size_v,3)
            new_grad = grid_avg_grad(input_grad, 2)
            current_grad[reindex_ctp_from_all(index_i),:] = new_grad
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
