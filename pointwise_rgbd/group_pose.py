"""Translation-first spatial left errors; rigid transforms generate all XYZ."""
import torch


def skew(x):
    a,b,c=x.unbind(-1);zero=torch.zeros_like(a)
    return torch.stack((zero,-c,b,c,zero,-a,-b,a,zero),-1).reshape(*x.shape[:-1],3,3)


def ad(x):
    out=x.new_zeros((*x.shape[:-1],6,6));w=skew(x[...,3:])
    out[...,:3,:3]=w;out[...,3:,3:]=w;out[...,:3,3:]=skew(x[...,:3]);return out


def exp(x):
    hat=x.new_zeros((*x.shape[:-1],4,4));hat[...,:3,:3]=skew(x[...,3:]);hat[...,:3,3]=x[...,:3]
    return torch.linalg.matrix_exp(hat)


def left_jacobian(x):
    block=x.new_zeros((*x.shape[:-1],12,12));block[...,:6,:6]=ad(x);block[...,:6,6:]=torch.eye(6,device=x.device,dtype=x.dtype)
    return torch.linalg.matrix_exp(block)[...,:6,6:]


def chart_reset(delta):
    out=delta.new_zeros((*delta.shape[:-1],12,12));out[...,:6,:6]=left_jacobian(delta[...,:6]);out[...,6:,6:]=torch.eye(6,device=delta.device,dtype=delta.dtype);return out


def transport(canonical,membership,poses):
    t=poses[membership]
    return (t[:,:3,:3]@canonical[:,:,None])[...,0]+t[:,:3,3]


def point_jacobian(xyz):
    out=xyz.new_zeros((*xyz.shape[:-1],3,6));out[...,:3]=torch.eye(3,device=xyz.device,dtype=xyz.dtype);out[...,3:]=-skew(xyz);return out


def marginal_position_covariance(xyz,membership,covariance):
    g=point_jacobian(xyz);return g@covariance[membership,:6,:6]@g.transpose(-1,-2)


def cross_position_covariance(x,y,pose_covariance):
    return point_jacobian(x)@pose_covariance@point_jacobian(y).transpose(-1,-2)
