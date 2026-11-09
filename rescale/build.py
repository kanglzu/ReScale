"""Build only the shared 5.73M ReScale predictor and its recursive geometry."""
from copy import deepcopy
from rescale.models.predictor import RecursiveUNet
from rescale.recursion.framework import RecursiveSegFramework

def build_model(cfg):
    m = deepcopy(cfg['model'])
    name = m.pop('name', 'RecursiveSegFramework')
    if name not in ('ReScale', 'RecursiveSegFramework'): raise ValueError('Only ReScale is included')
    for k in ('ds_weights', 'find_unused_parameters', 'broadcast_buffers'): m.pop(k, None)
    if m.pop('backbone', 'RecursiveUNet') != 'RecursiveUNet': raise ValueError('External backbones are not included')
    opts = deepcopy(cfg['innovations'])
    rec = opts['recursion']
    if rec.get('crop_source') != 'resized' or rec.get('rounds', 2) != 2 or not rec.get('enabled', True):
        raise ValueError('This release uses two passes with resized-input crops')
    if rec.get('separate_heads', False) or rec.get('separate_bn_statistics', False) or not rec.get('share_backbone', True):
        raise ValueError('The paper model shares all weights and heads')
    if rec.get('sampling_strategy', 'crop') != 'crop': raise ValueError('Only component-wise crops are supported')
    net = opts['network']
    if not all(net.get(k, True) for k in ('enabled', 'scb', 'scale_conditioning', 'bottleneck_mamba')) or net.get('morphology', False):
        raise ValueError('Use the released SCB + compressed Mamba configuration')
    if net.get('bottleneck_type', 'mamba') != 'mamba': raise ValueError('Use the compressed Mamba bottleneck')
    if net.get('scale_input', 'geometric') != 'geometric': raise ValueError('ReScale uses geometric crop extents')
    input_size = m.pop('input_size', cfg['data']['image_size'])
    mean, std = m.pop('mean', None), m.pop('std', None)
    backbone = RecursiveUNet(innovation=net, **m)
    return RecursiveSegFramework(backbone, input_size, innovations=opts, mean=mean, std=std)
