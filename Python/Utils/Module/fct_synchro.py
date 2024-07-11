import numpy as np
import logging

#from Utils.Module.fct_numpy import *


def find_min_dt(t_ref, t_array, side='right'):
    if side == 'right':
        # t_ref > t_array[idx] recherché => ref se passe après t_array[idx]
        dt = t_ref - t_array
    elif side == 'left':
        # t_ref < t_array[idx] recherché => ref se passe avant t_array[idx]
        dt = t_array - t_ref
    else:
        dt = np.abs(t_ref - t_array)
    logging.debug('dt = {}'.format(dt))
    if not dt[dt > 0].size == 0:
        which_idx = np.where(dt == np.min(dt[dt >= 0]))[0]
        if which_idx.size > 1:
            logging.debug('more than one idx found : {}'.format(which_idx))
            logging.debug('at dt = {} for t_ref = {} vs {}'.format(dt[which_idx],
                                                                   t_ref,
                                                                   t_array[which_idx]))
            if is_all_equal(t_array[which_idx]):
                logging.debug('ok c est same yuv => on prend celui le plus à droite pour le + 1 !')
                which_idx = which_idx[-1]
            else:
                logging.warning('normalement on devrait etre en abs la !')
                which_idx = which_idx[0] ## on prend celui a gauche !!
    else:
        which_idx = - 1  ## première pict est avant le batch loade

    return which_idx