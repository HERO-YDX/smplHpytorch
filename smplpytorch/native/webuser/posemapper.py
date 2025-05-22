import chumpy as ch
import numpy as np
import cv2


class Rodrigues(ch.Ch):
    dterms = 'rt'

    def compute_r(self):
        return cv2.Rodrigues(self.rt.r)[0]

    def compute_dr_wrt(self, wrt):
        if wrt is self.rt:
            return cv2.Rodrigues(self.rt.r)[1].T


def lrotmin(p):
    if isinstance(p, np.ndarray):
        p = p.ravel()[3:]
        return np.concatenate([(cv2.Rodrigues(np.array(pp))[0] - np.eye(3)).ravel() for pp in p.reshape((-1, 3))]).ravel()
    if p.ndim != 2 or p.shape[1] != 3:
        p = p.reshape((-1, 3))
    p = p[1:]
    return ch.concatenate([(Rodrigues(pp) - ch.eye(3)).ravel() for pp in p]).ravel()


def posemap(s):
    original_s_for_error_msg = str(s)
    original_type_for_error_msg = type(s)

    processed_s = s

    if isinstance(processed_s, np.ndarray):
        if processed_s.size == 1:
            processed_s = processed_s.item()
        else:
            pass

    s_str = processed_s
    if isinstance(s_str, np.bytes_):
        s_str = s_str.decode('utf-8')
    elif isinstance(s_str, bytes):
        s_str = s_str.decode('utf-8')

    if s_str == 'lrotmin':
        return lrotmin
    else:
        raise Exception('Unknown posemapping: %s (original type: %s, processed as type: %s, final string form: %s)' %
                        (original_s_for_error_msg, original_type_for_error_msg, type(processed_s), str(s_str)))

