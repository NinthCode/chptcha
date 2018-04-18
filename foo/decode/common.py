#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Nicot
# @Date:   2018-04-16 18:03:47
# @Last Modified by:   Nicot
# @Last Modified time: 2018-04-16 18:03:48
import numpy as np

def name2vec(name, captcha_char_num, char_set_len):
    vector = np.zeros(captcha_char_num * char_set_len)
    for i, c in enumerate(name):
        idx = 0
        if (c >= '0' and c <= '9'):
            idx = i * char_set_len + int(c)
        else:
            idx = i * char_set_len + ord(c) - 97 + 10
        vector[idx] = 1
        pass
    pass
    return vector
pass


def vec2name(vec):
    name = []
    for i in vec:
        if int(i) < 10:
            name.append(chr(int(i) + 48))
        else:
            name.append(chr(int(i) - 10 + 97))
        pass
    pass
    return "".join(name)
pass
