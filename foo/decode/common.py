#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Nicot
# @Date:   2018-04-16 18:03:47
# @Last Modified by:   Nicot
# @Last Modified time: 2018-04-16 18:03:48
import numpy as np


def zdw_name2vec(name):
    name = name.split('_')[2].lower()
    vector = np.zeros(4 * 36)
    for i, c in enumerate(name):
        idx = 0
        if '9' >= c >= '0':
            idx = i * 36 + int(c)
        else:
            idx = i * 36 + ord(c) - 97 + 10
        vector[idx] = 1
        pass
    pass
    return vector


pass


def zdw_vec2name(vec):
    name = []
    for i in vec:
        if int(i) < 10:
            name.append(chr(int(i) + 48))
        else:
            name.append(chr(int(i) - 10 + 97))
        pass
    return "".join(name)

pass


def invoice_name2vec(name):
    name = name.split('_')[1]
    vector = np.zeros(4 * 62)
    for i, c in enumerate(name):
        idx = 0
        if '9' >= c >= '0':
            idx = i * 62 + int(c)
        elif 'Z' >= c >= 'A':
            idx = i * 62 + ord(c) - 65 + 10
        else:
            idx = i * 62 + ord(c) - 97 + 36
        vector[idx] = 1
        pass
    pass
    return vector


pass


def invoice_vec2name(vec):
    name = []
    for i in vec:
        if int(i) < 10:
            name.append(chr(int(i) + 48))
        elif 36 > int(i) >= 10:
            name.append(chr(int(i) - 10 + 65))
        else:
            name.append(chr(int(i) - 36 + 97))
        pass
    return "".join(name)

pass

if __name__ == '__main__':
    vec = invoice_name2vec("a_UXD7")
    print(vec)
    vec1 = []
    for i, v in enumerate(vec):
        if int(v) == 1:
            vec1.append(i % 62)
    print(invoice_vec2name(vec1))
