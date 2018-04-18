#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Nicot
# @Date:   2018-04-13 21:24:12
# @Last Modified by:   Nicot
# @Last Modified time: 2018-04-13 21:24:59

import random

_UPPERCASELETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
_LOWERCASELETTERS = 'abcdefghijklmnopqrstuvwxyz'
_NUMBER = '0123456789'
_ALLSTRING = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'


def rdmchoices(choice, n=5):
    str = ''
    for i in range(n):
        str = str + random.choice(choice)
    return str


# 返回随机大小写数字字符串
rdmstr = (lambda n=5: rdmchoices(_ALLSTRING, n))
# 返回随机小写字母字符串
rdmlstr = (lambda n=5: rdmchoices(_LOWERCASELETTERS, n))
# 返回随机大写字母字符串
rdmustr = (lambda n=5: rdmchoices(_UPPERCASELETTERS, n))
# 返回随机数字字符串
rdmnstr = (lambda n=5: rdmchoices(_NUMBER, n))
#原random，random方法,返回字符串
rdm = (lambda : str(random.random()))


if __name__ == '__main__':
    print(rdmstr())
