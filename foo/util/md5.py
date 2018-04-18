#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Nicot
# @Date:   2018-04-13 10:44:20
# @Last Modified by:   Nicot
# @Last Modified time: 2018-04-13 11:30:12

import hashlib


def string(data, encoding="utf-8"):  # md5加密字符串
    m = hashlib.md5(data.encode(encoding=encoding))
    return m.hexdigest()


pass


def byte(data):  # md5加密byte
    return hashlib.md5(data).hexdigest()


pass

if __name__ == '__main__':
    print(string('123456'))
