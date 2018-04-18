#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Nicot
# @Date:   2018-04-13 10:58:51
# @Last Modified by:   Nicot
# @Last Modified time: 2018-04-14 00:42:21

import time
import os


class Logger():
    def __init__(self, modulename, timeformat='%Y-%m-%d %H:%M:%S', enablefile=False, filefullname='./out.log'):
        self._modulename = modulename
        self._enablefile = enablefile
        self._timeformat = timeformat
        self._filefullname = filefullname

    pass

    def _buildlog(self, data, level='DEBUG', e=''):
        logtime = time.strftime(self._timeformat, time.localtime(time.time()))
        return '%s %10s - [%s] - %s %s' % (logtime, self._modulename, level, str(data), e)

    pass

    def _echolog(self, logstr):
        print(logstr)
        if self._enablefile:
            cmd = 'echo %s >> %s' % (logstr, self._filefullname)
            os.system(cmd)

    def debug(self, data):
        self._echolog(self._buildlog(data, 'DEBUG'))

    pass

    def info(self, data):
        self._echolog(self._buildlog(data, 'INFO '))

    pass

    def error(self, data, e=''):
        self._echolog(self._buildlog(data, 'ERROR', str(e)))

    pass


if __name__ == '__main__':
    log = Logger('Logger', enablefile=True)
    log.info('hello logger')
    log.debug('aasasasasas')
