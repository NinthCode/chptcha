#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Nicot
# @Date:   2018-04-13 19:19:58
# @Last Modified by:   Nicot
# @Last Modified time: 2018-04-13 19:20:00
import sys
sys.path.append('../util/')

import time
import os
import sbconf
from logger import Logger
from dama2 import Dama2
from zdw import Zdw

log = Logger('main')

class Builder:
    def __init__(self, fatherforder, username, password, captchatype = 42, filenamesuffix='.png', samplenum=10, foldersize=1):
        self._fatherforder = fatherforder
        self._username = username
        self._password = password
        self._captchatype = captchatype
        self._samplenum = samplenum
        self._foldersize = foldersize
        self._filenamesuffix = filenamesuffix
        self._imagecount = 0
        self._folderimagecount = 61
        self._foldercount = 3
        self._nowfolder = self._createforder()
        self._nowfilename = self._createfilename()

    pass

    def _next(self):
        self._imagecount += 1
        self._folderimagecount += 1
        if self._folderimagecount >= self._foldersize:
            self._foldercount += 1
            self._nowfolder = self._createforder()
            self._folderimagecount = 0
        self._nowfilename = self._createfilename()
        pass

    pass

    def _createforder(self):
        forderpath = self._fatherforder + str(self._foldercount)
        if not os.path.exists(forderpath):
            log.debug('create new folder: ' + forderpath)
            os.makedirs(forderpath)
        return forderpath + '\\'

    pass

    def _createfilename(self):
        return time.strftime('%Y%m%d%H%M%S', time.localtime(time.time())) + '_' + str(
            self._folderimagecount)

    pass

    def _rename(self, captchastr, captchaid, status=''):
        oldname = self._nowfilename + self._filenamesuffix
        newname = self._nowfilename + '_' + captchastr + '_' + str(captchaid) + status + self._filenamesuffix
        log.debug('oldname: ' + oldname + ', newname: ' + newname)
        os.rename(os.path.join(self._nowfolder, oldname), os.path.join(self._nowfolder, newname))
        self._nowfilename = newname
    pass

    def build(self):
        d2 = Dama2(self._username, self._password)
        for i in range(self._samplenum):
            zdw = Zdw(self._nowfolder)
            if zdw.pullcaptcha(self._nowfilename + self._filenamesuffix) == 0:
                captchares = d2.decode(self._nowfolder + self._nowfilename + self._filenamesuffix, self._captchatype)
                if captchares['ret'] == 0:
                    captcha = captchares['result']
                    id = captchares['id']
                    self._rename(captcha, id, ('' if zdw.verifycaptcha(captcha) == 0 else '_E'))
                    log.info('captcha process done, filename: ' + self._nowfilename)
                elif captchares['ret'] == -304:
                    log.error('not sufficient funds !')
                elif captchares['ret'] == -303:
                    log.info('processing incomplete, filename: ' + self._nowfilename)
                    self._rename('', captchares['id'], '_U')
                else:
                    log.error('other errors')
                pass
            pass
            log.info('file processed, count is: ' + str(self._imagecount))
            self._next()
    pass


def main():
    builder = Builder(sbconf.main_image_temp_path, sbconf.main_dm2_username,
                      sbconf.main_dm2_password, samplenum=sbconf.main_dm2_samplenum,
                      foldersize=sbconf.main_dm2_foldersize)
    builder.build()



if __name__ == '__main__':
    main()
