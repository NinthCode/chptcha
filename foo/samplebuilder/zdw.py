#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Nicot
# @Date:   2018-04-13 18:12:55
# @Last Modified by:   Nicot
# @Last Modified time: 2018-04-13 18:13:13


import sys

sys.path.append('../util/')
import requests
import os
import sbconf
import rdm
from logger import Logger

log = Logger('zdw')


class Zdw:

    def __init__(self, savepath):
        self._session = requests.session()
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        pass
        self._savepath = savepath

    pass

    def pullcaptcha(self, savename):
        url = sbconf.zdw_pull_captcha_url + '?v=' + rdm.rdm()
        log.info('calling pull captch api, url: ' + url)
        response = self._session.get(url)
        if response.status_code == requests.codes.ok:
            f = None
            try:
                f = open(self._savepath + savename, 'wb')
                f.write(bytes(response.content))
                return 0
            except Exception as e:
                log.error('save captcha image faild, error is: ', e)
                return -1
            finally:
                f.close()
            pass
        else:
            log.error('pull captcha image faild, response code not 200, code is: ' + str(response.status_code))
            return 1
        pass

    pass

    def verifycaptcha(self, captchastr):
        data = {'password': rdm.rdmlstr(), 'validateCode': captchastr}
        url = sbconf.zdw_login_url + '&v=' + rdm.rdm()
        log.info('calling verify captch api, url: ' + url + ', data: ' + str(data))
        response = self._session.post(url, data=data)
        if response.status_code == requests.codes.ok:
            return 0 if (response.text.find('登录名不能为空') != -1 and response.text.find('校验码错误') == -1) else 1
        else:
            log.error('call verify captch api response code not 200, code is: ' + str(response.status_code))
            return -1
        pass

    pass


if __name__ == '__main__':
    zdw = Zdw('D:\\')
    zdw.pullcaptcha('sb.png')
    print(zdw.verifycaptcha('abcd'))
