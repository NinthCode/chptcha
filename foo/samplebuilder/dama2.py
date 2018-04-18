#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Nicot
# @Date:   2018-04-13 11:24:02
# @Last Modified by:   Nicot
# @Last Modified time: 2018-04-13 11:34:01
import sys

sys.path.append('../util/')
from logger import Logger
import md5
import requests
import base64
import sbconf

log = Logger('dama2')


class Dama2:

    def __init__(self, username, password):
        self._username = username
        self._password = md5.string(sbconf.dama2_key + md5.string(md5.string(username) + md5.string(password)))

    pass

    def _sign(self, param=b''):
        return (md5.byte(bytes(sbconf.dama2_key, encoding="utf8") + bytes(self._username, encoding='utf8') + param))[:8]

    pass

    def _callapi(self, url, data):
        log.info('calling dama2 api, url: ' + url + ', data: ' + str(data))
        response = requests.post(url, data=data)
        if response.status_code == requests.codes.ok:
            log.info("called dama2 api, resdata: " + str(response.json()))
            return response.json()
        else:
            log.error('dama2 response exception, return code is not 200, response code is ' + str(response.status_code))
            raise RuntimeError('http response error')
        pass

    pass

    def balances(self):
        data = {
            'appID': sbconf.dama2_id,
            'user': self._username,
            'pwd': self._password,
            'sign': self._sign()
        }
        jresp = self._callapi(sbconf.dama2_balance_url, data)
        if jresp['ret'] == 0:
            return jresp['balance']
        else:
            log.error('failed to query balance, error code is ' + str(jresp['ret']))
            return jresp['ret']
        pass

    pass

    def decode(self, filepath, type):
        try:
            f = open(filepath, 'rb')
            fdata = f.read()
            b64data = base64.b64encode(fdata)
            f.close()
            data = {'appID': sbconf.dama2_id,
                    'user': self._username,
                    'pwd': self._password,
                    'type': type,
                    'fileDataBase64': b64data,
                    'sign': self._sign(fdata)
                    }
            jresp = self._callapi(sbconf.dama2_d2file_url, data)
            if jresp['ret'] != 0:
                log.error('failed to decode, error code is ' + str(jresp['ret']))
            pass
            return jresp
        except Exception as e:
            log.error("failed to decode, exception is ", e)
            return None
        pass

    pass

    def result(self, id):
        data = {'appID': sbconf.dama2_id,
                'user': self._username,
                'pwd': self._password,
                'id': id,
                'sign': self._sign(id.encode(encoding="utf-8"))
                }
        jresp = self._callapi(sbconf.dama2_d2result_url, data)
        return jresp['ret']

    pass

    def reporterror(self, id):
        data = {'appID': sbconf.dama2_id,
                'user': self._username,
                'pwd': self._password,
                'id': id,
                'sign': self._sign(id.encode(encoding="utf-8"))
                }
        jresp = self._callapi(sbconf.dama2_d2reporterror_url, data)
        return jresp['ret']

    pass


if __name__ == '__main__':
    d2 = Dama2('test', 'test')
    print(d2.balances())
    print(d2.decode('D:\\a.png', 42))
