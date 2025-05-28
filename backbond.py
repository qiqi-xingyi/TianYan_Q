# --*-- conding:utf-8 --*--
# @time:5/28/25 14:26
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:backbond.py

from cqlib import TianYanPlatform
from cqlib.circuits import Circuit, Parameter

login_key = "rIzA7jf4sgF5tE+xr5IERprZGGDT/po2dTJQtiw68Wo="

if __name__ == '__main__':

    platform = TianYanPlatform(login_key=login_key)

    computer_list_data = platform.query_quantum_computer_list()

    for computer_data in computer_list_data:
        print(computer_data)