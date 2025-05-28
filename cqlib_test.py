# --*-- conding:utf-8 --*--
# @time:5/27/25 18:31
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:cqlib_test.py

import cqlib
print("cqlib:", dir(cqlib))

if __name__ == '__main__':

    try:
        import cqlib.remote as remote
        print("cqlib.remote:", dir(remote))
    except ImportError:
        print("cqlib.remote 不存在")

    for name in dir(cqlib):
        if name.lower().startswith(("client","api","service")):
            print("可疑入口:", name)
