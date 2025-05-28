# --*-- conding:utf-8 --*--
# @time:5/27/25 17:57
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:test_demo.py

from cqlib import TianYanPlatform
from cqlib.circuits import Circuit, Parameter

login_key = "rIzA7jf4sgF5tE+xr5IERprZGGDT/po2dTJQtiw68Wo="

if __name__ == '__main__':

    platform = TianYanPlatform(login_key=login_key)

    computer_list_data = platform.query_quantum_computer_list()

    for computer_data in computer_list_data:
        print(computer_data)

    platform.set_machine("tianyan176")

    circuit = Circuit(qubits=[0, 7])
    circuit.h(0)
    circuit.x(7)
    circuit.cx(0, 7)
    circuit.measure_all()
    print(circuit.draw())

    theta = Parameter('theta')
    circuit_para = Circuit(qubits=[0], parameters=[theta])
    circuit_para.rx(0, theta)

    print(f"带参数的线路: {circuit_para.qcis}")
    c1 = circuit_para.assign_parameters({'theta': 0.12})

    print(f"赋值后的线路: {c1.qcis}")

    query_id_single = platform.submit_experiment(
        circuit=circuit.qcis,
        num_shots=5000,
    )
    print(f'query_id: {query_id_single}')

    # # 也支持批量提交，一次最多支持 50 个任务
    # query_id_list = platform.submit_experiment(
    #     circuit=[circuit.qcis, circuit.qcis],
    #     num_shots=5000,
    # )
    # print(f'query_ids: {query_id_list}')

    # 查询原始数据
    exp_result = platform.query_experiment(query_id=query_id_single, max_wait_time=120, sleep_time=5)

    # 返回值为list，包含若干字典形式，
    # key："resultStatus"为线路执行的原始数据，共计1+num_shots个数据，第一个数据为测量的比特编号和顺序，
    # 如本例中[0, 6]，其余为每shot对应的结果，每shot结果按照比特顺序排列。
    # key："probability"为线路测量结果的概率统计，经过实时的读取修正后的统计结果。
    # key："experimentTaskId"为本次实验的查询id，主要用于批量实验时的结果对应确认。
    print(f"单个任务查询结果概率：{exp_result[0]['probability']}")

    # # 批量查询量子电路结果
    # exp_result = platform.query_experiment(query_id=query_id_list, max_wait_time=120, sleep_time=5)
    # print(f'输入的查询Id个数为: {len(query_id_list)}，查询到的实验结果个数为: {len(exp_result)}')
    #
    # print('第一个量子电路实验结果为：')
    # for res in exp_result:
    #     print(f"{res['experimentTaskId']} : {res['probability']}")


