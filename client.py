import os, io,json,time
import requests
from pathlib import Path
from pm.service_api.protocol import SegRequest,file_bytes_to_str

root = "/home/archie/Projects/data/TestSet/ATA-TestSample"
#http_url = "http://106.13.35.169:8001/v1/segment/file"
http_url = "http://127.0.0.1:8001/v1/segment/file"

def time_it(start_time):
    stop_time = time.time()
    print("耗时: {:.2f}秒".format(stop_time - start_time))
    return

simple_test_case=[
                '20150474_shell_occlusion_u',
                '20150474_shell_occlusion_l',
                '20182059_shell_occlusion_u',
                '20182059_shell_occlusion_l',
                '20181735_shell_occlusion_u',
                '20181735_shell_occlusion_l',
                '20180612_shell_occlusion_u',
                '20180612_shell_occlusion_l',
                # '1_d',
                # '1_u',
                # '2_d',
                # '2_u',
                # '3_d',
                # '3_u',
                # '4_d',
                # '4_u',
                # '5_d',
                # '5_u',
                # '6_d',
                # '6_u',
                # '7_d',
                # '7_u',
                # '8_d',
                # '8_u',
                # '9_d',
                # '9_u',
                # '10_d',
                # '10_u',
                # '11_d',
                # '11_u',
                # '12_d',
                # '12_u',
                # '13_d',
                # '13_u',
                # '14_d',
                # '14_u',
                # '15_d',
                # '15_u',
                # '16_d',
                # '16_u',
                # '17_d',
                # '17_u',
                # '18_d',
                # '18_u',
                # 'lower0321',
                # 'upper0321'
                ]

start_time = time.time()

for case in simple_test_case:
    file_path = Path(root) / "Separate" / "TestData" / f"{case}.stl"
    print(len(open(file_path, 'rb').read()))
    f_s = file_bytes_to_str(open(file_path, 'rb').read())
    req_raw = SegRequest(stl=f_s, s_o_i=case, threshhold=0.5)
    response = requests.post(http_url, json=req_raw.model_dump())
    stl = response.json()
    file_path = Path(root) / "Separate" / "TestData" / f"{case}.json"
    file = open(file_path,'w')
    file.write(json.dumps({'seg':stl['seg_result']}))
    file.close()

time_it(start_time)

