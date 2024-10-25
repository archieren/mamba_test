import os, io,json,tempfile
import requests
from pathlib import Path
from pm.service_api.protocol import SegRequest,file_bytes_to_str,str_to_file_bytes

root = "/home/archie/Projects/data/TestSet/ATA-TestSample"
http_url = "http://106.13.35.169:8001/v1/segment/file"

simple_test_case=['20150474_shell_occlusion_u',
                  '20150474_shell_occlusion_l',
                  '16_d',
                  '16_u',
                  '17_d',
                  '17_u',
                  '8_d',
                  '8_u',
                  'lower0321',
                  'upper0321']
for case in simple_test_case:
    file_path = Path(root) / "Separate" / "TestData" / f"{case}.stl"
    print(len(open(file_path, 'rb').read()))
    f_s = file_bytes_to_str(open(file_path, 'rb').read())
    req_raw = SegRequest(stl=f_s,threshhold=0.6)
    response = requests.post(http_url, json=req_raw.model_dump())
    stl = response.json()
    file_path = Path(root) / "Separate" / "TestData" / f"{case}.json"
    file = open(file_path,'w')
    file.write(json.dumps({'seg':stl['seg_result']}))
    file.close()

