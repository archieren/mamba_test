from pydantic import BaseModel,Base64Encoder
from typing import Dict,List
#TODO:有概念更好的Base64Bytes,Base64Str,但不熟练,经常被转晕.
def file_bytes_to_str(bytes:bytes): # 将图像文件内容转换成串， 利于json传送。
    bytes=Base64Encoder.encode(bytes)
    str=bytes.decode() 
    return str

def str_to_file_bytes(str:str):
    bytes= str.encode()
    bytes = Base64Encoder.decode(bytes)
    return bytes


class SegRequest(BaseModel):
    stl:str
    threshhold:float = 0.5
    s_o_i:str               # 用于判断是上下否！ superior or inferior

class SegResponse(BaseModel):
    seg_result:Dict[str, List[int]]