#coding=utf-8
import requests
import os


#default_url = ""
default_url = "http://192.168.1.131:8000/cvmart-trains/api/ft/algo-tool/usage/callback/1/1743"
class ToolReport:
    def __init__(self) -> None:
        self.url_post = os.getenv("ALGO_TOOL_USAGE_REPORT_URL", default_url)

    def report(self, algo_tool_type, algo_tool_name, algo_tool_version, remark):
        '''
        algo_tool_type:int, 1:基础模型/算法, 2:训练套件, 3:部署套件
        algo_tool_name:string
        algo_tool_version:string
        remark:string, 备注或其他信息
        '''
        data = {"algo_tool_type":algo_tool_type, "algo_tool_name":algo_tool_name,"algo_tool_version":algo_tool_version, "remark":remark}
        print("send report:{}".format(data))
        try:
            response = requests.post(self.url_post, json=data, timeout=5)
            print("post response:",response.json())
            if response.json()["code"] == 20000:
                print("tool report successfuly!!")
            else:
                print("tool report failed!!")
        except Exception as e:
            print("report failed:{}".format(e))


tool_report = ToolReport()

if __name__ == "__main__":

    tool_report.report(2, "ev_detection", "V1.0.0", "detection task")

