import requests
import argparse
import json
import math
import csv


class WeatherAPI:
    def __init__(self, args):
        self.link = args.link
        self.key = args.key
        self.num_of_rows = args.num_of_rows
        self.stn_id = args.stn_id
        self.start_date = args.start_date
        self.start_hour = args.start_hour
        self.end_date = args.end_date
        self.end_hour = args.end_hour
        self.path = args.path
        self.get_request_count()

    def get_request_count(self):
        result = self.request(1)
        total_count = result['response']['body']['totalCount']
        self.request_count = math.ceil(total_count / self.num_of_rows)

    def request(self, page_no):
        url = "%sServiceKey=%s" % (self.link, self.key)
        url += "&NumOfRows=%d" % self.num_of_rows
        url += "&pageNo=%d" % page_no
        url += "&dataCd=ASOS&dateCd=HR&dataType=JSON"
        url += "&stnIds=%s" % self.stn_id
        url += "&startDt=%s" % self.start_date
        url += "&startHh=%s" % self.start_hour
        url += "&endDt=%s" % self.end_date
        url += "&endHh=%s" % self.end_hour

        print(url)

        result = json.loads(requests.get(url).text)
        return result

    def item_to_dict(self, element):
        new_item = {
            "지점": element['stnId'],
            "일시": element['tm'],
            "기온(°C)": element['ta'],
            "강수량(mm)": element['rn'],
            "풍속(m/s)": element['ws'],
            "풍향(16방위)": element['wd'],
            "습도(%)": element['hm'],
            "이슬점온도(°C)": element['td'],
            "현지기압(hPa)": element['pa'],
            "일조(hr)": element['ss'],
            "시정(10m)": element['vs'],
            "지면온도(°C)": element['ts'],
            "증기압(hPa)": element['pv']
        }

        return new_item

    def request_all(self):
        for i in range(self.request_count):
            result = self.request(i + 1)
            response = result['response']
            body = response['body']
            items = body['items']
            item = items['item']

            new_list = list()
            for element in item:
                new_item = self.item_to_dict(element)
                new_list.append(new_item)

            self.save_to_csv(i, new_list)

    def save_to_csv(self, index, result):
        print(result[0]['일시'])

        csv_columns = ["지점", "일시", "기온(°C)",
                       "강수량(mm)", "풍속(m/s)", "풍향(16방위)",
                       "습도(%)", "이슬점온도(°C)", "현지기압(hPa)",
                       "일조(hr)", "시정(10m)", "지면온도(°C)", "증기압(hPa)"]

        with open(self.path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            if index == 0:
                writer.writeheader()
            for data in result:
                writer.writerow(data)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.link = 'http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList?'
    args.key = 'L%2FTS2BPl3w4GBLKwQW1cxubGbFKimMjJWCwockw0jBiG2f7I3T9JR4D8CCOXc%2FC7RS1gAkzoX8qAqUsSnRQ3AQ%3D%3D'
    args.num_of_rows = 24
    args.stn_id = 175
    args.start_date = '20200101'
    args.start_hour = '00'
    args.end_date = '20200819'
    args.end_hour = '23'
    args.path = 'SURFACE_ASOS_175_HR_2020_2020_2021.csv'

    api = WeatherAPI(args)
    api.request_all()
