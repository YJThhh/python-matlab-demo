from lxml.html import parse
from urllib.request import urlopen
import pandas as pd


def GetNextDateString(date):
    date = list(map(int, date.split('-')))
    import datetime

    timer = datetime.date(date[0], date[1], date[2])
    nextday = timer + datetime.timedelta(days=1)

    return str(nextday.year) + "-" + str(nextday.month) + "-" + str(nextday.day)


# 请输入日期 (年-月-日):
date_start = "2022-2-11"
next_date = date_start
date_end = "2022-2-18"  # 请注意，最后一天不被包含  3月不要写03同理3日

date_list = []

while GetNextDateString(next_date) != date_end:
    date_list.append(next_date)
    next_date = GetNextDateString(next_date)

# Step 1: init dataframe
dataFrame = pd.DataFrame(
    columns={"时次", "瞬时温度", "地面气压", "相对湿度", "瞬时风向", "瞬时风速", "1小时极大风速", "1小时降水", "10分钟平均能见度", "积雪深度"})

all_table_text_content_list = []

# Step 2: get heml tb
date_string = ''
for date_string in date_list:
    print("***开始爬取日期" + date_string)
    parsed = parse(urlopen('https://q-weather.info/weather/54511/history/?date=' + date_string))
    print("***开解析日期" + date_string)
    doc = parsed.getroot()
    tables = doc.findall('.//table')

    # thead 处理
    theads = tables[0].findall('.//thead')[0].findall('.//th')
    thead_columns_name = []
    for th in theads:
        thead_columns_name.append(th.text_content())
    # tbody 处理
    tbody = tables[0].findall('.//tbody')[0]
    trs = tbody.findall('.//tr')
    # 计算有多少行数据（不包括首行）
    num_row = len(trs)
    print("***日期" + date_string + "爬取完毕，总共" + str(num_row) + "行数据")
    # 然后一行行的向dataframe里面添加
    for i in range(num_row):
        new_row_dict = {}
        tds=trs[i].findall('.//td')
        for idx, td in enumerate(tds):
            new_row_dict[thead_columns_name[idx]] = td.text_content();
        new_row = pd.DataFrame(new_row_dict, index=[0])
        dataFrame = pd.concat([dataFrame, new_row], axis=0, join='outer')
print("***写入dataFrame完毕")
order = ["时次", "瞬时温度", "地面气压", "相对湿度", "瞬时风向", "瞬时风速", "1小时极大风速", "1小时降水", "10分钟平均能见度", "积雪深度"]
dataFrame = dataFrame[order]
dataFrame.to_excel('./data/wether_start_' + date_start + '_end_' + date_end + '.xlsx', index=False)

print("***写入文件完毕")
