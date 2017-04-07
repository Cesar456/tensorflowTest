import pymysql

conn = pymysql.connect(host='115.29.55.54', port=3306, user="root", passwd="shufe123", db="poemcesar", charset="utf8")
cur = conn.cursor()


# 获取所有诗歌内容
# TODO 先用500条试试水
def get_all_content():
    sql = "select content from poem limit 500"
    cur.execute(sql)
    return list(zip(*(cur.fetchall())))[0]

