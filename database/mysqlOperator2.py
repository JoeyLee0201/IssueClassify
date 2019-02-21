# -*- coding: UTF-8 -*-

import MySQLdb

con = MySQLdb.connect(host='10.131.252.160', port=3306, db='github', user='root', passwd='root', charset='utf8')
# con = MySQLdb.connect(host='localhost', port=3306, db='apichangeast', user='root', passwd='123456', charset='utf8')

cursor = con.cursor()    #创建一个游标对象


def close():
    cursor.close()
    con.close()


def selectAllIssues():
    SQL = """
    select  summary,description,type  from issue  limit 100000
    """
    try:
        cursor.execute(SQL)
        results = cursor.fetchall()
        return results
    except Exception, e:
        print e


if __name__ == '__main__':
    # projects = selectAllHighRepository()
    # for repo in projects:
    #     issues = selectAllIssueInOneRepo(repo[0])

    #     for issue in issues:
    #         links = selectTrueLinkInOneIssue(issue[1])
    #         print len(links),'\n'
    # print len(selectAllIssueInOneRepo(1459486))
    # print selectCommentInOneIssue('JakeWharton/ActionBarSherlock/issues/3')
    close()
