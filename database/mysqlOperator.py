# -*- coding: UTF-8 -*-

import MySQLdb

con = MySQLdb.connect(host='10.141.221.73', port=3306, db='codehub', user='root', passwd='root', charset='utf8')
# con = MySQLdb.connect(host='localhost', port=3306, db='apichangeast', user='root', passwd='123456', charset='utf8')

cursor = con.cursor()    #创建一个游标对象

# def selectApichangeById(id):
#     SQL = """
#     select * from apichange where apichange_id = %d
#     """
#     try:
#         cursor.execute(SQL % id)
#         res = cursor.fetchone()
#         while res:
#           print res
#           res = cursor.fetchone()    #fetchone只给出一条数据，然后游标后移。游标移动过最后一行数据后再fetch就得到None
#     except Exception,e:
#         print e

# def selectApichangeByRepositoryAndType(id, type):
#     SQL = """
#     select * from apichange where repository_id = %s and change_type = %s
#     """
#     try:
#         cursor.execute(SQL, (id, type))
#         res = cursor.fetchone()
#         while res:
#           print res
#           res = cursor.fetchone()    #fetchone只给出一条数据，然后游标后移。游标移动过最后一行数据后再fetch就得到None
#     except Exception,e:
#         print e


def selectAllHighRepository():
    SQL = """
    select repository_id, url from repository_high_quality
    """
    try:
        cursor.execute(SQL)
        results = cursor.fetchall()
        return results
    except Exception,e:
        print e


def selectAllIssues():
    SQL = """
    select  title,body,labels  from issue  limit 100000
    """
    try:
        cursor.execute(SQL)
        results = cursor.fetchall()
        return results
    except Exception, e:
        print e


def selectRepoByLabel(label):
    SQL = """
    SELECT distinct repository_id FROM issue 
    where labels like '%s' or labels like '%s,%%' or labels like '%%,%s,%%' or labels like '%%,%s'
    """
    try:
        cursor.execute(SQL % (label, label, label, label))
        results = cursor.fetchall()
        return results
    except Exception, e:
        print e


def selectAllIssueInOneRepo(repoId):
    SQL = """
    select repository_id,issue_index,created_at,closed_at,title,body,labels,type from issue where repository_id = %d and type = 'issue'
    """
    try:
        cursor.execute(SQL % repoId)
        results = cursor.fetchall()
        return results
    except Exception,e:
        print e


def selectTrueLinkInOneIssue(issueIndex):
    SQL = """
    select repository_id,issue_index,created_at,commit_id from issue_event where issue_index = '%s' and commit_id is not null
    """
    try:
        cursor.execute(SQL % issueIndex)
        results = cursor.fetchall()
        return results
    except Exception,e:
        print e


def selectCommentInOneIssue(issueIndex):
    SQL = """
    select repository_id,issue_index,created_at,updated_at,body from issue_comment where issue_index = '%s'
    """
    try:
        cursor.execute(SQL % issueIndex)
        results = cursor.fetchall()
        return results
    except Exception,e:
        print e


def randomSelectIssueWithStars(label, num, left_stars, right_stars,):
    SQL = """
            SELECT a.repository_id,a.title,a.body,a.labels, b.stars
            FROM codehub.issue as a
            inner join
            (SELECT repository_id,stars 
	         FROM codehub.repository_java
	         where repository_id in 
	            (SELECT distinct(repository_id) 
		         FROM codehub.issue 
                 where repository_id <> '-1') 
             order by cast(stars as UNSIGNED INTEGER)
             ) b
            on a.repository_id = b.repository_id
            where b.stars >= %d and b.stars< %d
            and (a.body != '' and length(a.body) =char_length(a.body) and length(a.title) =char_length(a.title) and
            (a.labels like '%s' or a.labels like '%s,%%' or 
            a.labels like '%%,%s,%%'or a.labels like '%%,%s'))
            order by rand() limit %d
            """
    try:
        cursor.execute(SQL % (left_stars, right_stars, label, label, label, label, num))
        results = cursor.fetchall()
        return results
    except Exception, e:
        print e


def randomSelectIssueWithRepository(label, num, repository_id,):
    SQL = """
        SELECT title,body,labels FROM issue 
        where body != '' and length(body) =char_length(body) and length(title) =char_length(title) and
        (labels like '%s' or labels like '%s,%%' or 
        labels like '%%,%s,%%'or labels like '%%,%s') and repository_id=%d
        order by rand() limit %d
            """
    try:
        cursor.execute(SQL % (label, label, label, label, repository_id, num))
        results = cursor.fetchall()
        return results
    except Exception, e:
        print e


def randomSelectIssueByLabel(label, num):
    SQL = """
        SELECT title,body,labels FROM issue 
        where body != '' and length(body) =char_length(body) and length(title) =char_length(title) and
        (labels like '%s' or labels like '%s,%%' or 
        labels like '%%,%s,%%'or labels like '%%,%s')
        order by rand() limit %d
        """
    try:
        cursor.execute(SQL % (label, label, label, label, num))
        results = cursor.fetchall()
        return results
    except Exception, e:
        print e


def countTrueLinkInOneIssue(issueIndex):
    SQL = """
    select count(event_id) from issue_event where issue_index = '%s' and commit_id is not null
    """
    try:
        cursor.execute(SQL % issueIndex)
        results = cursor.fetchone()
        return results
    except Exception,e:
        print e


def countRepoNumByLabel(label):
    SQL = """
        SELECT count(distinct repository_id) FROM issue where labels like '%%%s,%%' or labels like '%%,%s%%';
        """
    try:
        cursor.execute(SQL % (label, label))
        results = cursor.fetchone()
        return results[0]
    except Exception, e:
        print e


def close():
    cursor.close()
    con.close()


if __name__ == '__main__':
    # projects = selectAllHighRepository()
    # for repo in projects:
    #     issues = selectAllIssueInOneRepo(repo[0])

    #     for issue in issues:
    #         links = selectTrueLinkInOneIssue(issue[1])
    #         print len(links),'\n'
    # print len(selectAllIssueInOneRepo(1459486))
    print selectCommentInOneIssue('JakeWharton/ActionBarSherlock/issues/3')
    close()
