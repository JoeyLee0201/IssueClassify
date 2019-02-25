# -*- coding: UTF-8 -*-
from database import mysqlOperator as db
from database import mysqlOperator2 as db2
import json

f = open("./data/resource/label_map.json")
label_dic = json.loads(f.read())
f.close()


def label_map(input_label):
    for key, value in label_dic.items():
        if input_label in value:
            return key, value[input_label]


def fetch_issues():
    print 'fetch all issues...'
    issues = db.selectAllIssues()
    print 'fetch done'

    issue_list = []

    for i in issues:
        issue = {}
        issue['title'] = i[0]
        issue['body'] = i[1]
        issue['labels'] = i[2]
        issue_list.append(issue)

    res = json.dumps(issue_list, encoding="utf-8", indent=4)

    corpus = open('./output/issue_corpus.ic', "w")
    corpus.write(res)
    corpus.close()


def fetch_all_baseline_issues():
    print 'fetch all baseline issues...'
    issues = db2.selectAllIssues()
    print 'fetch done'
    res = []
    for i in issues:
        issue = {}
        issue['title'] = i[0]
        issue['body'] = i[1]
        issue['labels'] = i[2]
        res.append(issue)
    return res


def fetch_baseline_issue_with_id(id):
    print 'fetch  baseline issues with id '+id+'...'
    issues = db2.selectIssuesWithID(id)
    print 'fetch done'
    # issues 是二维的数组
    # print issues[0][2]
    issue = {}
    issue['title'] = issues[0][0]
    issue['body'] = issues[0][1]
    issue['labels'] = issues[0][2]
    return issue


def fetch_issues_random_with_detail(label, num=0, par_label=None):
    if par_label is None:
        par_label, _ = label_map(label)
    if num == 0:
        _, num = label_map(label)
    print 'fetch issue with label \''+label+'\''
    if '\'' in label:
        label = label.replace("'", "\\'")
    issues = db.randomSelectIssueByLabel(label, num)
    print 'fetch %d done' % len(issues)
    res = []
    for i in issues:
        issue = {}
        # 这段代码为错误代码，幸好不影响title的值
        # if label == "docs" or label == "c: documentation" or label == "type: documentation" \
        #         or label == "area - documentation" or label == "team: documentation" \
        #         or label == "doc":
        #     issue['title'] = "documentation"
        # else:
        #     issue['title'] = i[0]
        issue['title'] = i[0]
        issue['body'] = i[1]
        issue['labels'] = par_label
        res.append(issue)
    return res


def fetch_issues_random(par_label, limit=-1, is_limit=False, without=None):
    if limit > 0:
        is_limit = True
    labels = label_dic[par_label]
    res = []
    for label, num in labels.items():
        if label == without:
            continue
        if is_limit and limit - num <= 0:
            res += fetch_issues_random_with_detail(label, par_label=par_label, num=limit)
            return res
        res += fetch_issues_random_with_detail(label, par_label=par_label, num=num)
        limit -= num
    return res


def fetch_issues_random_with_detail_stars(label, left_star, right_star, num=0, par_label=None):
    if par_label is None:
        par_label, _ = label_map(label)
    if num == 0:
        _, num = label_map(label)
    print 'fetch issue with label \''+label+'\''
    if '\'' in label:
        label = label.replace("'", "\\'")
    issues = db.randomSelectIssueWithStars(label, num, left_star, right_star)
    print 'fetch %d done' % len(issues)
    res = []
    for i in issues:
        issue = {}
        # 这段代码为错误代码，幸好不影响title的值
        # if label == "docs" or label == "c: documentation" or label == "type: documentation" \
        #         or label == "area - documentation" or label == "team: documentation" \
        #         or label == "doc":
        #     issue['title'] = "documentation"
        # else:
        #     issue['title'] = i[0]
        issue['title'] = i[0]
        issue['body'] = i[1]
        issue['labels'] = par_label
        res.append(issue)
    return res


def fetch_issues_random_with_stars(par_label, left_star, right_star, limit=-1, is_limit=False, without=None):
    if limit > 0:
        is_limit = True
    labels = label_dic[par_label]
    res = []
    for label, num in labels.items():
        if label == without:
            continue
        if is_limit and limit - num <= 0:
            res += fetch_issues_random_with_detail_stars(label, left_star, right_star, par_label=par_label, num=limit)
            return res
        res += fetch_issues_random_with_detail_stars(label, left_star, right_star, par_label=par_label, num=num)
        limit -= num
    return res


def fetch_issues_random_with_detail_repository(label, repository_id, num=0, par_label=None):
    if par_label is None:
        par_label, _ = label_map(label)
    if num == 0:
        _, num = label_map(label)
    print 'fetch issue with label \''+label+'\''
    if '\'' in label:
        label = label.replace("'", "\\'")
    issues = db.randomSelectIssueWithRepository(label, num, repository_id)
    print 'fetch %d done' % len(issues)
    res = []
    for i in issues:
        issue = {}
        # 这段代码为错误代码，幸好不影响title的值
        # if label == "docs" or label == "c: documentation" or label == "type: documentation" \
        #         or label == "area - documentation" or label == "team: documentation" \
        #         or label == "doc":
        #     issue['title'] = "documentation"
        # else:
        #     issue['title'] = i[0]
        issue['title'] = i[0]
        issue['body'] = i[1]
        issue['labels'] = par_label
        res.append(issue)
    return res


def fetch_issues_random_with_repository(par_label, repository_id, limit=-1, is_limit=False, without=None):
    if limit > 0:
        is_limit = True
    labels = label_dic[par_label]
    res = []
    for label, num in labels.items():
        if label == without:
            continue
        if is_limit and limit - num <= 0:
            res += fetch_issues_random_with_detail_repository(label,repository_id, par_label=par_label, num=limit)
            return res
        res += fetch_issues_random_with_detail_repository(label, repository_id, par_label=par_label, num=num)
        limit -= num
    return res


def count(labels):
    f = open('./output/out.temp', "r")
    res = json.loads(f.read())
    f.close()

    length = len(labels)
    print 'length: ', length
    i = 0
    for key in labels:
        if key in res:
            i += 1
            print '%d / %d...............%.4f%%' % (i, length, i * 100.0 / length)
            continue
        res[key] = {}
        occur_num = 0
        repo = []
        print '%s' % key
        for label, value in labels[key].items():
            print ' ', label
            if '\'' in label:
                label = label.replace("\'", "\\\'")
            occur_num += value
            repo_list = db.selectRepoByLabel(label)
            # print repo_list
            for repo_id in repo_list:
                print '     repo_id:', repo_id[0]
                if repo_id[0] in repo:
                    continue
                repo.append(repo_id[0])
        res[key]['occur_num'] = occur_num
        res[key]['repo_num'] = len(repo)
        i += 1
        print '%d / %d...............%.4f%%' % (i, length, i * 100.0 / length)
        out = json.dumps(res, encoding="utf-8", indent=4)
        f = open('./output/out.temp', "w")
        f.write(out)
        f.close()

    return res


# fetch_issues_random("bug")
# fetch_baseline_issue_with_id("HTTPCLIENT-853")

