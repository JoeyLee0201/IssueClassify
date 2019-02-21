# -*- coding: UTF-8 -*-
import extractFromDB
import json
from preprocessor import preprocessor
import sys
import train_w2v as w2v
import nltk
reload(sys)
sys.setdefaultencoding('utf8')


def extract_issues():
    extractFromDB.fetch_issues()


def build_words_in_json():
    f = open('./output/issue_corpus.ic', "r")
    issue_corpus = json.loads(f.read())
    f.close()

    f = open('./output/words_corpus.ic', "w")
    f.write("[\n")
    i = 0
    length = len(issue_corpus)
    for issue in issue_corpus:
        content = unicode(issue['title'])+u': '+unicode(issue['body'])
        # print content
        res = preprocessor.preprocessToWord(content)
        f.write(json.dumps(res, encoding="utf-8"))
        if i != length:
            f.write(",\n")
        i += 1
        print '%d / %d...............%.4f%%' % (i, length, i * 100.0/length)
    f.write("\n]")
    f.close()


def divide_into_words(input_file, output_file, labels=None):
    res = []
    if labels is None:
        f = open(input_file, "r")
        issue_corpus = json.loads(f.read())
        f.close()

        i = 0
        length = len(issue_corpus)
        for issue in issue_corpus:
            content = unicode(issue['title']) + u': ' + unicode(issue['body'])
            words = preprocessor.preprocessToWord(content)
            res.append(words)
            # out = ""
            # for word in words:
            #     out += word+" "
            # f.write(out+"\n")
            i += 1
            print '%d / %d...............%.4f%%' % (i, length, i * 100.0 / length)
            # if i == length/2:
            #     f = open(output_file+"-part1", "w")
            #     f.write(json.dumps(res, encoding="utf-8"))
            #     f.close()
            #     res = []
        # f = open(output_file + "-part2", "w")
        f = open(output_file, "w")
        f.write(json.dumps(res, encoding="utf-8"))
        f.close()
    else:
        for label in labels:
            f = open(input_file+label+".ic", "r")
            issue_corpus = json.loads(f.read())
            f.close()

            i = 0
            length = len(issue_corpus)
            for issue in issue_corpus:
                content = unicode(issue['title']) + u': ' + unicode(issue['body'])
                words = preprocessor.preprocessToWord(content)
                res.append(words)
                # out = ""
                # for word in words:
                #     out += word+" "
                # f.write(out+"\n")
                i += 1
                print '%d / %d...............%.4f%%' % (i, length, i * 100.0 / length)
                # if i == length/2:
                #     f = open(output_file+"-part1", "w")
                #     f.write(json.dumps(res, encoding="utf-8"))
                #     f.close()
                #     res = []
            # f = open(output_file + "-part2", "w")
            f = open(output_file+label+".ic", "w")
            f.write(json.dumps(res, encoding="utf-8"))
            f.close()
            res = []
    return res


def build_corpus(output_file):
    res = extractFromDB.fetch_issues_random_with_detail("bug", num=4000)
    res += extractFromDB.fetch_issues_random("bug", without="bug", limit=4000)
    res += extractFromDB.fetch_issues_random("enhancement", limit=6000)
    res += extractFromDB.fetch_issues_random("question", limit=4000)
    res += extractFromDB.fetch_issues_random("feature", limit=3000)
    res += extractFromDB.fetch_issues_random("doc", limit=2000)
    f = open(output_file, "w")
    f.write(json.dumps(res, encoding="utf-8"))
    f.close()


def build_baseline_corpus(output_file):
    res = extractFromDB.fetch_all_baseline_issues()
    f = open(output_file, "w")
    f.write(json.dumps(res, encoding="utf-8"))
    f.close()


def build_test_corpus(output_file):
    res = extractFromDB.fetch_issues_random_with_detail("bug", num=500)
    res += extractFromDB.fetch_issues_random("bug", limit=500, without="bug")
    res += extractFromDB.fetch_issues_random("enhancement", limit=800)
    res += extractFromDB.fetch_issues_random("question", limit=700)
    res += extractFromDB.fetch_issues_random("feature", limit=700)
    # res += extractFromDB.fetch_issues_random("wontfix", limit=600)
    res += extractFromDB.fetch_issues_random("doc", limit=600)
    # res += extractFromDB.fetch_issues_random("invalid", limit=500)
    # res += extractFromDB.fetch_issues_random("help", limit=4000)
    f = open(output_file, "w")
    f.write(json.dumps(res, encoding="utf-8"))
    f.close()


def build_corpus_with_stars(output_file, left_star, right_star):
    res = extractFromDB.fetch_issues_random_with_detail_stars("bug", left_star, right_star, num=4000)
    res += extractFromDB.fetch_issues_random_with_stars("bug", left_star, right_star, without="bug", limit=4000)
    res += extractFromDB.fetch_issues_random_with_stars("enhancement", left_star, right_star, limit=6000)
    res += extractFromDB.fetch_issues_random_with_stars("question", left_star, right_star, limit=4000)
    res += extractFromDB.fetch_issues_random_with_stars("feature", left_star, right_star, limit=3000)
    res += extractFromDB.fetch_issues_random_with_stars("doc", left_star, right_star, limit=2000)
    f = open(output_file, "w")
    f.write(json.dumps(res, encoding="utf-8"))
    f.close()


def build_corpus_with_repository(output_file, repository_id):
    res = extractFromDB.fetch_issues_random_with_detail_repository("bug",repository_id, num=4000)
    res += extractFromDB.fetch_issues_random_with_repository("bug", repository_id, without="bug")
    res += extractFromDB.fetch_issues_random_with_repository("enhancement", repository_id)
    res += extractFromDB.fetch_issues_random_with_repository("question", repository_id)
    res += extractFromDB.fetch_issues_random_with_repository("feature", repository_id)
    res += extractFromDB.fetch_issues_random_with_repository("doc", repository_id)
    f = open(output_file, "w")
    f.write(json.dumps(res, encoding="utf-8"))
    f.close()


def main():
    # build_corpus("./data/output/issue_corpus_v4.ic")
    # build_test_corpus("./data/output/issue_corpus_random_test.ic")
    # build_corpus_with_stars("./data/output/issue_corpus_1000_1500.ic", 1000, 1500)
    # build_corpus_with_stars("./data/output/issue_corpus_1500_2000.ic", 1500, 2000)
    # build_corpus_with_stars("./data/output/issue_corpus_2000_2500.ic", 2000, 2500)
    # build_corpus_with_stars("./data/output/issue_corpus_2000_3000.ic", 2000, 3000)
    # build_corpus_with_repository("./data/output/issue_corpus_repository_3786237.ic", 3786237)
    # build_corpus_with_repository("./data/output/issue_corpus_repository_1625986.ic", 1625986)
    # build_corpus_with_repository("./data/output/issue_corpus_repository_6293402.ic", 6293402)
    # build_corpus_with_repository("./data/output/issue_corpus_repository_3148979.ic", 3148979)
    # build_corpus_with_repository("./data/output/issue_corpus_repository_1064563.ic", 1064563)
    # build_baseline_corpus("./data/output/baseline_issue_corpus.ic")
    # res = divide_into_words("./data/output/baseline_issue_corpus.ic",
    #                         "./data/output/baseline_issue_corpus_tokenize.ic")
    # res = divide_into_words("./data/output/issue_corpus/",
    #                         "./data/output/issue_corpus_tokenize/",
    #                         labels=["enhancement", "question", "feature", "doc"])
    # labels = ["bug", "enhancement", "question", "feature", "doc"]
    input = []
    # for label in labels:
    #     f = open("./data/output/issue_corpus_tokenize/"+label+".ic", "r")
    #     input += json.loads(f.read())
    #     f.close()
    # w2v.train(input, "./data/output/words_all.model")

    f = open("./data/output/baseline_issue_corpus_tokenize.ic", "r")
    input += json.loads(f.read())
    f.close()
    w2v.train(input, "./data/output/words_baseline.model")
    # res = preprocessor.preprocessToWord('''
    # This patch allows tests to be run in a specific order. Here is improvement: Sorting is enabled by adding the following to
    # `BuildConfig.groovy`:  ``` grails.testing.sortFiles = true ```  For finer control, a closure can be provided:
    #  ``` grails.testing.sortFiles = { a, b -> a <=> b } ```
    # ''')
    # print res

main()

