# -*- coding: utf-8 -*-
"""
__author__ = 'onion'
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import csv
import math
from gensim.models import Word2Vec
import re
import nltk
# nltk.download('punkt')
import string
from nltk.corpus import stopwords
import nltk.stem
from collections import OrderedDict


def get_downstream_size():
    l = []
    l2 = 0
    l10 = 0
    l20 = 0
    l50 = 0
    o50 = 0
    with open('./data/CPAN/DDG.json', 'r') as ddg_f:
        for line in ddg_f.readlines():
            line = json.loads(line)
            for key, value in line.items():
                print(len(value), value)
                l.append(len(value))
                if len(value) <= 5:
                    l2 += 1
                elif len(value) < 20:
                    l10 += 1
                elif len(value) < 50:
                    l20 += 1
                elif len(value) < 100:
                    l50 += 1
                else:
                    o50 += 1
    print(len(l), l2, l10, l20, l50, o50)
    num_list = ['<=5', '(5, 20]', '(20, 50]', '(50, 100]', '>100']
    num = [l2, l10, l20, l50, o50]
    print(num)
    print(np.average(l))
    plt.bar(num_list, num)
    plt.title('Distribution of Downstream Dependency Packages in CPAN')
    plt.xlabel('size')
    plt.ylabel('distribution')
    my_y_ticks = np.arange(0, 3500, 500)
    plt.yticks(my_y_ticks)
    plt.savefig('/Users/onion/Desktop/My_Paper/ddg/picture/Size distribution CPAN 2.png', dpi=300)
    plt.show()


def get_ddg_size():
    l = []
    l50 = 0
    l100 = 0
    with open('./data/CPAN/DDG.json', 'r') as ddg_f:
        for line in ddg_f.readlines():
            line = json.loads(line)
            for key, value in line.items():
                print(len(value), value)
                if len(value) >= 20:
                    l.append(len(value))

    num_list = ['<=5', '(5, 20]', '(20, 50]', '(50, 100]', '>100']
    num = [l50, l100]
    print(np.average(l), l)
    plt.bar(num_list, num)
    plt.title('Distribution of Downstream Dependent Packages in Cargo')
    plt.xlabel('size')
    plt.ylabel('distribution')
    plt.show()


def get_ddg_keywords():
    pac_keywords = {}
    all_pac_keywords = set()
    ddg_pac_keywords = set()
    with open('./data/CPAN/all_package.json', 'r') as fi:
        for line in fi.readlines():
            line = json.loads(line)
            pac_keywords[line['name']] = line['keywords']
            for k in line['keywords']:
                all_pac_keywords.add(k)

    ddgs = set()
    with open('./data/CPAN/DDG.json', 'r') as ddg_f:
        for line in ddg_f.readlines():
            line = json.loads(line)
            for key, value in line.items():
                print(len(value), value)
                if len(value) >= 20:
                    ddgs.add(key)
                    for v in value:
                        ddgs.add(v)
    print(len(ddgs), ddgs)

    for f in ddgs:
        for k in pac_keywords[f]:
            ddg_pac_keywords.add(k)
    print(len(ddg_pac_keywords), len(all_pac_keywords), len(ddg_pac_keywords)/len(all_pac_keywords))


def get_focal_features():
    focal_pac = []
    focal_ddg = {}

    with open('./data/RubyGems/DDG.json', 'r') as ddg_f:
        for line in ddg_f.readlines():
            line = json.loads(line)
            for key, value in line.items():
                if len(value) >= 20:
                    focal_ddg[key] = value
    print(len(focal_ddg))

    with open('./data/RubyGems/focal_feature2021.csv', 'a+') as fo:
        csv_writer = csv.writer(fo)
        with open('./data/RubyGems/all_package.json', 'r') as fi:
            for line in fi.readlines():
                line = json.loads(line)
                # features: size; focal ratio; age; version; contributor; star; fork; rank; keywords; dependents; repos
                for key, value in focal_ddg.items():
                    if line['name'] == key and line['name'] not in focal_pac:
                        focal_pac.append(line['name'])
                        print(key, line['versions'][0]['published_at'][:4])
                        # focal feature: name, size, age, star; fork; rank; keywords; dependents; repos; versions
                        csv_writer.writerow([key, len(value), 2021-int(line['versions'][0]['published_at'][:4]),
                                             line['stars'], line['forks'], line['rank'], len(line['keywords']),
                                             line['dependents_count'], line['dependent_repos_count'],
                                             len(line['versions'])])

    print(len(focal_pac))


def get_downstream_features():

    focal_ddg = {}

    with open('./data/RubyGems/DDG.json', 'r') as ddg_f:
        for line in ddg_f.readlines():
            line = json.loads(line)
            for key, value in line.items():
                if len(value) >= 20:
                    focal_ddg[key] = value
    print(len(focal_ddg))

    with open('./data/RubyGems/downstream_feature2021_median.csv', 'a+') as fo:
        csv_writer = csv.writer(fo)
        for key, value in focal_ddg.items():
            age = []
            star = []
            fork = []
            rank = []
            keywords = []
            dependents = []
            repos = []
            versions = []
            for v in value:
                with open('./data/RubyGems/all_package.json', 'r') as fi:
                    for line in fi.readlines():
                        line = json.loads(line)
                        # features: size; focal ratio; age; version; contributor; star; fork; rank;
                        # keywords; dependents; repos
                        if line['name'] == v:
                            age.append(2021-int(line['versions'][0]['published_at'][:4]))
                            star.append(line['stars'])
                            fork.append(line['forks'])
                            rank.append(line['rank'])
                            keywords.append(len(line['keywords']))
                            dependents.append(line['dependents_count'])
                            repos.append(line['dependent_repos_count'])
                            versions.append(len(line['versions']))

            print(key, np.average(star), np.average(dependents), np.average(versions))

            # downstream feature: name, size, age, star; fork; rank; keywords; dependents; repos; versions
            # csv_writer.writerow([key, len(value), np.average(age), np.average(star), np.average(fork),
            #                     np.average(rank), np.average(keywords), np.average(dependents),
            #                     np.average(repos), np.average(versions)])
            csv_writer.writerow([key, len(value), np.median(age), np.median(star), np.median(fork),
                                np.median(rank), np.median(keywords), np.median(dependents),
                                np.median(repos), np.median(versions)])


def get_downstream_features_top20():

    focal_ddg = {}

    pac_size = {}
    with open('./data/RubyGems/all_package.json', 'r') as fi:
        for line in fi.readlines():
            line = json.loads(line)
            pac_size[line['name']] = int(line['dependents_count'])

    with open('./data/RubyGems/CDDG.json', 'r') as ddg_f:
        for line in ddg_f.readlines():
            line = json.loads(line)
            for key, value in line.items():
                v_size = {}
                v_sort = []
                if len(value) >= 20:
                    l = int(0.2*len(value))
                    for v in value:
                        v_size[v] = pac_size[v]
                    print(v_size, len(value))
                    size_sort = sorted(v_size.items(), key=lambda item:item[1], reverse=True)
                    print(l, size_sort)
                    for i in range(l):
                        v_sort.append(size_sort[i][0])
                    print(v_sort)
                    focal_ddg[key] = v_sort
    print(focal_ddg)

    with open('./data/RubyGems/cddg_downstream_feature2021top20.csv', 'a+') as fo:
        csv_writer = csv.writer(fo)
        for key, value in focal_ddg.items():
            age = []
            star = []
            fork = []
            rank = []
            keywords = []
            dependents = []
            repos = []
            versions = []
            for v in value:
                with open('./data/RubyGems/all_package.json', 'r') as fi:
                    for line in fi.readlines():
                        line = json.loads(line)
                        # features: size; focal ratio; age; version; contributor; star; fork; rank;
                        # keywords; dependents; repos

                        # print(v)
                        if line['name'] == v:
                            age.append(2021-int(line['versions'][0]['published_at'][:4]))
                            star.append(line['stars'])
                            fork.append(line['forks'])
                            rank.append(line['rank'])
                            keywords.append(len(line['keywords']))
                            dependents.append(line['dependents_count'])
                            repos.append(line['dependent_repos_count'])
                            versions.append(len(line['versions']))

            print(key, np.average(star), np.average(dependents), np.average(versions))

            # downstream feature: name, size, age, star; fork; rank; keywords; dependents; repos; versions
            csv_writer.writerow([key, len(value), np.average(age), np.average(star), np.average(fork),
                                np.average(rank), np.average(keywords), np.average(dependents),
                                np.average(repos), np.average(versions)])


def get_downstream_features_collaborator():
    from itertools import combinations
    focal_ddg = {}

    with open('./data/Cargo/contributor.json', 'r') as fi:
        for line in fi.readlines():
            pac_con = json.loads(line)
    # print(pac_con['miniserver'], pac_con['actix-session'])
    # print(list(set(pac_con['miniserver']).intersection(set(pac_con['actix-session']))))

    with open('./data/Cargo/CDDG.json', 'r') as ddg_f:
        for line in ddg_f.readlines():
            line = json.loads(line)
            for key, value in line.items():
                print('key', key)
                value.append(key)
                # print(len(value), value)
                combine = list(combinations(value, 2))
                combine_count = []
                combine_result = set()
                for c in combine:
                    com_con = list(set(pac_con[c[0]]).intersection(set(pac_con[c[1]])))
                    # print(c, len(com_con))
                    combine_count.append([c[0], c[1], len(com_con)])
                # print('count', combine_count)
                combine_count = sorted(combine_count, key=lambda item: item[2], reverse=True)
                combine_sort = combine_count[:int(len(combine_count)*0.1)]
                # print('sort', combine_sort)
                for c in combine_sort:
                    combine_result.add(c[0])
                    combine_result.add(c[1])
                # print('result', len(combine_result), combine_result)
                focal_ddg[key] = list(combine_result)[:int(len(value)*0.2)]
                print(key, len(value), len(list(combine_result)[:int(len(value)*0.2)]))
                # return
    print(len(focal_ddg), focal_ddg['actix'])

    with open('./data/RubyGems/cddg_downstream_feature2021_collaborator20.csv', 'a+') as fo:
        csv_writer = csv.writer(fo)
        for key, value in focal_ddg.items():
            age = []
            star = []
            fork = []
            rank = []
            keywords = []
            dependents = []
            repos = []
            versions = []
            for v in value:
                with open('./data/RubyGems/all_package.json', 'r') as fi:
                    for line in fi.readlines():
                        line = json.loads(line)
                        # features: size; focal ratio; age; version; contributor; star; fork; rank;
                        # keywords; dependents; repos

                        # print(v)
                        if line['name'] == v:
                            age.append(2021-int(line['versions'][0]['published_at'][:4]))
                            star.append(line['stars'])
                            fork.append(line['forks'])
                            rank.append(line['rank'])
                            keywords.append(len(line['keywords']))
                            dependents.append(line['dependents_count'])
                            repos.append(line['dependent_repos_count'])
                            versions.append(len(line['versions']))

            print(key, np.average(star), np.average(dependents), np.average(versions))

            # downstream feature: name, size, age, star; fork; rank; keywords; dependents; repos; versions
            csv_writer.writerow([key, len(value), np.average(age), np.average(star), np.average(fork),
                                np.average(rank), np.average(keywords), np.average(dependents),
                                np.average(repos), np.average(versions)])

def get_wmd_semantic():
    focal_ddg = {}

    with open('./data/Cargo/DDG.json', 'r') as ddg_f:
        for line in ddg_f.readlines():
            line = json.loads(line)
            for key, value in line.items():
                if len(value) >= 20:
                    focal_ddg[key] = value
    print(len(focal_ddg))

    pac_description = {}
    pac_keywords = {}

    with open('./data/Cargo/all_package.json', 'r') as fi:
        for line in fi.readlines():
            line = json.loads(line)
            pac_keywords[line['name']] = line['keywords']
            pac_description[line['name']] = line['description']

    with open('./data/Cargo/downstream_ddg_description_semantic.csv', 'a+') as fo:
        csv_writer = csv.writer(fo)
        for key, value in focal_ddg.items():

            description_all = []
            keywords_all = []
            keywords_ = set()
            for v in value:
                if pac_keywords[v]:
                    keywords_all.append(pac_keywords[v])
                if pac_description[v]:
                    description_all.append(pac_description[v])
                for k in pac_keywords[v]:
                    keywords_.add(k)

            model = Word2Vec(description_all, size=100, window=2, min_count=1, workers=4, sg=1)

            semantic_results = []  # 关键字的语义一致性分析
            for i_first in range(len(value)):
                for i_second in range(len(value)):
                    if i_first >= i_second:
                        continue
                    result = model.wv.wmdistance(pac_description[value[i_first]],
                                                 pac_description[value[i_second]])
                    if math.isinf(result):
                        continue
                    semantic_results.append(result)
            print(key, np.average(semantic_results), keywords_)
            csv_writer.writerow([key, np.average(semantic_results)])


def get_jaccard_semantic():
    pac_keywords = {}
    with open('./data/RubyGems/all_package.json', 'r') as fi:
        for line in fi.readlines():
            line = json.loads(line)
            if line['keywords']:
                pac_keywords[line['name']] = line['keywords']

    focal_ddg = {}
    with open('./data/RubyGems/DDG.json', 'r') as ddg_f:
        for line in ddg_f.readlines():
            line = json.loads(line)
            for key, value in line.items():
                if len(value) >= 20:
                    focal_ddg[key] = value
    print(len(focal_ddg))

    with open('./data/RubyGems/downstream_ddg_jaccard_semantic.csv', 'a+') as fo:
        csv_writer = csv.writer(fo)
        for key, value in focal_ddg.items():
            semantic_results = []  # 关键字的语义一致性分析
            value.append(key)
            for i_first in range(len(value)):
                for i_second in range(len(value)):
                    if i_first >= i_second:
                        continue
                    if pac_keywords.get(value[i_first]) and pac_keywords.get(value[i_second]):
                        intersection = len(list(set(pac_keywords[value[i_first]]).
                                                intersection(pac_keywords[value[i_second]])))
                        union = (len(pac_keywords[value[i_first]]) + len(pac_keywords[value[i_second]])) - intersection
                        result = float(intersection) / union
                        semantic_results.append(result)
                        # print(result, pac_keywords[value[i_first]], pac_keywords[value[i_second]])
            print(key, np.average(semantic_results))
            csv_writer.writerow([key, np.average(semantic_results)])


def get_focal_ratio():
    ddg = {}
    with open('./data/RubyGems/DDG.json', 'r') as fi:
        for line in fi.readlines():
            line = json.loads(line)
            for key, value in line.items():
                ddg[key] = value
    print(ddg)

    focal_pac = set()
    for k, v in ddg.items():
        if len(v) >= 20:
            focal_pac.add(k)
    focal_pac = list(focal_pac)

    for key, value in ddg.items():
        # print(len(value), value)
        is_focal = set()
        for v in value:
            if v in focal_pac:
                is_focal.add(v)
        print(key, len(is_focal), len(value), len(is_focal) / len(value))


if __name__ == "__main__":
    get_downstream_size()
