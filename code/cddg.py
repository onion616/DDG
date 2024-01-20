# -*- coding: utf-8 -*-
"""
__author__ = 'onion'
"""

import csv
import json
import numpy as np


def get_cddg():
    with open('./data/CPAN/contributor.json', 'r') as fi:
        for line in fi.readlines():
            pac_contributor = json.loads(line)

    cddg_dict = {}
    with open('./data/CPAN/DDG.json', 'r') as fi:
        for line in fi.readlines():
            line = json.loads(line)
            # print(line)
            cddg = set()
            for key, value in line.items():
                for v in value:
                    if pac_contributor.get(v) and pac_contributor.get(key):
                        if set(pac_contributor[key]) & set(pac_contributor[v]):
                            cddg.add(v)
                if len(list(cddg)) >= 20:
                    cddg_dict[key] = list(cddg)
    print(cddg_dict)

    with open('./data/CPAN/CDDG.json', 'w') as fo:
        json.dump(cddg_dict, fo)


def cddg_focal_feature():
    focal_pac = set()
    with open('./data/RubyGems/CDDG.json', 'r') as fi:
        cddg = json.load(fi)
        for c in cddg:
            focal_pac.add(c)
    focal_pac = list(focal_pac)

    with open('./data/RubyGems/all_package.json', 'r') as fi:
        for line in fi.readlines():
            line = json.loads(line)
            if line['name'] in focal_pac:
                with open('./data/RubyGems/cddg_focal_feature2021.csv', 'a+') as fo:
                    csv_writer = csv.writer(fo)
                    csv_writer.writerow([line['name'], len(cddg[line['name']]),
                                         2021-int(line['versions'][0]['published_at'][:4]),
                                         line['stars'], line['forks'], line['rank'], len(line['keywords']),
                                         line['dependents_count'], line['dependent_repos_count'],
                                         len(line['versions'])])

                print(line['name'], len(cddg[line['name']]), 2021-int(line['versions'][0]['published_at'][:4]),
                      line['stars'], line['forks'], line['rank'], len(line['keywords']),
                      line['dependents_count'], line['dependent_repos_count'], len(line['versions']))


def cddg_downstream_feature():
    focal_pac = set()
    with open('./data/RubyGems/CDDG.json', 'r') as fi:
        cddg = json.load(fi)
        for c in cddg:
            focal_pac.add(c)
    focal_pac = list(focal_pac)

    for f in focal_pac:
        age = []
        star = []
        fork = []
        rank = []
        keywords = []
        dependents = []
        repos = []
        versions = []
        for v in cddg[f]:
            with open('./data/RubyGems/all_package.json', 'r') as fi:
                for line in fi.readlines():
                    line = json.loads(line)
                    if line['name'] == v:
                        age.append(2021 - int(line['versions'][0]['published_at'][:4]))
                        star.append(line['stars'])
                        fork.append(line['forks'])
                        rank.append(line['rank'])
                        keywords.append(len(line['keywords']))
                        dependents.append(line['dependents_count'])
                        repos.append(line['dependent_repos_count'])
                        versions.append(len(line['versions']))
        print(f, np.average(star), np.average(dependents), np.average(versions))

        with open('./data/RubyGems/cddg_downstream_feature2021.csv', 'a+') as fo:
            csv_writer = csv.writer(fo)
            csv_writer.writerow([f, len(cddg[f]), np.average(age), np.average(star), np.average(fork),
                                np.average(rank), np.average(keywords), np.average(dependents),
                                np.average(repos), np.average(versions)])


def focal_ratio():
    with open('./data/RubyGems/CDDG.json', 'r') as fi:
        cddg = json.load(fi)
        print(cddg)

    focal_pac = set()
    for k in cddg.keys():
        focal_pac.add(k)
    focal_pac = list(focal_pac)

    for key, value in cddg.items():
        # print(len(value), value)
        is_focal = set()
        for v in value:
            if v in focal_pac:
                is_focal.add(v)
        print(key, len(is_focal), len(value), len(is_focal)/len(value))


if __name__ == "__main__":
    cddg_focal_feature()
