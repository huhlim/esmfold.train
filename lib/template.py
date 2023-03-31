#!/usr/bin/env python

import os
import sys
import pathlib
import tempfile
import subprocess as sp
import typing as T

from openfold.data.parser import TemplateHit



class MMseqs_template_searcher(object):
    def __init__(self, mmseqs_binary_path, mmseqs_db_path):
        self.mmseqs_binary_path = mmseqs_binary_path
        self.mmseqs_db_path = mmseqs_db_path

    def parse(self, output: str) -> T.Sequence[TemplateHit]:
        hit_s = []
        index = -1
        for line in output.split("\n")[:-1]:
            if line.strip() == "":
                continue
            #
            index += 1

            query = line.strip().split()

    def query(self, sequence: str):
        pwd = os.getcwd()
        tmpdir = tempfile.TemporaryDirectory(prefix="mmseqs.")
        os.chdir(tmpdir.name)
        #
        with open("input.fa", "wt") as fout:
            fout.write(f">input\n{sequence}")
        #
        # createdb
        cmd = [self.mmseqs_binary_path] + "createdb input.fa input -v 0".split()
        sp.call(cmd)
        #
        # search
        cmd = (
            [self.mmseqs_binary_path]
            + "search input".split()
            + [self.mmseqs_db_path]
            + "output ./ --alignment-mode 3 -v 0".split()
        )
        sp.call(cmd)
        #
        # align
        cmd = (
            [self.mmseqs_binary_path]
            + "align input".split()
            + [self.mmseqs_db_path]
            + "output_new -a -v 0".split()
        )
        sp.call(cmd)
        #
        # convertalis
        cmd = (
            [self.mmseqs_binary_path]
            + "convertalis input".split()
            + +[self.mmseqs_db_path]
            + "output_new output.m8 -v 0".split()
            + '--format-output "query,target,fident,qlen,tlen,qstart,qend,tstart,tend,qaln,taln'.split()
        )
        sp.call(cmd)
        #
        with open("output.m8") as fp:
            output = self.parse(fp.read())
        #
        os.chdir(pwd)
        #
        return output
