#!/usr/bin/env bash

# Prerequisites:
# 1. Download Stanford NLP Java package from here:
# https://nlp.stanford.edu/software/stanford-parser-4.2.0.zip
# 2. unzip it inside tools directory

here=$(dirname "$0")
export CLASSPATH=$here/tools/stanford-parser-full-2020-11-17/stanford-parser.jar

stanford_parser="java edu.stanford.nlp.process.PTBTokenizer"

$stanford_parser
