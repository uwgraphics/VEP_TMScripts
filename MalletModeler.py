__author__ = 'Eric'

import subprocess
import argparse
import os
import shutil
import csv
import sys
import math
import json
import codecs
import time
from collections import defaultdict
from copy import deepcopy
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),os.path.pardir))
from RegexTokenizer import RegexTokenizer as RegT
import gzip

numRankingBins = 5 # Used to set this with a command-line argument, but Serendip really just likes 5

def runSalTM(args):
    fullStartTime = time.time()

    tokenizer = RegT(case_sensitive=False, preserve_original_strs=True)
    justWordTokenizer = RegT(case_sensitive=False,
                             preserve_original_strs=True,
                             excluded_token_types=(1,2,3)
                             )

    # Make and name necessary directories
    if os.path.isdir(args.corpus_path):
        corpus_dir = args.corpus_path
    else:
        print 'Invalid corpus_path %s. Argument is not a directory.' % args.corpus_path
        exit(1)
    if os.path.isdir(args.output_path):
        createDir(os.path.join(args.output_path, args.model_name), False)
        malletDir = os.path.join(args.output_path, args.model_name, 'Mallet')
        createDir(malletDir, args.forceOverwrite)
    else:
        print 'Invalid output_path %s. Argument is not a directory.' % args.output_path
        exit(1)
    if args.extra_stopword_path is not None and not os.path.isfile(args.extra_stopword_path):
        print 'Invalid stopword file %s. Argument is not a file.' % args.extra_stopword_path
        exit(1)

    # Read and parse the corpus
    print 'Reading corpus...'
    start = time.time()
    malletCorpusDir = os.path.join(malletDir, 'Corpus')
    createDir(malletCorpusDir, args.forceOverwrite)
    #tokens = {} # tokens[textName] -> tokens from tokenizer
    textList = os.listdir(corpus_dir)
    numTexts = len(textList)
    for i in range(len(textList)):
        textName = textList[i]
        msg = '\rReading text %d of %d: %s...' % (i+1, numTexts, textName)
        print msg,
        sys.stdout.flush()
        # Get the string from file and tokenize it
        textStr = ''
        with codecs.open(os.path.join(corpus_dir, textName), 'rb', encoding=args.encoding) as inF:
            textStr = inF.read()
            #for line in inF:
            #    textStr += line
        wordTokens = justWordTokenizer.tokenize(textStr)
        #tokens = tokenizer.tokenize(textStr)
        #tokens2 = t2.tokenize(textStr)

        # Get the words out for the topic modeler to deal with
        '''words = []
        for token in tokens:
            if token[RegT.INDEXES['TYPE']] == RegT.TYPES['WORD']:
                words.append(token[RegT.INDEXES['STRS']][0]) # Banking on there only being one word per token in this context
        '''
        words = [ wt[RegT.INDEXES['STRS']][0] for wt in wordTokens ]

        if args.chunk_size is not None:
            numChunks = int(len(words)/args.chunk_size) + 1
            i = 0
            while i < numChunks:
                strForMallet = u' '.join(words[i*args.chunk_size:(i+1)*args.chunk_size])
                with codecs.open(os.path.join(malletCorpusDir, textName + '__' + str(i).zfill(6)), 'wb', encoding=args.encoding) as outF: # TODO: what if 6 isn't big enough?
                    outF.write(strForMallet)
                i += 1
        else:
            # Join the words to create a Mallet-readable string and output to disk
            strForMallet = u' '.join(words)
            with codecs.open(os.path.join(malletCorpusDir, textName), 'wb', encoding=args.encoding) as outF:
                outF.write(strForMallet)
    print 'Done reading. (%.2f seconds)' % (time.time() - start)

    # Import corpus with Mallet
    print 'Importing corpus to Mallet...'
    start = time.time()
    malletCorpusFile = os.path.join(malletDir, args.model_name + '.mallet')
    #tokenRegex = '"[^\p{Space}]*"'
    #tokenRegex = '"\p{L}+"'
    #tokenRegex = '[\p{L}\p{M}]+'
    tokenRegex = '"[^\s]+"'
    callStr = 'mallet import-dir --input %s --output %s --keep-sequence --token-regex %s' \
              % (malletDir, malletCorpusFile, tokenRegex)
    if args.mallet_stopwords:
        callStr += ' --remove-stopwords'
        if args.extra_stopword_path is not None:
            callStr += ' --extra-stopwords %s' % args.extra_stopword_path
    elif args.extra_stopword_path is not None:
        callStr += ' --stoplist-file %s' % args.extra_stopword_path
    print callStr
    returncode = subprocess.call(callStr, shell=True)
    if returncode != 0:
        print 'mallet operation failed. May need to be installed on machine.'
        exit(1)
    print 'Done importing. (%.2f seconds)' % (time.time() - start)

    # Run model with Mallet
    print 'Training topics with Mallet...'
    start = time.time()
    malletOutputDir = os.path.join(malletDir, 'Output')
    createDir(malletOutputDir, args.forceOverwrite)
    def getParamPath(paramName):
        return os.path.join(malletOutputDir, paramName)
    callStr = 'mallet train-topics --input %s' % malletCorpusFile
    # Modeling parameters
    if args.num_topics is not None:
        callStr += ' --num-topics %d' % args.num_topics
    if args.num_iterations is not None:
        callStr += ' --num-iterations %d' % args.num_iterations
    if args.alpha is not None:
        callStr += ' --alpha %f' % args.alpha
    if args.beta is not None:
        callStr += ' --beta %f' % args.beta
    if args.randomSeed is not None:
        callStr += ' --random-seed %d' % args.randomSeed
    if args.optimizeInterval is not None:
        callStr += ' --optimize-interval %d' % args.optimizeInterval
    if args.optimizeBurnIn is not None:
        callStr += ' --optimize-burn-in %d' % args.optimizeBurnIn
    # Command-line output
    if args.showTopicsInterval is not None:
        callStr += ' --show-topics-interval %d' % args.showTopicsInterval
    # Output files. Probably don't need every one of these--TODO: get rid of unnecessary ones
    callStr += ' --output-doc-topics %s' % getParamPath('docTopics')
    callStr += ' --output-topic-keys %s' % getParamPath('topicKeys')
    callStr += ' --output-state %s' % getParamPath('finalState.gz')
    callStr += ' --output-model %s' % getParamPath('finalModel')
    callStr += ' --topic-word-weights-file %s' % getParamPath('topicWordWeights')
    callStr += ' --word-topic-counts-file %s' % getParamPath('wordTopicCounts')
    # Now call the damn thing
    print callStr
    returncode = subprocess.call(callStr, shell=True)
    if returncode != 0:
        print 'mallet operation failed. May need to be installed on machine.'
        exit(1)
    print 'Done training. (%.2f seconds)' % (time.time() - start)

    # Convert Mallet files
    serendipDir = os.path.join(args.output_path, args.model_name, 'TopicModel') #TODO: these directories should really be labeled "Serendip"
    createDir(serendipDir, args.forceOverwrite)

    print 'Building theta...'
    buildThetaAndMeta(os.path.join(malletOutputDir, 'docTopics'),
                      os.path.join(serendipDir, 'theta.csv'),
                      os.path.join(serendipDir, 'metadata.csv'),
                      args.chunk_size, args)

    print 'Building topics dir...'
    rankBins = buildTopicCSVs(os.path.join(malletOutputDir, 'wordTopicCounts'),
                                serendipDir,
                                args.num_topics,
                                numRankingBins,
                                args.forceOverwrite)

    # Tag and format the finalState (can use gzip library rather than extracting)
    print 'Tagging documents...'
    start = time.time()
    htmlDir = os.path.join(serendipDir, 'HTML')
    createDir(htmlDir, args.forceOverwrite)

    # Helper function tokenizes file, applies given tags, and spits out .csv file.
    # CSV file looks like:
    # token, tokenToMatch, endReason, tag, indexWithinTag
    def tagFileAsCSV(name, tags):
        with codecs.open(os.path.join(corpus_dir, name), 'rb', encoding=args.encoding) as inF:
            textStr = inF.read()
        currTokens = tokenizer.tokenize(textStr)
        rules = {}
        outList = []
        tagIndex = 0
        for tokenIndex in range(len(currTokens)):
            token = currTokens[tokenIndex]
            # If it's a word, write it out
            isWord = token[RegT.INDEXES['TYPE']] == RegT.TYPES['WORD']
            isPunc = token[RegT.INDEXES['TYPE']] == RegT.TYPES['PUNCTUATION']
            if isWord or isPunc:
                # Set joiner (how tokens will be pieced back together)
                try:
                    if currTokens[tokenIndex+1][RegT.INDEXES['TYPE']] == RegT.TYPES['WHITESPACE']:
                        joiner = 's'
                    elif currTokens[tokenIndex+1][RegT.INDEXES['TYPE']] == RegT.TYPES['NEWLINE']:
                        joiner = 'n'
                    else:
                        joiner = ''
                except IndexError: # Presumably this means that we're at the last token
                    if len(currTokens) == tokenIndex + 1:
                        joiner = ''
                    else:
                        raise
                csvLine = [token[RegT.INDEXES['STRS']][-1], token[RegT.INDEXES['STRS']][0], joiner]

                # If this is actually a tagged token, get the tag
                if isWord and tagIndex < len(tags) and token[RegT.INDEXES['STRS']][0] == tags[tagIndex][0]:
                    topic = int(tags[tagIndex][1])
                    word = tags[tagIndex][0]
                    # TODO: deal with "rules.json"
                    rule_name = 'topic_%d' % topic
                    if rule_name in rules:
                        rules[rule_name]['num_tags'] += 1
                        rules[rule_name]['num_included_tokens'] += 1
                    else:
                        rules[rule_name] = {
                            'name': rule_name,
                            'full_name': rule_name,
                            'num_tags': 1,
                            'num_included_tokens': 1
                        }
                    try:
                        freqRank, igRank, salRank = rankBins[topic][word]
                    except KeyError:
                        freqRank, igRank, salRank = rankBins[topic][word.encode('utf-8')]

                    # Since we only have one-word tags, all "index within tag" values are 0
                    csvLine += ['topic_%d' % topic, 0,
                                'freq%d' % freqRank, 0,
                                'ig%d' % igRank, 0,
                                'sal%d' % salRank, 0]

                    tagIndex += 1

                # Add the csv line to the outList
                outList.append(csvLine)

        if tagIndex != len(tags):
            raise Exception('Error: not all of tags read in file %s. This may be an issue with the text encoding (currently attempting to use %s, can provide other options through --encoding parameter).' % (name, args.encoding))
        nameSansExtension = name[:-4]
        currHTMLdir = os.path.join(htmlDir, nameSansExtension)
        createDir(currHTMLdir, args.forceOverwrite)
        # Write rules to json file
        with open(os.path.join(currHTMLdir, 'rules.json'), 'wb') as jsonF:
            jsonF.write(json.dumps(rules))
        # Write the tokens to CSV file
        with codecs.open(os.path.join(currHTMLdir, 'tokens.csv'), 'wb', encoding=args.encoding) as tokensF:
            tokensWriter = csv.writer(tokensF)
            tokensWriter.writerows(outList)

    # Helper function that helps when dealing with chunked files
    def getBasename(filename):
        baseWithChunkNum = os.path.basename(filename)
        if args.chunk_size is None:
            return baseWithChunkNum
        else:
            return baseWithChunkNum[:baseWithChunkNum.find('__')]

    # Read through all the lines of the finalState.
    # Whenever we get a complete file, tag and format it.
    # Once we gzip -d finalState.gz, we'll get a file that looks like this, starting on line 4:
    # doc source pos typeindex type topic
    with gzip.open(os.path.join(malletOutputDir, 'finalState.gz'), 'rb') as stateF:
        # Eat first three lines (which aren't useful to us yet)
        stateF.next()
        stateF.next()
        stateF.next()
        # Kick off loop
        currLine = stateF.next().split()
        currDocNum = 0
        currDocName = getBasename(currLine[1])
        currTags = []
        while True:
            msg = '\rTagging document %d...' % (currDocNum + 1)
            print msg,
            sys.stdout.flush()

            if getBasename(currLine[1]) == currDocName:
                currTags.append((currLine[4], int(currLine[5])))
            else:
                # Deal with currDoc
                tagFileAsCSV(currDocName, currTags)
                # Update things for next doc
                currDocNum += 1
                currDocName = getBasename(currLine[1])
                currTags = [currLine[4:]]
            try:
                currLine = stateF.next().split()
            except StopIteration:
                tagFileAsCSV(currDocName, currTags)
                break
    print 'Done tagging. (%.2f seconds)' % (time.time() - start)

    print 'Total time elapsed: %.2f seconds' % (time.time() - fullStartTime)

# This function takes a given docTopic file output from Mallet and turns it into theta.csv and metadata.csv
# docTopic file has lines of this form:
# docNum filename topic proportion topic proportion topic proportion ...
# TODO: remove the middle man. theta is no better than docTopic. Update Serendip for this.
# TODO: bug. This doesn't take into account the fact that the last chunk might be smaller than the rest when normalizing.
def buildThetaAndMeta(docTopicPath, thetaPath, metaPath, chunk_size, args):
    with open(docTopicPath, 'rb') as dtFile: # TODO: args.encoding?
        with open(thetaPath, 'wb') as tFile: # TODO: args.encoding?
            with open(metaPath, 'wb') as mFile: # TODO: args.encoding?
                # Prep csv writers
                thetaWriter = csv.writer(tFile)
                metaWriter = csv.writer(mFile)
                # Prep metadata.csv
                metaWriter.writerow(['id','filename'])
                metaWriter.writerow(['int','str'])
                # If we're not chunking, this is pretty straightforward
                if chunk_size is None:
                    # Loop through docTopics file, adding lines to theta and metadata
                    firstLine = True
                    for line in dtFile:
                        if firstLine:
                            firstLine = False
                        else:
                            line = line.split()
                            sparseLine = []
                            for i in range(2, len(line), 2):
                                if float(line[i+1]) > .001:
                                    sparseLine += line[i:i+2]
                            thetaWriter.writerow(sparseLine) # Take out docNum and filename for theta
                            metaWriter.writerow([line[0], os.path.basename(line[1])]) # Just include docNum and filename for metadata
                # If we are chunking, then we need to combine and normalize the rows for each file
                else:
                    def writeFileLine(docNum, basename, thetaLine, numChunks, thetaWriter, metaWriter):
                        # normalize the row
                        for topic in thetaLine:
                            thetaLine[topic] /= numChunks
                        # write it
                        row = []
                        for topic in thetaLine:
                            row.append(topic)
                            row.append(thetaLine[topic])
                        thetaWriter.writerow(row)
                        metaWriter.writerow([docNum, os.path.basename(basename)]) # TODO: make sure it corresponds with metadata
                    currDocNum = -1
                    firstLine = True
                    currBasename = ''
                    currThetaline = {}
                    numMiniFiles = 0
                    for line in dtFile:
                        if firstLine:
                            firstLine = False
                        else:
                            line = line.split()
                            name = line[1]
                            basename = name[:name.find('__')]
                            # If we're starting a new file, finish the last one
                            if basename != currBasename:
                                if currBasename != '':
                                    writeFileLine(currDocNum, currBasename, currThetaline, numMiniFiles, thetaWriter, metaWriter)
                                # update for next row
                                currDocNum += 1
                                currBasename = basename
                                numMiniFiles = 0
                                currThetaline = {}
                            for i in range(2, len(line), 2):
                                line[i] = int(line[i])
                                line[i+1] = float(line[i+1])
                                if line[i+1] > .001:
                                    if line[i] in currThetaline:
                                        currThetaline[line[i]] += line[i+1]
                                    else:
                                        currThetaline[line[i]] = line[i+1]
                            numMiniFiles += 1
                    writeFileLine(currDocNum, currBasename, currThetaline, numMiniFiles, thetaWriter, metaWriter)

# This function takes a given wordTopicCounts file output from Mallet and turns it into topic_#.csv files.
# The wordTopicCounts file has lines of this form:
# typeIndex type topic#:count topic#:count topic#:count ...
#TODO: word distributions
def buildTopicCSVs(wordTopicCountsPath, topicDirPath, num_topics, numRankingBins, forceOverwrite=False):
    # Read the wordTopicCounts file
    with open(wordTopicCountsPath, 'rb') as wtcF:
        topicDists = [ [] for i in range(num_topics) ] # Distributions across vocab for each topic
        p_topic = [ 0.0 for i in range(num_topics) ] # probability of each topic
        p_topicGivenWord = [ {} for i in range(num_topics) ] # probability of each topic given word
        p_word = [] # probability of each word across whole corpus
        totWordCount = 0.0 # Used to normalize p_topic
        for line in wtcF:
            line = line.split()
            word = line[1]
            countPairs = line[2:] # These are the topic#:count pairs
            thisWordCount = 0.0 # Used to normalize p_topicGivenWord
            topicsWithThisWord = []
            for countPair in countPairs:
                topic, count = map(int, countPair.split(':'))
                topicDists[topic].append([word, count])
                p_topic[topic] += count
                p_topicGivenWord[topic][word] = count
                topicsWithThisWord.append(topic)
                totWordCount += count
                thisWordCount += count
            # Normalize p_topicGivenWord
            for topic in topicsWithThisWord:
                p_topicGivenWord[topic][word] /= thisWordCount
            # Add word to corpus distribution
            p_word.append([word, thisWordCount])
        # Normalize p_topic
        for topic in range(num_topics):
            p_topic[topic] /= totWordCount
        # Normalize p_word
        for i in range(len(p_word)):
            p_word[i][1] /= totWordCount

    # Sort and normalize topicDists
    for topic in range(num_topics):
        topicDists[topic].sort(key=lambda x: x[1], reverse=True)
        tot = 0.0
        for i in range(len(topicDists[topic])):
            tot += topicDists[topic][i][1]
        for i in range(len(topicDists[topic])):
            topicDists[topic][i][1] /= tot
    # Sort p_word
    p_word.sort(key=lambda x:x[1], reverse=True)

    # Calculate and sort information gain lists
    igLists = [ [] for i in range(num_topics) ]
    igDicts = [ {} for i in range(num_topics) ]
    for topic in range(num_topics):
        p_t = p_topic[topic]
        totIG = 0.0
        for word, freq in topicDists[topic]:
            try:
                p_tGw = p_topicGivenWord[topic][word]
            except KeyError:
                p_tGw = 0.0
            if p_tGw == 0:
                iG = math.log(1 / (1-p_t))
            elif p_tGw == 1:
                iG = math.log(1 / p_t)
            else:
                iG = p_tGw * math.log(p_tGw / p_t) + (1-p_tGw) * math.log((1-p_tGw) / (1-p_t))
            igLists[topic].append([word, iG])
            igDicts[topic][word] = iG
            totIG += iG
        igLists[topic].sort(key=lambda x: x[1], reverse=True)
        # Normalizing, if only b/c binning is easy if everything sums to 1
        for i in range(len(igLists[topic])):
            igLists[topic][i][1] /= totIG

    # Calculate and sort saliency lists
    salLists = [ [] for i in range(num_topics) ]
    for topic in range(num_topics):
        totSal = 0.0
        for word, freq in topicDists[topic]:
            try:
                sal = freq * igDicts[topic][word]
                salLists[topic].append([word, sal])
                totSal += sal
            except KeyError:
                pass
        salLists[topic].sort(key=lambda x: x[1], reverse=True)
        # Normalizing, if only b/c binning is easy if everything sums to 1
        for i in range(len(salLists[topic])):
            salLists[topic][i][1] /= totSal

    # Output ranking lists to proper directories
    freqPath = os.path.join(topicDirPath, 'topics_freq')
    createDir(freqPath, forceOverwrite)
    igPath = os.path.join(topicDirPath, 'topics_ig')
    createDir(igPath, forceOverwrite)
    salPath = os.path.join(topicDirPath, 'topics_sal')
    createDir(salPath, forceOverwrite)
    for topic in range(num_topics):
        filename = 'topic_%d.csv' % topic
        with open(os.path.join(freqPath, filename), 'wb') as freqF:
            freqWriter = csv.writer(freqF)
            for freqPair in topicDists[topic]:
                freqWriter.writerow(freqPair)
        with open(os.path.join(igPath, filename), 'wb') as igF:
            igWriter = csv.writer(igF)
            for igPair in igLists[topic]:
                igWriter.writerow(igPair)
        with open(os.path.join(salPath, filename), 'wb') as salF:
            salWriter = csv.writer(salF)
            for salPair in salLists[topic]:
                salWriter.writerow(salPair)

    # Output full corpus distribution
    with open(os.path.join(topicDirPath, 'corpus_dist.csv'), 'wb') as distF:
            distWriter = csv.writer(distF)
            for pair in p_word:
                distWriter.writerow(pair)

    # Create ranking bins
    rankBins = [ {} for i in range(num_topics) ]
    for topic in range(num_topics):
        # frequency bins
        currFreqTot = 0.0
        currBin = 1
        for word, freq in topicDists[topic]:
            if currFreqTot > currBin * (1.0 / numRankingBins):
                currBin += 1
            rankBins[topic][word] = [currBin, 0, 0]
            currFreqTot += freq
        # info gain bins
        currIgTot = 0.0
        currBin = 1
        for word, ig in igLists[topic]:
            if currIgTot > currBin * (1.0 / numRankingBins):
                currBin += 1
            rankBins[topic][word][1] = currBin
            currIgTot += ig
        # saliency bins
        currSalTot = 0.0
        currBin = 1
        for word, sal in salLists[topic]:
            if currSalTot > currBin * (1.0 / numRankingBins):
                currBin += 1
            rankBins[topic][word][2] = currBin
            currSalTot += sal

    # Return rankBins to be used in HTML tagging
    return rankBins

# Helper function that creates new directories, overwriting old ones if necessary and desired.
def createDir(name, force=False):
    if os.path.exists(name):
        if force:
            shutil.rmtree(name)
        else:
            response = raw_input('%s already exists. Do you wish to overwrite it? (y/n) ' % name)
            if response.lower() == 'y' or response.lower() == 'yes':
                shutil.rmtree(name)
            elif response.lower() == 'n' or response.lower() == 'no':
                print 'Modeler aborted.'
                exit(0)
            else:
                print 'Response not understood.'
                print 'Modeler aborted.'
                exit(1)
    os.mkdir(name)

def main():
    parser = argparse.ArgumentParser(description='A Mallet-based topic modeler for Serendip')

    parser.add_argument('--model_name', help='name of the model', required=True)
    parser.add_argument('--output_path', help='path to output directory (new directory will be made for model_name', required=True)
    parser.add_argument('--corpus_path', help='full path to corpus directory', required=True)
    parser.add_argument('-f', '--forceOverwrite', help='force overwriting of previous corpus/model', action='store_true')

    parsingGroup = parser.add_argument_group('Corpus parsing parameters')
    parsingGroup.add_argument('-mS', '--mallet_stopwords', help='ignore standard Mallet stopwords', action='store_true')
    parsingGroup.add_argument('-eS', '--extra_stopword_path', help='path to file of extra stopwords to be ignored')
    parsingGroup.add_argument('--chunk_size', help='number of tokens per chunk (no chunking if left empty)', type=int)
    parsingGroup.add_argument('--encoding', help='text encoding type with which to read documents (default: utf-8)', default='utf-8')
    # other good encoding: latin-1

    modelingGroup = parser.add_argument_group('LDA modeling parameters')
    modelingGroup.add_argument('-n', '--num_topics', help='number of topics to infer', type=int, required=True)
    modelingGroup.add_argument('-i', '--num_iterations', help='number of iterations run by modeler', type=int, default=100)
    modelingGroup.add_argument('-a', '--alpha', help='alpha parameter for model', type=float)
    modelingGroup.add_argument('-b', '--beta', help='beta parameter for model', type=float)
    modelingGroup.add_argument('-r', '--randomSeed', help='starting seed for model iterations', type=int)
    modelingGroup.add_argument('-oI', '--optimizeInterval', help='iterations between parameter optimizations', type=int)
    modelingGroup.add_argument('-oBI', '--optimizeBurnIn', help='iterations before first parameter optimization', type=int)

    feedbackGroup = parser.add_argument_group('Command-line feedback parameters')
    feedbackGroup.add_argument('-sTI', '--showTopicsInterval', help='iterations between showing topics', type=int)

    parser.set_defaults(func=runSalTM)
    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()