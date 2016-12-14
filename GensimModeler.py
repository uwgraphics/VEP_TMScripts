__author__ = 'ealexand'

import os
import shutil
import sys
import codecs
import argparse
import time
import csv
from scipy.sparse import dok_matrix
import random
import json
import math

from gensim import corpora, models, similarities
from nltk.corpus import stopwords
#from nonnegfac.nmf import NMF
from RegexTokenizer import RegexTokenizer as RegT


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

def buildGSmodel(args):
    fullStartTime = time.time()

    # Make sure the corpus and output paths are directories, then make path for model
    if not os.path.isdir(args.corpus_path):
        print 'Invalid corpus_path %s. Argument is not a directory.' % args.corpus_path
        exit(1)
    if not os.path.isdir(args.output_path):
        print 'Invalid output_path %s. Argument is not a directory.' % args.output_path
        exit(1)
    modelDir = os.path.join(args.output_path, args.model_name)
    createDir(modelDir, False)

    # Handle stopwords as passed in by user
    stopwordList = []
    if args.nltk_stopwords:
        stopwordList += stopwords.words('english')
    if args.extra_stopword_path:
        if os.path.isfile(args.extra_stopword_path):
            with open(args.extra_stopword_path, 'rb') as extraStopsFile:
                extraStopsStr = extraStopsFile.read()
                extraStops = extraStopsStr.split()
                stopwordList += extraStops
        else:
            print 'Invalid extra_stopword_path %s. Argument is not a file.' % args.extra_stopword_path
            exit(1)

    # Build the tokenizer
    justWordTokenizer = RegT(case_sensitive=False,
                             preserve_original_strs=True,
                             excluded_token_types=(1,2,3))

    # Read and parse the corpus
    if not args.silent:
        print 'Reading corpus...'
        start = time.time()
    textList = os.listdir(args.corpus_path)
    numTexts = len(textList)
    trimmedTextTokens = [] # List of lists of trimmed tokens for each text (only including non-stopwords)

    # If we're tokenizing, create some extra variables for mapping from chunk to text
    if args.chunk_size:
        chunksPerText = {}      # Dict mapping textName to number of chunks
        chunkNumToTextName = [] # List mapping chunkNum to textName
        textNameToChunkNums = {}
        chunkNum = 0

    for i in range(numTexts):
        textName = textList[i]
        if not args.silent:
            msg = '\rReading text %d of %d: %s...' % (i+1, numTexts, textName)
            print msg,
            sys.stdout.flush()

        # Get string from file and tokenize
        with codecs.open(os.path.join(args.corpus_path, textName), 'rb', encoding=args.encoding) as inF:
            textStr = inF.read()
        wordTokens = justWordTokenizer.tokenize(textStr)

        # Extract the tokens from Joe's funky format and extract stopwords (NLTK-defined)
        # If chunking, split the files up
        if args.chunk_size:
            chunksPerText[textName] = int(math.ceil(float(len(wordTokens)) / args.chunk_size))
            tokenIndex = 0
            while tokenIndex < len(wordTokens):
                trimmedTokens = []
                for token in wordTokens[tokenIndex : tokenIndex + args.chunk_size]:
                    if not token[0][0] in stopwordList:
                        trimmedTokens.append(token[0][0])
                trimmedTextTokens.append(trimmedTokens)
                chunkNumToTextName.append(textName)
                if textName in textNameToChunkNums:
                    textNameToChunkNums[textName].append(chunkNum)
                else:
                    textNameToChunkNums[textName] = [chunkNum]
                chunkNum += 1
                tokenIndex += args.chunk_size
        # If not chunking, then just tokenize the full file string
        else:
            trimmedTokens = []
            for token in wordTokens:
                if not token[0][0] in stopwordList:
                    trimmedTokens.append(token[0][0])
            trimmedTextTokens.append(trimmedTokens)

    # Quick test to make sure we did chunking right
    if args.chunk_size:
        if not chunkNum == len(trimmedTextTokens):
            print 'ERROR: chunkNum and length of the chunked texts list do not match.'
            exit(1)

    if not args.silent:
        print 'Done reading. (%.2f seconds)' % (time.time() - start)

    if not args.silent:
        print 'Creating gensim corpus...'
        start = time.time()
    wordDict = corpora.Dictionary(trimmedTextTokens)
    gsCorpus = [wordDict.doc2bow(text) for text in trimmedTextTokens]
    corpora.MmCorpus.serialize(os.path.join(modelDir, 'corpus.mm'), gsCorpus)
    if not args.silent:
        print 'Done. (%.2f seconds)' % (time.time() - start)

    # Now build the model!
    if not args.silent:
        print 'Build LDA model...'
        start = time.time()
    #ldaModel = models.LdaMulticore(gsCorpus, id2word=wordDict, num_topics=args.num_topics, minimum_probability=.001)
    ldaModel = models.LdaModel(gsCorpus,
                               id2word=wordDict,
                               num_topics=args.num_topics,
                               alpha='auto',
                               eta='auto',
                               iterations=2000,
                               minimum_probability=.0001)
    if not args.silent:
        print 'Done. (%.2f seconds)' % (time.time() - start)
        print 'Saving model...'
        start = time.time()
    #ldaModel.save(os.path.join(modelDir, 'model_multi.save'))
    ldaModel.save(os.path.join(modelDir, 'model.save'))
    if not args.silent:
        print 'Done. (%.2f seconds)' % (time.time() - start)

    # Build Serendip Files
    serendipDir = os.path.join(modelDir, 'TopicModel')
    createDir(serendipDir)

    # Build theta, both writing out theta.csv and the object theta
    # theta is a list of dictionaries that map topic -> prop for each doc
    if not args.silent:
        thetaStart = time.time()
        print 'Writing out theta.csv file...'

    # If we're chunking, gotta be smart about calculating docs' proportions
    if args.chunk_size:
        splitThetaLists = ldaModel[gsCorpus]
        theta = [ {} for i in range(len(splitThetaLists)) ]
        with open(os.path.join(serendipDir, 'theta.csv'), 'wb') as tFile:
            thetaWriter = csv.writer(tFile)
            # Loop through all the document numbers
            for dNum in range(len(textList)):
                thetaRowDict = {}
                pTot = 0.0
                textName = textList[dNum]
                # First, add up all the probabilities for all the chunks in this text
                for chunkNum in textNameToChunkNums[textName]:
                    for topic, p in splitThetaLists[chunkNum]:
                        if topic in thetaRowDict:
                            thetaRowDict[topic] += p
                        else:
                            thetaRowDict[topic] = p
                        theta[chunkNum][topic] = p
                        pTot += p
                # Then, normalize into row
                thetaRow = []
                for topic in thetaRowDict:
                    thetaRowDict[topic] /= pTot
                    thetaRow += [topic, thetaRowDict[topic]]
                # Write the row out to the file
                thetaWriter.writerow(thetaRow)

    # If we're not chunking, just loop through each row
    else:
        # thetaLists has a list for each doc containing (topic, prop) tuples
        thetaLists = ldaModel[gsCorpus]
        theta = [{} for i in range(len(thetaLists))]
        with open(os.path.join(modelDir, 'theta.csv'), 'wb') as tFile:
            thetaWriter = csv.writer(tFile)
            for dNum in range(len(thetaLists)):
                thetaRow = []
                for topic, p in thetaLists[dNum]:
                    thetaRow += [topic, p]
                    theta[dNum][topic] = p
                thetaWriter.writerow(thetaRow)
    if not args.silent:
        print 'Done writing theta.csv. (%.2f seconds)' % (time.time() - thetaStart)

    writeTopicCSVs(ldaModel, serendipDir)
    writeDefaultMeta(textList, serendipDir)
    tag_corpus(args.corpus_path, textList, theta, ldaModel, serendipDir, textNameToChunkNums)

    if not args.silent:
        print 'Total time elapsed: %.2f seconds' % (time.time() - fullStartTime)

    return {
        'fnames': textList,
        'corpus': gsCorpus,
        'model': ldaModel
    }

def bow2matrix(bow, numDocs, numWords):
    s = dok_matrix((numWords, numDocs))
    for docNum in range(len(bow)):
        for wordId, count in bow[docNum]:
            s[wordId, docNum] = count
    return s

def writeDefaultMeta(filelist, modelDir):
    with open(os.path.join(modelDir, 'metadata.csv'), 'wb') as mFile:
        metaWriter = csv.writer(mFile)
        metaWriter.writerow(['id','filename'])
        metaWriter.writerow(['int','str'])
        for i in range(len(filelist)):
            metaWriter.writerow([i, filelist[i]])

# Given GenSim model and containing director, write topics to CSV files for use in Serendip
def writeTopicCSVs(model, modelDir, wordThreshold=None, densityThreshold=0.99, silent=False):
    if not silent:
        print 'Writing topics to CSV files...'
        topicStart = time.time()

    # If they don't give us a wordThreshold, set it to max (vocab size)
    if wordThreshold is None:
        wordThreshold = model.num_terms

    # Create directory for CSVs
    csvDir = os.path.join(modelDir, 'topics_freq') # TODO: sal, ig?
    createDir(csvDir)

    # Loop through topics, writing words and proportions to CSV file
    for topicNum in range(model.num_topics):
        with open(os.path.join(csvDir, 'topic_%d.csv' % topicNum), 'wb') as csvF:
            topicWriter = csv.writer(csvF)

            topicArray = model.show_topic(topicNum, min(model.num_terms, wordThreshold))
            cutoffIndex = 0
            currDensity = 0.0
            while currDensity < densityThreshold and cutoffIndex < wordThreshold:
                word, prob = topicArray[cutoffIndex]
                topicWriter.writerow([word, prob])
                currDensity += prob
                cutoffIndex += 1

    if not silent:
        print 'Done writing topic CSV files. (%.2f seconds)' % (time.time() - topicStart)

# Tag the files with the trained model, creating tokens.csv files and rules.json files
def tag_corpus(corpusDir, textNames, theta, model, modelDir, nameToChunkNums=None, silent=False):
    if not silent:
        print 'Tagging texts and writing token CSVs...'
        tokenStart = time.time()

    # Make the HTML directory for Serendip
    htmlDir = os.path.join(modelDir, 'HTML')
    createDir(htmlDir)

    # Create dictionaries for each topic mapping word to proportion
    topicDicts = [ {} for i in range(model.num_topics) ]
    for t in range(model.num_topics):
        tList = model.show_topic(t, model.num_terms)
        for i in range(len(tList)):
            topicDicts[t][tList[i][0]] = tList[i][1]
    if not silent:
        print 'Done making topic-word dictionaries. (%.2f seconds)' % (time.time() - tokenStart)

    # Query this p_tGwd[d][w] = ordered list of (topic, prop) descending by prop
    p_tGwd = [ {} for d in range(len(theta)) ]

    # Build tokenizer for files to be tagged
    taggingTokenizer = RegT(case_sensitive=False,
                             preserve_original_strs=True)#,
                             #excluded_token_types=(1,2,3))

    # Loop through the texts and tag 'em

    for textNum in range(len(textNames)):
        textName = textNames[textNum]
        with codecs.open(os.path.join(corpusDir, textName), 'rb', encoding=args.encoding) as textF:
            textStr = textF.read()
        currTokens = taggingTokenizer.tokenize(textStr)
        rules = {}
        outList = []
        if args.chunk_size:
            wordIndex = 0
            textChunks = nameToChunkNums[textName]
            chunkIndex = 0

        # Loop through all the tokens in the file
        for tokenIndex in range(len(currTokens)):
            token = currTokens[tokenIndex]
            # If it's a word or punc, write it out
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
                    if (len(currTokens)) == tokenIndex + 1:
                        joiner = ''
                    else:
                        raise
                csvLine = [token[RegT.INDEXES['STRS']][-1], token[RegT.INDEXES['STRS']][0], joiner]

                # If the word is in our model, get the tag
                word = token[RegT.INDEXES['STRS']][0]
                if isWord and word in model.id2word.token2id:
                    # First, decide whether we're indexing by text or by chunk, and calculate appropriately
                    if args.chunk_size:
                        if wordIndex == args.chunk_size:
                            chunkIndex += 1
                            wordIndex = 0
                        relevantTextIndex = textChunks[chunkIndex]
                        wordIndex += 1
                    else:
                        relevantTextIndex = textNum

                    # If we haven't already calculated p of topic given word, doc, do so
                    if not word in p_tGwd[relevantTextIndex]:
                        tmpTopicPropList = []
                        tot = 0.0
                        for t in range(model.num_topics):
                            try:
                                tmpProp = topicDicts[t][word] * theta[relevantTextIndex][t]
                                tmpTopicPropList.append([t, tmpProp])
                                tot += tmpProp
                                # May get a KeyError if word isn't in topicDicts[t] or t isn't in theta[textNum]

                            except KeyError:
                                continue
                        for i in range(len(tmpTopicPropList)):
                            tmpTopicPropList[i][1] /= tot
                        tmpTopicPropList.sort(key=lambda x: x[1], reverse=True)
                        p_tGwd[relevantTextIndex][word] = tmpTopicPropList

                    # Randomly sample from the p_tGwd distribution to create a tag for this instance
                    rand = random.random()
                    densityTot = 0.0
                    i = 0
                    while densityTot < rand:
                        topic, prop = p_tGwd[relevantTextIndex][word][i]
                        densityTot += prop
                        i += 1
                        # TODO: also get the rank_bin

                    # Add rule to rules for sampled topic
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

                    # Now, add the tag to the end
                    csvLine.append(rule_name) # TODO: freq, sal, ig

                # Finally, append the tagged token to the outList, which will be written to file
                outList.append(csvLine)

        # Build directory for it
        nameSansExtension = textName[:-4]
        currHTMLdir = os.path.join(htmlDir, nameSansExtension)
        createDir(currHTMLdir)
        # Write rules to json file
        with open(os.path.join(currHTMLdir, 'rules.json'), 'wb') as jsonF:
            jsonF.write(json.dumps(rules))
        # Write the tokens to CSV file
        with codecs.open(os.path.join(currHTMLdir, 'tokens.csv'), 'wb', encoding=args.encoding) as tokensF:
            tokensWriter = csv.writer(tokensF)
            tokensWriter.writerows(outList)

    if not silent:
        print 'Done tagging texts. (%.2f seconds)' % (time.time() - tokenStart)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='A gensim-based topic modeler for Serendip')

    parser.add_argument('--model_name', help='name of the model', required=True)
    parser.add_argument('--corpus_path', help='path to corpus directory', required=True)
    parser.add_argument('--output_path', help='path to output directory (new directory by will be made for model_name', required=True)
    parser.add_argument('-n', '--num_topics', help='number of topics to infer', type=int, required=True)
    parser.add_argument('--encoding', help='text encoding type with which to read documents (default: utf-8)', default='utf-8')
    parser.add_argument('--nltk_stopwords', help='extract nltk English stopwords from documents before modeling', action='store_true')
    parser.add_argument('--extra_stopword_path', help='path to file containing space-delimited stopwords to exclude before modeling')
    parser.add_argument('--chunk_size', help='if set, will split texts into chunks containing at most the given number of word tokens. (default is no chunking)', type=int)
    parser.add_argument('--silent', help='if set, will suppress console output', action='store_true')

    if 0:
        args = parser.parse_args([
            '--corpus_path', 'C:\Users\Admin\Projects\VEP_Core\\vep_core\Data\Corpora\Shakespeare',
            '--model_name', 'GS_Shake_50_chunked',
            '--output_path', 'C:\Users\Admin\Projects\VEP_Core\\vep_core\Data\Metadata',
            '--num_topics', '50',
            '--nltk_stopwords',
            '--extra_stopword_path', 'C:\Users\Admin\Projects\VEP_Core\\vep_core\Data\Stopwords\\nonStatisticalEMstopwords.txt',
            '--chunk_size', '2000'
        ])
    if 1:
        args = parser.parse_args()

    buildGSmodel(args)
