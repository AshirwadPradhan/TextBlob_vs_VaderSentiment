# vader is only for sentiment analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()
print('Vader Sentiment Analysis')

total_neg = 0
total_neg_correct = 0
with open('data/negative.txt', 'r') as f:
	for line in f.read().split('\n'):
		analysis = analyzer.polarity_scores(line)
		if not analysis['pos'] > 0.1:
			if analysis['pos'] - analysis['neg'] <=0:
				total_neg_correct += 1
			total_neg += 1

acc_neg = (total_neg_correct/total_neg) * 100

total_pos = 0
total_pos_correct = 0
with open('data/positive.txt', 'r') as f:
	for line in f.read().split('\n'):
		analysis = analyzer.polarity_scores(line)
		if not analysis['neg'] > 0.1:
			if analysis['pos'] - analysis['neg'] > 0:
				total_pos_correct += 1
			total_pos += 1

acc_pos = (total_pos_correct/total_pos) * 100


print('Positive Total : {total_pos}\n \
Postive Correct : {total_pos_correct}\n \
Postive Accuracy : {acc_pos}%\n\n'\
		.format(total_pos=total_pos, total_pos_correct=total_pos_correct,
			acc_pos=acc_pos))

print('Negative Total : {total_neg}\n \
Negative Correct : {total_neg_correct}\n \
Negative Accuracy : {acc_neg}%\n\n'\
		.format(total_neg=total_neg, total_neg_correct=total_neg_correct,
			acc_neg=acc_neg))
