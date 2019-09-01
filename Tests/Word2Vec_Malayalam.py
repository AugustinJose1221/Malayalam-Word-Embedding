from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from matplotlib import pyplot
# define training data
sentences = [['പ്രളയപുനര്‍നിര്‍മാണം', 'പരിസ്ഥിതി', 'സൗഹൃദമാക്കണമെന്ന്‌', 'മുഖ്യമന്ത്രി','പിണറായി','വിജയൻ','പറഞ്ഞു'],
			['ഹരിത','കേരള','മിഷന്റെ','ഭാഗമായി','വിവിധ','വകുപ്പുകള്‍','നടപ്പിലാക്കുന്ന','പദ്ധതികളുടെ','സംസ്ഥാനതല','ഉദ്ഘാടനം','കണ്ണൂർ','കടമ്പൂര്‍','കുഞ്ഞുമോലോം','ക്ഷേത്ര','പരിസരത്ത്','നിർവഹിച്ച്‌','സംസാരിക്കുകയായിരുന്നു','അദ്ദേഹം'],
			['പ്രളയാനന്തര','കേരളത്തിന്റെ','പുനര്‍നിര്‍മാണം','പൂര്‍ണമായും','പരിസ്ഥിതി','സൗഹൃദമാക്കാനാണ്','സര്‍ക്കാര്‍','ലക്ഷ്യമിടുന്നത്'],
			['ഇക്കാര്യത്തില്‍','എല്ലാവരുടെയും','സഹകരണമുണ്ടാവണം'],
			['ഓരോരുത്തര്‍ക്കും','തോന്നിയ','പോലെ','മണ്ണില്‍','ഇടപെടുന്ന','സ്ഥിതിക്ക്','മാറ്റം','വരണം']]
# train model
model = Word2Vec(sentences, min_count=1)
# fit a 2D PCA model to the vectors
X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)
# create a scatter plot of the projection
pyplot.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
	pyplot.annotate(i, xy=(result[i, 0], result[i, 1]))
	print(i," ",word)
pyplot.show()
