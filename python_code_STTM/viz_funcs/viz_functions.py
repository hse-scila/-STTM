from scipy.spatial.distance import cosine
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_similarity_matrix(model1, model2, similarity='cosine'):
    
    assert model1.num_terms == model2.num_terms
    
    topic_word_matrix = []
    for model in [model1, model2]:
        matrix = np.zeros(shape=(model.num_topics, model.num_terms))
        for i in range(model.num_topics):
            dist = np.array(model.get_topic_terms(i, topn=model.num_terms))
            matrix[i] = dist[dist[:, 0].argsort()][:, 1]
        topic_word_matrix.append(matrix)
        
    similarity_matrix = np.zeros(shape=(model1.num_topics, model2.num_topics))
    for i, k in zip(range(model1.num_topics), range(model1.num_topics)[::-1]):
        for j in range(model2.num_topics):
            if similarity == 'cosine':
                similarity_matrix[i][j] = 1 - cosine(topic_word_matrix[0][k], topic_word_matrix[1][j])
            
    return similarity_matrix, tuple(topic_word_matrix)

def plot_similarity_matrix(similarity_matrix, figsize=(10, 10)):
    
    plt.figure(figsize=figsize)
    ax = plt.gca()
    im = ax.imshow(similarity_matrix)

    plt.yticks(range(0, similarity_matrix.shape[0]), range(0, similarity_matrix.shape[0])[::-1])
    plt.xticks(range(0, similarity_matrix.shape[1]))

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(20)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(20)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)

    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=15)
    cbar.set_label('Similarity', fontsize=16, labelpad=15)

    plt.show()
    
def get_doc_topics_matrix(bow, topic_model):

    doc_topics_matrix = np.zeros(shape=(len(bow), topic_model.num_topics))
    for i in tqdm_notebook(range(doc_topics_matrix.shape[0])):
        doc_topics_matrix[i] = np.array(topic_model.get_document_topics(bow[i], minimum_probability=0))[:, 1]
        
    return doc_topics_matrix

def plot_ts_topic_modeling(timeSeries, topic_stream, casuality_effect, 
                           param=1.3, legend=True, alpha_border=0.2):
    
    assert timeSeries.shape == (topic_stream.shape[0],)
    assert casuality_effect.shape == (topic_stream.shape[1], )
    
    topics_ind = np.argsort(casuality_effect)
    topic_stream_df = pd.DataFrame(topic_stream, index=timeSeries.index)
    
    _neg = len(np.where(casuality_effect < 0)[0])
    _pos = len(np.where(casuality_effect > 0)[0])
    _zer = len(np.where(casuality_effect == 0)[0])
    cmap_pos = plt.cm.Blues(np.linspace(0, 1, round(param * _pos)))[-_pos:]
    cmap_neg = plt.cm.Reds(np.linspace(0, 1, round(param * _neg)))[::-1][:_neg]
    
    
    cmap = np.zeros(shape=(casuality_effect.shape[0], 4))
    
    base = 0
    if _neg:
        for i in range(_neg):
            cmap[base + i] = cmap_neg[i]
        base += i + 1
    if _zer:
        for j in range(_zer):
            cmap[base + j] = np.zeros(4)
        base += j + 1
    if _pos:
        for k in range(_pos):
            cmap[base + k] = cmap_pos[k]
        
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 1, 1)
    plt.plot(timeSeries, linewidth=3.0)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True)
    plt.legend(prop={'size':14})    
        
    plt.subplot(2, 1, 2)
    plt.fill_between(topic_stream_df.index, topic_stream_df[topics_ind[0]], 
                     label='Topic ID: {}'.format(topics_ind[0]), color=cmap[0])
    lower_topic = topic_stream_df[topics_ind[0]]

    for i in range(1, casuality_effect.shape[0]):
    
        upper_topic = lower_topic + topic_stream_df[topics_ind[i]]
        plt.fill_between(topic_stream_df.index, lower_topic, upper_topic, 
                         label='Topic ID: {}'.format(topics_ind[i]), color=cmap[i])
        plt.plot(upper_topic, color='black', alpha=alpha_border)
        lower_topic = upper_topic
    
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if legend:
        plt.legend(prop={'size':14})

    ax = plt.gca()
    ax.xaxis.grid(True)

    plt.show()
    
def create_circular_mask(h, w, center=None, radius=None):

    if center is None: 
        center = (int(w/2), int(h/2))
    if radius is None: 
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y-center[1]) **2)

    mask = dist_from_center <= radius
    mask = ~mask
    
    return mask
    
def plot_tonality_charged_words(screening_words, freq_vector, tone_vector, dictionary, 
                                 width=4000, height=2000, max_words=50, figsize=(20, 10)):
    
    vectors = pd.DataFrame(screening_words, columns=['word'])
    vectors['word'] = vectors['word'].apply(lambda x: dictionary[x])
    vectors['freq_vector'] = freq_vector
    vectors['tone_vector'] = tone_vector
    
    mask = create_circular_mask(height, width)
    mask = 255 * mask.astype(int)
    
    for sign in [True, False]:
        if sign:
            tmp = vectors[vectors.tone_vector >= 0].copy()
        else:
            tmp = vectors[vectors.tone_vector < 0].copy()
            tmp.tone_vector = np.abs(tmp.tone_vector)
        
        for vector in ['tone_vector', 'freq_vector']:
            frequencies = dict(tmp[['word', vector]].values)
            wc = WordCloud(width=width, height=height, max_words=max_words, 
                           background_color='white', mask=mask).generate_from_frequencies(frequencies)
            
            plt.figure(figsize=figsize)
            if sign:
                plt.title('Positive words: {}'.format(vector[:4]), fontsize=25, y=1.1)
            else:
                plt.title('Negative words: {}'.format(vector[:4]), fontsize=25, y=1.1)
            plt.imshow(wc)
            plt.tight_layout(pad=0)
            plt.axis("off")
            plt.show()
            
def plot_article_tonality_topics(article, tonality, words_dict, tm_model, dictionary, 
                                  significant_words_only=False):
        
    fig = plt.figure(figsize=(40, 20))
    grd.GridSpec(2, 3)

    topics_df = pd.DataFrame(tm_model.get_document_topics(article, minimum_probability=0), 
                             columns=['topic_id', 'probability'])
    topics_df['tonality'] = tonality
    topics_df = topics_df.sort_values(['probability'], ascending=False)
    topics_df = topics_df[:4]
    topics_id = topics_df.topic_id.values

    ax = plt.subplot2grid((2, 3), (0,0),  colspan=1, rowspan=2)
    plt.title('Topic distribution', fontsize=35)
    bars = ax.barh(range(len(topics_df)), topics_df['probability'][::-1], alpha=0.4)
    
    for i, bar in enumerate(bars[::-1]):
        if topics_df.iloc[i].tonality > 0:
            bar.set_color('g')
        elif topics_df.iloc[i].tonality < 0:
            bar.set_color('r')
            
    ax.set_yticks(range(len(topics_df)))
    ax.set_yticklabels(topics_df['topic_id'][::-1], horizontalalignment="left", fontsize=35)
    ax.tick_params(axis='y', direction='in', pad=-15)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(25)
    ax.grid(True)
    
    inds = [(i, j) for i in range(0, 2) for j in range(1, 3)]
    for i, ind in enumerate(inds):
        term_df = pd.DataFrame(tm_model.get_topic_terms(topics_id[i], topn=len(dictionary)), 
                               columns=['token_id', 'probability'])
        term_df['token'] = term_df['token_id'].apply(lambda x: dictionary[int(x)])
        
        if significant_words_only:
            term_df = term_df[term_df.token.isin(list(words_dict.keys()))].sort_values(by='probability', 
                                                                                       ascending=False)[:10]
        else:
            term_df = term_df[:10]
            
        ax = plt.subplot2grid((2, 3), ind)
        plt.title('Topic: {}'.format(str(topics_id[i])), fontsize=35)
        bars = ax.barh(range(len(term_df)), term_df['probability'][::-1], alpha=0.4)
        
        for j, bar in enumerate(bars[::-1]):
            if words_dict.get(term_df.iloc[j].token, np.nan) > 0:
                bar.set_color('g')
            elif words_dict.get(term_df.iloc[j].token, np.nan) < 0:
                bar.set_color('r')
                    
        ax.set_yticks(range(len(term_df))[::-1])
        ax.set_yticklabels(list(term_df['token']), horizontalalignment = "left", fontsize=35)  
        ax.tick_params(axis='y', direction='in', pad=-15)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(25)
        ax.grid(True)          

    fig.tight_layout()
    plt.show()