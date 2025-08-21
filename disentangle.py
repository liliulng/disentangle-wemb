import warnings
import random
import torch
import pandas as pd
import torch.nn.functional as F
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations, product
import os

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
sns.set_style('white')
colors = sns.color_palette('Set2')

def redo_adj_feats(feats):
    feats_map = {
        ('masc', 'sing'): ('masc', 'sing'),
        ('fem', 'sing'): ('fem', 'sing'),
        ('masc', 'plur'): ('masc', 'plur'),
        ('fem', 'plur'): ('fem', 'plur'),
        # ('sing',): (None, 'sing'),
        # ('plur',): (None, 'plur'),
        # ('masc',): ('masc', None),
        ('sing',): ('masc', 'sing'),
        ('plur',): ('fem', 'plur'),
        ('masc',): ('masc', 'sing'),
        ('masc', 'ord', 'sing'): ('masc', 'sing'),
        ('fem', 'ord', 'sing'): ('fem', 'sing')
    }
    if feats is None:
        return (None, None)
    return feats_map.get(feats, feats)

def expand_verb_feats(feats):
    feats_dict = {
        'gender': None,
        'mood': None,
        'number': None,
        'person': None,
        'tense': None,
        'voice': None,
        'finiteness': None,
    }

    for f in feats:
        if f == 'imp':
            '''
            There are 2 'imp': 
            'imp' for 'impératif' which is always the first of the tuple
            'imp' for 'imparfait'
            '''
            if feats.index(f) == 0:
                feats_dict['mood'] = f
            else:
                feats_dict['tense'] = f
        elif f in ['masc', 'fem']:
            feats_dict['gender'] = f
        elif f in ['sing', 'plur']:
            feats_dict['number'] = f
        elif f in ['1', '2', '3']:
            feats_dict['person'] = f
        elif f in ['pres', 'past', 'fut']:
            feats_dict['tense'] = f
        elif f in ['ind', 'cnd', 'sub']:
            feats_dict['mood'] = f
        elif f in ['act', 'pass']:
            feats_dict['voice'] = f
        elif f in ['fin', 'part', 'inf']:
            feats_dict['finiteness'] = f 
    return pd.Series(feats_dict)

def gp_statis(df, gp_col):
    '''
    Group by a column and get group name, size and number of groups.
    '''
    gp = df.groupby(gp_col)
    gp_statis = gp.size().value_counts().reset_index()
    gp_statis = gp_statis.rename(columns={gp_statis.columns[0]:"group_size", "count":"group_num"}).sort_values(by="group_size")
    gp_statis['group_name'] = [gp.size()[gp.size() == size].index.tolist() for size in gp_statis['group_size']]
    gp_statis = gp_statis.reset_index(drop=True)
    print(f"Got {len(gp)} groups.")
    return gp, gp_statis

def gp_mean_emb(gps):
    mean_embs = torch.stack([torch.mean(torch.stack(list(gp)), dim=0) for _, gp in gps])
    gp_mean = dict(zip(gps.groups.keys(), mean_embs))
    return gp_mean

def get_lex_mean(df, lex_col):
    '''Groupby lexeme and get the mean lexical embedding for each lexeme.'''
    gps = df.groupby(df.index)[lex_col]
    gp_mean = gp_mean_emb(gps)
    lex_mean = df.index.map(gp_mean)
    return lex_mean

def get_flex_mean(df, gp_col, flex_col, flex_mean_name):
    '''Group by inflection type and get the mean inflectional embedding for each group.'''
    gps = df.groupby(gp_col)[flex_col]
    gp_mean = gp_mean_emb(gps)
    # There are duplicated values in df.index, set gp_mean as a df and gp_col as the index, this avoids InvalidIndexError, ensures that the mapping is done based on unique values in gp_col.
    gp_mean = pd.DataFrame(gp_mean.items(), columns=[gp_col, flex_mean_name])
    gp_mean = gp_mean.set_index(gp_col)
    flex_mean = df[gp_col].map(gp_mean[flex_mean_name])
    return flex_mean

def split_lex_flex(df, gp_col, max_iter_time):
    '''
    Split the lexical and inflectional embeddings.
    gp_col: The feature column to group by: 'word_features', 'gender', 'number', etc.
    max_iter_time: The maximum number of iterations.
    '''
    warnings.simplefilter("ignore")

    # Drop None values
    none_count = df[gp_col].isna().sum()
    df = df.dropna(subset=gp_col)
    print(f"\nDropped {none_count} None values, got {len(df)} rows with {gp_col} values.")
    # lex : Group by lexical unit and get the initial mean lexical embedding for each lexeme 
    df['lex_mean'] = get_lex_mean(df, 'word_emb')  
    # flex : Group by inflection and get the initial mean inflectional embedding
    df['flex_mean'] = get_flex_mean(df, gp_col, 'word_emb', 'flex_mean') 

    df['delta_flex_list'] = [[]for i in range(len(df))]
    df['delta_lex_list'] = [[]for i in range(len(df))]

    for _ in range(max_iter_time):

        # print(f"\nIteration {i+1}")

        # 1. word_lex = word - flex_mean
        df['word_lex'] = df.word_emb - df.flex_mean

        # 2. new_lex_mean
        df['new_lex_mean'] = get_lex_mean(df, 'word_lex')

        # 3. delta_lex = new_lex_mean - lex_mean
        df['delta_lex'] = pd.Series(torch.linalg.vector_norm(torch.stack(list(df.new_lex_mean)) - torch.stack(list(df.lex_mean)), dim=1), index = df.index)
        
        for lst, delta in zip(df.delta_lex_list, df.delta_lex): # Do not use list comprehension, will cause problem of None value in list column.
          lst.append(delta)

        # 4. Update lex_mean
        df['lex_mean'] = df.new_lex_mean

        # 5. word_flex = word - new_lex_mean
        df['word_flex'] = df.word_emb - df.lex_mean

        # 6. new_flex_mean
        df['new_flex_mean'] = get_flex_mean(df, gp_col, 'word_flex', 'new_flex_mean')

        # 7. delta_flex = new_flex_mean - flex_mean
        # df['delta_flex'] = pd.Series(torch.linalg.vector_norm(torch.stack(list(df.new_flex_mean)) - torch.stack(list(df.flex_mean)), dim=1), index=df.index)
        new_flex_mean = torch.stack(df.new_flex_mean.tolist())
        flex_mean = torch.stack(df.flex_mean.tolist())
        df['delta_flex'] = pd.Series(torch.linalg.vector_norm(new_flex_mean - flex_mean, dim=1), index=df.index)
        
        # df['delta_flex'] = pd.Series(F.cosine_similarity(torch.stack(list(df.new_flex_mean)),torch.stack(list(df.flex_mean)),dim=1),index=df.index)

        for lst, delta in zip(df.delta_flex_list, df.delta_flex):
          lst.append(delta)

        # 8. Update flex_mean
        df['flex_mean'] = df.new_flex_mean
    
    return df

def delta_statis(df, gp_col, delta_col):
    # Get statistics of each group
    gp, statis = gp_statis(df, gp_col)
    statis.rename(columns={statis.columns[-1]:gp_col}, inplace=True)
    statis = statis.explode(gp_col, ignore_index=True)
    delta = gp[delta_col].first().reset_index(name=delta_col)
    statis = statis.merge(delta, on=gp_col)
    statis = statis.set_index(gp_col)
    return statis

def double_plot(df, cols, measure_method='cosine'):
    palette = sns.color_palette('Set2', len(cols))
    
    fig, (ax_box, ax_hist) = plt.subplots(
        nrows=2, 
        sharex=True,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": (0.3, 0.7)}
    )
    
    # Boxplot
    if len(cols) == 1:
        col = cols[0]
        sns.boxplot(
            x=df[col], ax=ax_box, orient='h',
            color=palette[0], fliersize=1, linewidth=1
        )
        mean_val = df[col].mean()
        ax_box.scatter(mean_val, 0, marker='^', color='black', s=60, zorder=3)
        ax_box.set_yticks([0])
        ax_box.set_yticklabels([col])
    else:
        box_data = pd.melt(df[cols], var_name='Variable', value_name='Value')
        sns.boxplot(
            x='Value', y='Variable', data=box_data, ax=ax_box, 
            orient='h', palette=palette, fliersize=1, linewidth=1
        )
        for idx, col in enumerate(cols):
            mean_val = df[col].mean()
            ax_box.scatter(mean_val, idx, marker='^', color='black', s=60, zorder=3)
        ax_box.set_yticks(range(len(cols)))
        ax_box.set_yticklabels(['before', 'after'])
    ax_box.set(xlabel=None)
    ax_box.set_ylabel('')
    
    # Histplot
    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, color=palette[i], label=col, ax=ax_hist, bins=30)
    # if len(cols) > 1:
    #     ax_hist.legend(loc='upper right', frameon=False)
    if measure_method == 'cosine':
        ax_hist.set_xlabel('Similarity')
        ax_hist.legend(['dist_before', 'dist_after'], loc='upper right', frameon=False)
    elif measure_method == 'l2':
        ax_hist.set_xlabel('L2 dist')
        ax_hist.legend(['l2_dist_before', 'l2_dist_after'], loc='upper right', frameon=False)
    else:
        ax_hist.set_xlabel(f"{measure_method}")
        ax_hist.legend(['before', 'after'], loc='upper right', frameon=False)
    for ax in (ax_box, ax_hist):
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()
    plt.show()

def pair_dist(pairs, measure_method):
    '''Calculate pairwise distance'''
    emb1, emb2 = zip(*pairs)
    if measure_method == 'cosine':
        dist = F.cosine_similarity(torch.stack(emb1), torch.stack(emb2), dim=1)
    elif measure_method == 'l2':
        dist = torch.linalg.vector_norm(torch.stack(emb1) - torch.stack(emb2), dim=1)
    return dist

def test_emb_entry(emb_entry):
    if isinstance(emb_entry, pd.Series):
        return emb_entry.tolist()
    elif isinstance(emb_entry, torch.Tensor):
        return [emb_entry]
    else:
        return []

def flex_pairs(df, emb_type, pair_num=10000):

    '''Get pairs of tokens with same feature value, but different lexeme'''
    toks = list(zip(df.index, df[emb_type]))
    pairs = []
    used_diff_pair = set()  # To track pairs of different lexeme and feature
    checked = 0
    rejected = 0
    max_possible = len(toks) * (len(toks) - 1) // 2
    if pair_num > max_possible:
        warnings.warn(f"Requested pair_num {pair_num} exceeds maximum possible ({max_possible}). Using {max_possible}.")
        pair_num = max_possible  
    
    while len(pairs) < pair_num:
        tok1, tok2 = random.sample(toks, 2)
        id1, emb1 = tok1
        id2, emb2 = tok2
        checked += 1
        if id1 != id2: # Ensure different lexeme
            diff_pair = tuple(sorted([id1, id2])) 
            if diff_pair not in used_diff_pair:
                used_diff_pair.add(diff_pair)
                pair = (emb1, emb2)
                pairs.append(pair)
            else:
                rejected += 1
        else:
            rejected += 1

    # print(f"Checked {checked} pairs, rejected {rejected} pairs, got {len(pairs)} pairs.")

    return pairs


def random_pairs(df, feature_name, emb_type, pair_num=10000):
    '''Get pairs of random tokens differ in feature value and different lexeme'''

    toks = list(zip(df.index, df[feature_name], df[emb_type]))
    pairs = []
    used_diff_pair = set()  
    checked = 0
    rejected = 0
    max_possible = len(toks) * (len(toks) - 1) // 2
    if pair_num > max_possible:
        warnings.warn(f"Requested pair_num {pair_num} exceeds maximum possible ({max_possible}). Using {max_possible}.")
        pair_num = max_possible  
    while len(pairs) < pair_num:
        tok1, tok2 = random.sample(toks, 2)
        id1, feat1, emb1 = tok1
        id2, feat2, emb2 = tok2
        checked += 1
        if id1 != id2 and feat1 != feat2:  # Ensure different lexeme and feature
            diff_pair = tuple(sorted([(id1, feat1), (id2, feat2)]))
            if diff_pair not in used_diff_pair:
                used_diff_pair.add(diff_pair)
                pair = (emb1, emb2)
                pairs.append(pair)
            else:
                rejected += 1
        else:
            rejected += 1
    
    # print(f"Checked {checked} pairs, rejected {rejected} pairs, got {len(pairs)} pairs.")

    return pairs

def eval_df(pairs_before, pairs_after, measure_method='l2'):
    dist_before = pair_dist(pairs_before, measure_method).mean().item()
    dist_after = pair_dist(pairs_after, measure_method).mean().item()
    dist_delta = (dist_after - dist_before) / dist_before
    result = pd.DataFrame({
        'dist_before': [round(dist_before, 4)],
        'dist_after': [round(dist_after, 4)],
        'dist_delta': [round(dist_delta, 4)]
    })
    # print(f"Mean distance before: {dist_before:.2f}, after: {dist_after:.2f}, delta: {dist_delta*100:.2f}%")
    return result

def repeat_func(func, pair_num=None, n_repeat=5, verbose=False, as_percentage=False, **kwargs):
    import inspect
    if pair_num is not None:
        if 'pair_num' in inspect.signature(func).parameters:
            kwargs['pair_num'] = pair_num
    records = []
    for _ in range(n_repeat):
        result = func(**kwargs)
        if isinstance(result, pd.DataFrame):
            records.append(result)
        else:
            raise ValueError(f"Function must return a pandas DataFrame, got {type(result)}.")

    result_df= pd.concat(records, ignore_index=True)
    mean_series = result_df.mean().round(4)
    # if as_percentage:
    #     for col in mean_series.index:
    #         if "delta" in col:
    #             mean_series[col] = mean_series[col] * 100
    result_df = mean_series.to_frame().T
    # if verbose:
#     # print(f"\nAverage over {n_repeat} runs:")
#     for col in result_df.columns:
#         if pd.api.types.is_numeric_dtype(result_df[col]):
#             mean_val = result_df[col].mean()
#             suffix = "%" if as_percentage and "delta" in col else ""
#             factor = 100 if as_percentage and "delta" in col else 1
#             print(f"  mean {col}: {mean_val * factor:.2f}{suffix}")
#         else:
#             print(f"  {col}: [non-numeric]")
    return result_df

def lex_pair_eval(df, feature_name, emb_type_before, emb_type_after, measure_method='l2'):

    '''Pairwise distance between tokens with different feature value of a lexeme'''
    print(f"\nGrammatical category : {feature_name}")
    groups = df.groupby(df.index)
    print(f"Got {len(groups)} lexemes.")
    group_feat_counts = groups[feature_name].nunique()
    # print(f"Lexemes with one unique feature: {(group_feat_counts==1).sum()}\nLexemes with multiple features: {(group_feat_counts!=1).sum()}")
    mul_feat_group = group_feat_counts[group_feat_counts!=1]
    df_multi_feat = df[df.index.isin(mul_feat_group.index)]
    gp_multi_feat = df_multi_feat.groupby(df_multi_feat.index)
    results = []
    for lexeme, gp in gp_multi_feat: # for each lexeme
        pairs_before =[]
        pairs_after =[]
        toks_before = [test_emb_entry(gp[gp[feature_name] == feat][emb_type_before]) for feat in gp[feature_name].unique()] 
        toks_after = [test_emb_entry(gp[gp[feature_name] == feat][emb_type_after]) for feat in gp[feature_name].unique()]
        for i, j in combinations(toks_before, 2):
            pairs_before.extend(list(product(i, j)))
        for i, j in combinations(toks_after, 2):
            pairs_after.extend(list(product(i, j)))
        result = eval_df(pairs_before, pairs_after, measure_method)
        # result['node_id'] = name
        results.append(result)
    results_df = pd.concat(results, ignore_index=True)
    results_df = results_df.mean().round(4).to_frame().T
    return results_df

def flex_pair_eval(df, feature_name, pair_num=10000):
    results = pd.DataFrame()
    for name, gp in df.groupby(feature_name):
        # print(f"\n{name}")
        result = repeat_func(pair_eval, pair_num=pair_num, n_repeat=10, type='flex', df=gp, emb_type_before='word_emb', emb_type_after='word_flex')
        result.index = [name]
        results = pd.concat([results, result], axis=0)
    return results

def debug_tensor_cols(df, cols):
    for col in cols:
        bad = df[col].apply(lambda x: not isinstance(x, torch.Tensor))
        if bad.any():
            print(f"[DEBUG] Column {col}: {bad.sum()} bad rows (non-Tensor)")
            print(df.loc[bad, [col, 'lexname', 'word_features']].head())

def ensure_tensor_series(s: pd.Series, ref: torch.Tensor) -> pd.Series:
    return s.apply(lambda x: x if isinstance(x, torch.Tensor) else torch.zeros_like(ref))

def remake_emb(df, df_lex, df_flex_dict, feature_names):
    # Initial lex and flex embeddings
    keep_cols = ['lexname', 'word_features', 'word_emb'] + feature_names
    df = df[keep_cols].copy()
    df['lex_mean'] = get_lex_mean(df, 'word_emb') # before lex: lex_mean
    for feat in feature_names: # feat : number, gender, mood, etc.
        df[f'flex_mean_{feat}'] = get_flex_mean(df, feat, 'word_emb', f'flex_mean_{feat}')

    # Before distillation
    flex_cols_before = [col for col in df.columns if col.startswith('flex_mean_')]
    emb0=df.word_emb.iloc[0]
    # Ensure all flex columns are tensors
    for col in flex_cols_before:
        df[col] = ensure_tensor_series(df[col], emb0)

    df['new_wemb_before'] = [
        row['lex_mean'] + torch.mean(torch.stack([row[col] for col in flex_cols_before]), dim=0)
        for _, row in df.iterrows()
    ]

    df['remake_error_before'] = pd.Series(torch.linalg.vector_norm(torch.stack(list(df.new_wemb_before)) - torch.stack(list(df.word_emb)), dim=1), index=df.index)

    # Final lex and flex embeddings
    df['word_lex'] = df_lex.word_lex
    for feat in feature_names:
        df_flex = df_flex_dict[feat]
        flex_vals = df_flex['word_flex'].tolist()
        k = 0
        res = []
        for _, row in df.iterrows():
            if pd.notna(row[feat]): 
                res.append(flex_vals[k])
                k += 1
            else:
                res.append(torch.zeros_like(emb0)) 
        df[f'word_flex_{feat}'] = res
        df[f'word_flex_{feat}'] = ensure_tensor_series(df[f'word_flex_{feat}'], emb0)
        
    # After distillation
    flex_cols_after = [col for col in df.columns if col.startswith('word_flex')]
    # df['new_wemb_after'] = df.word_lex + df[flex_cols_after].sum(axis=1)
    debug_tensor_cols(df, flex_cols_after + ['word_lex'])
    df['new_wemb_after'] = [
        row['word_lex'] + torch.mean(torch.stack([row[col] for col in flex_cols_after]), dim=0)
        for _, row in df.iterrows()
    ]
    df['remake_error_after'] = pd.Series(torch.linalg.vector_norm(torch.stack(list(df.new_wemb_after)) - torch.stack(list(df.word_emb)), dim=1), index=df.index)

    return df

def remake_eval(df):
    # Calculate the difference between initial and final remake error
    df['remake_error_delta'] = (df.remake_error_after - df.remake_error_before) / df.remake_error_before
    # print(f"Mean remake error before : {df.remake_error_before.mean():.2f}, after: {df.remake_error_after.mean():.2f}")
    # print(f"Mean remake error delta: {df.remake_error_delta.mean()*100:.2f}%")
    # double_plot(df, ['remake_error_before', 'remake_error_after'], measure_method='remake_error')
    result = pd.DataFrame({
        'remake_error_before': [round(df.remake_error_before.mean().item(), 4)],
        'remake_error_after': [round(df.remake_error_after.mean().item(), 4)],
        'remake_error_delta': [round(df.remake_error_delta.mean().item(), 4)]
    })
    return result
    

def pair_eval(type, df, emb_type_before, emb_type_after, measure_method='l2', feature_name=None, pair_num=10000):
    '''
    type : 'lex', 'flex', 'random', remake
    feature_name : 'number', 'gender', 'mood', etc.
    '''
    if type == 'random':
        pairs_before = random_pairs(df, feature_name, emb_type_before, pair_num=pair_num)
        pairs_after = random_pairs(df, feature_name, emb_type_after, pair_num=pair_num)
        result = eval_df(pairs_before, pairs_after, measure_method)

    elif type == 'flex':
        pairs_before = flex_pairs(df, emb_type_before, pair_num=pair_num)
        pairs_after = flex_pairs(df, emb_type_after, pair_num=pair_num)
        result = eval_df(pairs_before, pairs_after, measure_method)

    elif type == 'lex': 
        result = lex_pair_eval(df, feature_name, emb_type_before, emb_type_after, measure_method)

    elif type == 'remake':
        result = remake_eval(df)
        
    return result   

def distill_lex(df):  
    df = split_lex_flex(df, 'word_features', max_iter_time=10)
    df = df.drop(columns=['new_lex_mean', 'new_flex_mean'])
    # print(f"\n{delta_statis(df, 'word_features', 'delta_flex_list')}")
    return df

def eval_lex(df, n_repeat=10):
    print('\nEvaluation : lex')
    result_lex = pair_eval('lex', df, 'word_emb', 'word_lex', feature_name="word_features")
    print('\nEvaluation : random')
    result_rdm = repeat_func(func=pair_eval, pair_num=10000, n_repeat=n_repeat, type='random', feature_name="word_features", emb_type_before="word_emb", emb_type_after="word_lex", df=df)
    # make a dataframe with the results
    result = pd.concat([result_lex, result_rdm], axis=1)
    return result

def distill_flex(df, feature_name):
    df = split_lex_flex(df, feature_name, max_iter_time=10)
    df = df.drop(columns=['new_lex_mean', 'new_flex_mean'])
    # print(delta_statis(df, feature_name, 'delta_flex_list'))
    return df

def eval_flex(df, feature_name, n_repeat=10, pair_num=10000):
    print('\nEvaluation : flex')
    result_flex = flex_pair_eval(df, feature_name, pair_num=pair_num)
    print('\nEvaluation : random')
    result_rdm = repeat_func(func=pair_eval, pair_num=pair_num, n_repeat=n_repeat, type='random', feature_name=feature_name, emb_type_before="word_emb", emb_type_after="word_lex", df=df)
    return result_flex, result_rdm

# Data preparation
def prepare_df(df):
    print("Prepare data")
    df = df[['lexname', 'example', 'word_upos', 'word_features', 'word_offset', 'subtoken_index', 'embeddings', 'embeddings_shape', 'word_emb']]
    df.loc[:, 'word_features'] = [tuple(feats) if feats else None for feats in df.word_features] # Turn list to hashable tuple
    gp, statis = gp_statis(df, df.index) 
    df = df[df.index.isin([name for names in statis[statis.group_size>=4].group_name for name in names])] # Keep words with at least 4 examples
    emb_center = torch.stack(df.word_emb.tolist()).mean(dim=0, keepdim=True)
    df.loc[:,'word_emb'] = list(torch.stack(df.word_emb.tolist()) - emb_center) # Mean-centering original word embeddings

    # Nouns
    print('Exract nouns.')
    df_n = df[df.word_upos=='noun'].copy()
    print(f"Got {len(df_n)} nouns.")
    df_n = df_n.dropna(subset=['word_features'])
    print('Extract word features.')
    df_n['number'] = [tuple(['sing' if 'sing' in feats else 'plur' if 'plur' in feats else feats]) for feats in df_n.word_features] 
    df_n['number'] = df_n['number'].str[0]  # There are two types: sing, plur
    print(f"Got {len(df_n[df_n.number == 'sing'])} singular nouns, {len(df_n[df_n.number == 'plur'])} plural nouns.")
    # Adjectives
    print('Exract adjectives.')
    df_adj = df[df.word_upos=='adj'].copy()
    print(f"Got {len(df_adj)} adjectives.")
    print('Extracted word features.')
    df_adj['word_features'] = [redo_adj_feats(feats) for feats in df_adj.word_features]
    df_adj[['gender', 'number']] = pd.DataFrame(df_adj.word_features.tolist(), index=df_adj.index)
    print(f"Got {len(df_adj[df_adj.number == 'sing'])} singular adjectives, {len(df_adj[df_adj.number == 'plur'])} plural adjectives, {len(df_adj[df_adj.gender == 'masc'])} masculine adjectives,  {len(df_adj[df_adj.gender == 'fem'])} feminine adjectives.")
    # Verbs
    print('Exracted verbs.')
    df_v = df[df.word_upos=='verb'].copy()
    df_v = df_v.dropna(subset=['word_features'])
    print(f"Got {len(df_v)} verbs.")
    print('Extracted word features.')
    feats_expanded = pd.DataFrame([expand_verb_feats(feats) for feats in df_v.word_features], index=df_v.index)
    feats_expanded = feats_expanded.applymap(lambda x: 'v_'+x if x else None)
    df_v = pd.concat([df_v, feats_expanded], axis=1)
    return df_n, df_adj, df_v

def save_to_folder(df_list, folder):
        os.makedirs(folder, exist_ok=True)
        for name,df in df_list.items():
            file_path = os.path.join(folder, f"{name}.csv")
            df.to_csv(file_path)
            print(f"Saved {name} to {file_path}")

def process(df, layer_num):
    # df = pd.read_pickle('combined-stanza-lexemb-sum-checked-layer12.pkl')
    df_n, df_adj, df_v = prepare_df(df)

    # Distill the lexical vectors
    print('\nExtract lexical vectors.')
    print('\nExtract lexical vectors : Nouns.')
    df_n_lex = distill_lex(df_n)
    print('\nExtract lexical vectors : Adjectives.')
    df_adj_lex = distill_lex(df_adj)
    print('\nExtract lexical vectors : Verbs.')
    df_v_lex = distill_lex(df_v)

    # Evaluate the lexical vectors distillation
    print('\nEvaluate lexical vectors.')
    print('\nEvaluate lexical vectors : Nouns.')
    eval_n_lex = eval_lex(df_n_lex)
    print('\nEvaluate lexical vectors : Adjectives.')
    eval_adj_lex = eval_lex(df_adj_lex)
    print('\nEvaluate lexical vectors : Verbs.')
    eval_v_lex = eval_lex(df_v_lex)
    results_lex = pd.concat([eval_n_lex, eval_adj_lex, eval_v_lex], axis=0)
    results_lex.index = ['nouns', 'adjectives', 'verbs']
    new_cols = [("Lexeme", c) for c in results_lex.columns[:3]] + [("Random", c) for c in results_lex.columns[3:]]
    results_lex.columns = pd.MultiIndex.from_tuples(new_cols, names=["type", "pos"])
    # Output: results_lex
    
    # Distill the grammatical vectors
    print('\nExtract grammatical vectors.')
    print('\nExtract grammatical vectors : N-number.')
    df_n_num = distill_flex(df_n, 'number')
    print('\nExtract grammatical vectors : Adj-number')
    df_adj_num = distill_flex(df_adj, 'number')
    print('\nExtract grammatical vectors : Adj-gender')
    df_adj_gen = distill_flex(df_adj, 'gender')
    print("\nExtract grammatical vectors : V-mood")
    df_v_mood = distill_flex(df_v, 'mood')
    print("\nExtract grammatical vectors : V-number")
    df_v_num = distill_flex(df_v, 'number')
    print("\nExtract grammatical vectors : V-person")
    df_v_person = distill_flex(df_v, 'person')
    print("\nExtract grammatical vectors : V-tense")
    df_v_tense = distill_flex(df_v, 'tense')
    print("\nExtract grammatical vectors : V-finiteness")
    df_v_finit = distill_flex(df_v, 'finiteness')

    # Evaluate the grammatical vectors distillation
    print('\nEvaluate grammatical vectors.')
    print('\nEvaluate grammatical vectors : N-number.')
    eval_n_num, eval_n_rdn = eval_flex(df_n_num, 'number', pair_num=10000)
    print('\nEvaluate grammatical vectors : Adj-number.')
    eval_adj_num, eval_adj_num_rdn = eval_flex(df_adj_num, 'number', pair_num=5000)
    print('\nEvaluate grammatical vectors : Adj-gender.')
    eval_adj_gen, eval_adj_gen_rdn = eval_flex(df_adj_gen, 'gender', pair_num=5000)
    print("\nEvaluate grammatical vectors : V-mood")
    eval_v_mood, eval_v_mood_rdn = eval_flex(df_v_mood, 'mood', pair_num=1000)
    print("\nEvaluate grammatical vectors : V-number")
    eval_v_num, eval_v_num_rdn = eval_flex(df_v_num, 'number', pair_num=1000)
    print("\nEvaluate grammatical vectors : V-person")
    eval_v_person, eval_v_person_rdn = eval_flex(df_v_person, 'person', pair_num=300)
    print("\nEvaluate grammatical vectors : V-tense")
    eval_v_tense, eval_v_tense_rdn = eval_flex(df_v_tense, 'tense', pair_num=50)
    print("\nEvaluate grammatical vectors : V-finiteness")
    eval_v_finit, eval_v_finit_rdn = eval_flex(df_v_finit, 'finiteness', pair_num=1500)

    results_flex = pd.concat([
        eval_n_num, eval_adj_num, eval_adj_gen,
        eval_v_mood, eval_v_num, eval_v_person, eval_v_tense,
        eval_v_finit
    ], axis=0)

    # make multi-index columns
    new_index = ([("Noun",r) for r in results_flex.index[:2]] + [("Adjective",r) for r in results_flex.index[2:6]] + [("Verb",r) for r in results_flex.index[6:]])
    results_flex.index = pd.MultiIndex.from_tuples(new_index)

    results_flex_rdn = pd.concat([
        eval_n_rdn, eval_adj_num_rdn, eval_adj_gen_rdn,
        eval_v_mood_rdn, eval_v_num_rdn, eval_v_person_rdn, eval_v_tense_rdn,
        eval_v_finit_rdn
    ], axis=0)

    results_flex_rdn.index = ['N-number', 'Adj-number', 'Adj-gender', 'V-mood', 'V-number', 'V-person', 'V-tense', 'V-finiteness']
    # results_flex_rdn.index.name = 'random'

    # Output: results_flex, results_flex_rdn

    # Reconstruction evaluation
    print('\nReconstruction evaluation : Nouns.')
    df_flex_dict = {'number': df_n_num}
    feature_names = ['number']
    t = remake_emb(df_n, df_n_lex, df_flex_dict, feature_names)
    rem_n = remake_eval(t)

    print('\nReconstruction evaluation : Adjectives.')
    df_flex_dict = {'number': df_adj_num, 'gender':df_adj_gen}
    feature_names = ['number', 'gender']
    t = remake_emb(df_adj, df_adj_lex, df_flex_dict, feature_names)
    rem_adj = remake_eval(t)

    print('\nReconstruction evaluation : Verbs.')
    df_flex_dict = {'mood':df_v_mood, 'number': df_v_num, 'person': df_v_person, 'tense': df_v_tense, 'finiteness': df_v_finit}
    feature_names = ['mood','number', 'person', 'tense', 'finiteness']
    t = remake_emb(df_v, df_v_lex, df_flex_dict, feature_names)
    rem_v = remake_eval(t)

    rem_results = pd.concat([rem_n, rem_adj, rem_v], axis=0)
    rem_results.index = ['nouns', 'adjectives', 'verbs']

    # Save results

    all_results = {
        'results_lex': results_lex,
        'results_flex': results_flex,
        'results_flex_rdn': results_flex_rdn,
        'rem_results': rem_results
    }
    save_to_folder(all_results, f'results-layer-{layer_num}')
