import time
import torch
import pandas as pd
from disentangle import process
from transformers import CamembertTokenizer, CamembertModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "camembert-base"
tokenizer = CamembertTokenizer.from_pretrained(model_name)
model = CamembertModel.from_pretrained(model_name, output_hidden_states=True).to(device)
pd.set_option("display.max_rows", None)
pd.set_option('display.max_columns', None) 

pos_dict = {'préposition': 'adp',
 'verbe': ['verb', 'aux'],
 'nom': 'noun',
 'adverbe': 'adv',
 'adjectif': 'adj',
 'pronom': 'pron',
 'pronom relatif': 'pron',
 'conjonction': ['cconj', 'sconj'],
 'pronom déterminatif': 'pron',
 'numéral': 'num',
 'adjectif déterminatif': 'det',
 'pronom clitique': 'pron',
 'pronom personnel': 'pron',
 'article': 'det',
 'pronom interrogatif': 'pron',
 'préposition partitive': 'adp'}

pos_expect = [(k, v) if isinstance(v, str) else (k, vv) for k, v in pos_dict.items() for vv in (v if isinstance(v, list) else [v])]  

def get_tok_offset(tokens, example):
    tok_offsets = []
    prev_offset = 0
    for token in [token[1:] if token.startswith("▁") else token for token in tokens]:
        current_offset = example.find(token, prev_offset)
        tok_offsets.append((current_offset, current_offset + len(token)))
        prev_offset = current_offset + len(token)
    return tok_offsets

def get_word_subtoken(tokens, word_offset, tok_offsets):
    tokens_order = list(enumerate(tokens))
    subtoken_index = [
        [
            (tok[1], tok[0]+1)
        for tok, tok_offset, in zip(tokens_order, tok_offsets)
        if tok_offset[0] >= wd_offset[0] and tok_offset[1] <= wd_offset[1]
        ]
        for wd, wd_offset in word_offset]
    subtoken_index = [item for sublist in subtoken_index for item in sublist]
    return subtoken_index

def get_embedding(sent, subtoken_index, layer_num):   
    """
    sent: str, sentence to encode
    subtoken_index: list of tuple, start and end token index of the word
    """
    # encode the sentence
    inputs = tokenizer(sent, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states[layer_num] 

    # For simple lexeme
    if len(subtoken_index) == 1: 
        embedding = hidden_states[:, subtoken_index[0][1], :]
    # For idioms
    else:
        embedding = hidden_states[:, subtoken_index[0][1]:subtoken_index[-1][1]+1, :]
    # Back to CPU
    return embedding.detach().cpu()


def sum_emb(df):
    # Sum embeddings for each row
    df["sum_emb"] = None
    df["sum_emb"] = df["sum_emb"].astype(object)
    shape = df["embeddings_shape"].apply(tuple) == (1, 768)
    df.loc[shape, "sum_emb"] = pd.Series([torch.sum(emb, dim=0)
            for emb in df.loc[shape, "embeddings"]], index=df.index[shape])
    df.loc[~shape, "sum_emb"] = pd.Series([torch.sum(emb, dim=(0, 1))
            for emb in df.loc[~shape, "embeddings"]], index=df.index[~shape])

    # POS with inflection
    count_n = (df.POS_name == 'nom').sum()
    print(count_n) # 25153
    count_a = (df.POS_name == 'adjectif').sum()
    print(count_a) # 6630
    count_v = (df.POS_name == 'verbe').sum()
    print(count_v) # 6746
    print(sum([count_n, count_a, count_v])) # 38529

    return df

def merge_stanza(df, df_stanza):
    df_stanza = df_stanza[['node_id', 'word_offset', 'word_lemma', 'word_upos', 'word_features']]
    merged_df = pd.merge(df, df_stanza, on=['node_id', 'word_offset'])
    merged_df['word_features'] = [[v.lower() for v in feats.values()] if feats else None for feats in merged_df.word_features]
    merged_df.word_upos = merged_df.word_upos.str.lower()
    merged_df = merged_df.set_index('node_id')
    print(f"Got {len(merged_df)} rows of data.") #40933
    return merged_df

def check_stanza_errors(df):
    # Check stanza errors
    '''
    For checking stanza parsing errors, compare:
    'word_lemma', 'word_upos', 'word_features' in ex_stanza
    with 'lexname', 'POS_name', 'features_name' in lex_emb
    '''
    # POS
    # "POS_name" vs. "word_upos"
    df["pos_check"] = [True if (p1, p2) in pos_expect else False for p1, p2 in zip(df.POS_name, df.word_upos)]
    print(df.pos_check.value_counts())
    # Lemma
    # "lexname" vs. "word_lemma"
    df["lemma_check"] = [True if l1==l2 else False for l1, l2 in zip(df.lexname, df.word_lemma)]
    print(df.lemma_check.value_counts())
    # Drop pos errors then lemma errors
    df = df[df.pos_check]
    df = df[df.lemma_check]
    return df

def prepare_data():
    data = pd.read_pickle("data-LNfr.pkl")
    data = data[data.ph_status == 'lexie libre']
    data = data[['entry_id', 'std_name', 'lexname', 'POS_name', 'features_name', 'new_occurrence', 'example', 'word_offset']]
    data = data[data.new_occurrence.apply(len)==1] 

    # Eliminate 1381 rows with multiple occurrences: those with multiple occurrences or those with auxiliary verbs, difficult to calculate the embeddings

    data["tokens"] = data["example"].apply(tokenizer.tokenize)
    data["tok_offsets"] = [get_tok_offset(tokens, example) for tokens, example in zip(data.tokens, data.example)]
    data["subtoken_index"] = [get_word_subtoken(tokens, word_offset, tok_offsets) for tokens, word_offset, tok_offsets in zip(data.tokens, data.word_offset, data.tok_offsets)]
    data = data[data.word_offset.apply(len) == 1] 

    # We have eliminated 966 simple lexemes with multiple occurrences in the same example, this refers to those with auxiliary verbs or those tokenized into multiple subtokens. We get finally 40502 simple lexemes. 

    # Statistiques :
    # For lexemes with only one word offset, some are tokenized into multiple subtokens. There are 27590 with only one subtoken index, and 1998 of them are not starting with "▁".
    # len(data[data.subtoken_index.apply(len) == 1]) # 27590
    # len(data[data.subtoken_index.apply(len) == 1 & data.subtoken_index.apply(lambda x: not x[0][0].startswith("▁"))]) # 1998
    # one_subtok = data[data.subtoken_index.apply(len) == 1]
    # one_subtok[one_subtok.subtoken_index.apply(lambda x: not x[0][0].startswith("▁"))].sample(5)

    # Combine with stanza data and check errors
    data.word_offset = [lst[0] for lst in data.word_offset]
    data = data[['entry_id', 'lexname', 'POS_name', 'features_name', 'example', 'word_offset', 'subtoken_index']].reset_index()
    df_stanza = pd.read_pickle("stanza-LNfr.pkl")
    df = merge_stanza(data, df_stanza)
    df = check_stanza_errors(df)
    print(len(df))
    
    return df


def main():
    layer_num = 12
    df = prepare_data()
    print(f"Processing layer {layer_num}...")
    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    df["embeddings"] = [get_embedding(sent, subtoken_index, layer_num) for sent, subtoken_index in zip(df.example, df.subtoken_index)]
    df["embeddings_shape"] = [x.shape for x in df.embeddings]
    df = sum_emb(df)  
    df = df.rename(columns={'sum_emb': 'word_emb'})  # If using mean method, rename from 'mean_emb'
    df = df[['lexname', 'example', 'word_upos', 'word_features', 'word_offset', 'subtoken_index', 'embeddings', 'embeddings_shape', 'word_emb']]
    process(df, layer_num)  # Disentangle lexical and grammatical vectors, results will be saved in the 'results' folder
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.time()
    print(f"Layer {layer_num} processed in {end - start:.2f} seconds.")


if __name__ == "__main__":
    main()
