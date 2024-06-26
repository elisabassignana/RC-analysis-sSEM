import torch
import torch.nn as nn
import transformers


#
# Embeddings Base Class
#


class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_dim = None

    def __repr__(self):
        return f'<{self.__class__.__name__}: dim={self.emb_dim}>'

    def embed(self, sentences):
        """
        Returns a list of sentence embedding matrices for list of input sentences.

        Args:
            sentences: [['t_0_0', 't_0_1, ..., 't_0_N'], ['t_1_0', 't_1_1', ..., 't_1_M']]

        Returns:
            [np.Array(sen_len, emb_dim), ...]
        """
        raise NotImplementedError



class TransformerEmbeddings(Embeddings):
    def __init__(self, lm_name):
        super().__init__()
        # load transformer
        self._tok = transformers.AutoTokenizer.from_pretrained(lm_name, use_fast=True, add_prefix_space=True)
        self._lm = transformers.AutoModel.from_pretrained(lm_name, return_dict=True)

        # move model to GPU if available
        if torch.cuda.is_available():
            self._lm.to(torch.device('cuda'))

        # add special tokens
        self._tok.add_special_tokens({'additional_special_tokens': ['<E1>', '</E1>', '<E2>', '</E2>']})
        self._lm.resize_token_embeddings(len(self._tok))

        # public variables
        self.emb_dim = self._lm.config.hidden_size

    def embed(self, sentences):
        embeddings = []
        emb_words, att_words = self.forward(sentences)
        # gather non-padding embeddings per sentence into list
        for sidx in range(len(sentences)):
            embeddings.append(emb_words[sidx, :len(sentences[sidx]), :].cpu().numpy())
        return embeddings

    def forward(self, sentences):
        tok_sentences = self.tokenize(sentences)
        model_inputs = {k: tok_sentences[k] for k in ['input_ids', 'token_type_ids', 'attention_mask'] if k in tok_sentences}

        # perform embedding forward pass
        model_outputs = self._lm(**model_inputs, output_hidden_states=True)

        # extract embeddings from relevant layer
        hidden_states = model_outputs.hidden_states
        emb_pieces = hidden_states[-1]

        # return model-specific tokenization
        return emb_pieces, tok_sentences['attention_mask'], tok_sentences.encodings

    def tokenize(self, sentences):
        # tokenize batch: {input_ids: [[]], token_type_ids: [[]], attention_mask: [[]], special_tokens_mask: [[]]}
        tok_sentences = self._tok(
            [sentence.split(' ') for sentence in sentences],
            is_split_into_words=True, padding=True, truncation=True, return_tensors='pt'
        )
        # move input to GPU (if available)
        if torch.cuda.is_available():
            tok_sentences = tok_sentences.to(torch.device('cuda'))

        return tok_sentences


#
# Pooling Function
#


def get_marker_embeddings(token_embeddings, encodings, ent1, ent2):

    if torch.cuda.is_available():
        start_markers = torch.Tensor().to(torch.device('cuda'))
    else:
        start_markers = torch.Tensor()

    for embedding, word_id in zip(token_embeddings, encodings.word_ids):
        if word_id == ent1:
            start_markers = torch.cat([start_markers, embedding])
        elif word_id == ent2:
            start_markers = torch.cat([start_markers, embedding])

    return start_markers

def get_marker_embeddingsDoubleMarkers(token_embeddings, encodings, ent1, ent2, ent1end, ent2end):

    if torch.cuda.is_available():
        start_markers = torch.Tensor().to(torch.device('cuda'))
    else:
        start_markers = torch.Tensor()

    for embedding, word_id in zip(token_embeddings, encodings.word_ids):
        if word_id == ent1:
            start_markers = torch.cat([start_markers, embedding])
        elif word_id == ent2:
            start_markers = torch.cat([start_markers, embedding])
        elif word_id == ent1end:
            start_markers = torch.cat([start_markers, embedding])
        elif word_id == ent2end:
            start_markers = torch.cat([start_markers, embedding])

    return start_markers

