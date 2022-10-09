from multiprocessing import Pool
import spacy
# from spacy.matcher import Matcher
from tqdm import tqdm
import os
import nltk
import json
import string

import scispacy
from scispacy.linking import EntityLinker


# the lemma of it/them/mine/.. is -PRON-

# blacklist = set(["-PRON-", "actually", "likely", "possibly", "want",
#                  "make", "my", "someone", "sometimes_people", "sometimes", "would", "want_to",
#                  "one", "something", "sometimes", "everybody", "somebody", "could", "could_be"
#                  ])
#
#
# nltk.download('stopwords', quiet=True)
# nltk_stopwords = set(nltk.corpus.stopwords.words('english'))

# CHUNK_SIZE = 1

UMLS_VOCAB = None
# PATTERN_PATH = None
nlp = None
# matcher = None
linker = None


def load_umls_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf8") as fin:
        vocab = [l.strip() for l in fin]
    return vocab


def load_entity_linker(threshold=0.90):
    nlp = spacy.load("en_core_sci_sm")
    linker = EntityLinker(
        resolve_abbreviations=True,
        name="umls",
        threshold=threshold)
    nlp.add_pipe(linker)
    return nlp, linker

def entity_linking_to_umls(sentence, nlp, linker):
    doc = nlp(sentence)
    entities = doc.ents
    all_entities_results = []
    for mm in range(len(entities)):
        entity_text = entities[mm].text
        entity_start = entities[mm].start
        entity_end = entities[mm].end
        all_linked_entities = entities[mm]._.kb_ents
        all_entity_results = []
        for ii in range(len(all_linked_entities)):
            curr_concept_id = all_linked_entities[ii][0]
            curr_score = all_linked_entities[ii][1]
            curr_scispacy_entity = linker.kb.cui_to_entity[all_linked_entities[ii][0]]
            curr_canonical_name = curr_scispacy_entity.canonical_name
            curr_TUIs = curr_scispacy_entity.types
            curr_entity_result = {"Canonical Name": curr_canonical_name, "Concept ID": curr_concept_id,
                                  "TUIs": curr_TUIs, "Score": curr_score}
            all_entity_results.append(curr_entity_result)
        curr_entities_result = {"text": entity_text, "start": entity_start, "end": entity_end,
                                "start_char": entities[mm].start_char, "end_char": entities[mm].end_char,
                                "linking_results": all_entity_results}
        all_entities_results.append(curr_entities_result)
    return all_entities_results

def ground_mentioned_concepts(nlp, linker, sent):
    ent_link_results = entity_linking_to_umls(sent, nlp, linker)
      # [{'text': 'nausea',
      #   'start': 17,
      #   'end': 18,
      #   'start_char': 103,
      #   'end_char': 109,
      #   'linking_results': [{'Canonical Name': 'Nausea',
      #     'Concept ID': 'C0027497',
      #     'TUIs': ['T184'],
      #     'Score': 0.9999998807907104},
      #    {'Canonical Name': 'Have Nausea',
      #     'Concept ID': 'C2984057',
      #     'TUIs': ['T170'],
      #     'Score': 0.9999998807907104}]}, ...]
    mentioned_concepts = set()
    for ent_obj in ent_link_results:
        for ent_cand in ent_obj['linking_results']:
            CUI = ent_cand['Concept ID']
            if CUI in UMLS_VOCAB:
                mentioned_concepts.add(CUI)
    return mentioned_concepts

def ground_qa_pair(qa_pair):
    global nlp, linker
    if nlp is None or linker is None:
        print("Loading scispacy...")
        nlp, linker = load_entity_linker()
        print("Loaded scispacy.")

    s, a = qa_pair
    question_concepts = ground_mentioned_concepts(nlp, linker, s)
    answer_concepts = ground_mentioned_concepts(nlp, linker, a)
    question_concepts = question_concepts - answer_concepts
    if len(question_concepts) == 0:
        print(f"for {s}, concept not found in umls linking.")

    if len(answer_concepts) == 0:
        print(f"for {a}, concept not found in umls linking.")

    # question_concepts = question_concepts -  answer_concepts
    question_concepts = sorted(list(question_concepts))
    answer_concepts = sorted(list(answer_concepts))
    return {"sent": s, "ans": a, "qc": question_concepts, "ac": answer_concepts}


def match_mentioned_concepts(sents, answers, num_processes):
    assert len(sents) == len(answers), (len(sents), len(answers))
    res = []
    with Pool(num_processes) as p:
        res = list(tqdm(p.imap(ground_qa_pair, zip(sents, answers)), total=len(sents)))
    return res



def ground_umls(statement_path, umls_vocab_path, output_path, num_processes=1, debug=False):
    global UMLS_VOCAB
    if UMLS_VOCAB is None:
        UMLS_VOCAB = set(load_umls_vocab(umls_vocab_path))

    sents = []
    answers = []
    with open(statement_path, 'r') as fin:
        lines = [line for line in fin]

    if debug:
        lines = lines[192:195]
        print(len(lines))
    for line in lines:
        if line == "":
            continue
        j = json.loads(line)
        # {'answerKey': 'B',
        #   'id': 'b8c0a4703079cf661d7261a60a1bcbff',
        #   'question': {'question_concept': 'magazines',
        #                 'choices': [{'label': 'A', 'text': 'doctor'}, {'label': 'B', 'text': 'bookstore'}, {'label': 'C', 'text': 'market'}, {'label': 'D', 'text': 'train station'}, {'label': 'E', 'text': 'mortuary'}],
        #                 'stem': 'Where would you find magazines along side many other printed works?'},
        #   'statements': [{'label': False, 'statement': 'Doctor would you find magazines along side many other printed works.'}, {'label': True, 'statement': 'Bookstore would you find magazines along side many other printed works.'}, {'label': False, 'statement': 'Market would you find magazines along side many other printed works.'}, {'label': False, 'statement': 'Train station would you find magazines along side many other printed works.'}, {'label': False, 'statement': 'Mortuary would you find magazines along side many other printed works.'}]}

        for statement in j["statements"]:
            sents.append(statement["statement"])

        for answer in j["question"]["choices"]:
            ans = answer['text']
            # ans = " ".join(answer['text'].split("_"))
            try:
                assert all([i != "_" for i in ans])
            except Exception:
                print(ans)
            answers.append(ans)

    res = match_mentioned_concepts(sents, answers, num_processes)

    # check_path(output_path)
    os.system('mkdir -p {}'.format(os.path.dirname(output_path)))
    with open(output_path, 'w') as fout:
        for dic in res:
            fout.write(json.dumps(dic) + '\n')

    print(f'grounded concepts saved to {output_path}')
    print()


if __name__ == "__main__":
    pass
