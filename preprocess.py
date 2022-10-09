import argparse
from multiprocessing import cpu_count
from preprocess_utils.convert_csqa import convert_to_entailment
from preprocess_utils.convert_obqa import convert_to_obqa_statement
from preprocess_utils.conceptnet import extract_english, construct_graph
from preprocess_utils.umls import construct_graph_umls
from preprocess_utils.grounding import create_matcher_patterns, ground
from preprocess_utils.grounding_umls import ground_umls
from preprocess_utils.graph import generate_adj_data_from_grounded_concepts
from preprocess_utils.graph_with_glove import generate_adj_data_from_grounded_concepts__use_glove
from preprocess_utils.graph_with_LM import generate_adj_data_from_grounded_concepts__use_LM
from preprocess_utils.graph_umls_with_glove import generate_adj_data_from_grounded_concepts_umls__use_glove


input_paths = {
    'cpnet': {
        'csv': './data/cpnet/conceptnet-assertions-5.6.0.csv',
    },
    'csqa': {
        'train': './data/csqa/train_rand_split.jsonl',
        'dev': './data/csqa/dev_rand_split.jsonl',
        'test': './data/csqa/test_rand_split_no_answers.jsonl',
    },
    'riddle': {
        'train': './data/riddle/train.jsonl',
        'dev':   './data/riddle/devIH.jsonl',
        'test':  './data/riddle/testIH.jsonl',
        'test_hidden':  './data/riddle/test_hidden.jsonl',
    },
    'obqa': {
        'train': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/train.jsonl',
        'dev': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/dev.jsonl',
        'test': './data/obqa/OpenBookQA-V1-Sep2018/Data/Main/test.jsonl',
    },
}


output_paths = {
    'cpnet': {
        'csv': './data/cpnet/conceptnet.en.csv',
        'vocab': './data/cpnet/concept.txt',
        'patterns': './data/cpnet/matcher_patterns.json',
        'unpruned-graph': './data/cpnet/conceptnet.en.unpruned.graph',
        'pruned-graph': './data/cpnet/conceptnet.en.pruned.graph',
    },
    'umls': {
        'csv': './data/umls/umls.csv',
        'vocab': './data/umls/concepts.txt',
        'rel': './data/umls/relations.txt',
        'graph': './data/umls/umls.graph',
    },
    'csqa': {
        'statement': {
            'train': './data/csqa/statement/train.statement.jsonl',
            'dev': './data/csqa/statement/dev.statement.jsonl',
            'test': './data/csqa/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': './data/csqa/grounded/train.grounded.jsonl',
            'dev': './data/csqa/grounded/dev.grounded.jsonl',
            'test': './data/csqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/csqa/graph/train.graph.adj.pk',
            'adj-dev': './data/csqa/graph/dev.graph.adj.pk',
            'adj-test': './data/csqa/graph/test.graph.adj.pk',
        },
    },
    'riddle': {
        'statement': {
            'train': './data/riddle/statement/train.statement.jsonl',
            'dev':   './data/riddle/statement/dev.statement.jsonl',
            'test':  './data/riddle/statement/test.statement.jsonl',
            'test_hidden':  './data/riddle/statement/test_hidden.statement.jsonl',
        },
        'grounded': {
            'train': './data/riddle/grounded/train.grounded.jsonl',
            'dev':   './data/riddle/grounded/dev.grounded.jsonl',
            'test':  './data/riddle/grounded/test.grounded.jsonl',
            'test_hidden':  './data/riddle/grounded/test_hidden.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/riddle/graph/train.graph.adj.pk',
            'adj-dev':   './data/riddle/graph/dev.graph.adj.pk',
            'adj-test':  './data/riddle/graph/test.graph.adj.pk',
            'adj-test_hidden':  './data/riddle/graph/test_hidden.graph.adj.pk',
        },
    },
    'obqa': {
        'statement': {
            'train': './data/obqa/statement/train.statement.jsonl',
            'dev': './data/obqa/statement/dev.statement.jsonl',
            'test': './data/obqa/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': './data/obqa/grounded/train.grounded.jsonl',
            'dev': './data/obqa/grounded/dev.grounded.jsonl',
            'test': './data/obqa/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': './data/obqa/graph/train.graph.adj.pk',
            'adj-dev': './data/obqa/graph/dev.graph.adj.pk',
            'adj-test': './data/obqa/graph/test.graph.adj.pk',
        },
    },
}

for dname in ['medqa']:
    output_paths[dname] = {
        'statement': {
            'train': f'./data/{dname}/statement/train.statement.jsonl',
            'dev':   f'./data/{dname}/statement/dev.statement.jsonl',
            'test':  f'./data/{dname}/statement/test.statement.jsonl',
        },
        'grounded': {
            'train': f'./data/{dname}/grounded/train.grounded.jsonl',
            'dev':   f'./data/{dname}/grounded/dev.grounded.jsonl',
            'test':  f'./data/{dname}/grounded/test.grounded.jsonl',
        },
        'graph': {
            'adj-train': f'./data/{dname}/graph/train.graph.adj.pk',
            'adj-dev':   f'./data/{dname}/graph/dev.graph.adj.pk',
            'adj-test':  f'./data/{dname}/graph/test.graph.adj.pk',
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', default=['common', 'csqa', 'obqa'], nargs='+')
    parser.add_argument('-p', '--nprocs', type=int, default=cpu_count(), help='number of processes to use')
    parser.add_argument('--debug', action='store_true', help='enable debug mode')

    args = parser.parse_args()
    if args.debug:
        raise NotImplementedError()

    routines = {
        'common': [
            {'func': extract_english, 'args': (input_paths['cpnet']['csv'], output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'])},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['unpruned-graph'], False)},
            {'func': construct_graph, 'args': (output_paths['cpnet']['csv'], output_paths['cpnet']['vocab'],
                                               output_paths['cpnet']['pruned-graph'], True)},
            {'func': create_matcher_patterns, 'args': (output_paths['cpnet']['vocab'], output_paths['cpnet']['patterns'])},
        ],
        'umls': [
            {'func': construct_graph_umls, 'args': (output_paths['umls']['csv'], output_paths['umls']['vocab'], output_paths['umls']['rel'], output_paths['umls']['graph'], True)},
        ],
        'csqa': [
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['train'], output_paths['csqa']['statement']['train'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['dev'], output_paths['csqa']['statement']['dev'])},
            {'func': convert_to_entailment, 'args': (input_paths['csqa']['test'], output_paths['csqa']['statement']['test'])},
            {'func': ground, 'args': (output_paths['csqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['csqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['csqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['csqa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['csqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['csqa']['graph']['adj-test'], args.nprocs)},
        ],

        'riddle': [
            {'func': convert_to_entailment, 'args': (input_paths['riddle']['train'], output_paths['riddle']['statement']['train'])},
            {'func': convert_to_entailment, 'args': (input_paths['riddle']['dev'], output_paths['riddle']['statement']['dev'])},
            {'func': convert_to_entailment, 'args': (input_paths['riddle']['test'], output_paths['riddle']['statement']['test'])},
            {'func': convert_to_entailment, 'args': (input_paths['riddle']['test_hidden'], output_paths['riddle']['statement']['test_hidden'])},
            {'func': ground, 'args': (output_paths['riddle']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['riddle']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['riddle']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['riddle']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['riddle']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['riddle']['grounded']['test'], args.nprocs)},
            {'func': ground, 'args': (output_paths['riddle']['statement']['test_hidden'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['riddle']['grounded']['test_hidden'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['riddle']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['riddle']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['riddle']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['riddle']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['riddle']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['riddle']['graph']['adj-test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts, 'args': (output_paths['riddle']['grounded']['test_hidden'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['riddle']['graph']['adj-test_hidden'], args.nprocs)},
        ],

        'obqa': [
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['train'], output_paths['obqa']['statement']['train'])},
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['dev'], output_paths['obqa']['statement']['dev'])},
            {'func': convert_to_obqa_statement, 'args': (input_paths['obqa']['test'], output_paths['obqa']['statement']['test'])},
            {'func': ground, 'args': (output_paths['obqa']['statement']['train'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['train'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['dev'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['dev'], args.nprocs)},
            {'func': ground, 'args': (output_paths['obqa']['statement']['test'], output_paths['cpnet']['vocab'],
                                      output_paths['cpnet']['patterns'], output_paths['obqa']['grounded']['test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['train'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['dev'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts__use_LM, 'args': (output_paths['obqa']['grounded']['test'], output_paths['cpnet']['pruned-graph'], output_paths['cpnet']['vocab'], output_paths['obqa']['graph']['adj-test'], args.nprocs)},
        ],

        'medqa': [
            {'func': ground_umls, 'args': (output_paths['medqa']['statement']['dev'], output_paths['umls']['vocab'], output_paths['medqa']['grounded']['dev'], args.nprocs)},
            {'func': ground_umls, 'args': (output_paths['medqa']['statement']['test'], output_paths['umls']['vocab'], output_paths['medqa']['grounded']['test'], args.nprocs)},
            {'func': ground_umls, 'args': (output_paths['medqa']['statement']['train'], output_paths['umls']['vocab'], output_paths['medqa']['grounded']['train'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts_umls__use_glove, 'args': (output_paths['medqa']['grounded']['dev'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['medqa']['graph']['adj-dev'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts_umls__use_glove, 'args': (output_paths['medqa']['grounded']['test'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['medqa']['graph']['adj-test'], args.nprocs)},
            {'func': generate_adj_data_from_grounded_concepts_umls__use_glove, 'args': (output_paths['medqa']['grounded']['train'], output_paths['umls']['graph'], output_paths['umls']['vocab'], output_paths['medqa']['graph']['adj-train'], args.nprocs)},
        ],
    }

    for rt in args.run:
        for rt_dic in routines[rt]:
            rt_dic['func'](*rt_dic['args'])

    print('Successfully run {}'.format(' '.join(args.run)))


if __name__ == '__main__':
    main()
    # pass
