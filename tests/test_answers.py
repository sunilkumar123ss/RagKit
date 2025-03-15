"""Answer tests use local evidence and deterministic model responses, never paid APIs."""
from copy import deepcopy
import json
import tempfile
import unittest
from unittest.mock import patch

from ragkit.answers import ABSTENTION, MAX_CONTEXT_CHARS, MAX_QUERY_CHARS, answer
from ragkit.engine import RagKit


def source(text='Café evidence supports local retrieval.', index=1):
    return {'chunk_id': f'doc{index}:0', 'document_id': f'doc{index}', 'title': 'Synthetic note',
            'source': 'fixture.md', 'text': text, 'score': 1.0, 'lexical_score': 1.0, 'dense_score': 0.0,
            'citation': {'document_id': f'doc{index}', 'field': 'text', 'start': 3,
                         'end': 3 + len(text), 'quote': text}}


class StubEngine:
    def __init__(self, results=None):
        self.results = [source()] if results is None else results
        self.calls = []

    def search(self, query, **options):
        self.calls.append((query, options))
        return {'query': query, 'mode': options.get('mode', 'lexical'),
                'abstained': not self.results, 'results': self.results}


class AnswerTests(unittest.TestCase):
    def test_extractive_answer_has_exact_unicode_citations_and_no_model(self):
        with tempfile.TemporaryDirectory() as directory, RagKit(directory) as engine:
            original = '  Café 🦊 evidence\r\nwith exact offsets.'
            engine.ingest_text(original, 'Unicode note', document_id='unicode')
            with patch('ragkit.providers.generate') as generate:
                result = answer(engine, 'Café')
            self.assertFalse(result['abstained'])
            self.assertIn('Extractive answer', result['answer'])
            for citation in result['citations']:
                self.assertEqual(original[citation['start']:citation['end']], citation['quote'])
                self.assertIn('[' + citation['id'] + ']', result['answer'])
            generate.assert_not_called()

    def test_no_evidence_abstains_for_unconfigured_model_without_call(self):
        with patch.dict('os.environ', {}, clear=True), patch('ragkit.providers.generate') as generate:
            result = answer(StubEngine([]), 'unknown', provider='openai')
        self.assertTrue(result['abstained'])
        self.assertEqual(result['citations'], [])
        self.assertEqual(result['results'], [])
        generate.assert_not_called()

    def test_options_and_results_are_preserved(self):
        engine = StubEngine()
        before = deepcopy(engine.results)
        result = answer(engine, 'evidence', top_k=3, mode='hybrid', rerank=True, document_ids=['doc1'])
        self.assertEqual(engine.calls, [('evidence', {'top_k': 3, 'mode': 'hybrid', 'rerank': True, 'document_ids': ['doc1']})])
        self.assertEqual(engine.results, before)
        self.assertEqual(result['results'], before)

    def test_model_accepts_known_inline_references_and_selects_citations(self):
        engine = StubEngine([source(index=1), source(index=2)])
        raw = json.dumps({'answer': 'Local retrieval is supported [S2].', 'citations': ['S2'], 'abstained': False})
        with patch('ragkit.providers.generate', return_value=raw) as generate:
            result = answer(engine, 'retrieval', provider='anthropic')
        self.assertEqual([c['id'] for c in result['citations']], ['S2'])
        self.assertEqual(len(result['results']), 2)
        self.assertEqual(generate.call_args.args[0], 'anthropic')

    def test_unknown_and_malformed_references_are_rejected(self):
        bad = [(['S999'], 'Fact [S999]'), (['S1'], 'Fact [S1] and [S999]'),
               (['S1'], 'Fact [S1] and [s999]'), (['S1', 'S1'], 'Fact [S1]'),
               (['S1'], 'Fact [S1, S2]')]
        for refs, text in bad:
            with self.subTest(refs=refs, text=text), patch('ragkit.providers.generate', return_value=json.dumps(
                    {'answer': text, 'citations': refs, 'abstained': False})):
                with self.assertRaises(RuntimeError):
                    answer(StubEngine(), 'retrieval', provider='openai')

    def test_missing_references_and_wrong_schema_fail(self):
        cases = ['not JSON', '[]', '{}', json.dumps({'answer': 'Uncited claim', 'citations': [], 'abstained': False}),
                 json.dumps({'answer': 'Claim', 'citations': ['S1'], 'abstained': False}),
                 json.dumps({'answer': 'Claim [S1]', 'citations': 'S1', 'abstained': False}),
                 json.dumps({'answer': 'Claim [S1]', 'citations': ['S1'], 'abstained': 'false'})]
        for raw in cases:
            with self.subTest(raw=raw), patch('ragkit.providers.generate', return_value=raw):
                with self.assertRaises(RuntimeError):
                    answer(StubEngine(), 'retrieval', provider='ollama')

    def test_model_abstention_returns_standard_message(self):
        with patch('ragkit.providers.generate', return_value=json.dumps(
                {'answer': 'The sources cannot answer this.', 'citations': [], 'abstained': True})):
            result = answer(StubEngine(), 'retrieval', provider='openai')
        self.assertTrue(result['abstained'])
        self.assertEqual(result['answer'], ABSTENTION)
        self.assertEqual(result['citations'], [])

    def test_model_cannot_abstain_with_citations(self):
        with patch('ragkit.providers.generate', return_value=json.dumps(
                {'answer': 'No evidence [S1]', 'citations': ['S1'], 'abstained': True})):
            with self.assertRaisesRegex(RuntimeError, 'abstention'):
                answer(StubEngine(), 'retrieval', provider='openai')

    def test_context_bound_includes_json_escaping_and_preserves_offsets(self):
        evidence = [source('Evidence ' + '\x00"\\\n🦊' * 1000, index=i) for i in range(20)]
        with patch('ragkit.providers.generate', return_value=json.dumps(
                {'answer': 'Evidence is present [S1].', 'citations': ['S1'], 'abstained': False})) as generate:
            result = answer(StubEngine(evidence), 'evidence', top_k=20, provider='openai')
        prompt = json.loads(generate.call_args.args[2])
        self.assertLessEqual(len(json.dumps(prompt['sources'], ensure_ascii=False)), MAX_CONTEXT_CHARS)
        citation = result['citations'][0]
        self.assertEqual(citation['quote'], evidence[0]['text'][:citation['end'] - citation['start']])
        self.assertEqual(len(result['results']), 20)

    def test_invalid_core_citation_does_not_reach_provider(self):
        invalid = source()
        invalid['citation']['end'] += 1
        with patch('ragkit.providers.generate') as generate:
            with self.assertRaisesRegex(RuntimeError, 'invalid source citation'):
                answer(StubEngine([invalid]), 'evidence', provider='openai')
        generate.assert_not_called()

    def test_whitespace_only_prefix_does_not_hide_later_evidence(self):
        evidence = [source(' ' * 2000 + 'tail', index=1), source('Useful evidence.', index=2)]
        result = answer(StubEngine(evidence), 'evidence')
        self.assertFalse(result['abstained'])
        self.assertEqual([citation['document_id'] for citation in result['citations']], ['doc2'])

    def test_bad_query_or_provider_rejected_before_search(self):
        engine = StubEngine()
        for query in ['', '   ', None, 'x' * (MAX_QUERY_CHARS + 1)]:
            with self.subTest(query=str(query)[:20]), self.assertRaises(ValueError):
                answer(engine, query)
        with self.assertRaises(ValueError):
            answer(engine, 'evidence', provider='unknown')
        self.assertEqual(engine.calls, [])

    def test_provider_failure_is_not_a_silent_extractive_fallback(self):
        with patch('ragkit.providers.generate', side_effect=RuntimeError('Provider is unavailable')):
            with self.assertRaisesRegex(RuntimeError, 'unavailable'):
                answer(StubEngine(), 'retrieval', provider='openai')


if __name__ == '__main__':
    unittest.main()
