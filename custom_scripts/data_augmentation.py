import argparse
import json
import random
from typing import List, Dict, Tuple

import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def set_seed(seed: int = 42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Ensures deterministic behavior


def data_generate(
    examples: List[Dict[str, str]],
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    num_seqs: int = 5,
) -> List[Tuple[str, str, List[str]]]:
    """
    Generates multiple possible titles for each abstract using a language model.

    Args:
        examples (List[Dict[str, str]]): A list of dictionaries with keys 'title' and 'abstract'.
        tokenizer (PreTrainedTokenizer): Tokenizer for text processing.
        model (PreTrainedModel): Model to generate text.
        num_seqs (int, optional): Number of title variations to generate per abstract. Defaults to 5.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        List[Tuple[str, str, List[str]]]: A list of tuples where each tuple contains a lang, and original title
                                     and a list of generated titles.
    """

    abstracts = [ex['abstract'] for ex in examples]
    ids = [str(ex['id']) for ex in examples]

    inputs = tokenizer(
        abstracts,
        return_tensors='pt',
        max_length=512,
        padding=True,
        truncation=True
    ).to(model.device)

    model_outputs = model.generate(
        **inputs,
        max_length=512,
        no_repeat_ngram_size=2,
        num_beams=2 * num_seqs,
        num_return_sequences=num_seqs
    )

    gen_queries = tokenizer.batch_decode(
        model_outputs, skip_special_tokens=True)
    
    gen_by_example = [
        gen_queries[i * num_seqs: (i + 1) * num_seqs] for i in range(len(examples))]

    return [(ids[i], gen_by_example[i]) for i in range(len(ids))]


class BatchFileReader:
    def __init__(self, file_path: str,
                 batch_size: int,
                 debug: bool = False,
                 max_calls_if_debug: int = 10):
        self.file_path = file_path
        self.batch_size = batch_size

        self.file = open(file_path, 'r')
        self.num_lines = self._count_lines()

        self.num_batches = self.num_lines // self.batch_size
        self.num_batches += 1 if self.num_lines % self.batch_size else 0

        self.num_calls = 0
        self.max_calls = self.num_batches if not debug else max_calls_if_debug
        self.debug = debug

        if debug:
            self.num_batches = min(self.num_batches, max_calls_if_debug)
            self.num_lines = self.num_batches * self.batch_size

    def _count_lines(self):
        with open(self.file_path, 'r') as f:
            return sum(1 for _ in f)

    def read(self) -> List[str]:
        if self.num_calls >= self.num_batches:
            self.close()
            return []

        batch_content = [self.file.readline().strip()
                         for _ in range(self.batch_size)]
        self.num_calls += 1

        return [line for line in batch_content if line]

    def close(self):
        self.file.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,
                        required=True, help='Path to input file.')
    parser.add_argument('--output_file', type=str,
                        required=True, help='Path to output file.')
    parser.add_argument('--model_name', type=str, required=False,
                        default='google/mt5-base', help='Name of the model to use.')
    parser.add_argument('--model_path', type=str,
                        required=True, help='Path to the model to use.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for processing.')
    parser.add_argument('--num_seqs', type=int, default=5,
                        help='Number of title variations to generate per abstract.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility.')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')
    parser.add_argument(
        '--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu', help='Device to use.')
    args = parser.parse_args()

    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)

    model.to(args.device)

    reader = BatchFileReader(args.input_file, args.batch_size, args.debug)

    with open(args.output_file, 'w') as f:
        for _ in tqdm(range(reader.num_batches)):
            batch = reader.read()
            if not batch:
                break

            examples = [json.loads(line) for line in batch]
            for docid, gen_queries in data_generate(examples, tokenizer, model, args.num_seqs):
                for query in gen_queries:
                    f.write(json.dumps({'docid': docid, 'query': query}) + '\n')


if __name__ == '__main__':
    main()
