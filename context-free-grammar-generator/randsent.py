#!/usr/bin/env python3
"""
601.465/665 — Natural Language Processing
Assignment 1: Designing Context-Free Grammars

Assignment written by Jason Eisner
Modified by Kevin Duh
Re-modified by Alexandra DeLucia

Code template written by Alexandra DeLucia,
based on the submitted assignment with Keith Harrigian
and Carlos Aguirre Fall 2019
"""
import os
import sys
import random
import argparse
import re

# Want to know what command-line arguments a program allows?
# Commonly you can ask by passing it the --help option, like this:
#     python randsent.py --help
# This is possible for any program that processes its command-line
# arguments using the argparse module, as we do below.
#
# NOTE: When you use the Python argparse module, parse_args() is the
# traditional name for the function that you create to analyze the
# command line.  Parsing the command line is different from parsing a
# natural-language sentence.  It's easier.  But in both cases,
# "parsing" a string means identifying the elements of the string and
# the roles they play.

def parse_args():
    """
    Parse command-line arguments.

    Returns:
        args (an argparse.Namespace): Stores command-line attributes
    """
    # Initialize parser
    parser = argparse.ArgumentParser(description="Generate random sentences from a PCFG")
    # Grammar file (required argument)
    parser.add_argument(
        "-g",
        "--grammar",
        type=str, required=True,
        help="Path to grammar file",
    )
    # Start symbol of the grammar
    parser.add_argument(
        "-s",
        "--start_symbol",
        type=str,
        help="Start symbol of the grammar (default is ROOT)",
        default="ROOT",
    )
    # Number of sentences
    parser.add_argument(
        "-n",
        "--num_sentences",
        type=int,
        help="Number of sentences to generate (default is 1)",
        default=1,
    )
    # Max number of nonterminals to expand when generating a sentence
    parser.add_argument(
        "-M",
        "--max_expansions",
        type=int,
        help="Max number of nonterminals to expand when generating a sentence",
        default=450,
    )
    # Print the derivation tree for each generated sentence
    parser.add_argument(
        "-t",
        "--tree",
        action="store_true",
        help="Print the derivation tree for each generated sentence",
        default=False,
    )
    return parser.parse_args()

expansions_count = 0

class Grammar:
    def __init__(self, grammar_file):
        """
        Context-Free Grammar (CFG) Sentence Generator

        Args:
            grammar_file (str): Path to a .gr grammar file
        
        Returns:
            self
        """
        # Parse the input grammar file
        self.rules = {}
        self._load_rules_from_file(grammar_file)

    def _load_rules_from_file(self, grammar_file):
        """
        Read grammar file and store its rules in self.rules

        Args:
            grammar_file (str): Path to the raw grammar file 
        """
        with open(grammar_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                line = line.split('#', 1)[0].strip()
                
                parts = re.split(r'\s+', line, maxsplit=2)
                if len(parts) < 3: 
                    continue
                odds, lhs, rhs = parts
                odds = float(odds)
                # Remove any () or comment symbols from the RHS
                rhs_cleaned = re.sub(r'[()]', '', rhs)
                rhs_symbols = tuple(rhs_cleaned.split()) 

                if lhs not in self.rules:
                    self.rules[lhs] = []
                self.rules[lhs].append((odds, rhs_symbols))

    def sample(self, derivation_tree, max_expansions, start_symbol):
        """
        Sample a random sentence from this grammar

        Args:
            derivation_tree (bool): if true, the returned string will represent 
                the tree (using bracket notation) that records how the sentence 
                was derived
                               
            max_expansions (int): max number of nonterminal expansions we allow

            start_symbol (str): start symbol to generate from

        Returns:
            str: the random sentence or its derivation tree
        """
        global expansions_count
        expansions_count = 0 
        def expand(symbol):
            global expansions_count # Refer to global variable

            if expansions_count >= max_expansions:
                return "..." # In order to avoid long or infinite outputs
            if symbol not in self.rules:
                return symbol

            expansions_count += 1
            # Choose a rule based on the odds
            odds_and_rhs = self.rules[symbol]
            rhs_symbols = random.choices(
                population=[rhs for _, rhs in odds_and_rhs],  # The right-hand sides
                weights=[odds for odds, _ in odds_and_rhs],  # The corresponding odds
                k=1
            )[0]

            # Have to do another check 
            if expansions_count >= max_expansions:
              return "..."  # Truncate entire subtree if limit is reached

            # Recursively expand each symbol in the RHS
            expanded_symbols = [expand(sym) for sym in rhs_symbols]

            if derivation_tree:
                return f"({symbol} {' '.join(expanded_symbols)})"
            else:
                return ' '.join(expanded_symbols)
        return expand(start_symbol)
      


####################
### Main Program
####################
def main():
    # Parse command-line options
    args = parse_args()

    # Initialize Grammar object
    grammar = Grammar(args.grammar)

    # Generate sentences
    for i in range(args.num_sentences):
        # Use Grammar object to generate sentence
        sentence = grammar.sample(
            derivation_tree=args.tree,
            max_expansions=args.max_expansions,
            start_symbol=args.start_symbol
        )

        # Print the sentence with the specified format.
        # If it's a tree, we'll pipe the output through the prettyprint script.
        if args.tree:
            prettyprint_path = os.path.join(os.getcwd(), 'prettyprint')
            prettyprint_path = f'"{prettyprint_path}"'
            t = os.system(f"echo '{sentence}' | perl {prettyprint_path}")
        else:
            print(sentence)


if __name__ == "__main__":
    main()
