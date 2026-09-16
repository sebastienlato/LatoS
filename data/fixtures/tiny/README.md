# Tiny English fixture

Original English paragraphs authored with AI assistance for LatoS data-pipeline
tests on 2026-09-16, under the repository's MIT license. These are test inputs,
not an acquired dataset or evidence of language-model quality.

Three documents have fixed splits. The training document intentionally repeats
one test paragraph exactly and another with a single word changed. Preparation
keeps the held-out versions and removes the two training duplicates. All other
paragraphs should remain. Each document and group belongs to one split only.

The fixture is included in the source distribution for offline checks. It is
separate from `english-books-v1` and must not be mixed into held-out experiments.
