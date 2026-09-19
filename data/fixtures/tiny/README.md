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

The text payloads and JSON manifest are committed with LF line endings. Repository
attributes force LF checkout for fixture `.txt` and `.json` files, including when
`core.autocrlf=true`. Their original byte counts and SHA-256 hashes remain binding;
acquisition verifies raw bytes before any text cleaning. Do not regenerate the
manifest or normalize data at verification time to accommodate checkout changes.
