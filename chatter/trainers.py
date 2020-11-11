import itertools
import os
import csv
import time
from dateutil import parser as date_parser
from chatterbot.conversation import Statement
from chatterbot.tagging import PosLemmaTagger
from chatterbot.trainers import UbuntuCorpusTrainer


class CustomUbuntu(UbuntuCorpusTrainer):
    def train(self):
        import glob

        tagger = PosLemmaTagger(language=self.chatbot.storage.tagger.language)

        # Download and extract the Ubuntu dialog corpus if needed
        corpus_download_path = self.download(self.data_download_url)

        # Extract if the directory does not already exist
        if not self.is_extracted(self.extracted_data_directory):
            self.extract(corpus_download_path)

        extracted_corpus_path = os.path.join(self.extracted_data_directory, "**", "**", "*.tsv")

        # def chunks(items, items_per_chunk):
        #     for aslice in itertools.islice(items, items_per_chunk):
        #         yield aslice
        #     for start_index in range(0, len(items), items_per_chunk):
        #         end_index = start_index + items_per_chunk
        #         yield items[start_index:end_index]

        def grouper(iterable, n, fillvalue=None):
            """Collect data into fixed-length chunks or blocks"""
            # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
            args = [iter(iterable)] * n
            return itertools.zip_longest(*args, fillvalue=fillvalue)

        file_iter = glob.iglob(extracted_corpus_path)

        # file_groups = tuple(chunks(file_list, 10000))

        start_time = time.time()

        # for tsv_files in chunks(file_iter, 1000):

        statements_from_file = []

        group_count = 10000

        for x, tsv_file in enumerate(file_iter, start=1):
            if not x % group_count:
                self.chatbot.storage.create_many(statements_from_file)
                statements_from_file = []

            with open(tsv_file, "r", encoding="utf-8") as tsv:
                reader = csv.reader(tsv, delimiter="\t")

                previous_statement_text = None
                previous_statement_search_text = ""

                for row in reader:
                    if len(row) > 0:
                        statement = Statement(
                            text=row[3],
                            in_response_to=previous_statement_text,
                            conversation="training",
                            created_at=date_parser.parse(row[0]),
                            persona=row[1],
                        )

                        for preprocessor in self.chatbot.preprocessors:
                            statement = preprocessor(statement)

                        statement.search_text = tagger.get_text_index_string(statement.text)
                        statement.search_in_response_to = previous_statement_search_text

                        previous_statement_text = statement.text
                        previous_statement_search_text = statement.search_text

                        statements_from_file.append(statement)


        print("Training took", time.time() - start_time, "seconds.")
