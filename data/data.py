import os
from pathlib import Path
import csv
import datasets
from datasets.tasks import AutomaticSpeechRecognition

class TestDatasetConfig(datasets.BuilderConfig):
  

    def __init__(self, **kwargs):
        super(TestDatasetConfig, self).__init__(version=datasets.Version("2.0.1", ""), **kwargs)



class TestDataset(datasets.GeneratorBasedBuilder):
    """data dataset."""

    BUILDER_CONFIGS = [TestDatasetConfig(name="testd", description="'Clean' speech.")]


    def _info(self):
        return datasets.DatasetInfo(

            features=datasets.Features(
                {
                    "path": datasets.Value("string"),
                    "audio": datasets.Audio(sampling_rate=16_000),
                    "sentence": datasets.Value("string"),
                                    }
            ),
            supervised_keys=("path", "sentence"),

            task_templates=[AutomaticSpeechRecognition(audio_column="audio", transcription_column="sentence")],
        )


    def _split_generators(self, dl_manager):

        path_to_data = os.path.abspath(os.path.expanduser(dl_manager.manual_dir))

        if not os.path.exists(path_to_data):
            raise FileNotFoundError(
                f"{path_to_data} does not exist. Make sure you insert a manual dir via `datasets.load_dataset('data', data_dir=...)` that includes files unzipped from the TIMIT zip. Manual download instructions: {self.manual_download_instructions}"
            )


        path_to_clips = sorted(Path(path_to_data).glob(f"**/*.wav"))


        metadata_filepaths = {
            split: "/".join([path_to_data, f"{split}.csv"])
            for split in ["train", "validation"]
        }

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"metadata_filepath": metadata_filepaths["train"],
                                                                             "path_to_clips": path_to_clips}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"metadata_filepath": metadata_filepaths["validation"],
                                                                             "path_to_clips": path_to_clips}),
        ]

    def _generate_examples(self, metadata_filepath, path_to_clips):

        data_fields = list(self._info().features.keys())

        # audio is not a header of the csv files

        path_idx = data_fields.index("path")

        all_field_values = {}
        metadata_found = False


        with open(metadata_filepath, 'r') as csvfile:
            datareader = csv.reader(csvfile)

            for itr, row in enumerate(datareader):
                if itr == 0:

                    column_names = row
                    assert (
                            column_names == data_fields
                    ), f"The file should have {data_fields} as column names, but has {column_names}"
                else:
                    field_values = row
                    print(field_values)
                    result = {key: value for key, value in zip(data_fields, field_values)}


                    yield result['path'], result

def with_case_insensitive_suffix(path: Path, suffix: str):
    path = path.with_suffix(suffix.lower())
    path = path if path.exists() else path.with_suffix(suffix.upper())
    return path
