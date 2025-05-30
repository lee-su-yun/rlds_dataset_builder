from typing import Iterator, Tuple, Any
from PIL import Image
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class Piper5HZ_subtask(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('5.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '2.0.0': 'Validation',
      '2.5.0': 'Add Instruct(fine-tuning)',
      '3.0.0': 'Add validation episodes',
      '4.0.0': 'Change side-view into Top-View(train)',
      '4.5.0': 'Change instruction (remove plastic) (train) + remove first 5 frames(5Hz)',
      '5.0.0': 'Add three views + RGB change'

    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Main camera RGB observation.',
                        ),
                        'wrist_image': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='jpeg',
                            doc='Wrist camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(7,),
                            dtype=np.int64,
                            doc='Robot state',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.int64,
                        doc='Robot action',
                    ),
                    'discount': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Discount if provided, default to 1.'
                    ),
                    'reward': tfds.features.Scalar(
                        dtype=np.float32,
                        doc='Reward if provided, 1 on final step for demos.'
                    ),
                    'is_first': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on first step of the episode.'
                    ),
                    'is_last': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode.'
                    ),
                    'is_terminal': tfds.features.Scalar(
                        dtype=np.bool_,
                        doc='True on last step of the episode if it is a terminal step, True for demos.'
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Pick cups.'
                    ),
                    'language_embedding': tfds.features.Tensor(
                        shape=(512,),
                        dtype=np.float32,
                        doc='Kona language embedding. '
                            'See https://tfhub.dev/google/universal-sentence-encoder-large/5'
                    ),
                }),
                'episode_metadata': tfds.features.FeaturesDict({
                    'file_path': tfds.features.Text(
                        doc='/sdb1/piper_5hz_subtask/train/pick'
                    ),
                    'episode_id': tfds.features.Text(
                        doc='episode_id'
                    )
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/sdb1/piper_subtask_data/train/pick'),
            # 'val': self._generate_examples(path='/sdb1/piper_5hz/validation'),

        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

            episode = []
            instruction = episode_path.split("/")[-3].replace("_", " ").capitalize()
            language_embedding = self._embed([instruction])[0].numpy()

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i in range(30, len(data['index']), 6):
                # compute Kona language embedding
                #language_embedding = self._embed(['Pick the cup'])[0].numpy()
                ep =episode_path.split('/')[-2]
                # img = Image.open(f'{path}/{ep}/exo/color_img_{6*i}.jpeg')
                # img2 = Image.open(f'{path}/{ep}/wrist/color_img_{6*i}.jpeg')
                # print(data['observation.state'].shape)
                # print(data['observation.images.table'].shape)
                # print(data['action'].shape)
                img_exo = np.array(data['observation.images.exo'][i][0])
                img_wrist = np.array(data['observation.images.wrist'][i][0])
                img_table = np.array(data['observation.images.table'][i][0])
                img_exo = img_exo[...,::-1]
                img_wrist = img_wrist[...,::-1]
                img_table = img_table[...,::-1]


                episode.append({
                    'observation': {
                        'exo_image': img_exo,
                        'wrist_image': img_wrist,
                        'table_image': img_table,
                        'state': data['observation.state'][i][0]
                    },
                    'action': data['action'][i][0],
                    'discount': 1.0,
                    'reward': float(i == (len(data['index']) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data['index']) - 1),
                    'is_terminal': i == (len(data['index']) - 1),
                    'language_instruction': instruction,
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path,
                    'episode_id': str(data['episode_index']),
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(f"{path}/*/*/episode.pickle")
        #episode_paths = episode_paths[:20]

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )



