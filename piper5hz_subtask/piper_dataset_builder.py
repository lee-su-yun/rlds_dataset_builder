from typing import Iterator, Tuple, Any
from PIL import Image
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub


class Piper(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.5.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '1.5.0': '5hz.',
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
                            shape=(14,),
                            dtype=np.int64,
                            doc='Robot state, consists of [6x robot joint data, 6x end pose data, '
                            '2x gripper data].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.int64,
                        doc='Robot action, consists of [6x end pose data, '
                            '1x gripper data].',
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
                        doc='Align cups.'
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
                        doc='/sdb1/piper/'
                    ),
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/sdb1/piper_5hz'),

        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i in range(len(data['index'])):
                # compute Kona language embedding
                language_embedding = self._embed(['Align cups'])[0].numpy()
                ep =episode_path.split('/')[-2]
                img = Image.open(f'/sdb1/piper/{ep}/exo/color_img_{6*i}.jpeg')
                img2 = Image.open(f'/sdb1/piper/{ep}/wrist/color_img_{6*i}.jpeg')

                episode.append({
                    'observation': {
                        'image': np.array(img),
                        'wrist_image': np.array(img2),
                        'state': np.concatenate((data['state']['joint_data'][i],data['state']['end_pose_data'][i],
                                                 data['state']['gripper_data'][i],
                                                 )),
                    },
                    'action': np.concatenate((data['action']['end_pose_data'][i], [data['action']['gripper_data'][i][0]])),
                    'discount': 1.0,
                    'reward': float(i == (len(data['index']) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data['index']) - 1),
                    'is_terminal': i == (len(data['index']) - 1),
                    'language_instruction': 'Align cups.',
                    'language_embedding': language_embedding,
                })

            # create output data sample
            sample = {
                'steps': episode,
                'episode_metadata': {
                    'file_path': episode_path
                }
            }

            # if you want to skip an example for whatever reason, simply return None
            return episode_path, sample

        # create list of all examples
        episode_paths = glob.glob(f"{path}/aligncups_episode*/*.pickle")

        # for smallish datasets, use single-thread parsing
        for sample in episode_paths:
            yield _parse_example(sample)

        # for large datasets use beam to parallelize data parsing (this will have initialization overhead)
        # beam = tfds.core.lazy_imports.apache_beam
        # return (
        #         beam.Create(episode_paths)
        #         | beam.Map(_parse_example)
        # )

