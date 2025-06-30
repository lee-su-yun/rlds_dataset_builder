from typing import Iterator, Tuple, Any
from PIL import Image
import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import cv2

class Piper5HZ_subtask(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('6.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
      '2.0.0': 'Validation',
      '2.5.0': 'Add Instruct(fine-tuning)',
      '3.0.0': 'Add validation episodes',
      '4.0.0': 'Change side-view into Top-View(train)',
      '4.5.0': 'Change instruction (remove plastic) (train) + remove first 5 frames(5Hz)',
      '5.0.0': '4.5.0 + Table views + RGB change (train)',
      '5.5.0': 'Table views + RGB change + remove first 5 frames(5Hz) + Change instruction (Validation)',
      '6.0.0': 'Pick the grape and put it in the basket. training dataset (5hz, table, wrist view)'
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
                        doc='pick the grape and put it to the basket'
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
                        doc='/sdc1/piper_grape0627/pick the grape and put it to the basket'
                    ),
                    'episode_id': tfds.features.Text(
                        doc='episode_id'
                    )
                }),
            }))

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Define data splits."""
        return {
            'train': self._generate_examples(path='/sdb1/piper_grape0626/pick the grape and put it to the basket'),
            # 'val': self._generate_examples(path='/sdb1/piper_5hz/validation'),

        }

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        """Generator of examples for each split."""

        def _parse_example(episode_path):
            # load raw data --> this should change for your dataset
            # print(episode_path)
            data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

            # print(data.keys())
            # print(data['frame_index'])
            # print(data['episode_index'])
            # print(len(data['observation.images.table'][0]))
            # exit()
            # exit()
            instruction = 'pick the grape and put it to the basket'
            language_embedding = self._embed([instruction])[0].numpy()

            # assemble episode --> here we're assuming demos so we set reward to 1 at the end
            episode = []
            for i in range(0, len(data['frame_index']), 6):
                # compute Kona language embedding

                # img_wrist = np.array(data['observation.images.wrist'][i][0])
                # img_table = np.array(data['observation.images.table'][i][0])

                # decoded_img = cv2.imdecode(data['observation.images.table'], cv2.IMREAD_COLOR)

                if data['episode_index'][0].item() >= 80:
                    # image = np.asarray(data['observation.images.table'][i], dtype=np.uint8)
                    # wrist = np.asarray(data['observation.images.wrist'][i], dtype=np.uint8)

                    image = cv2.imdecode(data['observation.images.table'][i], cv2.IMREAD_COLOR)
                    wrist_image = cv2.imdecode(data['observation.images.wrist'][i], cv2.IMREAD_COLOR)
                else:
                    image = data['observation.images.table'][i][0]
                    wrist_image = data['observation.images.wrist'][i][0]

                episode.append({
                    'observation': {
                        # 'exo_image': img_exo,
                        'image': image,
                        'wrist_image': wrist_image,
                        'state': data['observation.state'][i][0]
                    },
                    'action': data['action'][i][0],
                    'discount': 1.0,
                    'reward': float(i == (len(data['frame_index']) - 1)),
                    'is_first': i == 0,
                    'is_last': i == (len(data['frame_index']) - 1),
                    'is_terminal': i == (len(data['frame_index']) - 1),
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
        episode_paths = glob.glob(f"{path}/*/episode.pickle")
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



